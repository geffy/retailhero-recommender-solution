import pandas as pd
import lightgbm as lgb
from src.learning_scripts.reranker_catboost import estimate_map


def rle_counts(arr):
    res = []
    prev = arr[0]
    cnt = 1
    for x in arr[1:]:
        if x != prev:
            res.append(cnt)
            cnt = 0
            prev = x
        cnt += 1
    res.append(cnt)
    return res


def build_pool(df, skip_features, ref_pool=None):
    num_names = [x for x in df.columns if not x.startswith("_") and x not in skip_features]
    pool = lgb.Dataset(
        data=df[num_names],
        label=df["_label"].values,
        group=rle_counts(df["_groupId"].values),
        reference=ref_pool,
    )
    return pool


if __name__ == "__main__":

    root = "../tmp/features/"

    skip_features = [
        "product_truncated_id",
        "product_level1_id",
        "product_level2_id",
        "product_level3_id",
        "product_level4_id",
        "product_segment_id",
        "product_vendor_id",
        "product_brand_id",
    ]

    train_pool = build_pool(
        pd.concat(
            (
                pd.read_parquet(root + "/train_8_v4.3_features.parquet"),
                pd.read_parquet(root + "/train_9_v4.3_features.parquet"),
                pd.read_parquet(root + "/train_10_v4.3_features.parquet"),
                pd.read_parquet(root + "/train_11_v4.3_features.parquet"),
            )
        ),
        skip_features,
    )

    valid_pool = build_pool(
        pd.read_parquet(root + "/train_15_v4.3_features.parquet"), skip_features, ref_pool=train_pool
    )

    params = {
        "boosting_type": "gbdt",
        "objective": "lambdarank",
        "metric": {"map", "ndcg"},
        "map_eval_at": 30,
        "ndcg_eval_at": 30,
        "num_leaves": 63,
        "learning_rate": 0.05,
        "bagging_freq": 1,
        "verbose": 1,
        "scale_pos_weight": 100,
        "random_seed": 43,
    }

    print("Starting training...")

    gbm = lgb.train(params, train_pool, num_boost_round=300, valid_sets=valid_pool)

    gbm.save_model("../tmp/model_4.3_reranker.lgb")
