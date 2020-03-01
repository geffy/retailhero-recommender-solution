import pandas as pd
import numpy as np
import json
import catboost
from catboost import CatBoost, Pool, MetricVisualizer
from src.utils import normalized_average_precision


def build_pool(df):
    num_names = [x for x in df.columns if not x.startswith("_")]
    return Pool(
        data=df[num_names],
        cat_features=[
            "product_truncated_id",
            "product_level1_id",
            "product_level2_id",
            "product_level3_id",
            "product_level4_id",
            "product_segment_id",
            "product_vendor_id",
            "product_brand_id",
        ],
        label=df["_label"],
        group_id=df["_groupId"],
    )


def estimate_map(part_15_preds):
    scores = []
    offset = 0
    for cur_aux in (json.loads(l) for l in open("../tmp/features/train_15_v4.3_aux.jsons")):
        _from = offset
        _to = _from + len(cur_aux["candidates"])

        cur_preds = part_15_preds[_from:_to]

        recommendations = np.array(cur_aux["candidates"])[np.argsort(-cur_preds)][:30]
        gt = cur_aux["gt"]
        scores.append(normalized_average_precision(gt, recommendations))

        offset += len(cur_aux["candidates"])
    assert offset == len(part_15_preds)
    return np.mean(scores)


if __name__ == "__main__":
    root = "../tmp/features/"

    train_pool = build_pool(
        pd.concat(
            (
                pd.read_parquet(root + "/train_8_v4.3_features.parquet"),
                pd.read_parquet(root + "/train_9_v4.3_features.parquet"),
                pd.read_parquet(root + "/train_10_v4.3_features.parquet"),
                pd.read_parquet(root + "/train_11_v4.3_features.parquet"),
            )
        )
    )
    valid_pool = build_pool(pd.read_parquet(root + "/train_15_v4.3_features.parquet"))
    train_pool.set_weight(train_pool.get_label() * 9 + 1)

    parameters = {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "QuerySoftMax",
        "grow_policy": "Depthwise",
        "score_function": "Cosine",
        "min_data_in_leaf": 20,
        "custom_metric": ["MAP:top=30"],
        "verbose": False,
        "random_seed": 44,
        "task_type": "GPU",
        "max_ctr_complexity": 2,
    }

    model = CatBoost(parameters)
    model.fit(train_pool, eval_set=valid_pool, plot=True)
    model.save_model("../tmp/model_v4.3_reranker.cbm")
    print("Estimated map: {}".format(estimate_map(model.predict(valid_pool))))
