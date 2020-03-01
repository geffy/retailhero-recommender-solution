import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tqdm import tqdm

import src.config as cfg
from src.config import BASE_SPLIT_POINT, PRODUCT_CSV_PATH, PURCHASE_CSV_PATH
import src.config as cfg
from src.featurizer.client_profile import ClientProfile
from src.featurizer.daily import days_between, get_date
from src.utils import iterate_shard

REF_DATE = get_date(BASE_SPLIT_POINT)


def estimate_prices():
    observed_prices = defaultdict(list)

    for df in tqdm(
        pd.read_csv(
            PURCHASE_CSV_PATH, chunksize=1000000, usecols=["product_id", "trn_sum_from_iss", "product_quantity"]
        )
    ):
        for row in df.itertuples():
            if row.product_quantity == 0:
                continue
            _price = row.trn_sum_from_iss / row.product_quantity
            observed_prices[row.product_id].append(_price)

    product_prices = {k: np.median(v) for k, v in observed_prices.items()}
    return product_prices


def estimate_times(ext_products_df, n_shards):
    stats = {}
    for pid in ext_products_df.product_id.values:
        stats[pid] = {"first_seen_day": 200, "last_seen_day": -200, "cnt": 0}

    for shard_id in range(n_shards):
        for js in tqdm(iterate_shard(shard_id)):
            for trans in js["transaction_history"]:
                curr_date = get_date(trans["datetime"])
                days = days_between(REF_DATE, curr_date)
                for product_item in trans["products"]:
                    pid = product_item["product_id"]
                    stats[pid]["cnt"] += 1
                    stats[pid]["first_seen_day"] = min(stats[pid]["first_seen_day"], days)
                    stats[pid]["last_seen_day"] = max(stats[pid]["last_seen_day"], days)
    stats_df = pd.DataFrame.from_dict(stats, orient="index").reset_index()
    return stats_df


def estimate_global_top(n_shards=3):
    cnt = defaultdict(int)
    for shard_id in range(n_shards):
        for js in tqdm(iterate_shard(shard_id)):
            for trans in js["transaction_history"]:
                for product in trans["products"]:
                    cnt[product["product_id"]] += 1

    _tmp = list(cnt.keys())
    return sorted(_tmp, key=lambda x: -cnt[x])


if __name__ == "__main__":
    # estimate global_top
    global_top = estimate_global_top()
    json.dump(global_top, open("../tmp/global_top.json", "w"))

    product_df = pd.read_csv(PRODUCT_CSV_PATH)
    # save product info as parquet for speed
    product_df.to_parquet(cfg.PRODUCT_PARQUET_PATH)

    # add estimated price for every product
    product_prices = estimate_prices()
    product_df["est_price"] = product_df["product_id"].map(product_prices).fillna(-1)

    # add first_seen / last_seen/ observed_cnt features
    timestats_df = estimate_times(product_df, n_shards=8)  # use only 8 first shards to estime time stats
    product_df = product_df.merge(timestats_df, how="left", left_on="product_id", right_on="index")
    del product_df["index"]
    product_df.to_parquet(cfg.PRODUCT_EXT_PARQUET_PATH)

    # save "actual" items list
    mask = (timestats_df.last_seen_day >= 0) & (timestats_df.cnt > 50)
    actual_items = list(timestats_df[mask]["index"].values)
    json.dump(actual_items, open("../tmp/actual_items.json", "w"))
