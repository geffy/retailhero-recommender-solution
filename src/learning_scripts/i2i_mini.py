import glob
import json
import os
import pickle
import sys

import implicit
import numpy as np
import pandas as pd
from scipy import sparse as sp
from tqdm import tqdm

import src.config as cfg
from src.utils import iterate_shard, normalized_average_precision
from src.utils_mini import ProductEncoderMini, TrainingSampleMini, make_coo_row_mini

if __name__ == "__main__":

    product_encoder = ProductEncoderMini(json.load(open("../tmp/actual_items.json")))

    rows = []
    for shard_id in range(8):
        for js in tqdm(iterate_shard(shard_id)):
            rows.append(make_coo_row_mini(js["transaction_history"], product_encoder))
    train_mat = sp.vstack(rows)

    for model, tag in [
        (implicit.nearest_neighbours.TFIDFRecommender(K=1), "tf1"),
        (implicit.nearest_neighbours.TFIDFRecommender(K=10), "tf10"),
    ]:

        model.fit(train_mat.T)
        out_dir = "../tmp/implicit_mini/" + tag
        os.makedirs(out_dir, exist_ok=True)
        print("Dump model to " + out_dir)
        pickle.dump(model, open(out_dir + "/model.pkl", "wb"))

        print("Estimate quiality...")
        scores = []
        for js in tqdm(iterate_shard(15)):
            row = make_coo_row_mini(js["transaction_history"], product_encoder).tocsr()
            raw_recs = model.recommend(
                userid=0, user_items=row, N=30, filter_already_liked_items=False, recalculate_user=True
            )
            recommended_items = product_encoder.toPid([idx for (idx, score) in raw_recs])
            gt_items = js["target"][0]["product_ids"]
            ap = normalized_average_precision(gt_items, recommended_items)
            scores.append(ap)
        print("\tmap: {}".format(np.mean(scores)))
