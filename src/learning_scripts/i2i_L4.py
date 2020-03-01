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
from src.utils import ProductEncoder, iterate_shard, make_coo_row

if __name__ == "__main__":

    product_encoder = ProductEncoder(cfg.PRODUCT_PARQUET_PATH)

    rows = []
    for shard_id in range(8):
        for js in tqdm(iterate_shard(shard_id)):
            rows.append(make_coo_row(js["transaction_history"], product_encoder, lvl="level_4"))
    train_mat = sp.vstack(rows)

    model, tag = (implicit.nearest_neighbours.CosineRecommender(K=10), "L4_cosine10")
    model.fit(train_mat.T)
    out_dir = "../tmp/implicit_full/{}/".format(tag)
    os.makedirs(out_dir, exist_ok=True)
    print("Dump model to " + out_dir)
    pickle.dump(model, open(out_dir + "/model.pkl", "wb"))
