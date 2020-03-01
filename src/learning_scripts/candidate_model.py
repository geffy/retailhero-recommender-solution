import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import pickle
import os

import catboost
from src.utils import iterate_shard
from src.featurizer.client_profile import ClientProfile
from src.two_stage_predictor import TwoStagePredictor

import glob
from collections import defaultdict


def limit_candidates_number(s, limit=1000):
    r = s.copy()
    r["candidates"] = r["candidates"][:limit]
    r["features"] = r["features"][:limit]
    return r


def produce_labels(s):
    return [(x in s["gt"]) * 1 for x in s["candidates"]]


def build_pool(samples):
    return catboost.Pool(
        data=np.vstack([s["features"] for s in samples]),
        group_id=np.concatenate([[s["client_id"],] * len(s["candidates"]) for s in samples]),
        label=np.concatenate([produce_labels(s) for s in samples]),
    )


if __name__ == "__main__":

    def load_part(part_id):
        raw_samples = pickle.load(open("../tmp/features/candidates_train_{:02d}.pickled".format(part_id), "rb"))
        return [limit_candidates_number(s, limit=1000) for s in raw_samples]

    part12 = load_part(12)
    part13 = load_part(13)

    train_pool = build_pool(part12[:20000] + part13)
    valid_pool = build_pool(part12[20000:])

    parameters = {
        "iterations": 200,
        "learning_rate": 0.05,
        "depth": 4,
        "loss_function": "PairLogit",
        "custom_metric": ["MAP:top=30", "RecallAt:top=250"],
        "verbose": False,
        "random_seed": 44,
        "task_type": "GPU",
        "max_ctr_complexity": 0,
    }

    model = catboost.CatBoost(parameters)
    model.fit(train_pool, eval_set=valid_pool, plot=True)
    model.save_model("../tmp/model_v4.3_candidates.cbm")
