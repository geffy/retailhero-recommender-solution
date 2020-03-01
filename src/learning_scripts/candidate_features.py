import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import pickle
import os

from src.utils import iterate_shard
from src.featurizer.client_profile import ClientProfile
from src.two_stage_predictor import TwoStagePredictor

import glob
from collections import defaultdict

from multiprocessing import Pool
import src.config as cfg
from functools import partial


def _extract(js, predictor: TwoStagePredictor):
    profile = ClientProfile(
        product_info_map=predictor.product_info_map,
        product_encoder=predictor.product_encoder,
        actual_product_encoder=predictor.actual_product_encoder,
        client_js=js,
    )

    precalc = predictor.feature_extractor.build_precalc(profile)
    candidates_list, candidates_features = predictor.candidate_selector.get_features(profile, precalc)
    gt = set(js["target"][0]["product_ids"])

    return {
        "gt": list(gt),
        "candidates": candidates_list,
        "features": candidates_features,
        "client_id": js["client_id"],
    }


def extract_batch(shard_id: str, predictor: TwoStagePredictor):
    map_func = partial(_extract, predictor=predictor)
    result = map(map_func, tqdm(iterate_shard(shard_id)))
    return list(result)


if __name__ == "__main__":
    predictor = TwoStagePredictor(assets_root="../tmp/", is_first_stage_train=True)
    out_dir = "../tmp/features/"
    os.makedirs(out_dir, exist_ok=True)

    pickle.dump(extract_batch(12, predictor), open(out_dir + "/candidates_train_12.pickled", "wb"))
    pickle.dump(extract_batch(13, predictor), open(out_dir + "/candidates_train_13.pickled", "wb"))
