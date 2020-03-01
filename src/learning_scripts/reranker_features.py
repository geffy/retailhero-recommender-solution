import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import pickle
import os

from src.utils import iterate_shard, md5_hex
from src.featurizer.client_profile import ClientProfile
from src.two_stage_predictor import TwoStagePredictor

import glob
from collections import defaultdict

from multiprocessing import Pool
import src.config as cfg
from functools import partial


def extract_features_as_df(shard_id: int, predictor: TwoStagePredictor):
    part_dfs = []
    aux = []
    md5_path = md5_hex(str(shard_id))
    for js in tqdm(iterate_shard(shard_id)):

        profile = ClientProfile(
            product_info_map=predictor.product_info_map,
            product_encoder=predictor.product_encoder,
            actual_product_encoder=predictor.actual_product_encoder,
            client_js=js,
        )

        precalc = predictor.feature_extractor.build_precalc(profile)
        candidates = predictor.candidate_selector.get_candidates(profile, precalc)

        rows = predictor.feature_extractor.build_features(
            profile, precalc, candidates, js["target"][0]["datetime"]
        )

        features = pd.DataFrame(rows)
        features = features[sorted(features.columns)]

        groupId = "{}:{}:0".format(md5_path, js["client_id"])
        features["_groupId"] = groupId
        gt = set(js["target"][0]["product_ids"])

        features["_label"] = [int(x in gt) for x in candidates]

        part_dfs.append(features)
        aux.append({"gt": list(gt), "candidates": candidates, "groupId": groupId})
    return part_dfs, aux


def save_assets(
    part_dfs, aux, tag, fmt_dfs="../tmp/features/{}_features.parquet", fmt_aux="../tmp/features/{}_aux.jsons"
):
    df = pd.concat(part_dfs, sort=True)
    df.fillna(-1).to_parquet(fmt_dfs.format(tag))
    with open(fmt_aux.format(tag), "w") as fout:
        for aux_js in aux:
            fout.write(json.dumps(aux_js) + "\n")


if __name__ == "__main__":
    predictor = TwoStagePredictor(assets_root="../tmp/", is_second_stage_train=True)
    out_dir = "../tmp/features/"
    os.makedirs(out_dir, exist_ok=True)

    for shard_id in [15, 8, 9, 10, 11]:
        part_dfs, aux = extract_features_as_df(shard_id, predictor)
        save_assets(part_dfs, aux, "train_{}_v4.3".format(shard_id))
