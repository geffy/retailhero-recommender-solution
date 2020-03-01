import itertools
import json
import os
from collections import defaultdict
from typing import List, Set

import numpy as np
import torch
from scipy import sparse as sp
from torch import nn
from tqdm import tqdm

from src.utils_mini import ProductEncoderMini, make_coo_row_mini

from src.utils import (
    coo_to_pytorch_sparse,
    iterate_shard,
    TrainingSample,
    normalized_average_precision,
)

from src.featurizer.nn import AwesomeModelV2

GLOBAL_COV_MAX = 0


def collect_train_data(
    shard_ids: List[int], product_encoder: ProductEncoderMini, is_train: bool = False
) -> List[TrainingSample]:
    samples = []
    for shard_id in shard_ids:
        for js in tqdm(iterate_shard(shard_id)):
            row = make_coo_row_mini(js["transaction_history"], product_encoder)
            target_items = product_encoder.toIdxWithFilter(js["target"][0]["product_ids"])

            # if train, add the next transaction to target
            if is_train and len(js["target"]) > 1:
                for target in js["target"][1:]:
                    target_items.extend(product_encoder.toIdxWithFilter(target["product_ids"]))

            # skip users with empty target
            if row is None or len(target_items) == 0:
                continue

            samples.append(TrainingSample(row=row, target_items=set(target_items), client_id=js["client_id"],))
    return samples


def make_target_coo_row(product_set, product_encoder: ProductEncoderMini):
    idx = list(product_set)
    values = np.ones(len(product_set)).astype(np.float32)
    return sp.coo_matrix((values, ([0] * len(idx), idx)), shape=(1, product_encoder.num_products))


def estimate(testing_model, testing_samples):
    _X = coo_to_pytorch_sparse(sp.vstack([sample.row for sample in testing_samples])).cuda()
    outs = testing_model.forward(_X)
    topk, indices = torch.topk(outs, 30, dim=1)
    covs = []
    scores = []
    for top_ids, sample in zip(indices.cpu().numpy(), testing_samples):
        covs.append(len(sample.target_items.intersection(top_ids)) / len(sample.target_items))
        scores.append(normalized_average_precision(sample.target_items, top_ids))
    return np.mean(covs), np.mean(scores)


def estimateAndSave(testing_model, testing_samples, overfiting_samples, out_path):
    global GLOBAL_COV_MAX

    epoch_result = {
        "va": estimate(testing_model, testing_samples),
        "tr": estimate(testing_model, overfiting_samples),
    }
    if epoch_result["va"][0] > GLOBAL_COV_MAX:
        GLOBAL_COV_MAX = epoch_result["va"][0]
        print("-> Annnd its a new champion!")
        torch.save(testing_model.state_dict(), out_path)
    return epoch_result


if __name__ == "__main__":
    # use only actual items as input/output for this net
    actual_items = json.load(open("../tmp/actual_items.json"))
    product_encoder = ProductEncoderMini(actual_items)

    train_samples = collect_train_data(range(8), product_encoder, is_train=True)
    valid_samples = collect_train_data([15,], product_encoder, is_train=False)

    model = AwesomeModelV2(product_encoder.num_products).cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones([19022]) * 100).cuda()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.003)
    target_dir = "../tmp/nn_v2/"
    os.makedirs(target_dir, exist_ok=True)

    history = []
    for epoch in range(256):
        for _ in tqdm(range(100)):
            optimizer.zero_grad()

            batch_samples = np.random.choice(train_samples, 1024, replace=False)

            _X = coo_to_pytorch_sparse(sp.vstack([sample.row for sample in batch_samples])).cuda()
            _sparse_Y = sp.vstack([make_target_coo_row(s.target_items, product_encoder) for s in batch_samples])
            _Y = coo_to_pytorch_sparse(_sparse_Y).cuda().to_dense()

            outs = model.forward(_X)
            loss = criterion(outs, _Y)

            loss.backward()
            optimizer.step()

        epoch_result = {"va": estimate(model, valid_samples), "tr": estimate(model, train_samples[:10000])}

        epoch_result = estimateAndSave(model, valid_samples, train_samples[:10000], target_dir + "/model.pth")
        print(epoch_result)
        history.append(epoch_result)
        json.dump(history, open(target_dir + "/history.json", "w"))
