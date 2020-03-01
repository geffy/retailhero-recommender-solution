import hashlib
import json
from typing import List, Set

import numpy as np
import pandas as pd
import torch
from scipy import sparse as sp

import src.config as cfg

# since they are used for check
blacklisted_clients = set(
    [
        "000012768d",
        "000036f903",
        "00010925a5",
        "0001f552b0",
        "00020e7b18",
        "000220a0a7",
        "00022fd34f",
        "0002ce2217",
        "00031cbbe6",
        "00035a21d9",
        "00038f9200",
        "0004231e2a",
        "0004254599",
        "00042a927a",
        "0004315e57",
        "00047b3720",
        "0004e1e14e",
        "00065f11c7",
        "00068fd5dc",
        "0006b9ad75",
        "0006f24465",
        "0006fca4bf",
        "000702109b",
        "00071890c8",
        "0007667c60",
        "00078c508d",
        "0007b4ca21",
        "0008244fb3",
        "00083b5b14",
        "0008b2cb41",
        "000940f00a",
        "000990be82",
        "0009e6bafa",
        "000a00419c",
        "000a400848",
        "000a9d12ff",
        "000ac12729",
        "000b0559be",
        "000b45b7ac",
        "000b9905d8",
        "000bc820f6",
        "000bc94494",
        "000bd5f2f1",
        "000bf8ff33",
        "000c049a1a",
        "000c216adb",
        "000ca87889",
        "000d599743",
        "000ddb6229",
        "000df9078a",
        "000efde438",
        "000f3b9860",
        "000f46bbfc",
        "0010082ab3",
        "0010f1f8ca",
        "001162084a",
        "001175d51b",
        "00127b29bb",
        "0012d1d4aa",
        "00134e091b",
        "001392b297",
        "0013c0cbab",
        "00140e5d34",
        "001566f916",
        "0015aa77ce",
        "00167a61e2",
        "0016b0d9ad",
        "00174b3561",
        "00177cee3e",
        "0017a7ebcb",
        "0017fdd057",
        "00184aab1b",
        "00184df0c9",
        "00184e8b0a",
        "00184f3b10",
        "0018650c30",
        "0018d2efac",
        "0018dea0ba",
        "0019a16b6b",
        "0019ca361b",
        "0019e0f07d",
        "0019fb86cb",
        "001a2412c6",
        "001b8d6788",
        "001c25b9e3",
        "001c2b565f",
        "001c8984f0",
        "001cef2991",
        "001d004e5e",
        "001d642f66",
        "001dac232d",
        "001de90d21",
        "001e840150",
        "001f46aa2c",
        "001fb70769",
        "00209f873d",
        "0020f90a83",
        "00211fcfaa",
        "00213be6fb",
        "0021e07838",
        "002283ef29",
    ]
)


class ProductEncoder:
    KEYS = ["level_1", "level_2", "level_3", "level_4"]

    def __init__(self, product_parquet_path):
        product_df = pd.read_parquet(product_parquet_path).fillna("NA")

        # fill product part
        self.product_idx = {}
        self.product_pid = {}
        self.product_lvl = {}
        for idx, row in enumerate(product_df.itertuples()):
            pid = row.product_id
            self.product_idx[pid] = idx
            self.product_pid[idx] = pid
            self.product_lvl[pid] = {f: row.__getattribute__(f) for f in ProductEncoder.KEYS}

        # fill level part
        self._lvl_mapper = {}
        for k in ProductEncoder.KEYS:
            values = sorted(set(product_df[k].values))
            self._lvl_mapper[k] = {level: idx for (level, idx) in zip(values, range(len(values)))}

    def toIdx(self, x):
        if type(x) == str:
            pid = x
            return self.product_idx[pid]
        return [self.product_idx[pid] for pid in x]

    def toPid(self, x):
        if type(x) == int:
            idx = x
            return self.product_pid[idx]
        return [self.product_pid[idx] for idx in x]

    @property
    def num_products(self):
        return len(self.product_idx)

    def lvlSize(self, lvl="product"):
        if lvl == "product":
            return self.num_products
        if lvl in ProductEncoder.KEYS:
            return len(self._lvl_mapper[lvl])
        raise RuntimeError("Unexpected lvl value: " + lvl)

    def lvlToIdx(self, x, lvl="product"):
        if lvl == "product":
            return self.toIdx(x)

        if lvl in ProductEncoder.KEYS:
            mapping = self._lvl_mapper[lvl]
            if type(x) == str:
                return mapping[self.product_lvl[x][lvl]]
            return [mapping[self.product_lvl[pid][lvl]] for pid in x]
        raise RuntimeError("Unexpected lvl value: " + lvl)


class TrainingSample:
    def __init__(self, row: sp.coo_matrix, target_items: Set[int], client_id: str = None):
        self.row = row
        self.target_items = target_items
        self.client_id = client_id


def make_coo_row(transaction_history, product_encoder: ProductEncoder, lvl="product"):
    idx = []
    values = []

    items = []
    for trans in transaction_history:
        items.extend([i["product_id"] for i in trans["products"]])
    n_items = len(items)

    for pid in items:
        idx.append(product_encoder.lvlToIdx(pid, lvl))
        values.append(1.0 / n_items)

    return sp.coo_matrix(
        (np.array(values).astype(np.float32), ([0] * len(idx), idx)), shape=(1, product_encoder.lvlSize(lvl)),
    )


def average_precision(actual, recommended, k=30):
    ap_sum = 0
    hits = 0
    for i in range(k):
        product_id = recommended[i] if i < len(recommended) else None
        if product_id is not None and product_id in actual:
            hits += 1
            ap_sum += hits / (i + 1)
    return ap_sum / k


def normalized_average_precision(actual, recommended, k=30):
    actual = set(actual)
    if len(actual) == 0:
        return 0.0

    ap = average_precision(actual, recommended, k=k)
    ap_ideal = average_precision(actual, list(actual)[:k], k=k)
    return ap / ap_ideal


def np_normalize_matrix(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (norm + 1e-6)


def coo_to_pytorch_sparse(M):
    """
    input: M is Scipy sparse matrix
    output: pytorch sparse tensor in GPU
    """
    M = M.astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    Ms = torch.sparse_coo_tensor(indices, values, shape)
    return Ms


def _get_shard_path(n_shard, jsons_dir=cfg.JSONS_DIR):
    return "{}/{:02d}.jsons.splitted".format(jsons_dir, n_shard)


def iterate_shard(n_shard):
    for js in (json.loads(l) for l in open(_get_shard_path(n_shard))):
        if js["client_id"] in blacklisted_clients:
            continue
        yield js


def md5_hash(x):
    return int(hashlib.md5(x.encode()).hexdigest(), 16)


def md5_hex(x):
    return hashlib.md5(x.encode()).hexdigest()[-8:]
