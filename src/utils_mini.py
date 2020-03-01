import hashlib
import json
from typing import List, Set

import numpy as np
import pandas as pd
import torch
from scipy import sparse as sp

import src.config as cfg


class ProductEncoderMini:
    def __init__(self, top_products):
        self._all_pids = top_products
        self.idx = {}
        self.pid = {}
        for idx, pid in enumerate(top_products):
            self.idx[pid] = idx
            self.pid[idx] = pid

    def isAllowed(self, pid):
        return pid in self.idx

    def filter(self, seq):
        return [x for x in seq if self.isAllowed(x)]

    def toIdx(self, x):
        if type(x) == str:
            pid = x
            return self.idx[pid]
        return [self.idx[pid] for pid in x]

    def toIdxWithFilter(self, x):
        if type(x) == str:
            pid = x
            return self.idx[pid]
        return [self.idx[pid] for pid in x if self.isAllowed(pid)]

    def toPid(self, x):
        if type(x) == int:
            idx = x
            return self.pid[idx]
        return [self.pid[idx] for idx in x]

    @property
    def num_products(self):
        return len(self.idx)


class TrainingSampleMini:
    def __init__(self, history: List[str], target_items: Set[str], row=None, client_id: str = None):
        self.history = history
        self.row = row
        self.target_items = target_items
        self.client_id = client_id


def squeeze_history(transaction_history):
    items = []
    for trans in transaction_history:
        items.extend([i["product_id"] for i in trans["products"]])
    return items


def make_coo_row_mini(transaction_history, product_encoder: ProductEncoderMini):
    idx = []
    values = []

    items = []
    for trans in transaction_history:
        items.extend([i["product_id"] for i in trans["products"]])
    items = [x for x in items if product_encoder.isAllowed(x)]
    n_items = len(items)

    for pid in items:
        idx.append(product_encoder.toIdx(pid))
        values.append(1.0 / n_items)

    return sp.coo_matrix(
        (np.array(values).astype(np.float32), ([0] * len(idx), idx)), shape=(1, product_encoder.num_products),
    )
