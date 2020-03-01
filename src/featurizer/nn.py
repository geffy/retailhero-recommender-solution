import itertools
import json
import os
from collections import defaultdict
from typing import List, Set, Dict

import numpy as np
import torch
from scipy import sparse as sp
from torch import nn
from tqdm.notebook import tqdm

from src.featurizer.client_profile import JsonType
from src.utils import TrainingSample, coo_to_pytorch_sparse, normalized_average_precision
from src.utils_mini import ProductEncoderMini, make_coo_row_mini


class AwesomeModelV2(nn.Module):
    def __init__(self, num_products):
        super(AwesomeModelV2, self).__init__()
        self._fc1 = nn.Linear(num_products, 512)
        self._fc2 = nn.Linear(512, num_products)

    def forward(self, x):
        x = self._fc1(x)
        return self._fc2(x)


class WrapperNnModel:
    def __init__(self, product_encoder_mini: ProductEncoderMini, model_path: str):
        self._pe = product_encoder_mini
        self._model = AwesomeModelV2(self._pe.num_products)
        self._model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage),)

    def predict(self, transaction_history: JsonType) -> Dict[str, float]:
        row = make_coo_row_mini(transaction_history, self._pe)
        X = coo_to_pytorch_sparse(row)
        outs = self._model.forward(X)
        return {pid: score for (pid, score) in zip(self._pe._all_pids, outs.data.numpy()[0])}
