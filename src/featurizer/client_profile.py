from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from src.featurizer.product_info import ProductInfo, ProductInfoMapType
from src.utils import ProductEncoder, make_coo_row
from src.utils_mini import ProductEncoderMini, make_coo_row_mini

JsonType = Dict[str, Any]


class ClientProfile:
    GENDER_MAPPER = {"M": 0, "F": 1, "U": -1}
    CATEGORICAL_KEYS = [
        "product_id",
        "level_1",
        "level_2",
        "level_3",
        "level_4",
        "segment_id",
        "brand_id",
        "vendor_id",
        "netto",
        "is_own_trademark",
        "is_alcohol",
    ]
    FLOAT_KEYS = ["netto", "est_price", "seen_cnt", "first_seen_day", "last_seen_day"]

    def __init__(
        self,
        product_info_map: ProductInfoMapType,
        product_encoder: ProductEncoder,
        actual_product_encoder: ProductEncoderMini,
        client_js: JsonType,
    ):

        self._js = client_js
        self._transactions = client_js["transaction_history"][-20:]

        self._pim = product_info_map
        self._pe = product_encoder
        self._ape = actual_product_encoder

        self._n_transactions = 0
        self._n_products = 0
        self._cnt = {}
        self._sum = {}
        for k in ClientProfile.CATEGORICAL_KEYS:
            self._cnt[k] = defaultdict(int)
            self._sum[k] = 0

        self._float_obs = {k: list() for k in ClientProfile.FLOAT_KEYS}

        self._age = client_js["age"]
        self._gender = ClientProfile.GENDER_MAPPER[client_js["gender"]]
        self._sparse_row_product = make_coo_row(
            client_js["transaction_history"], self._pe, lvl="product"
        ).tocsr()
        self._sparse_row_lvl4 = make_coo_row(client_js["transaction_history"], self._pe, lvl="level_4").tocsr()
        self._sparse_actual_products_row = make_coo_row_mini(
            client_js["transaction_history"], self._ape
        ).tocsr()
        self._visited_stores = set()

        for transaction in client_js["transaction_history"]:
            self._consume_transaction(transaction)

    @property
    def age(self) -> int:
        return self._age

    @property
    def gender(self) -> int:
        return self._gender

    @property
    def n_transactions(self) -> int:
        return self._n_transactions

    @property
    def n_products(self) -> int:
        return self._n_products

    @property
    def row_product(self):
        return self._sparse_row_product

    @property
    def row_actual_product(self):
        return self._sparse_actual_products_row

    @property
    def row_lvl4(self):
        return self._sparse_row_lvl4

    @property
    def seen_products(self):
        return self._cnt["product_id"].keys()

    @property
    def n_visited_stores(self):
        return len(self._visited_stores)

    def _consume_transaction(self, transaction: JsonType):
        self._n_transactions += 1
        self._visited_stores.add(transaction["store_id"])
        for product_description in transaction["products"]:
            self._consume_product(self._pim[product_description["product_id"]])

    def _consume_product(self, pi: ProductInfo):
        self._n_products += 1

        for k in ClientProfile.CATEGORICAL_KEYS:
            value = pi.__getattribute__(k)
            self._cnt[k][value] += 1
            self._sum[k] += 1

        for k in ClientProfile.FLOAT_KEYS:
            value = pi.__getattribute__(k)
            self._float_obs[k].append(value)

    def get_user_features(self):
        result = {
            "user_age": self.age,
            "user_gender": self.gender,
            "user_num_products": self.n_products,
            "user_num_transactions": self.n_transactions,
            "user_product_per_transactions": self.n_products / (self.n_transactions + 1e-5),
            "user_num_stores": self.n_visited_stores,
        }
        for k in ClientProfile.FLOAT_KEYS:
            _mean, _std = -1, -1
            if self.n_products > 0:
                _mean = np.mean(self._float_obs[k])
            if self.n_products > 1:
                _std = np.std(self._float_obs[k])
            result["user_{}_mean".format(k)] = _mean
            result["user_{}_std".format(k)] = _std
        return result

    def set_pairwise_features(self, pi: ProductInfo, attribute_names: List[str], result, eps=1e-5):
        for attribute_name in attribute_names:
            pairwise_cnt = self._cnt[attribute_name][pi.__getattribute__(attribute_name)]
            attribute_sum = self._sum[attribute_name] + eps
            result["pw_" + attribute_name + "_cnt"] = pairwise_cnt
            result["pw_" + attribute_name + "_rate"] = pairwise_cnt / attribute_sum
