from collections import namedtuple
from typing import Dict

import pandas as pd


class ProductInfo:
    def __init__(self, row: namedtuple):
        self._row = row

    @property
    def product_id(self) -> str:
        return self._row.product_id

    @property
    def level_1(self) -> str:
        return self._row.level_1

    @property
    def level_2(self) -> str:
        return self._row.level_2

    @property
    def level_3(self) -> str:
        return self._row.level_3

    @property
    def level_4(self) -> str:
        return self._row.level_4

    @property
    def segment_id(self) -> str:
        return self._row.segment_id

    @property
    def brand_id(self) -> str:
        return self._row.brand_id

    @property
    def vendor_id(self) -> str:
        return self._row.vendor_id

    @property
    def netto(self) -> int:

        return int(self._row.netto * 100)

    @property
    def is_own_trademark(self) -> int:
        return self._row.is_own_trademark

    @property
    def is_alcohol(self) -> int:
        return self._row.is_alcohol

    @property
    def est_price(self) -> float:
        return self._row.est_price

    @property
    def price_quantile_level_1(self) -> float:
        return self._row.price_quantile_level_1

    @property
    def price_quantile_level_2(self) -> float:
        return self._row.price_quantile_level_2

    @property
    def price_quantile_level_3(self) -> float:
        return self._row.price_quantile_level_3

    @property
    def price_quantile_level_4(self) -> float:
        return self._row.price_quantile_level_4

    @property
    def seen_cnt(self) -> int:
        return self._row.cnt

    @property
    def first_seen_day(self) -> int:
        return self._row.first_seen_day

    @property
    def last_seen_day(self) -> int:
        return self._row.last_seen_day


ProductInfoMapType = Dict[str, ProductInfo]


def init_product_info_map(product_parquet_path: str) -> ProductInfoMapType:
    product_info_map = {}
    product_df = pd.read_parquet(product_parquet_path).fillna(value=0)
    for row in product_df.itertuples():
        pi = ProductInfo(row)
        product_info_map[pi.product_id] = pi
    return product_info_map
