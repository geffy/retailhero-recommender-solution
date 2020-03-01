from typing import Any, List, Set

import numpy as np

from src.featurizer.client_profile import ClientProfile
from src.featurizer.product_info import ProductInfoMapType
from src.utils import ProductEncoder


class CandidatSelector:
    def __init__(
        self, model: Any, global_top: Set[str], product_info_map: ProductInfoMapType,
    ):
        self._model = model
        self._global_top = global_top
        self._pim = product_info_map

    def get_features(self, profile: ClientProfile, precalc) -> List[str]:
        candidates_set = set(self._global_top)
        candidates_set.update(profile.seen_products)
        for pairs in precalc["pairs"].values():
            candidates_set.update([pid for (pid, _) in pairs])

        candidates_list = sorted(list(candidates_set))
        candidates_map = {pid: idx for idx, pid in enumerate(candidates_list)}

        features = np.zeros((len(candidates_list), 6)).astype(np.float32) - 10
        # f0: tf1
        for (pid, score) in precalc["pairs"]["Mtf1"]:
            if pid not in candidates_map:
                continue
            features[candidates_map[pid], 0] = score

        # f1: tf10
        for (pid, score) in precalc["pairs"]["Mtf10"]:
            if pid not in candidates_map:
                continue
            features[candidates_map[pid], 1] = score

        # f2: nn22
        curr_map = precalc["map"]["nn22"]
        for idx, pid in enumerate(candidates_list):
            if pid not in curr_map:
                continue
            features[idx, 2] = curr_map[pid]

        # f3: seen in history
        for pid in profile.seen_products:
            features[candidates_map[pid], 3] = 1

        # f4: estimated popularity
        # f5: last_seen_date
        for idx, pid in enumerate(candidates_list):
            features[idx, 4] = self._pim[pid].seen_cnt
            features[idx, 5] = self._pim[pid].last_seen_day

        return (candidates_list, features)

    def get_candidates(self, profile: ClientProfile, precalc) -> List[str]:
        candidates_list, features = self.get_features(profile, precalc)
        scores = self._model.predict(features)
        idx = np.argsort(-scores)
        return [str(x) for x in np.array(candidates_list)[idx[:200]]]
