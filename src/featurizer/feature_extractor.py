import copy
import pickle
from collections import defaultdict
from typing import Any, List, Set, Tuple

import implicit
from scipy.stats import rankdata

from src.featurizer.client_profile import ClientProfile
from src.featurizer.daily import DailyScorer, split_date
from src.featurizer.nn import WrapperNnModel
from src.featurizer.product_info import ProductInfoMapType
from src.utils import ProductEncoder
from src.utils_mini import ProductEncoderMini, make_coo_row_mini


class ImplicitWrapperMini:
    def __init__(self, product_encoder, model_root):
        self.pe = product_encoder
        self.model = pickle.load(open(model_root + "/model.pkl", "rb"))

    def predict(self, actual_row, N=30):
        raw_recs = self.model.recommend(
            userid=0, user_items=actual_row, N=N, filter_already_liked_items=False, recalculate_user=True
        )
        return [(self.pe.toPid(int(idx)), score) for (idx, score) in raw_recs]


class FeatureExtractor:
    USE_KEYS = ["product_id", "level_1", "level_2", "level_3", "level_4", "segment_id", "vendor_id"]

    def __init__(
        self,
        product_info_map: ProductInfoMapType,
        product_encoder: ProductEncoder,
        global_top: Set[str],
        tagged_models: List[Tuple[str, Any]],
        mini_models: List[Tuple[str, ImplicitWrapperMini]],
        lvl4_model: Any,
        nn2_2: WrapperNnModel,
    ):
        self._pi = product_info_map
        self._pe = product_encoder
        self._global_top = set(global_top)
        self._tagged_models = tagged_models
        self._mini_models = mini_models
        self._lvl4_model = lvl4_model
        self._nn2_2 = nn2_2

    def build_precalc(self, profile: ClientProfile):
        precalc = {"pairs": {}, "map": {}}

        # batched calculation of models scores
        for model, tag in self._tagged_models:
            pairs = model.recommend(
                userid=0,
                user_items=profile.row_product,
                N=1000,
                recalculate_user=True,
                filter_already_liked_items=False,
            )
            precalc["pairs"][tag] = [(self._pe.toPid(int(idx)), score) for (idx, score) in pairs]

        # batched calculation of models scores
        for model, tag in self._mini_models:
            pairs = model.predict(profile._sparse_actual_products_row, N=1000)
            precalc["pairs"][tag] = pairs

        # nn models
        precalc["map"]["nn22"] = self._nn2_2.predict(profile._js["transaction_history"])

        return precalc

    def build_features(self, profile: ClientProfile, precalc, candidates: List[str], raw_date: str):
        rows = []

        lvl4_relevances = {
            idx: score
            for (idx, score) in self._lvl4_model.recommend(
                userid=0,
                user_items=profile.row_lvl4,
                N=1000,
                recalculate_user=True,
                filter_already_liked_items=False,
            )
        }

        iterate_over_models = [tag for _, tag in self._tagged_models]
        iterate_over_models.extend([tag for _, tag in self._mini_models])
        iterate_over_models.extend(["nn22", "fm22"])

        # fill with default values
        indexed_candidates = self._pe.toIdx(candidates)
        scores = defaultdict(dict)
        for idx in indexed_candidates:
            for tag in iterate_over_models:
                scores[idx]["model_{}_score".format(tag)] = -1

        # batched calculation of models scores
        for _, tag in self._tagged_models + self._mini_models:
            for (pid, score) in precalc["pairs"][tag]:
                scores[self._pe.toIdx(pid)]["model_{}_score".format(tag)] = score

        splitted_date = split_date(raw_date)

        # nn model + daily_scores
        for idx in indexed_candidates:
            pid = self._pe.toPid(idx)
            scores[idx]["model_nn22_score"] = precalc["map"]["nn22"].get(pid, -100)
            scores[idx]["model_fm22_score"] = -100  # rudiment

        # model ranks
        for tag in iterate_over_models:
            _rels = [scores[idx]["model_{}_score".format(tag)] for idx in indexed_candidates]
            _ranks = rankdata(_rels) / len(_rels)
            for idx, rank in zip(indexed_candidates, _ranks):
                scores[idx]["model_{}_rank".format(tag)] = rank

        static_user_features = profile.get_user_features()

        for product_id in candidates:
            pi = self._pi[product_id]

            # user features
            row = copy.copy(static_user_features)
            row["time_hour"] = splitted_date.hour
            row["time_weekday"] = splitted_date.weekday

            # product features
            row["product_truncated_id"] = product_id if product_id in self._global_top else "_RAREID"
            row["product_level4_id"] = str(pi.level_4)
            row["product_level3_id"] = str(pi.level_3)
            row["product_level2_id"] = str(pi.level_2)
            row["product_level1_id"] = str(pi.level_1)
            row["product_segment_id"] = str(pi.segment_id)
            row["product_vendor_id"] = str(pi.vendor_id)
            row["product_brand_id"] = str(pi.brand_id)

            for feature in ClientProfile.FLOAT_KEYS:
                row["product_" + feature] = pi.__getattribute__(feature)

            # pairwise part
            profile.set_pairwise_features(pi, FeatureExtractor.USE_KEYS, row)

            # model scores
            row.update(scores[self._pe.toIdx(product_id)])
            row["model_lvl4_score"] = lvl4_relevances.get(self._pe.lvlToIdx(product_id, lvl="level_4"), -1)
            rows.append(row)
        return rows
