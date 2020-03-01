import copy
import json
import pickle
from collections import defaultdict
from timeit import default_timer as timer

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
from scipy import sparse as sp
from scipy.stats import rankdata

from src.featurizer.candidate_selector import CandidatSelector
from src.featurizer.client_profile import ClientProfile
from src.featurizer.feature_extractor import FeatureExtractor, ImplicitWrapperMini
from src.featurizer.product_info import init_product_info_map
from src.featurizer.daily import DailyScorer
from src.featurizer.nn import WrapperNnModel, AwesomeModelV2
from src.utils import ProductEncoder, make_coo_row
from src.utils_mini import ProductEncoderMini


class TwoStagePredictor:
    def __init__(
        self, assets_root: str, is_first_stage_train: bool = False, is_second_stage_train: bool = False
    ):
        self.product_encoder = ProductEncoder(assets_root + "/products_orig.parquet")
        self.product_info_map = init_product_info_map(assets_root + "/products_ext_2.parquet")
        global_top_products = json.load(open(assets_root + "/global_top.json"))
        self.actual_products = json.load(open(assets_root + "/actual_items.json"))
        self.actual_product_encoder = ProductEncoderMini(self.actual_products)

        tagged_models = [
            (pickle.load(open(assets_root + "/implicit_full/cosine1/model.pkl", "rb")), "cosine1"),
        ]

        mini_models = [
            (ImplicitWrapperMini(self.actual_product_encoder, assets_root + "/implicit_mini/tf1/"), "Mtf1"),
            (ImplicitWrapperMini(self.actual_product_encoder, assets_root + "/implicit_mini/tf10/"), "Mtf10"),
        ]

        self.feature_extractor = FeatureExtractor(
            self.product_info_map,
            self.product_encoder,
            global_top=global_top_products[:2000],
            tagged_models=tagged_models,
            mini_models=mini_models,
            lvl4_model=pickle.load(open(assets_root + "/implicit_full/L4_cosine10/model.pkl", "rb")),
            nn2_2=WrapperNnModel(
                product_encoder_mini=self.actual_product_encoder, model_path=assets_root + "/nn_v2/model.pth",
            ),
        )

        candidate_model = None
        if not is_first_stage_train:
            candidate_model = CatBoost().load_model(assets_root + "/model_v4.3_candidates.cbm")

        self.candidate_selector = CandidatSelector(
            model=candidate_model, global_top=global_top_products[:50], product_info_map=self.product_info_map,
        )

        self.cb_reranker = None
        self.lgb_reranker = None

        if not is_first_stage_train and not is_second_stage_train:
            self.cb_reranker = CatBoost().load_model(assets_root + "/model_v4.3_reranker.cbm")
            self.lgb_reranker = lgb.Booster(model_file=assets_root + "/model_4.3_reranker.lgb")

    def rows_to_df(self, rows):
        features = pd.DataFrame(rows)
        return features[sorted(features.columns)].fillna(-1)

    def df_to_pool(self, df):
        num_names = [x for x in df.columns if not x.startswith("_")]
        return Pool(
            data=df[num_names],
            cat_features=[
                "product_truncated_id",
                "product_level1_id",
                "product_level2_id",
                "product_level3_id",
                "product_level4_id",
                "product_segment_id",
                "product_vendor_id",
                "product_brand_id",
            ],
        )

    def predict(self, js, k=30):
        profile = ClientProfile(
            product_info_map=self.product_info_map,
            product_encoder=self.product_encoder,
            actual_product_encoder=self.actual_product_encoder,
            client_js=js,
        )

        precalc = self.feature_extractor.build_precalc(profile)
        candidates = self.candidate_selector.get_candidates(profile, precalc)

        rows = self.feature_extractor.build_features(profile, precalc, candidates, js["query_time"])
        features_df = self.rows_to_df(rows)

        # cb part
        cb_pool = self.df_to_pool(features_df)
        cb_scores = self.cb_reranker.predict(cb_pool)

        # lgb part
        skip_features = [
            "product_truncated_id",
            "product_level1_id",
            "product_level2_id",
            "product_level3_id",
            "product_level4_id",
            "product_segment_id",
            "product_vendor_id",
            "product_brand_id",
        ]
        lgb_feature_names = [x for x in features_df.columns if not x.startswith("_") and x not in skip_features]
        lgb_scores = self.lgb_reranker.predict(features_df[lgb_feature_names])

        scores = rankdata(cb_scores) + rankdata(lgb_scores)
        return list(np.array(candidates)[np.argsort(-scores)][:30])
