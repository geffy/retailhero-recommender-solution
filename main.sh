#!/bin/bash
set -xeuo pipefail

mkdir -p tmp
cd src
export PYTHONPATH=..

# preprocess raw data
python3 prepare_scripts/purchases_to_jrows.py
python3 prepare_scripts/train_valid_split.py
python3 prepare_scripts/extend_products_info.py

# learn base models using shard 0-7
python3 learning_scripts/i2i_full.py
python3 learning_scripts/i2i_L4.py
python3 learning_scripts/i2i_mini.py
# this step requires gpu and was performed on single 1080
python3 learning_scripts/nn_v2.py

# learn candidate selection model on shards 12-13
python3 learning_scripts/candidate_features.py
python3 learning_scripts/candidate_model.py

# learn main reranking model on shards 8-11
python3 learning_scripts/reranker_features.py
python3 learning_scripts/reranker_catboost.py
python3 learning_scripts/reranker_lgb.py

# copy assets
cd ..
mkdir -p submit/solution/assests
rsync -avz --exclude "jsons" --exclude "features" ./tmp/ ./submit/solution/assets/

