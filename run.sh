
conda activate DIM
cd data
CUDA_VISIBLE_DEVICES='' python data_preprocess_s.py
cd ..
cd scripts
sh train_s.sh

sh test_s.sh

CUDA_VISIBLE_DEVICES='' python compute_recall_s.py

# debug
# export LC_ALL=C
# cd model
# CUDA_VISIBLE_DEVICES='' python data_helpers_s.py