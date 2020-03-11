
conda activate DIM
cd data
CUDA_VISIBLE_DEVICES='' python data_preprocess_s.py
cd ..
cd scripts
sh train_s.sh


# debug
# export LC_ALL=C
# cd model
# CUDA_VISIBLE_DEVICES='' python data_helpers_s.py