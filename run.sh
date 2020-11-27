
conda activate DIM
cd data
CUDA_VISIBLE_DEVICES='' python data_preprocess.py
cd ..
cd scripts
sh train.sh


# debug
# export LC_ALL=C