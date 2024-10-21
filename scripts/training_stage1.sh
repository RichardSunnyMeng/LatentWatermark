time=$(date "+%Y%m%d-%H%M%S")
NAME=${0%\.*}

cfg=./configs/inject_64_bits.json
gpu=$1


CUDA_VISIBLE_DEVICES=$gpu python ./training_stage1.py \
    --config $cfg 
