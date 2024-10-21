time=$(date "+%Y%m%d-%H%M%S")
NAME=${0%\.*}

cfg=./configs/inject_64_bits.json
gpu=$1


CUDA_VISIBLE_DEVICES=$gpu python ./extract.py \
    --config $cfg \
    --message_model_ckpt ./results/inject_64bits/checkpoints/message_model_epoch_1.pth \
    --fusing_mapping_model_ckpt ./results/inject_64bits/checkpoints/generator_epoch_1.pth