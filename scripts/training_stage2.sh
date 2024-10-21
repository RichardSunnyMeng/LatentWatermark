time=$(date "+%Y%m%d-%H%M%S")
NAME=${0%\.*}

cfg=./configs/inject_64_bits.json
gpu=$1


CUDA_VISIBLE_DEVICES=$gpu python ./training_stage2.py \
    --config $cfg \
    --pretrained_message_model ./results/inject_64bits/checkpoints/message_model_epoch_-1.pth
