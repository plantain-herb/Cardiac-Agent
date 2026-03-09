#!/bin/bash
MODEL_NAME="./weights/llava-med-v1.5-mistral-7b"
VISION_TOWER="./weights/clip-vit-large-patch14"
DATASET_LINK="./data"
OUTPUT_DIR="./checkpoints/test"
DATA_PATH="./data/train.json"

export DEEPSPEED_CACHE_PATH='/tmp/deepspeed-cache'
export TRITON_CACHE_DIR='/tmp/triton-cache'
# export CUDA_VISIBLE_DEVICES=5

deepspeed --include localhost:0,1 --master_port 29501 llava/train/train_mem_multi_image_aug.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $MODEL_NAME  \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $DATASET_LINK \
    --vision_tower $VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_steps 150 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --max_grad_norm 0.5
