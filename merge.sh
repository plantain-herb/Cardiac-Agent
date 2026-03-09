MODEL_NAME="./weights/llava-med-v1.5-mistral-7b"
OUTPUT_PATH="./merged_model/test"

CUDA_VISIBLE_DEVICES=0 python scripts/merge_lora_weights.py \
    --model-path $LORA_WEIGHTS_PATH \
    --model-base $MODEL_NAME \
    --save-model-path $OUTPUT_PATH
