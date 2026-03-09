MODEL_PATH="./merged_model/test"
QUESTION_FILES=(
    "./data/test.json"
)

for QUESTION_FILE in "${QUESTION_FILES[@]}"
do
    echo "Processing: $QUESTION_FILE"
    # IMAGE_FOLDER="/"
    IMAGE_FOLDER="./data"
    # 根据输入文件名生成输出文件名
    BASE_NAME=$(basename "$QUESTION_FILE" .json)
    OUTPUT_FILE="./test/${BASE_NAME}_test.jsonl"
    
    # 使用两轮对话推理脚本
    # 输出文件会自动生成为: ${OUTPUT_FILE}_round1.jsonl 和 ${OUTPUT_FILE}_round2.jsonl
    CUDA_VISIBLE_DEVICES=0 python llava/eval/model_vqa_multi_image_two_rounds.py \
        --model-path $MODEL_PATH \
        --question-file $QUESTION_FILE \
        --image-folder $IMAGE_FOLDER \
        --answers-file $OUTPUT_FILE \
        --conv-mode v1 \
        --temperature 0.2
done
