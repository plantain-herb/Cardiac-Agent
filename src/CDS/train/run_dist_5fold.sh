TRAIN_FILE=train.py
while getopts ":d:g:c:" opt; do
  case ${opt} in
  d)
    CUDA_VISIBLE_DEVICES=$OPTARG
    ;;
  g)
    nproc_per_node=$OPTARG
    ;;
  c)
    config=$OPTARG
    ;;
  \?)
    echo "Invalid Option: -$OPTARG" 1>&2
    exit 1
    ;;
  esac
done


# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --master_port 29503 --nproc_per_node=$nproc_per_node $TRAIN_FILE --config $config --launcher pytorch

for fold in 1 2 3 4 5; do
  echo "Running fold $fold"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port 29505 --nproc_per_node=2 train_5fold.py --config config/cine_class_config_5fold.py --fold $fold --launcher pytorch
done