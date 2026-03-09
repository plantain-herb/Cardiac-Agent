DATA_URL=172.30.0.54:/volume1/数据使用/模型训练数据集/肝胆产品线/胰腺分割模型/胰腺分割模型1.x版/1.x复盘版本/训练数据集/processed_data
DATA_DIR=train_data/processed_data

MODEL_DIR=checkpoints
MODEL_FIRST_URL=172.30.0.54:/volume1/数据使用/模型训练数据集/肝胆产品线/胰腺分割模型/胰腺分割模型1.x版/1.x复盘版本/模型参数文件/pretrain_first_stage.pth
MODEL_SECOND_URL=172.30.0.54:/volume1/数据使用/模型训练数据集/肝胆产品线/胰腺分割模型/胰腺分割模型1.x版/1.x复盘版本/模型参数文件/pretrain_second_stage.pth


ORI_DATA_ROOT=train_data/origin_data
ORI_DATA_URL=172.30.0.54:/volume1/数据使用/模型训练数据集/肝胆产品线/胰腺分割模型/胰腺分割模型1.x版/1.x复盘版本/训练数据集/ori_data

format:
	pre-commit run --all-files --show-diff-on-failure

# make USER_NAME=..
train_heart_seg_first:
	bash run_dist.sh -d 0,1,2,3 -g 4 -c ./config/seg_heart_first_config.py

train_heart_seg_second:
	bash run_dist.sh -d 0,1,2,3 -g 4 -c ./config/seg_heart_second_config.py

tensorboard:
	tensorboard serve --logdir checkpoints/ --host '0.0.0.0'

download:
	test $(USER_NAME)
	if [ ! -d "$(DATA_DIR)" ]; then \
		echo "Downloading training data..."; \
		rsync -avP $(USER_NAME)@$(DATA_URL) ./train_data; \
	fi

	if [ ! -d "$(MODEL_DIR)" ]; then \
		mkdir -p $(MODEL_DIR); \
		echo "Downloading first stage pretrain model..."; \
		rsync -avP $(USER_NAME)@$(MODEL_FIRST_URL) ./checkpoints/; \
		rsync -avP $(USER_NAME)@$(MODEL_SECOND_URL) ./checkpoints/; \
	fi


generate_data:
	test $(USER_NAME)
	if [ ! -d "$(ORI_DATA_ROOT)"  ]; then \
		echo "Downloading origin data..."; \
		rsync -avP $(USER_NAME)@$(ORI_DATA_URL) ./train_data; \
	fi
	python custom/utils/generate_dataset_ms.py


save_torchscript: check_torchscript
	python ./custom/utils/save_torchscript.py --model_path=${model_path} --config_path=${config_path} --output_path=${output_path}

check_torchscript:
ifeq ($(strip $(model_path)),)
	echo "model_path cannot be empty."
		false
endif
ifeq ($(strip $(config_path)),)
	echo "config_path cannot be empty."
		false
endif
ifeq ($(strip $(output_path)),)
	echo "output_path cannot be empty."
		false
endif



clean:
	rm -rf $(MODEL_ROOT)
	rm -rf $(DATA_DIR)
