MODEL_NAME=seg_heart_model_$(VERSION).tar
MODEL_URL=ip:../模型参数文件/$(MODEL_NAME)
DATA_URL=ip:../input

MODEL_PATH=data/model
INPUT_PATH=data/input
OUTPUT_PATH=data/output
CONSISTENCY_PATH=./consistency

GPU=0

run: prepare
	python main.py \
	--gpu=$(GPU) \
	--input_path=$(INPUT_PATH)/dicom \
	--output_path=$(OUTPUT_PATH) \
	--model_path=$(MODEL_PATH)/$(MODEL_NAME)

consistency: prepare
	python main.py \
	--gpu=$(GPU) \
	--input_path=$(INPUT_PATH)/dicom \
	--output_path=$(OUTPUT_PATH) \
	--consistency_path=$(CONSISTENCY_PATH) \
	--model_path=$(MODEL_PATH)/$(MODEL_NAME)

benchmark: prepare
	mkdir -p benchmark
	mprof run python -m cProfile -o benchmark.cprof main.py \
	--gpu=$(GPU) \
	--input_path=$(INPUT_PATH)/dicom/ \
	--output_path=$(OUTPUT_PATH) \
	--model_path=$(MODEL_PATH)/$(MODEL_NAME) \
	| tee benchmark/${VERSION}-output.txt; \
	mprof plot -o benchmark/${VERSION}-memory.png; \
	python3 -c 'import pstats; pstats.Stats("benchmark.cprof").sort_stats("cumulative").print_stats(50);' > benchmark/${VERSION}-time.txt; \
	rm *.dat benchmark.cprof;


prepare: check
	if [ ! -f $(MODEL_PATH)/$(shell basename $(MODEL_URL)) ]; then \
		echo "Downloading model..."; \
		mkdir -p $(MODEL_PATH); \
		rsync -avP $(USER_NAME)@$(MODEL_URL) $(MODEL_PATH);\
	fi
	if [ ! -d $(INPUT_PATH) ]; then \
		echo "Downloading input data..."; \
		mkdir -p $(INPUT_PATH); \
		rsync -avP $(USER_NAME)@$(DATA_URL)/ $(INPUT_PATH)/dicom/; \
	fi
	if [ ! -d $(OUTPUT_PATH) ]; then \
		mkdir -p $(OUTPUT_PATH); \
	fi

check:
ifeq ($(strip $(VERSION)),)
		echo "VERSION cannot be empty."
		false
endif
ifeq ($(strip $(USER_NAME)), )
		echo "USER_NAME cannot be empty."
		false
endif
		echo "welcome $(USER_NAME), version now is $(VERSION)"
