# Getting Started
Follow the instructions to run this example

## Prerequisites
- python
- virtualenv

## Installation

In this folder:

    $ virtualenv venv -p python && source venv/bin/activate && python -m pip install -r requirements.txt

胰腺分割模型是两阶段模型，一阶段粗分割模型和二阶段精分割模型
## 运行demo
### 1.下载模型训练的数据和预训练模型，运行下列命令, 会在train文件夹下生成train_data和checkpoints存放训练数据和预训练模型：
    $ USER_NAME=${USER} make download 

### 2. 也可以通过下列命令下载原数据，自行生成模型训练数据，放在train_data文件夹下，这里一阶段和二阶段模型训练使用的同一个生成的npz数据:
    $ USER_NAME=${USER} make generate_data

### 3. 训练模型,有两阶段模型，所以有两个命令分别训练模型，可以在配置文件里修改配置:
#### 胰腺一阶段分割模型训练(note: torch >=1.7)
    $ USER_NAME=${USER} make train_heart_seg_first

#### 胰腺二阶段分割模型训练
    $ USER_NAME=${USER} make train_heart_seg_second


## 模型pth转换成静态图pt文件, 自行替换模型路径和模型配置路径和输出模型路径。

    $ USER_NAME=${USER} model_path=./checkpoints/v1/epoch_1.pth config_path=./config/seg_3d_config.py  output_path=./checkpoints/v1 make save_torchscript

## 文件结构说明

```
├── config
│   ├── seg_heart_first_config.py
│   └── seg_heart_second_config.py
├── custom
│   ├── dataset
│   │   ├── dataset_ms.py
│   │   ├── dataset.py
│   │   └── __init__.py
│   ├── __init__.py
│   ├── model
│   │   ├── backbones
|   |   |  |── ResUnet.py
|   |   |  └── Unet3D.py
│   │   ├── __init__.py
│   │   ├── msnet
|   |   |  |── ms_head.py
|   |   |  |── ms_net.py
|   |   |  └── ms_network.py
│   │   ├── seg_head.py
│   │   └── seg_network.py
│   └── utils
│       ├── eda.py
│       ├── generate_dataset_ms.py
│       ├── generate_dataset.py
│       ├── __init__.py
│       ├── save_torchscript.py
│       ├── split_dataset.py
│       └── test_aug.py
├── Makefile
├── README.md
├── requirements.txt
├── run_dist.sh
└── train.py
```


- train.py: 训练代码入口，需要注意的是，在train.py里import custom，训练相关需要注册的模块可以直接放入到custom文件夹下面，会自动进行注册; 一般来说，训练相关的代码务必放入到custom文件夹下面!!!<br>

- ./custom/dataset/dataset.py: dataset类，需要@DATASETS.register_module进行注册方可被识别,一阶段模型dataset<br>

- ./custom/dataset/dataset_ms.py: dataset类，需要@DATASETS.register_module进行注册方可被识别,二阶段模型dataset<br>

- ./custom/dataset/generate_dataset_ms.py: 从原始数据生成输入到模型的数据

- ./custom/model/backbones/ResUnet.py: 模型backbone，需要@BACKBONES.register_module进行注册方可被识别

- ./custom/model/backbones/Unet3D.py: 模型backbone，需要@BACKBONES.register_module进行注册方可被识别

- ./custom/model/seg_head.py: 一阶段模型head文件，需要@HEADS.register_module进行注册方可被识别

- ./custom/model/seg_network.py: 一阶段整个网络部分，训练阶段构建模型，forward方法输出loss的dict, 通过@NETWORKS.register_module进行注册

- ./custom/model/msnet/ms_head.py: 二阶段模型head文件，需要@HEADS.register_module进行注册方可被识别

- ./custom/model/msnet/ms_network.py: 二阶段整个网络部分，训练阶段构建模型，forward方法输出loss的dict, 通过@NETWORKS.register_module进行注册

- ./custom/model/msnet/ms_net.py: 二阶段模型backbone，需要@BACKBONES.register_module进行注册方可被识别

- ./config/seg_heart_first_config.py: 一阶段模型配置文件

- ./config/seg_heart_second_config.py: 二阶段模型配置文件

- run_dist.sh: 分布式训练的运行脚本
