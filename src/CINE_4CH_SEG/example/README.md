## Prerequisites
- python
- virtualenv

## Installation

In this folder:

    $ virtualenv venv -p python && source venv/bin/activate && python -m pip install -r requirements.txt

## run demo

    $ USER_NAME=${USER} VERSION=m5 make

## note: main.py 为matrix工程封装所用文件，输出分割结果和json文件   main_infer.py 为模型推理所用文件，输出分割结果
