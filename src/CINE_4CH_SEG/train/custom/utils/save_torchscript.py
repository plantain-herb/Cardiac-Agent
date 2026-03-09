"""生成torchscript很难直接从Config构建的模型进行转换，需要剥离出组件."""

import argparse
import os
import sys

import torch
from mmengine import Config
from ..model.utils import build_network

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import custom  # noqa: F401


def load_model(config, model_path):
    seg_net = build_network(config.model)
    checkpoint = torch.load(model_path)
    seg_net.load_state_dict(checkpoint["state_dict"])
    seg_net = seg_net.cuda()
    seg_net.eval()
    return seg_net


def update_config_batchnorm(cfg):
    from mmcv.utils.config import ConfigDict

    for key in cfg.keys():
        if key == "apply_sync_batchnorm":
            cfg["apply_sync_batchnorm"] = False
        elif isinstance(cfg[key], ConfigDict):
            update_config_batchnorm(cfg[key])


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="epoch_100.pth",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="seg_heart_second_config.py",
    )
    parser.add_argument("--output_path", type=str, default="../../../example/data/model/m1")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["IS_SCRIPTING"] = "1"
    args = parse_args()
    config = Config.fromfile(args.config_path)
    update_config_batchnorm(config)
    model = load_model(config, args.model_path)
    res_unet_jit = torch.jit.script(model)
    res_unet_jit.save(os.path.join(args.output_path, "seg_heart_second.pt"))
