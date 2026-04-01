# from mmcv.utils import Registry
from mmengine.registry import Registry
BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
LOSSES = Registry('loss')
NETWORKS = Registry('network')