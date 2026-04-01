import os
from typing import List, Union
from torch.utils.data import Dataset
from custom.dataset.registry import DATASETS
import inspect
from mmengine.registry import Registry
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
import copy
def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)

def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
    if "type" not in cfg:
        raise KeyError(f'the cfg dict must contain the key "type", but got {cfg}')
    if not isinstance(registry, Registry):
        raise TypeError("registry must be an mmcv.Registry object, " f"but got {type(registry)}")
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError("default_args must be a dict or None, " f"but got {type(default_args)}")

    args = cfg.copy()
    obj_type = args.pop("type")
    if is_str(obj_type):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {registry.name} registry")
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)

import collections
from custom.dataset.registry import PIPELINES


class Compose(object):

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def build_pipelines(pipelines):
    return Compose(pipelines)


@DATASETS.register_module()
class CustomDataset(Dataset):
    """Custom dataset, default data_lst_file is npz file list
    Args:
        data_lst_file (str): The input file of data list.
    """

    CLASSES = None

    def __init__(self, data_lst_file):
        assert os.path.exists(data_lst_file)
        self.data_lst = self._load_files(data_lst_file)

    def __len__(self):
        return len(self.data_lst)

    def _load_file_list(self, dst_list_file):
        data_file_list = []
        with open(dst_list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line and os.path.exists(line) and os.path.isfile(line):
                    data_file_list.append(line)
        assert len(data_file_list) != 0, 'has no avilable file in dst_list_file'
        return data_file_list

    def _load_files(self, file_list: Union[List[str], str]):
        data_file_list = []
        if isinstance(file_list, str):
            return self._load_file_list(file_list)
        elif isinstance(file_list, (tuple, list)):
            for i in file_list:
                data_file_list.extend(self._load_file_list(i))
            return data_file_list
        else:
            raise ValueError('dst_list_file only support str, list, tuple')

    def _load_source_data(self, filename):
        raise NotImplementedError

    def __getitem__(self, index):
        return self._load_source_data(self.data_lst[index])


@DATASETS.register_module()
class DefaultSampleDataset(object):
    """Default SampleDataset for SampleDataLoader All subclasses should
    implement the following APIs:

    - ``sampled_data_count()``
    - ``__getitem__()``
    - ``sample_source_data()``
    - ``source_data_count()``

    Args:
        data_lst_file (str): The input file of data list.
    """

    CLASSES = None

    def __init__(self, data_lst_file):
        self.data_lst_file = data_lst_file

    @property
    def sampled_data_count(self):
        raise NotImplementedError

    def _load_file_list(self, dst_list_file):
        data_file_list = []
        with open(dst_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and os.path.exists(line) and os.path.isfile(line):
                    data_file_list.append(line)
        assert len(data_file_list) != 0, 'has avilable file in dst_list_file'
        return data_file_list

    def _load_files(self, file_list: Union[List[str], str]):
        data_file_list = []
        if isinstance(file_list, str):
            return self._load_file_list(file_list)
        elif isinstance(file_list, (tuple, list)):
            for i in file_list:
                data_file_list.extend(self._load_file_list(i))
            return data_file_list
        else:
            raise ValueError('dst_list_file only support str, list, tuple')

    @property
    def source_data_count(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return self.source_data_count

    def sample_source_data(self, idx, source_data):
        raise NotImplementedError

def _concat_dataset(cfg, default_args=None):
    data_files = cfg['data_file']

    datasets = []
    num_dset = len(data_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        data_cfg['data_file'] = data_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets)


def build_dataset(cfg, default_args=None):
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif isinstance(cfg.get('data_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES if hasattr(datasets[0], 'CLASSES') else None


@DATASETS.register_module()
class RepeatDataset(object):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES if hasattr(dataset, 'CLASSES') else None

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return int(self.times * self._ori_len)

