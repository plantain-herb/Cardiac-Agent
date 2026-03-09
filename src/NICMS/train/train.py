import warnings
warnings.filterwarnings("ignore")
import argparse
import os
# from mmcv import Config
from mmengine import Config
import copy
import cv2
import inspect
import itertools
import logging
import mmcv
import multiprocessing as python_multiprocessing
import numpy as np
import os.path as osp
import pickle
import queue
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
import torchvision
import traceback
from collections import abc, defaultdict, OrderedDict
from custom.dataset.utils import DefaultSampleDataset, build_dataset
from custom.model.utils import build_network
from functools import partial
from mmcv.parallel import DataContainer as DC
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, collate
from mmcv.runner import DistSamplerSeedHook, init_dist, get_dist_info, Hook, OptimizerHook
from mmcv.runner.base_runner import BaseRunner
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.utils import get_host_info
# from mmcv.utils import Registry
from mmengine.registry import Registry
from torch import nn
from torch._utils import _flatten_dense_tensors, _take_tensors, _unflatten_dense_tensors
from torch._utils import ExceptionWrapper
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler, _utils, DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL, signal_handling
from torch.utils.data._utils.worker import ManagerWatchdog, WorkerInfo
get_worker_info = _utils.worker.get_worker_info
default_collate = _utils.collate.default_collate
logger_initialized = {}
TRAINNERS = Registry('trainner')
OPTIMIZERS = Registry('optimizers')
RUNNERS = Registry('runners')
DATALOADERS = Registry('dataloaders')
def register_torch_optimizers():
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            OPTIMIZERS.register_module()(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers

TORCH_OPTIMIZERS = register_torch_optimizers()


def build_trainner(cfg):
    trainner_config = cfg.pop('trainner')
    cfg = dict(cfg=cfg)
    trainner = build_from_cfg(trainner_config, TRAINNERS, cfg)
    return trainner

def get_logger(name, log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, "w")
        handlers.append(file_handler)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            "logger should be either a logging.Logger object, str, " f'"silent" or None, but got {type(logger)}'
        )

@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            if self.batch_processor is None:
                outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
            else:
                outputs = self.batch_processor(self.model, data_batch, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('"batch_processor()" or "model.train_step()"' ' must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                if self.batch_processor is None:
                    outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
                else:
                    outputs = self.batch_processor(self.model, data_batch, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('"batch_processor()" or "model.val_step()"' ' must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        self._max_epochs = max_epochs
        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s', get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        self.call_hook('before_run')

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(f'runner has no method named "{mode}" to run an ' 'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError('mode in workflow must be a str, but got {}'.format(type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epochs:
                        return
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(
        self, out_dir, filename_tmpl='epoch_{}.pth', save_optimizer=True, meta=None, create_symlink=True
    ):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            mmcv.symlink(filename, osp.join(out_dir, 'latest.pth'))


def get_batchsize_from_dict(input_dict: dict):
    """
    Args:
        input_dict: dict()

    Returns: num_samples

    """
    num_samples_list = []
    for key, value in input_dict.items():
        if type(value) in [list, tuple]:
            for i in value:
                if torch.is_tensor(i) or isinstance(i, DC):
                    num_samples_list.append(len(i))  # 兼容mmdetection MultiScaleFlipAug
        elif torch.is_tensor(value) or isinstance(value, DC):
            num_samples_list.append(len(value))
    num_samples = np.argmax(np.bincount(np.array(num_samples_list)))
    return num_samples


class SSDataParallel(MMDataParallel):

    def __init__(self, *args, dim=0, **kwargs):
        super(SSDataParallel, self).__init__(*args, dim=dim, **kwargs)
        self.dim = dim

    def single_test(self, *inputs, **kwargs):
        """add single_test.

        The main difference lies in the CPU inference where the datas in
        :class:`DataContainers` will still be gathered.
        """
        if not self.device_ids:
            # We add the following line thus the module could gather and
            # convert data containers as those in GPU inference
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module.single_test(*inputs[0], **kwargs[0])
        else:
            return super().single_test(*inputs, **kwargs)


class SSDistributedDataParallel(MMDistributedDataParallel):

    def single_test(self, *inputs, **kwargs):
        if self.require_forward_param_sync:
            self._sync_buffers()

        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.single_test(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.single_test(*inputs, **kwargs)

        self.require_forward_param_sync = False

        return output

def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)

def single_gpu_test(model, data_loader, show=False):
    if show:
        raise NotImplementedError('not inplemented for plot during validation, please　' 'plot result offline')
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model.single_test(**data)
        results.append(result)

        if show:
            model.module.show_result(data, result)

        batch_size = get_batchsize_from_dict(data)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model.single_test(**data)
        results.append(result)

        if rank == 0:
            batch_size = get_batchsize_from_dict(data)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [part_tensor.new_zeros(shape_max) for _ in range(world_size)]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results

def cast_tensor_type(inputs, src_type, dst_type):
    if isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type)
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({k: cast_tensor_type(v, src_type, dst_type) for k, v in inputs.items()})
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs

class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

@DATALOADERS.register_module()
class SampleDataLoader(object):
    __initialized = False

    def __init__(
        self,
        dataset: DefaultSampleDataset,
        batch_size=1,
        source_batch_size=1,
        imgs_per_gpu=2,
        shuffle=False,
        sampler=None,
        num_workers=0,
        collate_fn=None,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        source_thread_count=1,
        source_prefetch_count=2,
        dist_info=None
    ):
        torch._C._log_api_usage_once('python.data_loader')
        assert isinstance(dataset, DefaultSampleDataset), 'dataset must be instance of DefaultSampleDataset.'

        if num_workers < 0:
            raise ValueError(
                'num_workers option should be non-negative; '
                'use num_workers=0 to disable multiprocessing.'
            )

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        self.dataset = dataset
        self.num_workers = num_workers
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.source_thread_count = source_thread_count
        self.source_prefetch_count = source_prefetch_count

        source_sampler = sampler

        if source_sampler is None:  # give default samplers
            if shuffle:
                source_sampler = RandomSampler(dataset)
            else:
                source_sampler = SequentialSampler(dataset)
        self._origin_source_sampler = source_sampler
        source_sampler = BatchSampler(source_sampler, source_batch_size, drop_last)

        if dist_info is None:
            sampler = RandomSampler(range(dataset.sampled_data_count))
        else:
            rank, world_size = dist_info
            sampler = DistributedSampler(range(dataset.sampled_data_count), world_size, rank, shuffle=False)
            batch_size = imgs_per_gpu
        sampler = BatchSampler(sampler, batch_size, drop_last=drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self._source_sampler = source_sampler
        self._sampler = sampler
        self.batch_sampler = sampler

        if collate_fn is None:
            collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn
        self.__initialized = True

    @property
    def sampler(self):
        return self._origin_source_sampler

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if not multiprocessing._supports_context:
                    raise ValueError(
                        'multiprocessing_context relies on Python >= 3.4, with '
                        'support for different start methods'
                    )

                if isinstance(multiprocessing_context, str):
                    valid_start_methods = multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            (
                                'multiprocessing_context option '
                                'should specify a valid start method in {}, but got '
                                'multiprocessing_context={}'
                            ).format(valid_start_methods, multiprocessing_context)
                        )
                    multiprocessing_context = multiprocessing.get_context(multiprocessing_context)

                if not isinstance(multiprocessing_context, python_multiprocessing.context.BaseContext):
                    raise ValueError(
                        (
                            'multiprocessing_context option should be a valid context '
                            'object or a string specifying the start method, but got '
                            'multiprocessing_context={}'
                        ).format(multiprocessing_context)
                    )
            else:
                raise ValueError(
                    (
                        'multiprocessing_context can only be used with '
                        'multi-process loading (num_workers > 0), but got '
                        'num_workers={}'
                    ).format(self.num_workers)
                )

        self.__multiprocessing_context = multiprocessing_context

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'batch_sampler', 'sampler', 'drop_last', 'dataset'):
            raise ValueError(
                '{} attribute should not be set after {} is '
                'initialized'.format(attr, self.__class__.__name__)
            )

        super(SampleDataLoader, self).__setattr__(attr, val)

    def __iter__(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)

    @property
    def _index_sampler(self):
        return self._sampler

    @property
    def _source_index_sampler(self):
        return self._source_sampler

    def __len__(self):
        return len(self._index_sampler)  # with iterable-style dataset, this will error

class _BaseDataLoaderIter(object):

    def __init__(self, loader):
        self._dataset = loader.dataset
        self._index_sampler = loader._index_sampler
        self._source_index_sampler = loader._source_index_sampler
        self._num_workers = loader.num_workers
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._source_sample_iter_cycle = itertools.cycle(self._source_index_sampler)
        self._batch_size = loader.batch_size
        self._base_seed = torch.empty((), dtype=torch.int64).random_().item()

    def __iter__(self):
        return self

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def __next__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self._index_sampler)

    def __getstate__(self):
        raise NotImplementedError('{} cannot be pickled', self.__class__.__name__)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):

    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0
        self._source_datas = []
        self._idx_sod = 0

    def __next__(self):
        index = self._next_index()  # may raise StopIteration
        data = self._get_data(index)
        return data

    def _get_data(self, index):
        data = []
        idx_batch = 0
        while idx_batch < self._batch_size:
            while len(self._source_datas) != 0:
                self._idx_sod = self._idx_sod % len(self._source_datas)
                idx, sod = self._source_datas[self._idx_sod]
                sample_data = self._dataset.sample_source_data(idx, sod)
                if sample_data is None:
                    self._source_datas.pop(self._idx_sod)
                else:
                    self._source_datas[self._idx_sod][0] += 1
                    self._idx_sod += 1
                    idx_batch += 1
                    data.append(sample_data)
                    if idx_batch == self._batch_size:
                        break
            else:
                source_index = next(self._source_sample_iter_cycle)
                source_datas = [self._dataset[idx] for idx in source_index]
                self._source_datas = [[0, sod] for sod in source_datas]
                self._idx_sod = 0

        data = self._collate_fn(data)
        return data

    next = __next__  # Python 2 compatibility


def _cycle_push_loop(data_iter, data_queue, done_event):
    try:
        while not done_event.is_set():
            data = next(data_iter)
            while not done_event.is_set():
                try:
                    data_queue.put(data, timeout=MP_STATUS_CHECK_INTERVAL)
                    break
                except queue.Full:
                    continue
    except:  # noqa: E722
        traceback.print_exc()


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):

    def __init__(self, loader):
        super(_MultiProcessingDataLoaderIter, self).__init__(loader)

        assert self._num_workers > 0

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._source_index_queue = multiprocessing_context.Queue(len(self._dataset))
        self._source_index_done_event = threading.Event()
        self._source_index_push_thread = threading.Thread(
            target=_cycle_push_loop,
            args=(self._source_sample_iter_cycle, self._source_index_queue, self._source_index_done_event)
        )
        self._source_index_push_thread.daemon = True
        self._source_index_push_thread.start()

        self._source_thread_count = loader.source_thread_count
        self._source_prefetch_count = loader.source_prefetch_count

        self._worker_init_fn = loader.worker_init_fn
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        self._worker_result_queue = multiprocessing_context.Queue()
        self._worker_pids_set = False
        self._shutdown = False
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        # information about data not yet yielded, i.e., model_zoo w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._task_info = {}
        self._tasks_outstanding = 0  # always equal to count(v for v in task_info.values() if len(v) == 1)
        self._workers_done_event = multiprocessing_context.Event()

        self._index_queues = []
        self._workers = []

        self._workers_status = []
        for i in range(self._num_workers):
            index_queue = multiprocessing_context.Queue()
            # index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=_worker_loop,
                args=(
                    self._batch_size, self._dataset, index_queue, self._source_index_queue, self._worker_result_queue,
                    self._workers_done_event, self._collate_fn, self._base_seed + i, self._worker_init_fn, i,
                    self._num_workers, self._source_thread_count, self._source_prefetch_count
                )
            )
            w.daemon = True

            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)
            self._workers_status.append(True)

        self._data_queue = self._worker_result_queue

        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True

        # prime the prefetch loop
        for _ in range(2 * self._num_workers):
            self._try_put_index()

    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._shutdown_worker(worker_id)
            if len(failed_workers) > 0:
                pids_str = ', '.join(str(w.pid) for w in failed_workers)
                raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str))
            if isinstance(e, queue.Empty):
                return (False, None)
            raise

    def _get_data(self):
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self._timeout))
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def __next__(self):
        while True:
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[worker_id]:  # has data or is still active
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                self._shutdown_workers()
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data, )
            else:
                del self._task_info[idx]
                return self._process_data(data)

    next = __next__  # Python 2 compatibility

    def _try_put_index(self):
        assert self._tasks_outstanding < 2 * self._num_workers
        try:
            index = self._next_index()
        except StopIteration:
            return
        for _ in range(self._num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx, )
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_index()
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        return data

    def _shutdown_worker(self, worker_id):

        assert self._workers_status[worker_id]

        q = self._index_queues[worker_id]
        q.put(None)

        self._workers_status[worker_id] = False

    def _shutdown_workers(self):
        python_exit_status = _utils.python_exit_status
        if python_exit_status is True or python_exit_status is None:
            return
        if not self._shutdown:
            self._shutdown = True
            try:
                # Exit workers now.
                self._workers_done_event.set()
                for worker_id in range(len(self._workers)):
                    if self._workers_status[worker_id]:
                        self._shutdown_worker(worker_id)
                for w in self._workers:
                    w.join()
                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()

                self._source_index_done_event.set()
                self._source_index_push_thread.join()
                self._source_index_queue.cancel_join_thread()
                self._source_index_queue.close()
            finally:
                if self._worker_pids_set:
                    _utils.signal_handling._remove_worker_pids(id(self))
                    self._worker_pids_set = False

    def __del__(self):
        self._shutdown_workers()


_worker_info = get_worker_info()


def _source_data_loop(index_queue, data_queue, dataset, done_event):
    try:
        while not done_event.is_set():
            index = index_queue.get()
            data = [dataset[idx] for idx in index]
            while not done_event.is_set():
                try:
                    data_queue.put(data, timeout=MP_STATUS_CHECK_INTERVAL)
                    break
                except queue.Full:
                    continue
    except:  # noqa: E722
        traceback.print_exc()


def _worker_loop(
    batch_size,
    dataset,
    index_queue,
    source_index_queue,
    data_queue,
    done_event,
    collate_fn,
    seed,
    init_fn,
    worker_id,
    num_workers,
    source_thread_count=1,
    source_prefetch_count=2
):
    try:
        signal_handling._set_worker_signal_handlers()

        torch.set_num_threads(1)
        random.seed(seed)
        torch.manual_seed(seed)

        global _worker_info
        _worker_info = WorkerInfo(id=worker_id, num_workers=num_workers, seed=seed, dataset=dataset)

        init_exception = None

        try:
            if init_fn is not None:
                init_fn(worker_id)
        except Exception:
            init_exception = ExceptionWrapper(where='in DataLoader worker process {}'.format(worker_id))
        iteration_end = False

        source_data_queue = queue.Queue(source_prefetch_count)
        source_done_event = threading.Event()
        source_data_prefetch_threads = [
            threading.Thread(
                target=_source_data_loop, args=(source_index_queue, source_data_queue, dataset, source_done_event)
            ) for _ in range(source_thread_count)
        ]

        for t in source_data_prefetch_threads:
            t.daemon = True
            t.start()

        watchdog = ManagerWatchdog()
        source_datas = source_data_queue.get()
        source_datas = [[0, sod] for sod in source_datas]
        idx_sod = 0
        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if r is None:
                # Received the final signal
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                # `done_event` is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue
            idx_sample, index = r
            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    data = []
                    idx_batch = 0
                    while idx_batch < batch_size:
                        while len(source_datas) != 0:
                            idx_sod = idx_sod % len(source_datas)
                            idx, sod = source_datas[idx_sod]
                            sample_data = dataset.sample_source_data(idx, sod)
                            if sample_data is None:
                                source_datas.pop(idx_sod)
                            else:
                                source_datas[idx_sod][0] += 1
                                idx_sod += 1
                                idx_batch += 1
                                data.append(sample_data)
                                if idx_batch == batch_size:
                                    break
                        else:
                            source_datas = source_data_queue.get()
                            source_datas = [[0, sod] for sod in source_datas]
                            idx_sod = 0

                    data = collate_fn(data)
                except Exception as e:
                    data = ExceptionWrapper(where='in DataLoader worker process {}'.format(worker_id))
            data_queue.put((idx_sample, data))
            del data, idx_sample, index, r  # save memory
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()

    source_done_event.set()
    for t in source_data_prefetch_threads:
        t.join()


class DistEvalHook(Hook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self, dataloader, interval=1, gpu_collect=False, **eval_kwargs):
        if not isinstance(dataloader, DataLoader) and not isinstance(dataloader, SampleDataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader or SampleDataLoader, but got {}'.format(type(dataloader))
            )
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.eval_kwargs = eval_kwargs

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        results = multi_gpu_test(
            runner.model, self.dataloader, tmpdir=osp.join(runner.work_dir, '.eval_hook'), gpu_collect=self.gpu_collect
        )
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True


class EvalHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, **eval_kwargs):
        if not isinstance(dataloader, DataLoader) and not isinstance(dataloader, SampleDataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader or SampleDataLoader, but got {}'.format(type(dataloader))
            )
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True



def collect_env():
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    if cuda_available:
        from torch.utils.cpp_extension import CUDA_HOME
        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output('"{}" -V | tail -n1'.format(nvcc), shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            env_info['GPU ' + ','.join(devids)] = name

    gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
    gcc = gcc.decode('utf-8').strip()
    env_info['GCC'] = gcc

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = torch.__config__.show()

    env_info['TorchVision'] = torchvision.__version__

    env_info['OpenCV'] = cv2.__version__

    env_info['MMCV'] = mmcv.__version__
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print('{}: {}'.format(name, val))


def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)

def allreduce_grads(params, coalesce=True, bucket_size_mb=-1):
    grads = [param.grad.data for param in params if param.requires_grad and param.grad is not None]
    world_size = dist.get_world_size()
    if coalesce:
        _allreduce_coalesced(grads, world_size, bucket_size_mb)
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))

class Fp16OptimizerHook(OptimizerHook):
    """FP16 optimizer hook.

    The steps of fp16 optimizer is as follows.
    1. Scale the loss value.
    2. BP in the fp16 model.
    2. Copy gradients from fp16 model to fp32 weights.
    3. Update fp32 weights.
    4. Copy updated parameters from fp32 weights to fp16 model.

    Refer to https://arxiv.org/abs/1710.03740 for more details.

    Args:
        loss_scale (float): Scale factor multiplied with loss.
    """

    def __init__(self, grad_clip=None, coalesce=True, bucket_size_mb=-1, loss_scale=512., distributed=True):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.loss_scale = loss_scale
        self.distributed = distributed

    def before_run(self, runner):
        # keep a copy of fp32 weights
        runner.optimizer.param_groups = copy.deepcopy(runner.optimizer.param_groups)
        # convert model to fp16
        wrap_fp16_model(runner.model)

    def copy_grads_to_fp32(self, fp16_net, fp32_weights):
        """Copy gradients from fp16 model to fp32 weight copy."""
        for fp32_param, fp16_param in zip(fp32_weights, fp16_net.parameters()):
            if fp16_param.grad is not None:
                if fp32_param.grad is None:
                    fp32_param.grad = fp32_param.data.new(fp32_param.size())
                fp32_param.grad.copy_(fp16_param.grad)

    def copy_params_to_fp16(self, fp16_net, fp32_weights):
        """Copy updated params from fp32 weight copy to fp16 model."""
        for fp16_param, fp32_param in zip(fp16_net.parameters(), fp32_weights):
            fp16_param.data.copy_(fp32_param.data)

    def after_train_iter(self, runner):
        # clear grads of last iteration
        runner.model.zero_grad()
        runner.optimizer.zero_grad()
        # scale the loss value
        scaled_loss = runner.outputs['loss'] * self.loss_scale
        scaled_loss.backward()
        # copy fp16 grads in the model to fp32 params in the optimizer
        fp32_weights = []
        for param_group in runner.optimizer.param_groups:
            fp32_weights += param_group['params']
        self.copy_grads_to_fp32(runner.model, fp32_weights)
        # allreduce grads
        if self.distributed:
            allreduce_grads(fp32_weights, self.coalesce, self.bucket_size_mb)
        # scale the gradients back
        for param in fp32_weights:
            if param.grad is not None:
                param.grad.div_(self.loss_scale)
        if self.grad_clip is not None:
            self.clip_grads(fp32_weights)
        # update fp32 params
        runner.optimizer.step()
        # copy fp32 params to the fp16 model
        self.copy_params_to_fp16(runner.model, fp32_weights)


def wrap_fp16_model(model):
    # convert model to fp16
    model.half()
    # patch the normalization layers to make it work in fp32 mode
    patch_norm_fp32(model)
    # set `fp16_enabled` flag
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True


def patch_norm_fp32(module):
    if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.GroupNorm)):
        module.float()
        if isinstance(module, nn.GroupNorm) or torch.__version__ < '1.3':
            module.forward = patch_forward_method(module.forward, torch.half, torch.float)
    for child in module.children():
        patch_norm_fp32(child)
    return module


def patch_forward_method(func, src_type, dst_type, convert_output=True):
    """Patch the forward method of a module.

    Args:
        func (callable): The original forward method.
        src_type (torch.dtype): Type of input arguments to be converted from.
        dst_type (torch.dtype): Type of input arguments to be converted to.
        convert_output (bool): Whether to convert the output back to src_type.

    Returns:
        callable: The patched forward method.
    """

    def new_forward(*args, **kwargs):
        output = func(*cast_tensor_type(args, src_type, dst_type), **cast_tensor_type(kwargs, src_type, dst_type))
        if convert_output:
            output = cast_tensor_type(output, dst_type, src_type)
        return output

    return new_forward


class DistOptimizerHook(OptimizerHook):

    def __init__(self, grad_clip=None, coalesce=True, bucket_size_mb=-1):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        runner.optimizer.step()

def build_runner(runner_config, **kwargs):
    return build_from_cfg(runner_config, RUNNERS, kwargs)

def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 4 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult,
                  dwconv_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.
                  `dwconv_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of depthwise conv layers.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Example:
        >>> import torch
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001)
        >>> optimizer = build_optimizer(model, optimizer_cfg)
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        params = model.parameters()
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg['lr']
        base_wd = optimizer_cfg.get('weight_decay', None)
        # weight_decay must be explicitly specified if mult is specified
        if (
            'bias_decay_mult' in paramwise_options or 'norm_decay_mult' in paramwise_options
            or 'dwconv_decay_mult' in paramwise_options
        ):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
        dwconv_decay_mult = paramwise_options.get('dwconv_decay_mult', 1.)
        named_modules = dict(model.named_modules())
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                # FP16 training needs to copy gradient/weight between master
                # weight copy and model weight, it is convenient to keep all
                # parameters here to align with model.parameters()
                params.append(param_group)
                continue

            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith('.bias'):
                param_group['lr'] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult

            module_name = name.replace('.weight', '').replace('.bias', '')
            if module_name in named_modules and base_wd is not None:
                module = named_modules[module_name]
                # if this Conv2d is depthwise Conv2d
                if isinstance(module, torch.nn.Conv2d) and \
                        module.in_channels == module.groups:
                    param_group['weight_decay'] = base_wd * dwconv_decay_mult
            # otherwise use the global settings

            params.append(param_group)

    optimizer_cfg['params'] = params

    return build_from_cfg(optimizer_cfg, OPTIMIZERS)



def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloader(
    dataset,
    imgs_per_gpu,
    workers_per_gpu,
    num_gpus=1,
    dataloader_cfg=None,
    dist=True,
    shuffle=True,
    seed=None,
    drop_last=False,
    **kwargs
):
    """Build DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        imgs_per_gpu (int): Number of images on each GPU, i.e., batch size of
            each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=shuffle)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None

    if dataloader_cfg is not None:
        default_args = {
            'batch_size': batch_size,
            'imgs_per_gpu': imgs_per_gpu,
            'num_workers': num_workers,
            'collate_fn': partial(collate, samples_per_gpu=imgs_per_gpu),
            'dataset': dataset,
            'shuffle': shuffle,
            'sampler': sampler,
            'worker_init_fn': init_fn,
            'drop_last': drop_last
        }
        if dist:
            default_args['dist_info'] = (rank, world_size)
        else:
            default_args['dist_info'] = None
        for key, value in default_args.items():
            dataloader_cfg[key] = value
        data_loader = build_from_cfg(dataloader_cfg, DATALOADERS, default_args=default_args)
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            shuffle=shuffle if sampler is None else False,
            **kwargs
        )
    return data_loader


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@TRAINNERS.register_module()
class Trainner(object):

    def __init__(self, cfg, runner_config):
        self.cfg = cfg
        self.runner_config = runner_config

        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        if cfg.get('work_dir', None) is None:
            cfg.work_dir = '../'
        if cfg.get('autoscale_lr', None) is not None and cfg.get('autoscale_lr'):
            cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8
        if cfg.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(cfg.launcher, **cfg.dist_params)
        if cfg.get('optimizer_config', None) is None:
            cfg.optimizer_config = {}

        self.distributed = distributed

        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
        logger = get_logger(cfg.get('task_name', ''), log_file=log_file, log_level=cfg.log_level)
        self.timestamp = timestamp

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([('{}: {}'.format(k, v)) for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
        meta['env_info'] = env_info

        # log some basic info
        logger.info('Distributed training: {}'.format(distributed))
        logger.info('Config:\n{}'.format(cfg.text))

        # set random seeds
        cfg.seed = cfg.get('seed', None)
        if cfg.get('seed') is not None:
            logger.info('Set random seed to {}, deterministic: {}'.format(cfg.seed, cfg.deterministic))
            set_random_seed(cfg.seed, deterministic=cfg.deterministic)
        meta['seed'] = cfg.seed
        self.meta = meta

        # 构造模型和datasets
        self.model = self._build_model()
        self.datasets = self._build_datasets(self.cfg.data.train)

        if len(self.cfg.workflow) == 2:
            val_dataset = copy.deepcopy(self.cfg.data.val)
            if self.cfg.data.train.get('pipeline', None) is not None:
                val_dataset.pipeline = self.cfg.data.train.get('pipeline')
            self.datasets.append(self._build_datasets(val_dataset)[0])
        # add an attribute for visualization convenience
        self.model.CLASSES = self.datasets[0].CLASSES if hasattr(self.datasets[0], 'CLASSES') else None

    def _build_model(self):
        self.cfg.train_cfg = self.cfg.get('train_cfg', None)
        self.cfg.test_cfg = self.cfg.get('test_cfg', None)
        model = build_network(self.cfg.model, train_cfg=self.cfg.train_cfg, test_cfg=self.cfg.test_cfg)
        return model

    def _build_datasets(self, data_cfg):
        return [build_dataset(data_cfg)]

    def _build_optimizer(self, model, optimizer):
        return build_optimizer(model, optimizer)

    def _builder_runner(self, model, _batch_processor, optimizer, work_dir, logger, meta):
        return build_runner(
            self.runner_config,
            model=model,
            batch_processor=_batch_processor,
            optimizer=optimizer,
            work_dir=work_dir,
            logger=logger,
            meta=meta
        )

    def _build_dataloader(self, dataset, dist):
        dataloader_cfg = self.cfg.data.get('dataloader', None)
        return build_dataloader(
            dataset=dataset,
            imgs_per_gpu=self.cfg.data.imgs_per_gpu,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            num_gpus=self.cfg.gpus,
            dataloader_cfg=dataloader_cfg,
            dist=dist,
            shuffle=self.cfg.data.shuffle if self.cfg.data.get('shuffle', None) is not None else True,
            seed=self.cfg.seed,
            drop_last=self.cfg.data.drop_last if self.cfg.data.get('drop_last', None) is not None else False,
        )

    def _dist_train(self, model, dataset, cfg, validate=False, logger=None, timestamp=None, meta=None):
        dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
        data_loaders = [(self._build_dataloader(ds, dist=True)) for ds in dataset]
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        if hasattr(model, 'train_step') or hasattr(model, 'val_step'):
            self._batch_processor = None
        model = SSDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters
        )

        # build runner
        optimizer = self._build_optimizer(model, cfg.optimizer)
        runner = self._builder_runner(model, self._batch_processor, optimizer, cfg.work_dir, logger=logger, meta=meta)
        runner.timestamp = timestamp

        # fp16 setting
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config, **fp16_cfg)
        else:
            optimizer_config = DistOptimizerHook(**cfg.optimizer_config)

        # register hooks
        runner.register_training_hooks(cfg.lr_config, optimizer_config, cfg.checkpoint_config, cfg.log_config)
        runner.register_hook(DistSamplerSeedHook())
        # register eval hooks
        if validate:
            val_dataset = build_dataset(cfg.data.val)
            dataloader_cfg = cfg.data.get('dataloader', None)
            val_dataloader = build_dataloader(
                val_dataset,
                imgs_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=True,
                shuffle=False,
                dataloader_cfg=dataloader_cfg
            )
            eval_cfg = cfg.get('evaluation', {})
            runner.register_hook(DistEvalHook(val_dataloader, **eval_cfg))

        if cfg.resume_from:
            runner.resume(cfg.resume_from)
        elif cfg.load_from:
            runner.load_checkpoint(cfg.load_from)
        runner.run(data_loaders, cfg.workflow, cfg.total_epochs)

    def _non_dist_train(self, model, dataset, cfg, validate=False, logger=None, timestamp=None, meta=None):
        # prepare data loaders
        dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
        data_loaders = [(self._build_dataloader(ds, dist=False)) for ds in dataset]
        # put model on gpus
        if hasattr(model, 'train_step') or hasattr(model, 'val_step'):
            self._batch_processor = None
        model = SSDataParallel(model, device_ids=range(cfg.gpus)).cuda()

        # build runner
        optimizer = self._build_optimizer(model, cfg.optimizer)
        runner = self._builder_runner(model, self._batch_processor, optimizer, cfg.work_dir, logger=logger, meta=meta)
        # an ugly walkaround to make the .log and .log.json filenames the same
        runner.timestamp = timestamp
        # fp16 setting
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config, **fp16_cfg, distributed=False)
        else:
            optimizer_config = cfg.optimizer_config
        runner.register_training_hooks(cfg.lr_config, optimizer_config, cfg.checkpoint_config, cfg.log_config)

        if validate:
            val_dataset = build_dataset(cfg.data.val)
            dataloader_cfg = cfg.data.get('dataloader', None)
            val_dataloader = build_dataloader(
                val_dataset,
                imgs_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=True,
                shuffle=False,
                dataloader_cfg=dataloader_cfg
            )
            eval_cfg = cfg.get('evaluation', {})
            runner.register_hook(EvalHook(val_dataloader, **eval_cfg))

        if cfg.resume_from:
            runner.resume(cfg.resume_from)
        elif cfg.load_from:
            runner.load_checkpoint(cfg.load_from)
        runner.run(data_loaders, cfg.workflow, cfg.total_epochs)

    def _train(self, model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None):
        logger = get_logger(name=cfg.get('task_name', ''), log_level=cfg.log_level)

        # start training
        if distributed:
            self._dist_train(model, dataset, cfg, validate=validate, logger=logger, timestamp=timestamp, meta=meta)
        else:
            self._non_dist_train(model, dataset, cfg, validate=validate, logger=logger, timestamp=timestamp, meta=meta)

    def _batch_processor(self, model, data):
        """Process a data batch.

        This method is required as an argument of Runner, which defines how to
        process a data batch and obtain proper outputs.

        Args:
            model (nn.Module): A PyTorch model.
            data (dict): The data batch in a dict.

        Returns:
            dict: A dict containing losses and log vars.
        """
        losses = model(**data)
        loss, log_vars = self._parse_losses(losses)

        num_samples = get_batchsize_from_dict(data)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError('{} is not a tensor or list of tensors'.format(loss_name))

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def run(self):
        self._train(
            self.model,
            self.datasets,
            self.cfg,
            distributed=self.distributed,
            validate=self.cfg.validate,
            timestamp=self.timestamp,
            meta=self.meta
        )

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=""
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--launcher", type=str, default="none")
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    for arg in vars(args):
        cfg[arg] = getattr(args, arg)
    if cfg["launcher"] != "none":
        cfg["find_unused_parameters"] = True
    t = build_trainner(cfg)
    t.run()
