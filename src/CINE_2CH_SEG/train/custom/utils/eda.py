"""生成模型输入数据."""

import argparse
import glob
import json
import os
import random
import sys
import traceback

import numpy as np
import SimpleITK as sitk
import threadpool
import threading
from queue import Queue
import torch
import torch.nn.functional as F
from scipy.ndimage import morphology
from scipy.ndimage.interpolation import zoom
from skimage.morphology import skeletonize
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='/home/tx-deepocean/data1/liver_data/Train_Data')
    parser.add_argument('--ct_set', type=str, default='0513')
    parser.add_argument('--tgt_path', type=str,
                        default='')
    args = parser.parse_args()
    return args


def load_scans(dcm_path):
    reader = sitk.ImageSeriesReader()
    name = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(name)
    img = reader.Execute()
    vol = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    spacing = spacing[::-1]
    return vol, img, spacing


def load_nii(nii_path):
    tmp_img = sitk.ReadImage(nii_path)
    spacing = tmp_img.GetSpacing()
    spacing = spacing[::-1]
    origin_coord = tmp_img.GetOrigin()
    data_np = sitk.GetArrayFromImage(tmp_img)
    return data_np, spacing, origin_coord


def get_seg_max_box(seg, padding=None):
    points = np.argwhere(seg > 0)
    if len(points) == 0:
        return np.zeros([0, 0, 0, 0, 0, 0], dtype=np.int_)
    zmin, zmax = np.min(points[:, 0]), np.max(points[:, 0])
    ymin, ymax = np.min(points[:, 1]), np.max(points[:, 1])
    xmin, xmax = np.min(points[:, 2]), np.max(points[:, 2])
    if padding is not None:
        zmin = max(0, zmin - padding)
        zmax = min(seg.shape[0], zmax + padding)
        ymin = max(0, ymin - padding)
        ymax = min(seg.shape[1], ymax + padding)
        xmin = max(0, xmin - padding)
        xmax = min(seg.shape[2], xmax + padding)
    return np.array([zmin, ymin, xmin, zmax, ymax, xmax], dtype=np.int_)

def _get_unique_labels(input_points, count=100):
    # obtaining unique labels from the input listing of points, two different functions got the same results but the latter one is more productive
    #all_labels = np.unique(input_points)

    bin_count = np.bincount(input_points)
    all_labels = set(
        [idx for idx, c_v in enumerate(bin_count) if c_v > count and idx > 0])
    return list(all_labels), bin_count

def _dynamic_window_level(vol, indices):
    points = vol[indices[:, 0], indices[:, 1], indices[:, 2]]
    max_hu = np.max(points)
    min_hu = np.min(points)
    vol = np.clip(vol, min_hu, max_hu)
    vol_ = vol - min_hu
    points = vol_[indices[:, 0], indices[:, 1], indices[:, 2]]
    all_labels, bin_count = _get_unique_labels(points)

    win_min = all_labels[0]
    win_max = all_labels[-1]
    vol_ = np.clip(vol_, win_min, win_max)
    vol_ = 1.0 * (vol_ - win_min) / (win_max - win_min)
    return vol_
    
def gen_single_data(info):
    try:
        pid, vol_dir, tgt_dir, seg_file, sitk_lock = info
        flag = True
        if os.path.exists(os.path.join(tgt_dir, f'{pid}.nii.gz')):
            print(f"pid: {pid} skip already exists")
            try:
                vol, spacing, _ = load_nii(os.path.join(tgt_dir, f'{pid}.nii.gz'))
                mb = vol.size * vol.itemsize / (1024 * 1024)
                if mb < 20:
                    os.remove(os.path.join(tgt_dir, f'{pid}.nii.gz'))
                    print(f"delete: {os.path.join(tgt_dir, {pid}.nii.gz)}")
            except Exception as e:
                print(f"load {pid} error: {e}, delete: {os.path.join(tgt_dir, {pid}.nii.gz)}")
                os.remove(os.path.join(tgt_dir, f'{pid}.nii.gz'))
            return None

        for file in [vol_dir, seg_file]:
            if not os.path.exists(file):
                print(f'dont find file {file}')
                flag = False
        if not flag:
            return None
        sitk_lock.acquire()
        vol, sitk_img, spacing_vol = load_scans(vol_dir)
        sitk_lock.release()
        seg, spacing, _ = load_nii(seg_file)
        seg = seg.astype(np.uint8)
        spacing_missing = np.linalg.norm((np.array(spacing_vol) - np.array(spacing)))
        if vol.shape != seg.shape or spacing_missing > 0.1:
            print(f'shape :{vol.shape}!={seg.shape}, spacing: {spacing_vol}!={spacing}')
            return None

        indices = np.argwhere(seg > 0)
        if indices.shape != 0:
            vol_norm = _dynamic_window_level(vol, indices)
            vol_norm = sitk.GetImageFromArray(vol_norm)
            vol_norm.CopyInformation(sitk_img)
            sitk.WriteImage(vol_norm, os.path.join(tgt_dir, f'{pid}.nii.gz'))

        ret_string = f'{pid}.npz'
        print(f'{pid} successed')
        return ret_string
    except:
        traceback.print_exc()
        sitk_lock.release()
        print(f'{pid} failed')
        return None


def write_list(request, result):
    write_queue.put(result)


if __name__ == '__main__':
    sitk_lock = threading.RLock()
    write_queue = Queue()
    args = parse_args()
    src_dir = args.src_path
    tgt_dir = args.tgt_path
    os.makedirs(tgt_dir, exist_ok=True)
    ct_set = args.ct_set.split(',')
    data_lst = []
    for cs in ct_set:
        src_dcm_dir = os.path.join(src_dir, cs, 'dcm')
        src_seg_dir = os.path.join(src_dir, cs, 'anno')
        for seg_file in sorted(os.listdir(src_seg_dir)):
            if '-seg.nii.gz' not in seg_file:
                print(f'abnormal seg file :{seg_file}')
                continue
            pid = seg_file.replace("_color-seg.nii.gz", "").replace('-seg.nii.gz', '')
            vol_dir = os.path.join(src_dcm_dir, pid)
            seg_file = os.path.join(src_seg_dir, seg_file)
            info = [pid, vol_dir, tgt_dir, seg_file, sitk_lock]
            # gen_single_data(info)
            data_lst.append(info)

    pool = threadpool.ThreadPool(16)
    requests = threadpool.makeRequests(gen_single_data, data_lst, write_list)
    ret_lines = [pool.putRequest(req) for req in requests]
    pool.wait()
    print(f'finshed {len(data_lst)} patient.')
