"""生成模型输入数据."""

import argparse
import glob
import os
from multiprocessing import Pool

from queue import Queue
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from skimage.morphology import skeletonize
import torch
from scipy.ndimage.filters import gaussian_filter
import threadpool
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='/home/qutaiping/nas/ori_data/AZHC/dul/del/2ch/image/train')
    parser.add_argument('--tgt_path', type=str,
                        default="/home/qutaiping/nas/processed_data/processed_2CH_dy_stage2")
    args = parser.parse_args()
    return args


def gen_lst(tgt_path):
    save_file = os.path.join(tgt_path, "train.lst")
    data_list = glob.glob(os.path.join(tgt_path, "*.npz"))
    print("num of traindata: ", len(data_list))
    with open(save_file, "w") as f:
        for data in data_list:
            f.writelines(data + "\r\n")


def load_scans(dcm_path):
    if dcm_path.endswith(".nii.gz"):
        sitk_img = sitk.ReadImage(dcm_path)
    else:
        reader = sitk.ImageSeriesReader()
        name = reader.GetGDCMSeriesFileNames(dcm_path)
        reader.SetFileNames(name)
        sitk_img = reader.Execute()
    spacing = sitk_img.GetSpacing()
    return sitk_img, np.array(spacing)


def binary_dilation(vol, size, iter=1):
    from scipy.ndimage import morphology

    str_3D = np.ones(size, dtype="float32")
    vol_out = morphology.binary_dilation(vol, str_3D, iterations=iter)
    vol_out = vol_out.astype("uint8")
    return vol_out


def get_smooth_seg_by_skeleton(seg):
    sk_seg = skeletonize(seg)
    skeleton_dil = binary_dilation(sk_seg, [3, 3, 3], iter=1)
    seg = (seg + skeleton_dil) > 0
    seg = seg.astype(np.uint8)
    return seg, skeleton_dil, sk_seg

def _get_cross_points(sk_line, shape, radius = 3):
    cross_mask = np.zeros_like(sk_line)
    seed_list = np.argwhere(sk_line > 0)
    def _is_cross(p, cl):
        p_s = p - 1
        p_e = p + 2
        if np.sum(cl[max(0, p_s[0]): min(p_e[0], shape[0]), max(0, p_s[1]): min(p_e[1], shape[1]), max(0, p_s[2]): min(p_e[2], shape[2])]) > 3:
            return True
        else:
            return False
    for p in seed_list:
        if _is_cross(p, sk_line):
            p_s = p - radius
            p_e = p + radius + 1
            cross_mask[p_s[0]:p_e[0], p_s[1]:p_e[1], p_s[2]:p_e[2]] = 1
    return cross_mask

def get_seg_max_box(seg, padding=40):
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

def singe_generate_data(inputs):
    
    dcm_path, mask_path, heart_path, tgt_path, pid = inputs
    # ret_string = f'{pid}.npz'
    # print(f'{pid} successed')
    # return ret_string
    for file in [dcm_path, mask_path, heart_path]:
        if os.path.exists(file)==False:
            print('file not exist', file)
            return
    

    img_itk, spacing_vol = load_scans(dcm_path)
    vol = sitk.GetArrayFromImage(img_itk)
    # vol = np.flip(vol, axis=1)

    mask_img = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(mask_img)
    mask[mask > 2] = 2
    # mask = np.flip(mask, axis=1)

    heart_mask = sitk.GetArrayFromImage(sitk.ReadImage(heart_path))
    # heart_mask = np.flip(heart_mask, axis=1)
    # heart_range_crop = get_seg_max_box(heart_mask == 51, padding=2)
    heart_range_crop = get_seg_max_box(heart_mask > 0, padding=20)
    zmin, ymin, xmin, zmax, ymax, xmax = heart_range_crop
    print(heart_range_crop, vol.shape, pid, np.unique(mask))
    # in_vessel_mask_dila_all = np.zeros_like(mask)
    #切块以加快训练速度
    vol = vol[zmin:zmax,ymin:ymax,xmin:xmax]
    mask = mask[zmin:zmax,ymin:ymax,xmin:xmax]
    heart_mask = heart_mask[zmin:zmax,ymin:ymax,xmin:xmax]


    
    loc = np.where(heart_mask > 0)
    sample_region = [np.min(loc[0]), np.max(loc[0]), np.min(loc[1]), np.max(loc[1]), np.min(loc[2]), np.max(loc[2])]
    tp_points = np.argwhere(mask)
    np.savez_compressed(
        os.path.join(tgt_path, f"{pid}.npz"),
        vol=vol,
        mask=mask,
        bbox=sample_region,
        spacing=spacing_vol[::-1],
        tp_points=tp_points,
    )
    # print(dcm_path.split("/")[-1])
    # return 
    ret_string = f'{pid}.npz'
    print(f'{pid} succeeded')
    return ret_string


def write_list(request, result):
    write_queue.put(result)


if __name__ == "__main__":
    args = parse_args()
    write_queue = Queue()
    src_dir = args.src_path
    tgt_dir = args.tgt_path
    os.makedirs(tgt_dir, exist_ok=True)
    data_lst = []

    all_files = os.listdir(src_dir)
    image_files = {re.match(r"(\d{5})\.nii\.gz", f).group(1): f
                   for f in all_files if re.match(r"\d{5}\.nii\.gz", f)}
    seg_files = {re.match(r"(\d{5})_seg\.nii\.gz", f).group(1): f
                 for f in all_files if re.match(r"\d{5}_seg\.nii\.gz", f)}
    
    for pid in sorted(image_files.keys() & seg_files.keys()):
        vol_path = os.path.join(src_dir, image_files[pid])
        seg_path = os.path.join(src_dir, seg_files[pid])
        heart_path = seg_path
        info = [vol_path, seg_path, heart_path, tgt_dir, pid]
        data_lst.append(info)

    pool = threadpool.ThreadPool(1)
    requests = threadpool.makeRequests(singe_generate_data, data_lst, write_list)
    ret_lines = [pool.putRequest(req) for req in requests]
    pool.wait()
    print(f'finshed {len(data_lst)} patient.')
    gen_lst(tgt_dir)
