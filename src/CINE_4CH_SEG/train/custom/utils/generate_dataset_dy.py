"""生成模型输入数据."""

import argparse
import glob
import os
import traceback

import numpy as np
import SimpleITK as sitk
import threadpool
import threading
from queue import Queue
from scipy.ndimage.interpolation import zoom
from skimage.morphology import skeletonize
from tqdm import tqdm
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='/home/qutaiping/nas/cmr_heart_models2/datasets/4CH_train')
    #parser.add_argument('--ct_set', type=str, default='0513')
    parser.add_argument('--tgt_path', type=str,
                        default="/home/qutaiping/nas/cmr_heart_models2/processed_data/processed_4CH_dy_stage1")

    args = parser.parse_args()
    return args


# def load_scans(dcm_path):
#     reader = sitk.ImageSeriesReader()
#     name = reader.GetGDCMSeriesFileNames(dcm_path)
#     reader.SetFileNames(name)
#     img = reader.Execute()
#     vol = sitk.GetArrayFromImage(img)
#     spacing = img.GetSpacing()
#     spacing = spacing[::-1]
#     return vol, img, spacing


# def load_nii(nii_path):
#     tmp_img = sitk.ReadImage(nii_path)
#     spacing = tmp_img.GetSpacing()
#     spacing = spacing[::-1]
#     origin_coord = tmp_img.GetOrigin()
#     data_np = sitk.GetArrayFromImage(tmp_img)
#     return data_np, spacing, origin_coord

def load_scans(dcm_path):
    if dcm_path.endswith(".nii.gz"):
        sitk_img = sitk.ReadImage(dcm_path)
    else:
        reader = sitk.ImageSeriesReader()
        name = reader.GetGDCMSeriesFileNames(dcm_path)
        reader.SetFileNames(name)
        sitk_img = reader.Execute()

    spacing = sitk_img.GetSpacing()
    spacing = np.array(spacing[::-1])
    return sitk_img, spacing



def gen_lst(tgt_path):
    save_file = os.path.join(tgt_path, "train.lst")
    data_list = glob.glob(os.path.join(tgt_path, "*.npz"))
    print("num of traindata: ", len(data_list))
    with open(save_file, "w") as f:
        for data in data_list:
            f.writelines(data + "\r\n")


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

## 自适应窗宽窗位，取粗分割结果对应的hu值
# def count_win(vol, coarse_seg):
#     vol_hou = vol * (coarse_seg > 0)
#     hu_points = np.argwhere(vol_hou)
#     hu_sum = vol_hou.sum()
#     hu_mean = hu_sum / hu_points.shape[0]
#     point = vol_hou[hu_points[: , 0], hu_points[: , 1], hu_points[: , 2]]
#     hu_max = point.max()
#     hu_min = point.min()
#     #print(f"hu_sum={hu_sum}, hu_mean:{hu_mean}, hu_max:{hu_max}, hu_min:{hu_min}")
#     win_level = int(hu_mean)
#     a_10 = np.percentile(point, 10)
#     a_90 = np.percentile(point, 90)
#     print('__________', a_10, a_90, hu_min, hu_max)
#     return {'wim_mean':win_level, "hu_min":hu_min, "hu_max":hu_max,"p10":a_10, "p90":a_90 }

def gen_single_data(info):
    try:
        pid, vol_dir, tgt_dir, seg_file,  sitk_lock = info
        flag = True
        for file in [vol_dir, seg_file]:
            if not os.path.exists(file):
                print(f'dont find file {file}')
                flag = False
        if not flag:
            return None
        sitk_lock.acquire()
        vol_img, spacing_vol = load_scans(vol_dir)
        print(spacing_vol)
        vol = sitk.GetArrayFromImage(vol_img)  
        # vol = np.flip(vol, axis=1)
        sitk_lock.release()
        seg_img, spacing = load_scans(seg_file)
        seg = sitk.GetArrayFromImage(seg_img)
        # seg = np.flip(seg, axis=1)
        seg = seg.astype(np.uint8)

        #print("**********************", coarse_file)

        spacing_missing = np.linalg.norm((np.array(spacing_vol) - np.array(spacing)))
        if vol.shape != seg.shape or spacing_missing > 0.1:
            print(f'shape :{vol.shape}!={seg.shape}, spacing: {spacing_vol}!={spacing}')
            return None

        # heart_range = get_seg_max_box(seg > 0, padding=2)

        # heart_range_crop = get_seg_max_box(seg > 0, padding=60)
        # zmin, ymin, xmin, zmax, ymax, xmax = heart_range_crop
        # vol = vol[zmin: zmax, ymin: ymax, xmin: xmax]
        # seg = seg[zmin: zmax, ymin: ymax, xmin: xmax]

        # heart_range[:3] = heart_range[:3] - heart_range_crop[:3]
        # heart_range[3:] = heart_range[3:] - heart_range_crop[:3]
        # if np.sum(seg > 1) == 0:
        #     data_type = 1
        # else:
        #     data_type = 0
        
        # 求边缘
        # seg_img = sitk.GetImageFromArray((seg > 0).astype("uint8"))
        # border = sitk.BinaryContour(seg_img, fullyConnected=True)
        # border_arr = sitk.GetArrayFromImage(border)
        # border_points = np.argwhere(border_arr)
        
        #dic = count_win(vol, coarse_seg)
        
        # print(pid, ':win_level', win_level, win_width, hu_min, hu_max)
        # vol_1 = window_array(vol, win_level, win_width)
        # img = sitk.GetImageFromArray(vol_1)
        # img.CopyInformation(sitk_img)
        # os.makedirs('./debug', exist_ok=True)
        # #sitk.WriteImage(img, os.path.join('./debug', f'{pid[:-7]}_win.nii.gz'))
        # sitk.WriteImage(sitk.GetImageFromArray(vol), os.path.join('./debug', f'{pid[:-7]}.nii.gz'))
        save_file = os.path.join(tgt_dir, f'{pid}.npz')
        np.savez_compressed(
            save_file,
            vol=vol,
            mask=(seg > 0).astype("uint8"),
            #heart_range=heart_range,
            src_spacing=spacing_vol,
            #data_type=data_type,
            #border_points=border_points,
            # win_dic=dic,
            
        )

        # debug
        # os.makedirs('./debug', exist_ok=True)
        # zmin, ymin, xmin, zmax, ymax, xmax = liver_range
        # sitk.WriteImage(sitk.GetImageFromArray(seg[zmin: zmax, ymin: ymax, xmin: xmax]), os.path.join('./debug', f'{pid}-seg.nii.gz'))
        # sitk.WriteImage(border, os.path.join('./debug', f'{pid}_border-seg.nii.gz'))

        ret_string = f'{pid}.npz'
        print(f'{pid} successed')
        return ret_string
    except:
        traceback.print_exc()
        #sitk_lock.release()
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
    # ct_set = args.ct_set.split(',')
    data_lst = []
    # for cs in ct_set:

    # 获取所有image和seg文件
    all_files = os.listdir(src_dir)
    image_files = {re.match(r"(\d{5})\.nii\.gz", f).group(1): f
                   for f in all_files if re.match(r"\d{5}\.nii\.gz", f)}
    seg_files = {re.match(r"(\d{5})untitled_original\.nii\.gz", f).group(1): f
                 for f in all_files if re.match(r"\d{5}untitled_original\.nii\.gz", f)}
    
    # 只处理两者都存在的编号
    for pid in sorted(image_files.keys() & seg_files.keys()):
        vol_path = os.path.join(src_dir, image_files[pid])
        seg_path = os.path.join(src_dir, seg_files[pid])
        info = [pid, vol_path, tgt_dir, seg_path, sitk_lock]
        data_lst.append(info)


    pool = threadpool.ThreadPool(12)
    requests = threadpool.makeRequests(gen_single_data, data_lst)
    ret_lines = [pool.putRequest(req) for req in requests]
    pool.wait()
    print(f'finshed {len(data_lst)} patient.')

    gen_lst(tgt_dir)
