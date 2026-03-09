"""生成模型输入数据."""

import argparse
import glob
import os
import traceback

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import threadpool
import pandas
import multiprocessing

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dicom_dir", type=str, default="/home/qutaiping/nas/ori_data/diagnosis_second/patients_refine/patients"
    )
    parser.add_argument(
        "--src_dicom_val_dir", type=str, default="/home/qutaiping/nas/ori_data/diagnosis_second/split_datasets/val"
    )
    parser.add_argument(
        "--src_labels_dir", type=str, default="/home/qutaiping/nas/zhaocan/heart_diagnosis"
    )
    parser.add_argument(
        "--tgt_dir", type=str, default="/home/qutaiping/nas/processed_data/processed_data_diag_second_refine"
    )
    parser.add_argument(
        "--tgt_val_dir", type=str, default="/home/qutaiping/nas/processed_data/processed_data_diag_second/val"
    )

    args = parser.parse_args()
    return args

def load_nii(nii_path):
    tmp_img = sitk.ReadImage(nii_path)
    spacing = tmp_img.GetSpacing()
    spacing = spacing[::-1]
    data_np = sitk.GetArrayFromImage(tmp_img)
    return data_np, tmp_img, spacing


def load_scans(dcm_path):
    if dcm_path.endswith(".nii.gz"):
        sitk_img = sitk.ReadImage(dcm_path)
    else:
        reader = sitk.ImageSeriesReader()
        name = reader.GetGDCMSeriesFileNames(dcm_path)
        reader.SetFileNames(name)
        sitk_img = reader.Execute()
    
    dcm_array = sitk.GetArrayFromImage(sitk_img)
    spacing = np.array(sitk_img.GetSpacing()[::-1])
    return dcm_array, sitk_img, spacing


def gen_lst(tgt_path, filename):
    save_file = os.path.join(tgt_path, filename+".lst")
    data_list = glob.glob(os.path.join(tgt_path, "*.npz"))
    print("num of " + filename + "data: ", len(data_list))
    with open(save_file, "w") as f:
        for data in data_list:
            f.writelines(data + "\r\n")

def create_zero_image():
    shape = (90, 200, 200)
    spacing = (0.1123, 0.9375, 0.9375)
    origin = (0.0, 0.0, 0.0)
    direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)

    arr = np.zeros(shape, dtype=np.int16)

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    return img

def gen_single_data(info):
    idx, id, dcm_files, save_file, gt = info
    
    dcm_arrays = []
    spacings = []
    
    for dcm_path in dcm_files:
        dcm_array, _, spacing = load_scans(dcm_path)
        dcm_arrays.append(dcm_array.astype(np.float32))
        spacings.append(spacing)

    np.savez_compressed(
        save_file,
        cine_4 = dcm_arrays[0],    # cine 4ch
        cine_s = dcm_arrays[1],    # cine sa
        lge_s = dcm_arrays[2],    # lge sa
        gt = gt,
        spacing = spacings,
    )

    print(f"{id} succeed, in process idx {idx}")


if __name__ == "__main__":
    args = parse_args()

    tgt_dir = os.path.join(os.getcwd(), args.tgt_dir)
    tgt_val_dir = os.path.join(os.getcwd(), args.tgt_val_dir)
    os.makedirs(tgt_dir, exist_ok=True)
    os.makedirs(tgt_val_dir, exist_ok=True)
    data_lst = []
    idx = 0

    src_dcm_dir = os.path.join(os.getcwd(), args.src_dicom_dir)
    src_dcm_val_dir = os.path.join(os.getcwd(), args.src_dicom_val_dir)
    src_labels_dir = os.path.join(os.getcwd(), args.src_labels_dir)
    excel_file = os.path.join(src_labels_dir, "diag_second_data.csv")

    column_name = 'id'
    df = pandas.read_csv(excel_file, dtype={column_name: str})
    df = df.set_index([column_name])

    for id in os.listdir(src_dcm_dir):
        id_path = os.path.join(src_dcm_dir, id)
        if not os.path.isdir(id_path):
            continue

        save_file = os.path.join(tgt_dir, f"{id}.npz")
        if os.path.exists(save_file):
            print(f"{save_file} 已存在，跳过。")
            continue

        dcm_files = [os.path.join(id_path, f) for f in os.listdir(id_path)]

        # 检查是否缺失 2ch 或 4ch 序列
        # required_sequences = ['2ch', '4ch']
        # for seq in required_sequences:
        #     if not any(seq in os.path.basename(dcm_file) for dcm_file in dcm_files):
        #         print(f"Warning: {id} 缺失 {seq} 序列，使用全0图像替代。")
        #         zero_image = create_zero_image()
        #         # zero_seg = create_zero_image("seg")

        #         zero_image_path = os.path.join(id_path, f"{id}_{seq}_image.nii.gz")
        #         # zero_seg_path = os.path.join(id_path, f"{id}_{seq}_seg.nii.gz")

        #         sitk.WriteImage(zero_image, zero_image_path)
        #         # sitk.WriteImage(zero_seg, zero_seg_path)

        #         dcm_files.append(zero_image_path)
        #         # seg_files.append(zero_seg_path)
        
        dcm_files = sorted(dcm_files)
        # seg_files = sorted(seg_files)

        if len(dcm_files) != 3:
            print(f"Warning: {id} has {len(dcm_files)} image files, expected 3.")
            continue

        gt = [df.loc[id, "label"]]
        # info = [idx, id, dcm_files, seg_files, save_file, gt]
        info = [idx, id, dcm_files, save_file, gt]
        idx += 1
        data_lst.append(info)

        # pool = threadpool.ThreadPool(1)
        # requests = threadpool.makeRequests(gen_single_data, data_lst)
        # ret_lines = [pool.putRequest(req) for req in requests]
        # pool.wait()


    # for id in os.listdir(src_dcm_val_dir):
    #     id_path = os.path.join(src_dcm_val_dir, id)
    #     if not os.path.isdir(id_path):
    #         continue

    #     save_file = os.path.join(tgt_val_dir, f"{id}.npz")
    #     if os.path.exists(save_file):
    #         print(f"{save_file} 已存在，跳过。")
    #         continue

    #     dcm_files = [os.path.join(id_path, f) for f in os.listdir(id_path)]
        
    #     # seg_files = []
    #     # for dcm_file in dcm_files:
    #     #     seg_file = dcm_file.replace('_image.nii.gz', '_seg.nii.gz')
    #     #     seg_files.append(seg_file)

    #     # 检查是否缺失 2ch 或 4ch 序列
    #     # required_sequences = ['2ch', '4ch']
    #     # for seq in required_sequences:
    #     #     if not any(seq in os.path.basename(dcm_file) for dcm_file in dcm_files):
    #     #         print(f"Warning: {id} 缺失 {seq} 序列，使用全0图像替代。")
    #     #         zero_image = create_zero_image()
    #     #         # zero_seg = create_zero_image("seg")

    #     #         zero_image_path = os.path.join(id_path, f"{id}_{seq}_image.nii.gz")
    #     #         # zero_seg_path = os.path.join(id_path, f"{id}_{seq}_seg.nii.gz")

    #     #         sitk.WriteImage(zero_image, zero_image_path)
    #     #         # sitk.WriteImage(zero_seg, zero_seg_path)

    #     #         dcm_files.append(zero_image_path)
    #     #         # seg_files.append(zero_seg_path)
        
    #     dcm_files = sorted(dcm_files)
    #     # seg_files = sorted(seg_files)
        
    #     if len(dcm_files) != 3:
    #         print(f"Warning: {id} has {len(dcm_files)} image files, expected 3.")
    #         # continue
            
    #     gt = [df.loc[id, "label"]]
    #     # info = [idx, id, dcm_files, seg_files, save_file, gt]
    #     info = [idx, id, dcm_files, save_file, gt]
    #     idx += 1
    #     data_lst.append(info)

    # 多进程加速处理
    print(f"Start processing {len(data_lst)} samples with multiprocessing ...")
    pool = multiprocessing.Pool(processes=16)  # 根据CPU核数调整
    pool.map(gen_single_data, data_lst)
    pool.close()
    pool.join()

    gen_lst(args.tgt_dir, "add")
    # gen_lst(args.tgt_val_dir, "val")