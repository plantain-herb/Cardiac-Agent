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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dicom_dir", type=str, default="/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/ori_data/KNCG_train"
        )
    parser.add_argument(
        "--src_dicom_val_dir", type=str, default="/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/ori_data/KNCG_test"
        )
    parser.add_argument(
        "--src_labels_dir", type=str, default="/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/ori_data/labels"
    )


    parser.add_argument(
        "--src_infar_mask_path", type=str,
        default="/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/ori_data/infar_segtoclass_deal2"
    )

    parser.add_argument(
        "--tgt_dir", type=str, default="/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/processed_data_KNCG_class"
    )
    parser.add_argument(
        "--tgt_val_dir", type=str, default="/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/processed_data_KNCG_class/val"
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


def gen_single_data(info):
    idx, pid, dcm_file, infar_file, save_file, gt = info
    # try:
    dcm_array, dcm_img, spacing = load_scans(dcm_file)
    dcm_shape = dcm_array.shape
    infar_array, infar_img, infar_spacing = load_nii(infar_file)

    dcm_array = dcm_array.astype(np.float32)


    loc = np.where(infar_array > 0)
    constant_shift = 5
    bbox = np.array(
            [np.min(loc[0]), np.max(loc[0]), np.min(loc[1]), np.max(loc[1]), np.min(loc[2]), np.max(loc[2])]
        )
    bbox[0] = max(0, bbox[0] - constant_shift)
    bbox[1] = min(dcm_shape[0], bbox[1] + constant_shift)
    bbox[2] = max(0, bbox[2] - constant_shift)
    bbox[3] = min(dcm_shape[1], bbox[3] + constant_shift)
    bbox[4] = max(0, bbox[4] - constant_shift)
    bbox[5] = min(dcm_shape[2], bbox[5] + constant_shift)

    infar_array[infar_array > 1] = 0

    np.savez_compressed(
        save_file, 
        dcm=dcm_array, 
        infar=infar_array.astype("uint8"), 
        spacing=spacing, 
        infar_box=bbox, 
        gt=gt,
    )

    # img_itk = sitk.GetImageFromArray(dcm_array.astype(np.int32))
    # img_itk.CopyInformation(dcm_img)
    # sitk.WriteImage(img_itk, save_file.replace('.npz', '-dcm.nii.gz'))
    # img_itk = sitk.GetImageFromArray(seg_array.astype(np.uint8))
    # img_itk.CopyInformation(dcm_img)
    # sitk.WriteImage(img_itk, save_file.replace('.npz', '-seg.nii.gz'))

    print(f"{pid} succeed, in process id {idx}")
    # except Exception:
    #     traceback.print_exc()
    #     print(f"{pid} failed , in process id {idx}")


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
    src_infar_dir = os.path.join(os.getcwd(),args.src_infar_mask_path)
    excel_file = os.path.join(src_labels_dir, "KNCG.xlsx")

    df = pandas.read_excel(excel_file)
    column_name = 'ID'
    df[column_name] = df[column_name].astype(str)
    df = df.set_index(["ID"])
    # src_path_arr = [src_dcm_dir, src_dcm_val_dir]
    # tgt_path_arr = [tgt_dir, tgt_val_dir]
    src_path_arr = [src_dcm_dir]
    tgt_path_arr = [tgt_dir]
    # src_path_arr = [src_dcm_val_dir]
    # tgt_path_arr = [tgt_val_dir]

    for i in range(len(src_path_arr)):
        for pid in os.listdir(src_path_arr[i]):
            pid = pid.replace(".nii.gz","")
            # if pid != '52216149':
            #     continue
            infar_file = os.path.join(src_infar_dir, pid + ".nii.gz")
            dcm_file = os.path.join(src_path_arr[i], pid + ".nii.gz")
            save_file = os.path.join(tgt_path_arr[i], f"{pid}.npz")
            gt = [df.loc[pid, "LABEL"]]
            info = [idx, pid, dcm_file, infar_file, save_file, gt]
            idx += 1
            data_lst.append(info)

        pool = threadpool.ThreadPool(1)
        requests = threadpool.makeRequests(gen_single_data, data_lst)
        ret_lines = [pool.putRequest(req) for req in requests]
        pool.wait()

    gen_lst(args.tgt_dir, "train")
    # gen_lst(args.tgt_val_dir, "validation")
