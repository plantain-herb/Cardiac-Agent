"""infer推理所用的mian文件"""
import argparse
import os
import sys
import tarfile
import time
import traceback
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
# from memory_profiler import profile
# from profiler import Profiler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from infer.predictor_DY import CtAbdomenSegDYModel, CtAbdomenSegDYPredictor
except Exception:
    raise

import json
with open('example/id_slice_info_sa.json', 'r') as f:
    nocal_dict = json.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Test Heart Segmentation")
    parser.add_argument("--gpu", default=2, type=int)
    # parser.add_argument("--input_path", default="/SMMN-Share/data_share/test_data/cine_seg_sa/img", type=str)
    # parser.add_argument("--output_path", default="/SMMN-Share/data_share/test_data/cine_seg_sa/pre", type=str)
    parser.add_argument("--input_path", default="/home/qutaiping/nas/ori_data/AZHC/mapping_out/CINE/sa", type=str)
    parser.add_argument("--output_path", default="/home/qutaiping/nas/ori_data/AZHC/mapping_out/CINE/sa_pred2", type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--model_file_DY_first", default='/SMMN-Share/data_share/test_data/checkpoints/cine_seg_first_SA/latest.pth', type=str)
    parser.add_argument("--model_file_DY_second", default="/SMMN-Share/data_share/test_data/checkpoints/cine_seg_second_SA/epoch_45.pth", type=str)
    parser.add_argument("--network_file_DY_first", default="train/config/seg_mrdy_stage1.py", type=str)
    parser.add_argument("--network_file_DY_second", type=str, default="train/config/seg_mrdy_stage2.py")

    parser.add_argument("--config_file", type=str, default='example/heart_seg.yaml')
    args = parser.parse_args()
    return args

def inference(predictor: CtAbdomenSegDYPredictor, hu_volume, spacing):
    pred_array = predictor.DY_predict(hu_volume, spacing)
    return pred_array


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


def main(input_path, output_path, gpu, args):
    if (
        args.model_file_DY_first is not None
        and args.network_file_DY_first is not None
        and args.model_file_DY_second is not None
        and args.network_file_DY_second is not None
        and args.config_file is not None
    ):
        model_seg_3d = CtAbdomenSegDYModel(
            model_DY_f=args.model_file_DY_first,
            model_DY_crop_f=args.model_file_DY_second,
            network_DY_f=args.network_file_DY_first,
            network_DY_crop_f=args.network_file_DY_second,
            config_f=args.config_file,
        )
        predictor_seg_3d = CtAbdomenSegDYPredictor(gpu=gpu, model=model_seg_3d,)
    else:
        with tarfile.open(args.model_path, "r") as tar:
            predictor_seg_3d = CtAbdomenSegDYPredictor.build_predictor_from_tar(tar=tar, gpu=gpu)

    os.makedirs(output_path, exist_ok=True)

    pred_files = os.listdir(output_path)

    for pid in tqdm(os.listdir(input_path)):
        try:
            id_ = pid.split('.nii.gz')[0]
            dicom_path = os.path.join(input_path, pid)
            print(f"predicting {pid}")
            print(dicom_path)
            sitk_img, spacing = load_scans(dicom_path)
            hu_volume = sitk.GetArrayFromImage(sitk_img)  # for testing nii gz format

            seg_id = id_ + '-seg.nii.gz'
            if seg_id in pred_files:
                continue

            # heart_path = os.path.join(input_heart_path, id_ + '_seg.nii.gz')
            # heart_img, _ = load_scans(heart_path)
            # heart_mask = sitk.GetArrayFromImage(heart_img)  # for testing nii gz format
            # heart_mask = np.flip(heart_mask, axis=1)
            # heart_mask[heart_mask != 51] = 0
            # heart_mask[heart_mask > 0] = 1
            # pid = pid.replace('.nii.gz', '')

            pid = pid.replace('_sa_image.nii.gz', '')

            DY_mask = inference(predictor_seg_3d, hu_volume, spacing)
            try:
                nocal = nocal_dict.get(pid)
            except:
                nocal = DY_mask.shape[0] // 10
            DY_mask[-nocal:, :, :] = 0
            DY_mask[:nocal, :, :] = 0
            DY_img = sitk.GetImageFromArray(DY_mask)
            DY_img.CopyInformation(sitk_img)
            sitk.WriteImage(DY_img, os.path.join(output_path, f"{pid}_sa_image-seg.nii.gz"))

            # hu_volume = np.flip(hu_volume, axis=1)
            # img = sitk.GetImageFromArray(hu_volume)
            # img.CopyInformation(sitk_img)
            # sitk.WriteImage(img, os.path.join(output_path, f"{pid}.nii.gz"))
        except:  # noqa: E722
            traceback.print_exc()
            continue

if __name__ == "__main__":
    args = parse_args()
    main(
        input_path=args.input_path, output_path=args.output_path, gpu=args.gpu, args=args,
    )
