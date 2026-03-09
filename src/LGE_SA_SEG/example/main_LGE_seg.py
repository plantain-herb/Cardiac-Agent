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
    from infer.predictor_LGE_seg import LGESegHeartModel, LGESegHeartPredictor
except Exception:
    raise


def parse_args():
    parser = argparse.ArgumentParser(description="Test Heart Segmentation")
    parser.add_argument("--gpu", default=0, type=int)
    # parser.add_argument("--input_path", default="/SMMN-Share/data_share/test_data/lge_sa", type=str)
    # parser.add_argument("--output_path", default="~/demo_outpath", type=str)
    # parser.add_argument("--model_path", default=None, type=str)
    # parser.add_argument("--model_file_heart_first", default='/SMMN-Share/data_share/test_data/checkpoints/first_LGE_SA/latest.pth', type=str)
    # parser.add_argument("--model_file_heart_second", default="/SMMN-Share/data_share/test_data/checkpoints/second_LGE_SA/latest.pth", type=str)
    parser.add_argument("--input_path", default="/home/qutaiping/nas/lh/LGE_SA/test/img", type=str)
    parser.add_argument("--output_path", default="/home/qutaiping/nas/ori_data/LGE_SA/LGE_SA/test/pred-refine", type=str)
    # parser.add_argument("--input_path", default="/home/qutaiping/nas/ori_data/HNWB/LGE/dcm", type=str)
    # parser.add_argument("--output_path", default="/home/qutaiping/nas/ori_data/HNWB/LGE/pred-refine", type=str)

    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--model_file_heart_first", default='/home/qutaiping/nas/checkpoints/first_LGE_seg_refine_agent/latest.pth', type=str)
    parser.add_argument("--model_file_heart_second", default="/home/qutaiping/nas/checkpoints/second_LGE_seg_agent160-refine/latest.pth", type=str)
    parser.add_argument("--network_file_heart_first", default="../train/config/seg_LGE_first_config.py", type=str)
    parser.add_argument("--network_file_heart_second", type=str, default="../train/config/seg_LGE_second_config.py")
    parser.add_argument("--config_file", type=str, default='heart_seg.yaml')
    args = parser.parse_args()
    return args


def inference(predictor: LGESegHeartPredictor, hu_volume):
    pred_array = predictor.predict(hu_volume)
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
        args.model_file_heart_first is not None
        and args.model_file_heart_second is not None
        and args.network_file_heart_first is not None
        and args.network_file_heart_second is not None
        and args.config_file is not None
    ):
        model_seg_2d = LGESegHeartModel(
            model_heart_f=args.model_file_heart_first,
            model_heart_crop_f=args.model_file_heart_second,
            network_heart_f=args.network_file_heart_first,
            network_heart_crop_f=args.network_file_heart_second,
            config_f=args.config_file,
        )
        predictor_seg_2d = LGESegHeartPredictor(gpu=gpu, model=model_seg_2d,)
    else:
        with tarfile.open(args.model_path, "r") as tar:
            predictor_seg_2d = LGESegHeartPredictor.build_predictor_from_tar(tar=tar, gpu=gpu)

    os.makedirs(output_path, exist_ok=True)

    for pid in tqdm(os.listdir(input_path)):
        try:
            dicom_path = os.path.join(input_path, pid)
            print(f"predicting {pid}")
            sitk_img, spacing = load_scans(dicom_path)
            hu_volume = sitk.GetArrayFromImage(sitk_img)  # for testing nii gz format
            pid = pid.replace('.nii.gz', '')

            heart_mask = inference(predictor_seg_2d, hu_volume)
            print(heart_mask.shape,hu_volume.shape)
            heart_img = sitk.GetImageFromArray(heart_mask)
            heart_img.CopyInformation(sitk_img)
            sitk.WriteImage(heart_img, os.path.join(output_path, f"{pid}-seg.nii.gz"))
        except:  # noqa: E722
            traceback.print_exc()
            break

if __name__ == "__main__":
    args = parse_args()
    main(
        input_path=args.input_path, output_path=args.output_path, gpu=args.gpu, args=args,
    )
