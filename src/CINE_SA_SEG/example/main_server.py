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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from infer.predictor_DY import CtAbdomenSegDYModel, CtAbdomenSegDYPredictor
except Exception:
    raise


def parse_args():
    parser = argparse.ArgumentParser(description="Test Heart Segmentation")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--input_path", default="/SMMN-Share/data_share/test_data/cine_sa", type=str)
    parser.add_argument("--output_path", default="output", type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--model_file_DY_first", default='/SMMN-Share/data_share/test_data/checkpoints/first_SA/latest.pth', type=str)
    parser.add_argument("--model_file_DY_second", default="/SMMN-Share/data_share/test_data/checkpoints/second_SA/latest.pth", type=str)
    parser.add_argument("--network_file_DY_first", default="cmr_heart_models/train/config/seg_mrdy_stage1.py", type=str)
    parser.add_argument("--network_file_DY_second", type=str, default="cmr_heart_models/train/config/seg_mrdy_stage2.py")

    parser.add_argument("--config_file", type=str, default='cmr_heart_models/example/heart_seg.yaml')
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


def main(input_path, output_path="output"):
    # print(f"input_path: {input_path}")
    try:
        # print(input_path)
        sitk_img, spacing = load_scans(input_path)
        hu_volume = sitk.GetArrayFromImage(sitk_img)  # for testing nii gz format

        pid = os.path.basename(input_path).replace('.nii.gz', '')

        DY_mask = inference(predictor_seg_3d, hu_volume, spacing)
        DY_img = sitk.GetImageFromArray(DY_mask)
        DY_img.CopyInformation(sitk_img)
        sitk.WriteImage(DY_img, os.path.join(output_path, f"{pid}-seg.nii.gz"))

    except:
        traceback.print_exc()


    # print(f"input_path: {input_path}")
args = parse_args()
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

    predictor_seg_3d = CtAbdomenSegDYPredictor(gpu=args.gpu, model=model_seg_3d,)
else:
    with tarfile.open(args.model_path, "r") as tar:
        predictor_seg_3d = CtAbdomenSegDYPredictor.build_predictor_from_tar(tar=tar, gpu=args.gpu)

# main(
#     input_path=input_path,
#     output_path=args.output_path,
# )
    # print("successfully run the main_seg function")
    
# input_path = "/SMMN-Share/data_share/test_data/cine_sa/AZ192068_SBPW_ALLS.nii.gz"
# main_seg(input_path)


# if __name__ == "__main__":
#     args = parse_args()
#     main(
#         input_path=args.input_path, output_path=args.output_path, gpu=args.gpu, args=args,
#     )
