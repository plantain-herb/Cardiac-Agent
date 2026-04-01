import argparse
import glob
import os
import sys
import tarfile
import traceback

import time
import numpy as np
import pandas 
import pydicom
import SimpleITK as sitk
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer.predictor_LGE_class import (
    LGEClassificationModel,
    LGEClassificationPredictor,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test segmask_3d")

    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument(
        "--input_dicom_path", default="/home/can/anzhen/datasets/CMR0515/NII/sec/LGE_4Ch_2Ch",type=str
    )
    parser.add_argument("--output_path", default="/home/can/anzhen/datasets/class_LGE_CINE", type=str)
    parser.add_argument(
        "--model_path",
        default=glob.glob("./data/model/*.tar")[0] if len(glob.glob("./data/model/*.tar")) > 0 else None,
        # default=None,
        type=str,
    )
    parser.add_argument(
        "--model_cls_file", 
        default='/home/can/anzhen/datasets/cmr-lge-master/class_checkpoint/LGE_class_v1/epoch_100.pth',
        type=str,
    )
    parser.add_argument(
        "--network_cls_file", 
        default="/home/can/anzhen/datasets/cmr-lge-master/train/config/LGE_class_config.py", 
        type=str,
    )
    parser.add_argument(
        "--config_file",
        default="/home/can/anzhen/datasets/cmr-lge-master/example/cls.yaml",
        type=str,
    )
    args = parser.parse_args()
    return args


def inference(
    predictor: LGEClassificationPredictor,
    hu_volume: np.ndarray,
):
    # st = time.time()
    is_art_cls = predictor.predict(hu_volume)
    # ed = time.time()
    # print(ed-st)

    return is_art_cls


def load_scans(dcm_path):
    if dcm_path.endswith(".nii.gz"):
        sitk_img = sitk.ReadImage(dcm_path)
    else:
        reader = sitk.ImageSeriesReader()
        name = reader.GetGDCMSeriesFileNames(dcm_path)
        reader.SetFileNames(name)
        sitk_img = reader.Execute()
    return sitk_img


def main(input_dicom_path, output_path, gpu, args):
    if args.model_cls_file is not None and args.network_cls_file is not None and args.config_file is not None:
        model_segUrinary_vessel = LGEClassificationModel(
            model_f=args.model_cls_file, network_f=args.network_cls_file, config_f=args.config_file,
        )
        predictor_segUrinary_vessel = LGEClassificationPredictor(gpu=gpu, model=model_segUrinary_vessel,)
    else:
        print('tar:', args.model_path)
        with tarfile.open(args.model_path, "r") as tar:
            predictor_segUrinary_vessel = LGEClassificationPredictor.build_predictor_from_tar(tar=tar, gpu=gpu)


    os.makedirs(output_path, exist_ok=True)
    patient_cls={}

    #df = pandas.read_excel(gt_labels_path).set_index(["pid"])
    result = {"pid":[], "pred_cls": [], "pred_prob":[]}

    for patient_dir in tqdm(os.listdir(input_dicom_path)):
        print(patient_dir)
        # if patient_dir != 'CS591003-CT1598162':
        #     continue
        pid = patient_dir.split('.nii.gz')[0]
        cls_dict={}
        # try:
        cls_path = os.path.join(input_dicom_path, patient_dir)
        
        sitk_img = load_scans(cls_path)
        hu_volume = sitk.GetArrayFromImage(sitk_img)

        # else: 
            # infar_mask = np.ones(hu_volume.shape, dtype="uint8")
            # print("infar not exists, inputs the whole dcm.")
        pred = inference(predictor_segUrinary_vessel, hu_volume)
        pred_num = np.argmax(pred, 0)
        result["pid"].append(pid)
        result["pred_cls"].append(pred_num+1)
        result["pred_prob"].append(pred)
        
        patient_cls[patient_dir] = cls_dict
        # except:  # noqa: E722
        #     break
    
    df = pandas.DataFrame(result)
    # df.to_excel(os.path.join(output_path, "CMR0514-LGE_sa.xlsx"), index=False)
    df.to_csv(os.path.join(output_path, "CMR0515-sec-LGE_4Ch_2Ch.csv"), index=False)


def read_cls_data(path: str):
    result = dict()
    with open(path) as f:
        for line in f.readlines():
            dicom, cls = line.split()
            result[int(cls)] = dicom
    return result


if __name__ == "__main__":
    args = parse_args()
    main(
        input_dicom_path=args.input_dicom_path,
        output_path=args.output_path,
        gpu=args.gpu,
        args=args,
    )
