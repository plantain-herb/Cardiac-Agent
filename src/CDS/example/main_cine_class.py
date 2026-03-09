import argparse
import glob
import os
import sys
import tarfile
import traceback

import time
import numpy as np
import pandas 
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer.predictor_cine_class import (
    CineClassificationModel,
    CineClassificationPredictor,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test segmask_3d")

    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--input_path", default="/home/qutaiping/nas/ori_data/diagnosis_first/test_cropped",type=str)
    parser.add_argument("--output_path", default="/home/qutaiping/nas/code/diagnosis_first_refine/results", type=str)
    parser.add_argument(
        "--model_path",
        default=glob.glob("./data/model/*.tar")[0] if len(glob.glob("./data/model/*.tar")) > 0 else None,
        # default=None,
        type=str,
    )
    parser.add_argument(
        "--model_cls_file", 
        default='/home/qutaiping/nas/checkpoints/diagnosis_first_refine4/epoch_28.pth',
        type=str,
    )
    parser.add_argument(
        "--network_cls_file", 
        default="/home/qutaiping/nas/code/diagnosis_first_refine/train/config/cine_class_config.py", 
        type=str,
    )
    parser.add_argument(
        "--config_file", 
        default="/home/qutaiping/nas/code/diagnosis_first_refine/example/cls.yaml",
        type=str, 
    )
    parser.add_argument(
        "--csv_path",
        default="/home/qutaiping/nas/zhaocan/heart_diagnosis/diag_first_data.csv",
        type=str,
        help="Path to the ground truth labels CSV file",
    )
    args = parser.parse_args()
    return args


def inference(
    predictor: CineClassificationPredictor,
    vols: list[np.ndarray]  # List of numpy arrays for 2ch, 4ch, and sa volumes
):
    is_art_cls = predictor.predict(vols)
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


def load_gt_labels(gt_csv_path):
    """Load ground truth labels from a CSV file."""
    gt_data = pd.read_csv(gt_csv_path, dtype={'id': str})
    gt_dict = dict(zip(gt_data['id'], gt_data['label']))
    return gt_dict


def calculate_accuracy(predictions, gt_labels):
    """Calculate accuracy based on predictions and ground truth labels."""
    correct = 0
    total = len(predictions)

    for id, pred_cls in zip(predictions['id'], predictions['pred_cls']):
        if (id in gt_labels) and (gt_labels[id] == pred_cls):
            correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


def main(input_path, output_path, gpu, args):
    # Load ground truth labels
    gt_labels_path = args.csv_path
    gt_labels = load_gt_labels(gt_labels_path)

    if args.model_cls_file is not None and args.network_cls_file is not None and args.config_file is not None:
        model_segUrinary_vessel = CineClassificationModel(
            model_f=args.model_cls_file, network_f=args.network_cls_file, config_f=args.config_file,
        )
        predictor_segUrinary_vessel = CineClassificationPredictor(gpu=gpu, model=model_segUrinary_vessel,)
    else:
        print('tar:', args.model_path)
        with tarfile.open(args.model_path, "r") as tar:
            predictor_segUrinary_vessel = CineClassificationPredictor.build_predictor_from_tar(tar=tar, gpu=gpu)


    os.makedirs(output_path, exist_ok=True)
    patient_cls={}

    #df = pandas.read_excel(gt_labels_path).set_index(["pid"])
    result = {"id":[], "pred_cls": [], "gt":[], "pred_prob":[]}

    for patient_dir in tqdm(os.listdir(input_path)):
        print(f"processing {patient_dir}")

        id = patient_dir
        cls_dict={}
        # try:
        cls_path = os.path.join(input_path, patient_dir)
        
        # sitk_img = load_scans(cls_path)
        # hu_volume = sitk.GetArrayFromImage(sitk_img)

        nii_files = sorted([
            os.path.join(cls_path, f) for f in os.listdir(cls_path)
            if f.endswith('.nii.gz')
        ])
        if len(nii_files) != 3:
            print(f"Warning: {id} has {len(nii_files)} files, expected 3.")
            continue

        vols = []
        for nii_file in nii_files:
            sitk_img = load_scans(nii_file)
            vol = sitk.GetArrayFromImage(sitk_img)
            vols.append(vol.astype(np.float32))     # vols[0]: 2ch, vols[1]: 4ch, vols[2]: sa

        # pred = inference(predictor_segUrinary_vessel, vols)
        preds, avg_pred = inference(predictor_segUrinary_vessel, vols)
        pred_num = np.argmax(avg_pred, 0)
        
        result["id"].append(id)
        result["pred_cls"].append(pred_num)
        result["gt"].append(gt_labels.get(id, -1))
        result["preds_prob"].append(preds)
        result["avg_pred"].append(avg_pred) 

        # patient_cls[patient_dir] = cls_dict
        # except:  # noqa: E722
        #     break
    
    result_df = pandas.DataFrame(result)
    result_df.to_csv(os.path.join(output_path, "result_testset_test.csv"), index=False)

    # Calculate accuracy
    accuracy = calculate_accuracy(result_df, gt_labels)
    print(f"Accuracy: {accuracy:.4f}")


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
        input_path=args.input_path,
        output_path=args.output_path,
        gpu=args.gpu,
        args=args,
    )
