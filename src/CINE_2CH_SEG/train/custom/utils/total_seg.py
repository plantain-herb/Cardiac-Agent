import nibabel as nib
from totalsegmentator.python_api import totalsegmentator
import os
import numpy as np
if __name__ == "__main__":

    data_path = '/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/ori_data/CMR_dcm/'
    out_path = '/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/ori_data/CMR_totalseg'
    pids = os.listdir(data_path)
    for pid in pids:
        try:
            cur_id = pid.split('.nii.gz')[0]
            input_path = os.path.join(data_path, pid)
            output_path = os.path.join(out_path, cur_id + '_seg.nii.gz')

            input_img = nib.load(input_path)
            if np.sum(input_img.affine) == 1.0:
                continue
            totalsegmentator(input_img, output_path, task="total_mr")
        except:
            continue
            # nib.save(output_img, output_path)
