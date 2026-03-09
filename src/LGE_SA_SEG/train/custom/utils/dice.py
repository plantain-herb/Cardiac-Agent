import numpy as np
import os
import SimpleITK as sitk


def dice_coef(np1, np2):
    """计算两个malb的dice系数"""
    sum1 = np.sum(np1)
    sum2 = np.sum(np2)
    if (sum1 + sum2) == 0:
        return 1
    else:
        dice = (2 * np.sum((np1 == np2) * (np1 > 0) * (np2 > 0))) / (np.sum(np1 > 0) + np.sum(np2 > 0))
        return dice


import numpy as np

def dice_coefficient(mask1, mask2, class_id):
    """
    计算指定类别的Dice系数
    
    参数:
        mask1: 第一个mask数组
        mask2: 第二个mask数组
        class_id: 要计算的类别ID
        
    返回:
        该类别的Dice系数
    """
    mask1_class = (mask1 == class_id).astype(int)
    mask2_class = (mask2 == class_id).astype(int)
    
    intersection = np.sum(mask1_class * mask2_class)
    sum_masks = np.sum(mask1_class) + np.sum(mask2_class)
    
    if sum_masks == 0:
        return 1.0  # 如果两个mask都没有该类，则Dice为1
    return (2.0 * intersection) / sum_masks

def calculate_all_dice(mask1, mask2, num_classes):
    """
    计算所有类别的Dice系数和平均Dice
    
    参数:
        mask1: 第一个mask数组
        mask2: 第二个mask数组
        num_classes: 类别数量(在您的情况中是7)
        
    返回:
        (各类别Dice系数字典, 平均Dice系数)
    """
    dice_scores = {}
    total_dice = 0.0
    valid_classes = 0
    
    for class_id in range(1, num_classes + 1):
        dice = dice_coefficient(mask1, mask2, class_id)
        dice_scores[f'Class {class_id}'] = dice
        
        # 只有当至少一个mask包含该类时才计入平均
        if np.any(mask1 == class_id) or np.any(mask2 == class_id):
            total_dice += dice
            valid_classes += 1
    
    # 计算平均Dice(忽略两个mask都不存在的类别)
    mean_dice = total_dice / valid_classes if valid_classes > 0 else 0
    
    return dice_scores, mean_dice

save_file = "LGE_SA_heart_dice2.lst"
ours_path = '/home/qutaiping/nas/ori_data/LGE_SA/LGE_SA/test/pred-refine'
lb_path = '/home/qutaiping/nas/ori_data/LGE_SA/LGE_SA/test/seg'
for pid in os.listdir(lb_path):
    pid_ = pid.split('_seg.nii.gz')[0]
    ours = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(ours_path, pid_ + '_image-seg.nii.gz')))
    lb = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(lb_path, pid)))
    color_table = {0:0, 1:1, 2:2, 3:3, 4:3}
    for k, v in color_table.items():
        lb[lb == k] = v

    dice = calculate_all_dice(ours, lb, 3)
    with open(save_file, "a") as f:
        f.writelines(pid + ' ' + str(dice) + "\r\n")