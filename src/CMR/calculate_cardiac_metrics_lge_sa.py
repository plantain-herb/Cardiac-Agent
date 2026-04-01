import numpy as np
import nibabel as nib
import pandas as pd

# 医学参数定义
# LGE相关参数
LGE_LABEL3_ID = 3     # LGE mask中的label 3
LGE_TISSUE_DENSITY = 1.05  # LGE组织密度，假设与心肌密度相同

def calculate_label3_mass(lge_sa_mask_path):
    """
    计算LGE SA mask文件中label 3的质量
    
    Args:
        lge_sa_mask_path: LGE分割掩码文件路径 (str)
    
    Returns:
        float: label 3的质量（克），如果失败返回None
    """
    try:
        # 加载LGE分割数据
        lge_img = nib.load(lge_sa_mask_path)
        lge_data = np.round(lge_img.get_fdata()).astype(np.int16)
        
        # 获取spacing信息
        spacing = lge_img.header.get_zooms()[:3]  # (x, y, z)
        
        # 计算label 3的总体积
        label3_pixels = np.sum(lge_data == LGE_LABEL3_ID)
        label3_volume_ml = label3_pixels * spacing[0] * spacing[1] * spacing[2] / 1000.0  # 转换为ml
        label3_mass_g = label3_volume_ml * LGE_TISSUE_DENSITY  # 计算质量（g）
        
        return label3_mass_g
        
    except Exception:
        return None

def calculate_lge_sa_metrics(lge_sa_mask_path):
    """
    计算LGE SA mask文件中label 3的质量
    
    Args:
        lge_sa_mask_path: LGE分割掩码文件路径 (str)
    
    Returns:
        float: LGE SA label 3的质量（克），如果失败返回None
    """
    try:
        result = {}
        result['LGE_SA_Label3_Mass'] = calculate_label3_mass(lge_sa_mask_path)
        return result
    except Exception:
        return None

if __name__ == "__main__":
    lge_sa_mask_path = "/Users/zhanglantian/Documents/BAAI/Code/code/measure/data_lge/0000375_sa_lge-seg.nii.gz"
    result = calculate_lge_sa_metrics(lge_sa_mask_path)
    print(result)

    