import numpy as np
import nibabel as nib
import pandas as pd

        
         
LGE_LABEL3_ID = 3                        
LGE_TISSUE_DENSITY = 1.05                     

def calculate_label3_mass(lge_sa_mask_path):
    """Compute mass (g) of label 3 in an LGE SA mask.

    Args:
        lge_sa_mask_path (str): Path to the LGE SA segmentation mask (NIfTI).

    Returns:
        float | None: Mass of label 3 in grams. Returns None on failure.
    """
    try:
                   
        lge_img = nib.load(lge_sa_mask_path)
        lge_data = np.round(lge_img.get_fdata()).astype(np.int16)
        
                     
        spacing = lge_img.header.get_zooms()[:3]  # (x, y, z)
        
                       
        label3_pixels = np.sum(lge_data == LGE_LABEL3_ID)
        label3_volume_ml = label3_pixels * spacing[0] * spacing[1] * spacing[2] / 1000.0         
        label3_mass_g = label3_volume_ml * LGE_TISSUE_DENSITY           
        
        return label3_mass_g
        
    except Exception:
        return None

def calculate_lge_sa_metrics(lge_sa_mask_path):
    """Compute LGE SA metrics.

    Currently returns only the mass (g) of label 3.

    Args:
        lge_sa_mask_path (str): Path to the LGE SA segmentation mask (NIfTI).

    Returns:
        dict | None: Metrics dict. Returns None on failure.
    """
    try:
        result = {}
        result['LGE_SA_Label3_Mass'] = calculate_label3_mass(lge_sa_mask_path)
        return result
    except Exception:
        return None

if __name__ == "__main__":
    lge_sa_mask_path = "0000375_sa_lge-seg.nii.gz"
    result = calculate_lge_sa_metrics(lge_sa_mask_path)
    print(result)

