import pandas as pd
import os
import json
import numpy as np
from calculate_cardiac_metrics_cine_4ch import calculate_cine_4ch_metrics
from calculate_cardiac_metrics_cine_sa import calculate_cine_sa_metrics

def pad_image_key(image_key):
    return str(image_key).zfill(7)

# def read_id_mapping():
#     # Read the CSV files
#     mapping_path1 = "id_rename_mapping.csv"
#     mapping_path2 = "id_rename_mapping2.csv"
    
#     # Read both CSV files if they exist
#     id_mapping = {}
    
#     if os.path.exists(mapping_path1):
#         df1 = pd.read_csv(mapping_path1)
#         for _, row in df1.iterrows():
#             new_name_ = row['new_name'].split(".")[0]
#             id_mapping[new_name_] = row['id']
    
#     if os.path.exists(mapping_path2):
#         df2 = pd.read_csv(mapping_path2)
#         for _, row in df2.iterrows():
#             new_name_ = row['new_name'].split(".")[0]
#             id_mapping[new_name_] = row['id']
    
#     return id_mapping

def read_id_mapping2():
    mapping_path2 = "id_mapping_714.json"
    with open(mapping_path2, "r") as f:
        id_mapping2 = json.load(f)
    id_mapping2 = {v: k for k, v in id_mapping2.items()}
    return id_mapping2

def read_id_slice_sax():
    mapping_path2 = "id_slice_info_sa.json"
    with open(mapping_path2, "r") as f:
        id_slice = json.load(f)
    # id_slice = {v: k for k, v in id_slice.items()}
    return id_slice

def read_id_slice_4ch():
    mapping_path2 = "id_slice_info_4ch.json"
    with open(mapping_path2, "r") as f:
        id_slice = json.load(f)
    # id_slice = {v: k for k, v in id_slice.items()}
    return id_slice

if __name__ == "__main__":
    # print(read_id_mapping())
    # id_mapping = read_id_mapping()
    id_mapping2 = read_id_mapping2()
    result = []
    error_list = []

    for id_map in os.listdir('CINE/sa_pred_fh'):
        id_map = id_map.split('_sa_image-seg.nii.gz')[0]
        # print(id_map)
        # image_key = id_mapping2[id_map]
        # padded_image_key = pad_image_key(image_key)
        
                                           
        # if id_map not in id_mapping2:
            # error_list.append({
            #     "id": id_map,
            #     "error": "ID not found in id_mapping2"
            # })
            # continue

                   
        cine_4ch_mask_path = f"CINE/4ch_pred/{id_map}_4ch_image-seg.nii.gz"
        cine_sa_mask_path = f"CINE/sa_pred2/{id_map}_sa_image-seg.nii.gz"
        if not os.path.exists(cine_sa_mask_path):
            cine_sa_mask_path = f"CINE/sa_pred2/{id_map}.nii.gz"
        # print(cine_4ch_mask_path)
        # print(cine_sa_mask_path)
        slice_nums_sax = read_id_slice_sax()
        # print(id_map)
        try:
            slice_num_sax = slice_nums_sax[str(id_map)]
        except:
            continue
        slice_nums_4ch = read_id_slice_4ch()
        # print(id_map)
        try:
            slice_num_4ch = slice_nums_4ch[str(id_map)]
        except:
            continue
                 
        cine_sa_metrics = calculate_cine_sa_metrics(cine_sa_mask_path, slice_num_sax)
        cine_4ch_metrics = calculate_cine_4ch_metrics(cine_4ch_mask_path, slice_num_4ch)

                   
        try:
            result_dict = {"id": id_mapping2[id_map]}
            
                         
            try:
                result_dict["LA_LD"] = cine_4ch_metrics['LA_ED_Long_Diameter']
            except:
                result_dict["LA_LD"] = None
                print(f"Warning: LA_LD not available for {id_map}")
            
            try:
                result_dict["RA_LD"] = cine_4ch_metrics['RA_ED_Long_Diameter']
            except:
                result_dict["RA_LD"] = None
            
            try:
                result_dict["LV_LD"] = cine_sa_metrics['LV_ED_Long_Diameter']
            except:
                result_dict["LV_LD"] = None
            
            try:
                result_dict["RV_LD"] = cine_sa_metrics['RV_ED_Long_Diameter']
            except:
                result_dict["RV_LD"] = None
            
                             
            def get_metric_safe(metrics_dict, key, default=None):
                try:
                    return metrics_dict[key]
                except:
                    print(f"Warning: {key} not available for {id_map}")
                    return default
            
                         
            segments_info = [
                                              
                ("LV_BS", 1, "Basal_anteroseptal", "max"),
                ("LV_BS", 2, "Basal_anterior", "max"),
                ("LV_BS", 3, "Basal_lateral", "max"),
                ("LV_BS", 4, "Basal_posterior", "max"),
                ("LV_BS", 5, "Basal_inferior", "max"),
                ("LV_BS", 6, "Basal_inferoseptal", "max"),
                ("LV_IP", 7, "Mid_anteroseptal", "max"),
                ("LV_IP", 8, "Mid_anterior", "max"),
                ("LV_IP", 9, "Mid_lateral", "max"),
                ("LV_IP", 10, "Mid_posterior", "max"),
                ("LV_IP", 11, "Mid_inferior", "max"),
                ("LV_IP", 12, "Mid_inferoseptal", "max"),
                ("LV_SP", 13, "Apical_anterior", "max"),
                ("LV_SP", 14, "Apical_lateral", "max"),
                ("LV_SP", 15, "Apical_inferior", "max"),
                ("LV_SP", 16, "Apical_septal", "max"),
            ]
            
                    
            for prefix, num, name, metric_type in segments_info:
                key = f"{prefix}_{num:02d}_{metric_type}"
                try:
                    metric_key = f'ED_Segment_{num:02d}_{name}_Thickness_max'
                    result_dict[key] = cine_sa_metrics[metric_key]
                except:
                    result_dict[key] = None
            
                     
            for prefix, num, name, metric_type in segments_info:
                key = f"{prefix}_{num:02d}_mean"
                try:
                    metric_key = f'ED_Segment_{num:02d}_{name}_Thickness_mean'
                    result_dict[key] = cine_sa_metrics[metric_key]
                except:
                    result_dict[key] = None
            
                    
            for prefix, num, name, metric_type in segments_info:
                key = f"{prefix}_{num:02d}_min"
                try:
                    metric_key = f'ED_Segment_{num:02d}_{name}_Thickness_min'
                    result_dict[key] = cine_sa_metrics[metric_key]
                except:
                    result_dict[key] = None
            
                         
            try:
                result_dict["LV_TP_17_max"] = cine_4ch_metrics['ED_LV_Apex_Thickness_max']
            except:
                result_dict["LV_TP_17_max"] = None
            
            try:
                result_dict["LV_TP_17_mean"] = cine_4ch_metrics['ED_LV_Apex_Thickness_mean']
            except:
                result_dict["LV_TP_17_mean"] = None
            
            try:
                result_dict["LV_TP_17_min"] = cine_4ch_metrics['ED_LV_Apex_Thickness_min']
            except:
                result_dict["LV_TP_17_min"] = None
            
                      
            rv_thickness_keys = ['ED_RV_Wall_Thickness_Div_1', 'ED_RV_Wall_Thickness_Div_2', 'ED_RV_Wall_Thickness_Div_3']
            rv_prefixes = ['RV_BS_01', 'RV_IP_02', 'RV_SP_03']
            
            for rv_key, prefix in zip(rv_thickness_keys, rv_prefixes):
                try:
                    result_dict[prefix] = cine_4ch_metrics[rv_key]
                except:
                    result_dict[prefix] = None
            
                       
            lv_function_keys = ['LV_EDV', 'LV_ESV', 'LV_SV', 'LV_EF', 'LV_CO', 'LV_Mass']
            for key in lv_function_keys:
                try:
                    result_dict[key] = cine_sa_metrics[key]
                except:
                    result_dict[key] = None
            
            rv_function_keys = ['RV_EDV', 'RV_ESV', 'RV_SV', 'RV_EF', 'RV_CO']
            for key in rv_function_keys:
                try:
                    result_dict[key] = cine_sa_metrics[key]
                except:
                    result_dict[key] = None
            
                              
            critical_metrics = ['LV_EF', 'RV_EF', 'LV_EDV', 'RV_EDV']
            missing_critical = [m for m in critical_metrics if result_dict.get(m) is None]
            
            if missing_critical:
                print(f"Warning for {id_map}: Missing critical metrics: {missing_critical}")
            
                        
            result.append(result_dict)
            
        except Exception as e:
            print(f"Major error for {id_map}: {e}")
            error_list.append({
                "id": id_mapping2.get(id_map, id_map),
                "error": str(e)
            })


    def convert_for_json(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        else:
            return obj

               
    result_serializable = convert_for_json(result)
    # save result to json
    with open("measure_result_v15_fh.json", "w", encoding="utf-8") as f:
        json.dump(result_serializable, f, ensure_ascii=False, indent=4)

    with open("error_result_v15_fh.json", "w", encoding="utf-8") as f:
        json.dump(error_list, f, ensure_ascii=False, indent=4)