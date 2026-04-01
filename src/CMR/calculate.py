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

    for id_map in os.listdir('/home/taiping-qu/code/nas/ori_data/AZHC/mapping_out/CINE/sa_pred_fh'):
        id_map = id_map.split('_sa_image-seg.nii.gz')[0]
        # print(id_map)
        # image_key = id_mapping2[id_map]
        # padded_image_key = pad_image_key(image_key)
        
        # 判断padded_image_key是否在id_mapping2中
        # if id_map not in id_mapping2:
            # error_list.append({
            #     "id": id_map,
            #     "error": "ID not found in id_mapping2"
            # })
            # continue

        # 1. 分割结果目录
        cine_4ch_mask_path = f"/home/taiping-qu/code/nas/ori_data/AZHC/mapping_out/CINE/4ch_pred/{id_map}_4ch_image-seg.nii.gz"
        cine_sa_mask_path = f"/home/taiping-qu/code/nas/ori_data/AZHC/mapping_out/CINE/sa_pred2/{id_map}_sa_image-seg.nii.gz"
        if not os.path.exists(cine_sa_mask_path):
            cine_sa_mask_path = f"/home/taiping-qu/code/nas/ori_data/AZHC/mapping_out/CINE/sa_pred2/{id_map}.nii.gz"
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
        # 2. 测量结果
        cine_sa_metrics = calculate_cine_sa_metrics(cine_sa_mask_path, slice_num_sax)
        cine_4ch_metrics = calculate_cine_4ch_metrics(cine_4ch_mask_path, slice_num_4ch)

        # 3. 重新整理结果
        try:
            result_dict = {"id": id_mapping2[id_map]}
            
            # 1. 处理主要直径测量
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
            
            # 2. 批量处理LV壁厚度的函数
            def get_metric_safe(metrics_dict, key, default=None):
                try:
                    return metrics_dict[key]
                except:
                    print(f"Warning: {key} not available for {id_map}")
                    return default
            
            # 使用列表推导式批量处理
            segments_info = [
                # (前缀, 编号, 名称, 是否max/mean/min)
                ("LV_BS", 1, "基底前间隔", "max"),
                ("LV_BS", 2, "基底前壁", "max"),
                ("LV_BS", 3, "基底侧壁", "max"),
                ("LV_BS", 4, "基底后壁", "max"),
                ("LV_BS", 5, "基底下壁", "max"),
                ("LV_BS", 6, "基底下间隔", "max"),
                ("LV_IP", 7, "中间前间隔", "max"),
                ("LV_IP", 8, "中间前壁", "max"),
                ("LV_IP", 9, "中间侧壁", "max"),
                ("LV_IP", 10, "中间后壁", "max"),
                ("LV_IP", 11, "中间下壁", "max"),
                ("LV_IP", 12, "中间下间隔", "max"),
                ("LV_SP", 13, "心尖前壁", "max"),
                ("LV_SP", 14, "心尖侧壁", "max"),
                ("LV_SP", 15, "心尖下壁", "max"),
                ("LV_SP", 16, "心尖间隔", "max"),
            ]
            
            # 处理max值
            for prefix, num, name, metric_type in segments_info:
                key = f"{prefix}_{num:02d}_{metric_type}"
                try:
                    metric_key = f'ED_Segment_{num:02d}_{name}_Thickness_max'
                    result_dict[key] = cine_sa_metrics[metric_key]
                except:
                    result_dict[key] = None
            
            # 处理mean值
            for prefix, num, name, metric_type in segments_info:
                key = f"{prefix}_{num:02d}_mean"
                try:
                    metric_key = f'ED_Segment_{num:02d}_{name}_Thickness_mean'
                    result_dict[key] = cine_sa_metrics[metric_key]
                except:
                    result_dict[key] = None
            
            # 处理min值
            for prefix, num, name, metric_type in segments_info:
                key = f"{prefix}_{num:02d}_min"
                try:
                    metric_key = f'ED_Segment_{num:02d}_{name}_Thickness_min'
                    result_dict[key] = cine_sa_metrics[metric_key]
                except:
                    result_dict[key] = None
            
            # 3. 特殊处理心尖厚度
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
            
            # 4. RV壁厚度
            rv_thickness_keys = ['ED_RV_Wall_Thickness_Div_1', 'ED_RV_Wall_Thickness_Div_2', 'ED_RV_Wall_Thickness_Div_3']
            rv_prefixes = ['RV_BS_01', 'RV_IP_02', 'RV_SP_03']
            
            for rv_key, prefix in zip(rv_thickness_keys, rv_prefixes):
                try:
                    result_dict[prefix] = cine_4ch_metrics[rv_key]
                except:
                    result_dict[prefix] = None
            
            # 5. 心室功能指标
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
            
            # 6. 检查是否有任何关键指标缺失
            critical_metrics = ['LV_EF', 'RV_EF', 'LV_EDV', 'RV_EDV']
            missing_critical = [m for m in critical_metrics if result_dict.get(m) is None]
            
            if missing_critical:
                print(f"Warning for {id_map}: Missing critical metrics: {missing_critical}")
            
            # 7. 添加到结果列表
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

    # 在dump之前转换
    result_serializable = convert_for_json(result)
    # save result to json
    with open("measure_result_v15_fh.json", "w", encoding="utf-8") as f:
        json.dump(result_serializable, f, ensure_ascii=False, indent=4)

    with open("error_result_v15_fh.json", "w", encoding="utf-8") as f:
        json.dump(error_list, f, ensure_ascii=False, indent=4)