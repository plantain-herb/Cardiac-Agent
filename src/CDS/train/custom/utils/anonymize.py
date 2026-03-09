import os
import re
import json
from collections import defaultdict

def anonymize_nii_files(root_dir, output_dir, mapping_file="id_mapping.json"):
    """
    将指定目录下的NIfTI文件按患者ID进行匿名化处理
    
    参数:
    root_dir (str): 包含CINE和LGE文件夹的根目录
    output_dir (str): 输出匿名化文件的目录
    mapping_file (str): 存储ID映射关系的JSON文件路径
    """
    # 创建输出目录结构
    os.makedirs(output_dir, exist_ok=True)
    for subdir in ["CINE", "LGE"]:
        for view in ["sa", "2ch", "4ch"]:
            os.makedirs(os.path.join(output_dir, subdir, view), exist_ok=True)
    
    # 正则表达式模式，用于提取患者ID
    id_pattern = re.compile(r'^(\d+)_(.*?\.nii\.gz)$')
    
    # 存储ID映射关系
    id_mapping = {}
    next_anonymous_id = 1
    
    # 检查是否存在已有的映射文件
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            id_mapping = json.load(f)
        if id_mapping:
            next_anonymous_id = max(int(k) for k in id_mapping.values()) + 1
    
    # 遍历所有文件
    for subdir in ["CINE", "LGE"]:
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        for view in ["sa", "2ch", "4ch"]:
            view_path = os.path.join(subdir_path, view)
            if not os.path.isdir(view_path):
                continue
                
            for filename in os.listdir(view_path):
                if filename.endswith('.nii.gz'):
                    match = id_pattern.match(filename)
                    if match:
                        patient_id, file_suffix = match.groups()
                        
                        # 获取或创建匿名ID
                        if patient_id not in id_mapping:
                            id_mapping[patient_id] = f"{next_anonymous_id:07d}"
                            next_anonymous_id += 1
                            
                        # 构建新文件名
                        new_filename = f"{id_mapping[patient_id]}_{file_suffix}"
                        
                        # 源文件和目标文件路径
                        src_path = os.path.join(view_path, filename)
                        dst_path = os.path.join(output_dir, subdir, view, new_filename)
                        
                        # 复制文件
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        os.link(src_path, dst_path)  # 使用硬链接以提高效率，可改为shutil.copy2以保留所有元数据
                        print(f"已匿名化: {src_path} -> {dst_path}")
    
    # 保存ID映射关系
    with open(mapping_file, 'w') as f:
        json.dump(id_mapping, f, indent=4)
    
    print(f"匿名化完成! ID映射关系已保存到 {mapping_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NIfTI文件匿名化工具')
    parser.add_argument('--input', default='/home/qutaiping/nas/ori_data/AZHC/all')
    parser.add_argument('--output', default='/home/qutaiping/nas/ori_data/AZHC/test_mapping_out')
    parser.add_argument('--mapping', default='/home/qutaiping/nas/ori_data/AZHC/test_mapping_out/id_mapping.json', help='ID映射文件路径')
    
    args = parser.parse_args()
    
    anonymize_nii_files(args.input, args.output, args.mapping)    