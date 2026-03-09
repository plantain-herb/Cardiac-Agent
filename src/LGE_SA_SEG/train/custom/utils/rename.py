import os
import shutil

def copy_and_rename_files(txt_path, source_base, dest_base):
    """
    复制并按规则重命名文件
    
    参数:
    txt_path: 包含新PID、原始PID对应关系的txt文件路径
    source_base: 源文件根目录（包含img和seg子文件夹）
    dest_base: 目标文件根目录（会创建img和seg子文件夹）
    """
    # 创建目标文件夹（如果不存在）
    dest_img = os.path.join(dest_base, 'img')
    dest_seg = os.path.join(dest_base, 'seg')
    os.makedirs(dest_img, exist_ok=True)
    os.makedirs(dest_seg, exist_ok=True)
    
    # 读取txt文件，建立新PID到原始PID的映射
    pid_mapping = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        # 跳过表头
        next(f)
        next(f)  # 跳过分隔线
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 分割行数据（处理可能的空格分隔）
            parts = line.split()
            if len(parts) >= 2:
                new_pid = parts[0]
                original_pid = parts[1]
                pid_mapping[new_pid] = original_pid
    
    # 处理文件复制和重命名
    for new_pid, original_pid in pid_mapping.items():
        # 源文件路径
        src_img = os.path.join(source_base, 'img', f'{new_pid}_sa_image.nii.gz')
        src_seg = os.path.join(source_base, 'seg', f'{new_pid}_sa_seg.nii.gz')
        
        # 目标文件路径
        dst_img = os.path.join(dest_img, f'{original_pid}_sa_image.nii.gz')
        dst_seg = os.path.join(dest_seg, f'{original_pid}_sa_seg.nii.gz')
        
        # 复制文件
        try:
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
                print(f"已复制: {src_img} -> {dst_img}")
            else:
                print(f"警告: 未找到图像文件 {src_img}")
                
            if os.path.exists(src_seg):
                shutil.copy2(src_seg, dst_seg)
                print(f"已复制: {src_seg} -> {dst_seg}")
            else:
                print(f"警告: 未找到分割文件 {src_seg}")
        except Exception as e:
            print(f"处理 {new_pid} 时出错: {str(e)}")

if __name__ == "__main__":
    # 配置路径
    txt_file_path = "/home/qutaiping/nas/ori_data/LGE_SA/LGE_SA1/WBN.txt"  # 请替换为你的txt文件实际路径
    source_directory = "/home/qutaiping/nas/ori_data/LGE_SA/LGE_SA1"
    destination_directory = "/home/qutaiping/nas/ori_data/LGE_SA/LGE_SA"
    
    # 执行复制重命名操作
    copy_and_rename_files(txt_file_path, source_directory, destination_directory)
    print("操作完成！")