import os
import numpy as np
import SimpleITK as sitk

def calculate_global_mean_std(data_dir, seg_dir):
    """
    计算文件夹中所有影像前景区域的全局均值和方差（含5%-95%分位裁剪，去除极端值）
    
    参数:
    data_dir: 影像文件（xxximage.nii.gz）文件夹路径
    seg_dir: 分割掩码文件（xxxseg.nii.gz）文件夹路径
    
    返回:
    global_mean: 全局均值
    global_std: 全局标准差
    global_var: 全局方差
    """
    total_sum = 0.0       
    total_sq_sum = 0.0    
    total_valid_count = 0  # 统计裁剪后有效前景像素总数

    for filename in os.listdir(data_dir):
        if filename.endswith("image.nii.gz"):
            # 提取文件名前缀，匹配对应的分割文件
            file_prefix = filename.replace("image.nii.gz", "")
            img_path = os.path.join(data_dir, filename)
            seg_path = os.path.join(seg_dir, f"{file_prefix}seg.nii.gz")

            try:
                # 读取影像和分割掩码
                image = sitk.ReadImage(img_path)
                seg_image = sitk.ReadImage(seg_path)
                vol = sitk.GetArrayFromImage(image).astype(np.float32)
                seg = sitk.GetArrayFromImage(seg_image).astype(np.int8)

                # 1. 提取前景区域（掩码>0的部分）
                foreground_mask = seg > 0
                foreground_pixels = vol[foreground_mask]

                if len(foreground_pixels) == 0:
                    print(f"警告: {filename} 无前景像素，跳过")
                    continue

                # 2. 5%-95%分位裁剪，去除极端异常值
                lower = np.percentile(foreground_pixels, 1)   # 5分位数
                upper = np.percentile(foreground_pixels, 99)  # 95分位数
                # 裁剪并保留在[lower, upper]范围内的像素
                clipped_pixels = foreground_pixels[(foreground_pixels >= lower) & (foreground_pixels <= upper)]

                if len(clipped_pixels) == 0:
                    print(f"警告: {filename} 裁剪后无有效像素，跳过")
                    continue

                # 3. 累加统计量
                total_sum += clipped_pixels.sum()
                total_sq_sum += (clipped_pixels ** 2).sum()
                total_valid_count += len(clipped_pixels)

                print(f"已处理: {filename} | 裁剪后前景像素数: {len(clipped_pixels)} | 累计有效像素数: {total_valid_count}")

            except Exception as e:
                print(f"处理文件 {filename} 出错: {str(e)}")
                continue

    if total_valid_count == 0:
        raise ValueError("未找到有效裁剪后的前景像素，请检查数据或分位参数")

    # 计算全局统计量
    global_mean = total_sum / total_valid_count
    global_var = (total_sq_sum / total_valid_count) - (global_mean ** 2)
    global_std = np.sqrt(global_var)

    return global_mean, global_std, global_var

if __name__ == "__main__":
    # 配置路径
    data_directory = "/home/qutaiping/nas/ori_data/LGE_SA/LGE_SA/train/img"  
    seg_directory = "/home/qutaiping/nas/ori_data/LGE_SA/LGE_SA/train/seg"  
    
    # 计算统计量
    mean, std, var = calculate_global_mean_std(data_directory, seg_directory)
    
    # 输出并保存结果
    print("\n===== 5%-95%裁剪后前景全局统计结果 =====")
    print(f"均值 (mean): {mean:.6f}")
    print(f"标准差 (std): {std:.6f}")
    print(f"方差 (var): {var:.6f}")
    
    with open("clipped_global_stats.txt", "w") as f:
        f.write(f"分位裁剪：5%-95%\n")
        f.write(f"mean: {mean:.6f}\n")
        f.write(f"std: {std:.6f}\n")
        f.write(f"var: {var:.6f}\n")
    print("\n结果已保存到 clipped_global_stats.txt")