import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt



# 可视化结果
# def visualize_cam_3d(cam, original_volume, slice_idx=144):
#     # 选择中间切片
#     cam_slice = cam[18, :, :]
#     original_slice = original_volume[slice_idx, :, :]
    
#     # 调整大小和归一化
#     cam_slice = cv2.resize(cam_slice, (128, 128))
#     cam_slice = np.uint8(255 * cam_slice)
#     heatmap = cv2.applyColorMap(cam_slice, cv2.COLORMAP_JET)
    
#     original_slice = np.uint8(255 * (original_slice - np.min(original_slice)) / 
#                     (np.max(original_slice) - np.min(original_slice)))
#     original_slice = cv2.cvtColor(original_slice, cv2.COLOR_GRAY2BGR)
    
#     # 叠加热图
#     superimposed = heatmap * 0.4 + original_slice * 0.6
#     superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    
#     # 显示结果
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(original_slice)
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(superimposed)
#     plt.show()
def enhance_hotspots(gradcam, alpha=2):
    """
    使用幂律变换增强Grad-CAM中的热点区域
    
    参数:
    - gradcam: 原始Grad-CAM热力图
    - alpha: 幂指数，值越大，增强效果越明显
    
    返回:
    - 增强后的热力图
    """
    # 复制原图以避免修改原始数据
    result = gradcam.copy()
    
    # 归一化到[0,1]
    if np.max(result) > 0:
        result = result / np.max(result)
    
    # 幂律变换
    result = np.power(result, alpha)
    
    # 再次归一化
    if np.max(result) > 0:
        result = result / np.max(result)
    
    return result
def visualize_cam_3d(cam, original_volume, slice_idx=124):
    # 选择中间切片 yangxing 34   144
    # ying 124 20
    cam_slice = cam[20, :, :]
    # cam_slice2 = cam[33, :, :]
    # cam_slice = (cam_slice1 + cam_slice2) / 2
    original_slice = original_volume[slice_idx, :, :]
    # resized_mask_ = resized_mask[slice_idx, :, :]
    
    # 调整大小和归一化
    cam_slice = cv2.resize(cam_slice, (128, 128))
    cam_slice = enhance_hotspots(cam_slice)
    # cam_slice = cam_slice * resized_mask_
    normalized_cam = (cam_slice - np.min(cam_slice)) / (np.max(cam_slice) - np.min(cam_slice) + 1e-8)
    
    # 创建热力图
    cmap = plt.get_cmap('jet')
    heatmap = cmap(normalized_cam)
    heatmap = np.delete(heatmap, 3, 2)  # 删除alpha通道
    heatmap = np.uint8(255 * heatmap)
    
    # 归一化原始切片并转换为BGR
    original_slice = np.uint8(255 * (original_slice - np.min(original_slice)) / 
                    (np.max(original_slice) - np.min(original_slice) + 1e-8))
    original_slice = cv2.cvtColor(original_slice, cv2.COLOR_GRAY2RGB)
    # 叠加热图
    superimposed = (heatmap * 0.4 + original_slice * 0.6).astype(np.uint8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=600)  # 设置全局dpi
    save_path = '/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/ori_data/grad_cam_output2'
    if save_path:
        # 保存原始图像（通过Matplotlib保存，支持dpi设置）
        original_fig, ax_original = plt.subplots(figsize=(7, 5), dpi=600)
        ax_original.imshow(original_slice)
        # ax_original.set_title('Original Image')
        ax_original.axis('off')
        original_fig.tight_layout()
        original_fig.savefig(f"/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/ori_data/grad_cam_output2/51056641original1.tiff", dpi=600, bbox_inches='tight')
        plt.close(original_fig)  # 关闭独立图像窗口
        
    if save_path:
        fig = plt.figure(figsize=(7, 5), dpi=600)  # 调整为单个图像的尺寸
        ax = fig.add_subplot(111)
        ax.imshow(superimposed)
        # ax.set_title('Grad-CAM Superimposed')
        ax.axis('off')
        
        # 添加颜色条
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Attention Intensity', rotation=270, labelpad=15)
        
        # 保存带颜色条的叠加图
        fig.tight_layout()
        fig.savefig(f"/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/ori_data/grad_cam_output2/51056641superimposed1.tiff", dpi=600, bbox_inches='tight')
        plt.close(fig)

    # 显示结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 显示原始图像
    ax1.imshow(original_slice)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # 显示叠加图像
    im = ax2.imshow(superimposed)
    ax2.set_title('Grad-CAM Superimposed')
    ax2.axis('off')
    
    # 添加热力图图例
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [左, 下, 宽, 高]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Attention Intensity', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.show()

def extract_and_pad(mask, bbox, target_size=[288, 128, 128]):
    """
    根据边界框提取区域并进行零填充以达到目标尺寸
    
    参数:
    - mask: 原始掩码数组
    - bbox: 边界框数组 [min_z, max_z, min_y, max_y, min_x, max_x]
    - target_size: 目标尺寸 [depth, height, width]
    
    返回:
    - 处理后的掩码数组，尺寸为target_size
    """
    # 提取边界框坐标
    min_z, max_z, min_y, max_y, min_x, max_x = bbox
    
    # 计算边界框尺寸
    bbox_depth = max_z - min_z + 1
    bbox_height = max_y - min_y + 1
    bbox_width = max_x - min_x + 1
    
    # 创建目标尺寸的零数组
    padded_mask = np.zeros(target_size, dtype=mask.dtype)
    
    # 计算在目标数组中的中心位置
    center_z = target_size[0] // 2
    center_y = target_size[1] // 2
    center_x = target_size[2] // 2
    
    # 计算提取区域在目标数组中的起始和结束位置
    z_start = max(center_z - bbox_depth // 2, 0)
    z_end = min(z_start + bbox_depth, target_size[0])
    
    y_start = max(center_y - bbox_height // 2, 0)
    y_end = min(y_start + bbox_height, target_size[1])
    
    x_start = max(center_x - bbox_width // 2, 0)
    x_end = min(x_start + bbox_width, target_size[2])
    
    # 计算原始边界框中需要提取的区域（处理裁剪情况）
    extract_z_start = max(0, (bbox_depth - (z_end - z_start)) // 2)
    extract_z_end = extract_z_start + (z_end - z_start)
    
    extract_y_start = max(0, (bbox_height - (y_end - y_start)) // 2)
    extract_y_end = extract_y_start + (y_end - y_start)
    
    extract_x_start = max(0, (bbox_width - (x_end - x_start)) // 2)
    extract_x_end = extract_x_start + (x_end - x_start)
    
    # 提取区域并填充到目标数组
    padded_mask[z_start:z_end, y_start:y_end, x_start:x_end] = \
        mask[min_z + extract_z_start:min_z + extract_z_end,
             min_y + extract_y_start:min_y + extract_y_end,
             min_x + extract_x_start:min_x + extract_x_end]
    
    return padded_mask


def _normalization(vol):
    hu_max = np.max(vol)
    hu_min = np.min(vol)
    vol_normalized = (vol - hu_min) / (hu_max - hu_min + 1e-8)
    return vol_normalized

from scipy.ndimage import zoom
from scipy import ndimage
import SimpleITK as sitk
import os
cam = sitk.GetArrayFromImage(sitk.ReadImage('/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/ori_data/grad_cam_output2/51056641_right.nii.gz'))
input_tensor = sitk.GetArrayFromImage(sitk.ReadImage('/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/ori_data/grad_cam_output2/51056641-img.nii.gz'))
# mask = sitk.GetArrayFromImage(sitk.ReadImage('/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/ori_data/CG_seg_resample/51127570.nii.gz'))
# input_tensor = _normalization(input_tensor)
# loc = np.where(mask > 0)
# constant_shift = 5
# bbox = np.array(
#         [np.min(loc[0]), np.max(loc[0]), np.min(loc[1]), np.max(loc[1]), np.min(loc[2]), np.max(loc[2])]
#     )
# min_z, max_z, min_y, max_y, min_x, max_x = bbox

# extracted = input_tensor[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
# extracted_mask = mask[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
# extracted_mask[extracted_mask > 1] = 0
# target_size=[288, 128, 128]
# 计算当前尺寸和目标尺寸的比例
# current_size = np.array(extracted.shape)
# zoom_factors = np.array(target_size) / current_size

# 使用scipy.ndimage.zoom进行线性插值
# resized = zoom(extracted, zoom_factors, order=1)
# resized_mask = zoom(extracted_mask, zoom_factors, order=1)
# structure = np.ones((10, 10, 10), dtype=bool)
# resized_mask = ndimage.binary_dilation(resized_mask, structure=structure, iterations=1)
# processed_img = extract_and_pad(input_tensor, bbox, target_size=[288, 128, 128])
# print(bbox)
visualize_cam_3d(cam, input_tensor)