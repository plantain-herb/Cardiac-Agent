import os
import nibabel as nib
import numpy as np
import warnings
import re
import logging
import math
import cv2
from scipy.ndimage import zoom
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# --- 核心参数配置 ---
# 裁剪控制
CROP_MARGIN = 5     # 裁剪时在边界外保留的像素余量
from scipy.stats import mode 
# 医学参数定义
BACKGROUND_ID = 0     # 背景
LV_MYOCARDIUM_ID = 1  # 左心室心肌
LV_BLOOD_POOL_ID = 2  # 左心室血腔
RV_BLOOD_POOL_ID = 3  # 右心室血腔
RV_MYOCARDIUM_ID = 4  # 右心室心肌
MYOCARDIUM_DENSITY = 1.05
ASSUMED_HEART_RATE = 70

# 分块参数
# BLOCK_SIZES = [30]

# 核心参数
MAX_SLICES_PER_BLOCK = 8  # 在每块中选择前N层切片来计算指标
SKIP_HEAD_SLICES_PER_BLOCK = 1  # 每块前面跳过的切片数
SKIP_TAIL_SLICES_PER_BLOCK = 2  # 每块后面跳过的切片数
TARGET_SLICE_INDEX = 3  # 用于计算短径的切片索引（第4张，索引从0开始）

# 等分分析参数
SEGMENTATION_SLICE_INDICES = [1, 3, 5]  # 对应第2、4、6张切片（索引从0开始）
SEGMENTATION_DIVISIONS = [4, 6, 6]      # 对应第2、4、6张切片的等分数

# 心脏分区命名映射（基于标准17分区模型）
SEGMENT_NAMES = {
    # 第2张切片（心尖部，4等分）- 对应标准17分区的13-16
    1: {
        1: {'id': 13, 'name': '心尖前壁'},
        2: {'id': 14, 'name': '心尖侧壁'}, 
        3: {'id': 15, 'name': '心尖下壁'},
        4: {'id': 16, 'name': '心尖间隔'}
    },
    2: {
        1: {'id': 13, 'name': '心尖前壁'},
        2: {'id': 14, 'name': '心尖侧壁'}, 
        3: {'id': 15, 'name': '心尖下壁'},
        4: {'id': 16, 'name': '心尖间隔'}
    },
    3: {
        1: {'id': 13, 'name': '心尖前壁'},
        2: {'id': 14, 'name': '心尖侧壁'}, 
        3: {'id': 15, 'name': '心尖下壁'},
        4: {'id': 16, 'name': '心尖间隔'}
    },
    # 第4张切片（中部，6等分）- 对应标准17分区的7-12
    4: {
        1: {'id': 7, 'name': '中间前间隔'},
        2: {'id': 8, 'name': '中间前壁'},
        3: {'id': 9, 'name': '中间侧壁'},
        4: {'id': 10, 'name': '中间后壁'},
        5: {'id': 11, 'name': '中间下壁'},
        6: {'id': 12, 'name': '中间下间隔'}
    },
    5: {
        1: {'id': 7, 'name': '中间前间隔'},
        2: {'id': 8, 'name': '中间前壁'},
        3: {'id': 9, 'name': '中间侧壁'},
        4: {'id': 10, 'name': '中间后壁'},
        5: {'id': 11, 'name': '中间下壁'},
        6: {'id': 12, 'name': '中间下间隔'}
    },
    # 第6张切片（基底部，6等分）- 对应标准17分区的1-6
    6: {
        1: {'id': 1, 'name': '基底前间隔'},
        2: {'id': 2, 'name': '基底前壁'},
        3: {'id': 3, 'name': '基底侧壁'},
        4: {'id': 4, 'name': '基底后壁'},
        5: {'id': 5, 'name': '基底下壁'},
        6: {'id': 6, 'name': '基底下间隔'}
    },
    7: {
        1: {'id': 1, 'name': '基底前间隔'},
        2: {'id': 2, 'name': '基底前壁'},
        3: {'id': 3, 'name': '基底侧壁'},
        4: {'id': 4, 'name': '基底后壁'},
        5: {'id': 5, 'name': '基底下壁'},
        6: {'id': 6, 'name': '基底下间隔'}
    },
    8: {
        1: {'id': 1, 'name': '基底前间隔'},
        2: {'id': 2, 'name': '基底前壁'},
        3: {'id': 3, 'name': '基底侧壁'},
        4: {'id': 4, 'name': '基底后壁'},
        5: {'id': 5, 'name': '基底下壁'},
        6: {'id': 6, 'name': '基底下间隔'}
    },
    9: {
        1: {'id': 1, 'name': '基底前间隔'},
        2: {'id': 2, 'name': '基底前壁'},
        3: {'id': 3, 'name': '基底侧壁'},
        4: {'id': 4, 'name': '基底后壁'},
        5: {'id': 5, 'name': '基底下壁'},
        6: {'id': 6, 'name': '基底下间隔'}
    },
    10: {
        1: {'id': 1, 'name': '基底前间隔'},
        2: {'id': 2, 'name': '基底前壁'},
        3: {'id': 3, 'name': '基底侧壁'},
        4: {'id': 4, 'name': '基底后壁'},
        5: {'id': 5, 'name': '基底下壁'},
        6: {'id': 6, 'name': '基底下间隔'}
    },
    11: {
        1: {'id': 1, 'name': '基底前间隔'},
        2: {'id': 2, 'name': '基底前壁'},
        3: {'id': 3, 'name': '基底侧壁'},
        4: {'id': 4, 'name': '基底后壁'},
        5: {'id': 5, 'name': '基底下壁'},
        6: {'id': 6, 'name': '基底下间隔'}
    }
}

warnings.filterwarnings('ignore')

# Line2D类和相关函数 - 从lesion_tools.py移植
class Line2D:
    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c

    def __repr__(self) -> str:
        return str(self.__dict__)

def build_line2D(p1, p2) -> Line2D:
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = -a * p1[0] - b * p1[1]
    return Line2D(a, b, c)

def euclidean_distance(p1, p2):
    distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return distance

def line2d_normal_to_point(l, p):  # noqa: E741
    return Line2D(-l.b, l.a, l.b * p[0] - l.a * p[1])

def line_intersection(l1, l2):
    if l1.a * l2.b - l2.a * l1.b == 0:
        return [0, 0], False

    x = (l2.c * l1.b - l1.c * l2.b) / (l1.a * l2.b - l2.a * l1.b)
    y = (l2.c * l1.a - l1.c * l2.a) / (l1.b * l2.a - l2.b * l1.a)

    return [x, y], True

def line2d_project_to_point(l, p):  # noqa: E741
    norm_line = line2d_normal_to_point(l, p)
    p, _ = line_intersection(l, norm_line)
    return p

def build_diameter(p1, p2, pixel_spacing_xy):
    dx = (p1[0] - p2[0]) * pixel_spacing_xy[0]
    dy = (p1[1] - p2[1]) * pixel_spacing_xy[1]
    length = math.sqrt(dx * dx + dy * dy)
    return length

def get_long_diameter_start_end_index(contour):
    try:
        idx1, idx2 = -1, -1
        if len(contour) <= 2:
            return idx1, idx2, False

        # WARNING: ConvexHull is not stable when all points of contour are on a line
        hull = ConvexHull(contour)

        p1s, p2s = [], []
        for p in hull.points:
            p1s.append(p)
            p2s.append(p)

        maxd = -1.0
        for pt1 in p1s:
            for pt2 in p2s:
                if euclidean_distance(pt1, pt2) > maxd:
                    maxd = euclidean_distance(pt1, pt2)
                    p1 = pt1
                    p2 = pt2

        for i, p in enumerate(contour):
            if p[0] == p1[0] and p[1] == p1[1]:
                idx1 = i
            if p[0] == p2[0] and p[1] == p2[1]:
                idx2 = i

        if idx1 < 0 or idx2 < 0:
            return idx1, idx2, False

        return idx1, idx2, True
    except Exception as e:
        logging.error(f"获取长径失败: {e}")
        return -1, -1, False

def get_short_diameter_start_end_index(contour, long_diameter_point_idx1, long_diameter_point_idx2):
    idx1, idx2 = -1, -1
    if len(contour) < 4:
        return idx1, idx2, False
    if (
        long_diameter_point_idx1 < 0
        or long_diameter_point_idx1 >= len(contour)
        or long_diameter_point_idx2 < 0
        or long_diameter_point_idx2 >= len(contour)  # noqa: W503
    ):
        return idx1, idx2, False

    st_idx, ed_idx = long_diameter_point_idx1, long_diameter_point_idx2
    long_p1, long_p2 = contour[long_diameter_point_idx1], contour[long_diameter_point_idx2]

    long_line = build_line2D(long_p1, long_p2)

    side1 = []
    side2 = []

    i = st_idx
    while i != ed_idx:
        side1.append(i)
        i = (i + 1) % len(contour)
    side1.append(i)

    i = st_idx
    while i != ed_idx:
        side2.append(i)
        i = (i - 1) % len(contour)
    side2.append(i)

    side1 = sorted(
        side1, key=lambda x: euclidean_distance(contour[st_idx], line2d_project_to_point(long_line, contour[x]))
    )
    side2 = sorted(
        side2, key=lambda x: euclidean_distance(contour[st_idx], line2d_project_to_point(long_line, contour[x]))
    )

    if len(side1) < 2 or len(side2) < 2:
        return idx1, idx2, False

    max_d = -1
    short_p1, short_p2 = long_p1, long_p1

    idx1_idx, idx2_idx = 0, 0
    while idx2_idx < len(side2):
        p1, p2 = contour[side1[idx1_idx]], contour[side2[idx2_idx]]
        d1 = euclidean_distance(contour[st_idx], line2d_project_to_point(long_line, p1))
        d2 = euclidean_distance(contour[st_idx], line2d_project_to_point(long_line, p2))
        if d2 > d1:
            idx1_idx += 1
            if idx1_idx >= len(side1):
                break
            continue

        d = euclidean_distance(p1, p2)
        if d > max_d:
            max_d = d
            short_p1 = p1
            short_p2 = p2

        idx2_idx += 1

    idx1_idx, idx2_idx = 0, 0
    while idx1_idx < len(side1):
        p1, p2 = contour[side1[idx1_idx]], contour[side2[idx2_idx]]
        d1 = euclidean_distance(contour[st_idx], line2d_project_to_point(long_line, p1))
        d2 = euclidean_distance(contour[st_idx], line2d_project_to_point(long_line, p2))
        if d1 > d2:
            idx2_idx += 1
            if idx2_idx >= len(side2):
                break
            continue

        d = euclidean_distance(p1, p2)
        if d > max_d:
            max_d = d
            short_p1 = p1
            short_p2 = p2

        idx1_idx += 1

    if short_p1[0] == short_p2[0] and short_p1[1] == short_p2[1]:
        return idx1, idx2, False

    for i, p in enumerate(contour):
        if p[0] == short_p1[0] and p[1] == short_p1[1]:
            idx1 = i
        if p[0] == short_p2[0] and p[1] == short_p2[1]:
            idx2 = i

    if idx1 < 0 or idx2 < 0:
        return idx1, idx2, False

    return idx1, idx2, True

# def calculate_myocardium_thickness(mask, spacing_xy):
#     """
#     计算左心室心肌厚度
#     通过计算心内膜（LV_BLOOD_POOL_ID边界）和心外膜（LV_MYOCARDIUM_ID外边界）之间的距离
    
#     Args:
#         mask: 分割掩码
#         spacing_xy: XY方向的像素间距
    
#     Returns:
#         average_thickness: 平均心肌厚度 (mm)
#         thickness_map: 厚度分布图
#     """
#     try:
#         # 获取左心室血腔和心肌的掩码
#         lv_blood_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
#         lv_myo_mask = (mask == LV_MYOCARDIUM_ID).astype(np.uint8)
        
#         if np.sum(lv_blood_mask) == 0 or np.sum(lv_myo_mask) == 0:
#             return 0.0, None
            
#         # 找到心内膜轮廓（血腔边界）
#         endo_contours, _ = cv2.findContours(lv_blood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         if not endo_contours:
#             return 0.0, None
            
#         # 找到心外膜轮廓（心肌外边界）
#         epi_contours, _ = cv2.findContours(lv_myo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         if not epi_contours:
#             return 0.0, None
            
#         # 选择最大的轮廓
#         endo_contour = max(endo_contours, key=cv2.contourArea)
#         epi_contour = max(epi_contours, key=cv2.contourArea)
        
#         # 计算心内膜每个点到心外膜的最短距离
#         endo_points = np.squeeze(endo_contour)
#         epi_points = np.squeeze(epi_contour)
        
#         if len(endo_points.shape) == 1:
#             endo_points = endo_points.reshape(1, -1)
#         if len(epi_points.shape) == 1:
#             epi_points = epi_points.reshape(1, -1)
            
#         thicknesses = []
#         thickness_map = np.zeros_like(mask, dtype=np.float32)
        
#         for endo_point in endo_points:
#             # 计算到心外膜所有点的距离
#             distances = []
#             for epi_point in epi_points:
#                 dx = (endo_point[0] - epi_point[0]) * spacing_xy[0]
#                 dy = (endo_point[1] - epi_point[1]) * spacing_xy[1]
#                 dist = np.sqrt(dx*dx + dy*dy)
#                 distances.append(dist)
            
#             # 最短距离即为该点的厚度
#             min_thickness = min(distances)
#             thicknesses.append(min_thickness)
            
#             # 在厚度图上标记
#             if 0 <= endo_point[1] < thickness_map.shape[0] and 0 <= endo_point[0] < thickness_map.shape[1]:
#                 thickness_map[endo_point[1], endo_point[0]] = min_thickness
        
#         average_thickness = np.mean(thicknesses) if thicknesses else 0.0
        
#         return average_thickness, thickness_map
        
#     except Exception as e:
#         logging.error(f"计算心肌厚度失败: {e}")
#         return 0.0, None

# --------------------------------------------------------计算左心室心肌厚度---------------------------------------------
def test_thickness_calculation(ed_slice, original_spacing):
    """修正版厚度计算 - 使用径向测量确保准确性"""
    global LV_BLOOD_POOL_ID, LV_MYOCARDIUM_ID
    
    LV_BLOOD_POOL_ID = 2  # 左心室血腔
    LV_MYOCARDIUM_ID = 1  # 左心室心肌
    
    # 使用径向测量方法计算厚度
    thickness_max, thickness_mean, thickness_min, thickness_map, message, _ = calculate_thickness_radial_accurate(ed_slice, original_spacing)
    
    # 可视化结果
    # if thickness > 0 and thickness_map is not None:
    #     # 创建可视化
    #     fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
    #     # 1. 原始分割
    #     axes[0, 0].imshow(ed_slice, cmap='tab10')
    #     axes[0, 0].set_title('原始分割掩码')
    #     axes[0, 0].axis('off')
        
    #     # 2. 准确的厚度分布图（基于径向测量）
    #     im = axes[0, 1].imshow(thickness_map, cmap='hot', vmin=0, vmax=15)
    #     axes[0, 1].set_title('心肌厚度分布 (mm) - 径向测量')
    #     axes[0, 1].axis('off')
    #     plt.colorbar(im, ax=axes[0, 1])
        
    #     # 3. 显示径向测量线
    #     axes[1, 0].imshow(ed_slice, cmap='tab10')
        
    #     # 绘制所有径向测量线
    #     for line in measurement_lines:
    #         endo = line['endo_point']
    #         epi = line['epi_point']
            
    #         # 画测量线
    #         axes[1, 0].plot([endo[0], epi[0]], [endo[1], epi[1]], 
    #                        'red', linewidth=1.5, alpha=0.8)
            
    #         # 标记端点
    #         axes[1, 0].scatter([endo[0]], [endo[1]], c='blue', s=20, marker='o', alpha=0.7)
    #         axes[1, 0].scatter([epi[0]], [epi[1]], c='green', s=20, marker='s', alpha=0.7)
        
    #     # 标记中心点（小标记避免遮挡）
    #     if measurement_lines:
    #         center = measurement_lines[0]['center']
    #         axes[1, 0].scatter([center[0]], [center[1]], c='yellow', 
    #                          s=40, marker='+', linewidth=2)
        
    #     axes[1, 0].set_title(f'径向测量线 ({len(measurement_lines)}个方向)')
    #     axes[1, 0].axis('off')
        
    #     # 4. 厚度分布直方图
    #     thickness_values = [line['thickness'] for line in measurement_lines]
    #     if thickness_values:
    #         axes[1, 1].hist(thickness_values, bins=15, alpha=0.7, 
    #                        color='skyblue', edgecolor='black')
    #         axes[1, 1].set_xlabel('厚度 (mm)')
    #         axes[1, 1].set_ylabel('测量方向数')
    #         axes[1, 1].set_title(f'厚度分布直方图\n平均: {np.mean(thickness_values):.2f}mm')
    #         axes[1, 1].axvline(np.mean(thickness_values), color='red', 
    #                          linestyle='--', linewidth=2)
            
    #         # 添加统计信息
    #         stats_text = f'统计信息:\n总方向数: {len(thickness_values)}\n平均厚度: {np.mean(thickness_values):.2f}mm\n厚度范围: {np.min(thickness_values):.1f}-{np.max(thickness_values):.1f}mm'
    #         axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
    #                        verticalalignment='top', fontsize=10,
    #                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    #     else:
    #         axes[1, 1].text(0.5, 0.5, '无厚度数据', ha='center', va='center', 
    #                       transform=axes[1, 1].transAxes)
    #         axes[1, 1].set_title('厚度分布直方图')
        
    #     plt.suptitle(f'心肌厚度分析结果: {message}', fontsize=16, y=0.98)
    #     plt.tight_layout()
    #     plt.show()
    
    return thickness_max, thickness_mean, thickness_min, thickness_map, message

def calculate_thickness_radial_accurate(mask, spacing_xy, num_angles=36):
    """
    准确的径向厚度测量方法
    确保厚度图完整覆盖心肌区域
    """
    try:
        blood_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        myo_mask = (mask == LV_MYOCARDIUM_ID).astype(np.uint8)
        
        # 找到血腔中心
        blood_points = np.where(blood_mask > 0)
        if len(blood_points[0]) == 0:
            return 8.0, 8.0, 8.0, None, "未找到血腔区域", []
        
        center_y = np.mean(blood_points[0])
        center_x = np.mean(blood_points[1])
        center = np.array([center_x, center_y])
        
        # 记录测量线
        measurement_lines = []
        thickness_values = []
        
        # 创建厚度图 - 初始化为0
        thickness_map = np.zeros_like(mask, dtype=np.float32)
        
        # 在多个方向上测量
        for angle_idx in range(num_angles):
            angle = angle_idx * (360 / num_angles)
            rad = np.radians(angle)
            direction = np.array([np.cos(rad), np.sin(rad)])
            
            # 从中心向外寻找血腔边界（心内膜）
            endo_point = find_boundary_along_ray_accurate(center, direction, blood_mask)
            if endo_point is None:
                continue
            
            # 从心内膜点继续向外寻找心肌外边界（心外膜）
            epi_point = find_boundary_along_ray_accurate(endo_point, direction, myo_mask)
            if epi_point is None:
                continue
            
            # 计算物理厚度
            dx = (epi_point[0] - endo_point[0]) * spacing_xy[0]
            dy = (epi_point[1] - endo_point[1]) * spacing_xy[1]
            thickness = np.sqrt(dx*dx + dy*dy)
            if 1 <= thickness <= 40.0:  # 合理的厚度范围
                measurement_lines.append({
                    'angle': angle,
                    'endo_point': endo_point,
                    'epi_point': epi_point,
                    'thickness': thickness,
                    'center': center
                })
                thickness_values.append(thickness)
                
                # 在厚度图上填充从心内膜到心外膜的线段
                fill_thickness_line(thickness_map, endo_point, epi_point, thickness, spacing_xy)
        
        if not thickness_values:
            return 8.0, 8.0, 8.0, None, None, []
        q1 = np.percentile(thickness_values, 25)
        q3 = np.percentile(thickness_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr  # 下界（1.5*IQR是统计学常用阈值）
        upper_bound = q3 + 1.5 * iqr  # 上界
        filtered_thickness = [t for t in thickness_values if lower_bound <= t <= upper_bound]
        if len(filtered_thickness) < 3:  # 至少保留3个有效值保证可靠性
            filtered_thickness = thickness_values.copy()

        if len(measurement_lines) >= 5:  # 测量方向足够多时启用
            sorted_lines = sorted(measurement_lines, key=lambda x: x['angle'])
            sorted_ts = [line['thickness'] for line in sorted_lines]
            valid_ts = []
            
            for i in range(len(sorted_ts)):
                left_idx = (i - 1) % len(sorted_ts)
                right_idx = (i + 1) % len(sorted_ts)
                neighbors = [sorted_ts[left_idx], sorted_ts[right_idx]]
                current_t = sorted_ts[i]
                
                if abs(current_t - np.mean(neighbors)) / np.mean(neighbors) <= 0.1:
                    valid_ts.append(current_t)
            
            if len(valid_ts) >= 3:
                filtered_thickness = valid_ts

        # if len(filtered_thickness) >= 2:  # 至少有2个值才进行筛选（避免单值误判）
        #     max_t = np.max(filtered_thickness)
        #     # 保留：与最大值相差＜25%的数值（即 t ≥ max_t * 0.75）
        #     # 逻辑：(max_t - t) / max_t ≤ 0.25 → t ≥ max_t * (1 - 0.25) = max_t * 0.75
        #     filtered_thickness = [t for t in filtered_thickness if t >= max_t * 0.75]

        # # 步骤4：最终数据量校验（新增，避免筛选后数据过少）
        if len(filtered_thickness) < 3:
            # 若筛选后有效数据不足3个，回退到空间一致性验证后的结果（或IQR过滤结果）
            # 优先保留空间一致性验证后的结果（更可靠），若无则用IQR过滤结果
            if 'valid_ts' in locals() and len(valid_ts) >= 3:
                filtered_thickness = valid_ts
            elif len([t for t in thickness_values if lower_bound <= t <= upper_bound]) >= 3:
                filtered_thickness = [t for t in thickness_values if lower_bound <= t <= upper_bound]
            else:
                # 极端情况：仅保留原始数据（保证至少有数据用于计算）
                filtered_thickness = thickness_values.copy()
        # 步骤4：计算复合指标（满足不同场景需求）
        if len(filtered_thickness) >= 1:
            if np.min(filtered_thickness) > 2 + 2 * spacing_xy[0] * spacing_xy[1]:
                max_thickness = np.max(filtered_thickness) - 2 * spacing_xy[0] * spacing_xy[1]      # 最直线距离
                mean_thickness = np.mean(filtered_thickness) - 2 * spacing_xy[0] * spacing_xy[1] 
                min_thickness = np.min(filtered_thickness) - 2 * spacing_xy[0] * spacing_xy[1] 
        else:
            max_thickness = None
            mean_thickness = None
            min_thickness = None

        # 步骤5：选择最终输出值（根据你的核心需求）
        # 场景1：临床诊断/病理评估（优先最大厚度，符合CMR测量规范
        # if max_thickness > 5:
        #     final_thickness = max_thickness - 2 # 目前预测的mask稍微厚一圈
        
        message = f"径向测量: {len(thickness_values)}个方向, 厚度 {max_thickness:.2f}mm"
        
        return float(max_thickness), float(mean_thickness), float(min_thickness), thickness_map, message, measurement_lines
        
    except Exception as e:
        return 8.0, 8.0, 8.0, None, f"径向测量失败: {str(e)}", []

def find_boundary_along_ray_accurate(start_point, direction, mask, max_steps=100):
    """
    准确寻找边界点 - 沿着射线方向找到第一个边界
    """
    step_size = 1.0
    current_point = start_point.copy()
    
    # 检查起始点是否在mask内
    x, y = int(round(current_point[0])), int(round(current_point[1]))
    if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
        return None
    
    was_inside = mask[y, x] > 0
    
    for step in range(max_steps):
        current_point = current_point + direction * step_size
        x, y = int(round(current_point[0])), int(round(current_point[1]))
        
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
            return None
        
        is_inside = mask[y, x] > 0
        
        # 寻找从内到外的边界
        if was_inside and not is_inside:
            # 找到边界，返回前一个点（在mask内的最后一个点）
            boundary_point = current_point - direction * (step_size / 2)
            return boundary_point
        
        was_inside = is_inside
    
    return None

def fill_thickness_line(thickness_map, start_point, end_point, thickness, spacing_xy):
    """
    在厚度图上填充从起点到终点的线段
    """
    # 计算线段上的点
    length_pixels = np.linalg.norm(end_point - start_point)
    num_points = max(2, int(length_pixels))
    
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        point = start_point + t * (end_point - start_point)
        x, y = int(round(point[0])), int(round(point[1]))
        
        if 0 <= y < thickness_map.shape[0] and 0 <= x < thickness_map.shape[1]:
            thickness_map[y, x] = thickness

def calculate_myocardium_thickness_simple(mask, spacing_xy):
    """
    修改后的简单厚度计算方法 - 使用径向测量作为主要方法
    """
    try:
        thickness_max, thickness_mean, thickness_min, thickness_map, message, _ = calculate_thickness_radial_accurate(mask, spacing_xy)
        return thickness_max, thickness_mean, thickness_min, thickness_map, message
    except Exception as e:
        return 0.0, 0.0, 0.0, None, f"计算失败: {str(e)}"
# --------------------------------------------------------------------------------------------------------------------
def create_slice_segmentation(mask, num_divisions=6, start_angle_degrees=0):
    """
    将切片按照放射状等分
    
    Args:
        mask: 2D掩码
        num_divisions: 等分数量
        start_angle_degrees: 起始角度（度数），0度为正X轴方向
        
    Returns:
        segmented_mask: 等分后的掩码，每个扇形区域有不同的标签
        center: 质心坐标
    """
    try:
        # 只处理左心室血腔
        lv_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        
        if np.sum(lv_mask) == 0:
            return None, None
            
        # 找到质心
        moments = cv2.moments(lv_mask)
        if moments['m00'] == 0:
            return None, None
            
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
        center = (center_x, center_y)
        
        # 创建等分掩码
        segmented_mask = np.zeros_like(mask, dtype=np.int32)
        h, w = mask.shape
        
        # 为每个像素计算角度并分配到对应的扇形
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # 计算每个像素相对于质心的角度
        dx = x_coords - center_x
        dy = y_coords - center_y
        angles = np.arctan2(dy, dx)  # 范围 [-π, π]
        
        # 转换起始角度为弧度
        start_angle_radians = np.radians(start_angle_degrees)
        
        # 调整角度以适应起始角度
        angles = angles - start_angle_radians
        angles = np.where(angles < -np.pi, angles + 2*np.pi, angles)
        angles = np.where(angles > np.pi, angles - 2*np.pi, angles)
        
        # 转换到 [0, num_divisions) 范围
        angles = (angles + np.pi) / (2 * np.pi) * num_divisions
        angles = np.floor(angles).astype(np.int32)
        angles = np.clip(angles, 0, num_divisions - 1)
        
        # 只在原始掩码的有效区域内分配扇形标签
        for div in range(num_divisions):
            division_mask = (angles == div) & (mask > 0)
            segmented_mask[division_mask] = div + 1  # 标签从1开始
            
        return segmented_mask, center
        
    except Exception as e:
        logging.error(f"切片等分失败: {e}")
        return None, None

def analyze_slice_segments_for_thickness(mask, segmented_mask, spacing_xy, num_divisions):
    """
    为各分区计算心肌厚度
    
    Args:
        mask: 原始掩码
        segmented_mask: 等分后的掩码
        spacing_xy: 像素间距
        num_divisions: 等分数量
        
    Returns:
        segment_thickness_stats: 每个扇形区域的心肌厚度统计
    """
    try:
        segment_thickness_stats = {}
        
        for div in range(1, num_divisions + 1):
            segment_area_mask = (segmented_mask == div)
            
            # 创建该分区的掩码
            segment_mask = mask.copy()
            segment_mask[~segment_area_mask] = 0  # 只保留当前分区的数据
            
            # 计算该分区的心肌厚度
            thickness_max, thickness_mean, thickness_min, thickness_map, _ = test_thickness_calculation(segment_mask, spacing_xy)
            
            segment_thickness_stats[div] = {
                'thickness_mm_max': thickness_max,
                'thickness_mm_mean': thickness_mean,
                'thickness_mm_min': thickness_min,
            }
            
        return segment_thickness_stats
        
    except Exception as e:
        logging.error(f"分析分区心肌厚度失败: {e}")
        return {}

def calculate_diameter_from_mask(mask, spacing_xy, target_id=LV_BLOOD_POOL_ID):
    """
    计算掩码的长短径
    Args:
        mask: 分割掩码
        spacing_xy: XY方向的像素间距
        target_id: 要计算直径的目标ID（默认为左心室血腔）
    Returns:
        long_diameter, short_diameter: 长径和短径
    """
    # 只处理目标ID对应的区域
    binary_mask = (mask == target_id).astype(np.uint8)
    
    # 找到轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        logging.warning("未找到轮廓")
        return -1.0, -1.0
    
    # 找到最大的轮廓
    out_cont = contours[0]
    if len(contours) > 1:
        area = cv2.contourArea(contours[0])
        for cont in contours[1:]:  
            cont_area = cv2.contourArea(cont)
            if cont_area > area:
                area = cont_area
                out_cont = cont
    
    if len(out_cont) == 0:
        logging.warning("轮廓为空")
        return 0.0, 0.0
    
    # 处理特殊情况
    x, y = np.unique(out_cont[:, :, 0]), np.unique(out_cont[:, :, 1])
    if len(x) == 1 or len(y) == 1:
        l = (max(x) - min(x) + 1) * spacing_xy[0]  # noqa: E741
        r = (max(y) - min(y) + 1) * spacing_xy[1]
        return max(l, r), min(l, r)
    
    # 计算长径
    cont = np.squeeze(out_cont)
    st_idx, ed_idx, ok = get_long_diameter_start_end_index(cont)
    if not ok:
        logging.warning("无法找到长径")
        return -1.0, -1.0
    
    p1 = cont[st_idx]
    p2 = cont[ed_idx]
    long_diameter = build_diameter(p1, p2, spacing_xy)
    
    # 计算短径
    st_idx, ed_idx, ok = get_short_diameter_start_end_index(cont, st_idx, ed_idx)
    if not ok:
        logging.warning("无法找到短径")
        return -1.0, -1.0
    
    p1 = cont[st_idx]
    p2 = cont[ed_idx]
    short_diameter = build_diameter(p1, p2, spacing_xy)
    
    return long_diameter, short_diameter

def create_3d_blocks(data, num_blocks):
    """
    将3D数据按指定块数分块，每块包含按间隔采样的切片
    
    参数:
        data: 3D numpy数组，形状为 (H, W, D)
        num_blocks: 要分成的块数 (30 或 25)
    
    返回:
        blocks: 列表，每个元素是一个3D numpy数组
        original_blocks: 原始块列表（用于后续分析）
    """
    total_slices = data.shape[2]
    
    if total_slices < num_blocks:
        logging.warning(f"    总切片数 {total_slices} 小于所需块数 {num_blocks}，无法分块")
        return None, None
    
    slices_per_block = total_slices // num_blocks
    if slices_per_block == 0:
        logging.warning(f"    每块切片数为0，无法分块")
        return None, None
    
    logging.info(f"    总切片数: {total_slices}, 分成 {num_blocks} 块, 每块 {slices_per_block} 层")
    
    blocks = []
    original_blocks = []  # 保存原始完整块
    
    for block_idx in range(num_blocks):
        slice_indices = []
        for i in range(slices_per_block):
            slice_idx = block_idx + i * num_blocks
            if slice_idx < total_slices:
                slice_indices.append(slice_idx)
        
        if len(slice_indices) == 0:
            logging.warning(f"    块 {block_idx} 没有有效切片，跳过")
            continue
            
        # 保存原始完整块
        original_block_data = data[:, :, slice_indices]
        original_blocks.append(original_block_data)
        
        # print(original_block_data.shape[2])

        if original_block_data.shape[2] <= 10:
            effective_slice_indices = slice_indices
        # 应用跳过切片逻辑，选择8层切片

        # if original_block_data.shape[2] == 9:
        #     effective_slice_indices = slice_indices[:-1]
        # if original_block_data.shape[2] == 10:
        #     effective_slice_indices = slice_indices[1:-1]
            # effective_slice_indices = effective_slice_indices[:-1]
        if original_block_data.shape[2] > 10:
            effective_slice_indices = slice_indices[1:-1]
            # effective_slice_indices = slice_indices[SKIP_HEAD_SLICES_PER_BLOCK:]
            # if len(effective_slice_indices) > SKIP_TAIL_SLICES_PER_BLOCK:
            #     effective_slice_indices = effective_slice_indices[:-SKIP_TAIL_SLICES_PER_BLOCK]
        # 从有效切片中选择最多8层
        # if len(effective_slice_indices) > MAX_SLICES_PER_BLOCK:
        #     effective_slice_indices = effective_slice_indices[:MAX_SLICES_PER_BLOCK]
        
        if len(effective_slice_indices) == 0:
            logging.warning(f"    块 {block_idx} 跳过切片后没有有效切片，跳过")
            blocks.append(None)
            continue
            
        # 重新获取8层切片数据
        # effective_indices_in_data = [slice_indices[i] for i in range(len(slice_indices)) 
        #                            if i >= SKIP_HEAD_SLICES_PER_BLOCK and 
        #                               i < len(slice_indices) - SKIP_TAIL_SLICES_PER_BLOCK and
        #                               i - SKIP_HEAD_SLICES_PER_BLOCK < MAX_SLICES_PER_BLOCK]
        
        if len(effective_slice_indices) == 0:
            logging.warning(f"    块 {block_idx} 有效切片索引为空，跳过")
            blocks.append(None)
            continue
            
        block_data = data[:, :, effective_slice_indices]
        blocks.append(block_data)
        
        logging.info(f"      块 {block_idx}: 原始切片 {slice_indices[:5]}{'...' if len(slice_indices) > 5 else ''}")
        logging.info(f"              有效切片 {effective_slice_indices}, 最终形状: {block_data.shape}")
    
    logging.info(f"    成功创建 {len(blocks)} 个块（包含 {len([b for b in blocks if b is not None])} 个有效块）")
    
    # import numpy as np
    # import matplotlib.pyplot as plt
    # plt.subplot(1,1,1)
    # plt.imshow(blocks[0][:,:,7])
    # plt.show()

    return blocks, original_blocks

# ------下面为计算左右心室横径---------
def find_interventricular_septum_center_robust(mask):
    """
    鲁棒的室间隔中心点查找方法
    """
    try:
        # 获取左右心室血腔掩码
        lv_blood_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        rv_blood_mask = (mask == RV_BLOOD_POOL_ID).astype(np.uint8)
        
        print(f"左心室血腔像素数: {np.sum(lv_blood_mask)}")
        print(f"右心室血腔像素数: {np.sum(rv_blood_mask)}")
        
        if np.sum(lv_blood_mask) == 0 or np.sum(rv_blood_mask) == 0:
            print("左心室或右心室血腔为空")
            return None
        
        # 方法1: 使用质心连线方法
        septum_center = find_septum_by_centroid_line(lv_blood_mask, rv_blood_mask)
        if septum_center is not None:
            return septum_center
        
        # 方法2: 使用距离变换方法
        septum_center = find_septum_by_distance_transform(lv_blood_mask, rv_blood_mask)
        if septum_center is not None:
            return septum_center
        
        print("所有方法都失败")
        return None
        
    except Exception as e:
        print(f"寻找室间隔中心失败: {e}")
        return None

def find_septum_by_centroid_line(lv_mask, rv_mask):
    """
    使用质心连线方法找到室间隔中心
    """
    try:
        # 计算两个心室的质心
        lv_moments = cv2.moments(lv_mask)
        rv_moments = cv2.moments(rv_mask)
        
        if lv_moments["m00"] == 0 or rv_moments["m00"] == 0:
            return None
        
        lv_center_x = lv_moments["m10"] / lv_moments["m00"]
        lv_center_y = lv_moments["m01"] / lv_moments["m00"]
        
        rv_center_x = rv_moments["m10"] / rv_moments["m00"]
        rv_center_y = rv_moments["m01"] / rv_moments["m00"]
        
        lv_center = np.array([lv_center_x, lv_center_y])
        rv_center = np.array([rv_center_x, rv_center_y])
        
        print(f"左心室质心: {lv_center}")
        print(f"右心室质心: {rv_center}")
        
        # 计算两个质心的中点
        centers_midpoint = (lv_center + rv_center) / 2
        
        # 计算连线方向
        direction = rv_center - lv_center
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
        
        # 在中点附近寻找实际的接触点
        contact_points = []
        search_range = 15  # 搜索范围
        step_size = 2
        
        # 垂直方向
        perpendicular = np.array([-direction[1], direction[0]])
        
        for offset in range(-search_range, search_range + 1, step_size):
            search_point = centers_midpoint + perpendicular * offset
            
            # 向左右两个方向寻找边界
            lv_boundary = find_closest_boundary(search_point, -direction, lv_mask, 30)
            rv_boundary = find_closest_boundary(search_point, direction, rv_mask, 30)
            
            if lv_boundary is not None and rv_boundary is not None:
                # 计算中点
                contact_midpoint = (lv_boundary + rv_boundary) / 2
                contact_points.append(contact_midpoint)
        
        if contact_points:
            # 计算所有接触点的平均值
            contact_array = np.array(contact_points)
            septum_center = np.mean(contact_array, axis=0)
            print(f"质心连线方法找到室间隔中心: {septum_center} (基于{len(contact_points)}个点)")
            return septum_center
        
        # 如果没有找到接触点，返回质心中点
        print(f"使用质心中点作为室间隔中心: {centers_midpoint}")
        return centers_midpoint
        
    except Exception as e:
        print(f"质心连线方法失败: {e}")
        return None

def find_closest_boundary(start_point, direction, mask, max_steps=50):
    """
    沿着指定方向找到最近的边界
    """
    try:
        step_size = 1.0
        
        # 检查起始点是否在mask内
        x, y = int(round(start_point[0])), int(round(start_point[1]))
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
            return None
        
        is_inside = mask[y, x] > 0
        
        # 沿着方向搜索边界
        for step in range(max_steps):
            test_point = start_point + direction * step * step_size
            x, y = int(round(test_point[0])), int(round(test_point[1]))
            
            if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
                return None
            
            current_inside = mask[y, x] > 0
            
            # 找到边界
            if is_inside != current_inside:
                boundary_point = test_point - direction * (step_size / 2)
                return boundary_point
        
        return None
        
    except Exception as e:
        return None

def find_septum_by_distance_transform(lv_mask, rv_mask):
    """
    使用距离变换方法找到室间隔中心
    """
    try:
        # 创建联合掩码
        combined_mask = lv_mask | rv_mask
        
        # 计算距离变换
        dist_transform = cv2.distanceTransform(combined_mask, cv2.DIST_L2, 5)
        
        # 反转距离变换，找到两个心室之间的区域
        inverted_dist = cv2.distanceTransform(255 - combined_mask * 255, cv2.DIST_L2, 5)
        
        # 找到距离两个心室都较近的区域
        septum_region = (dist_transform > 5) & (inverted_dist > 5)
        
        if np.sum(septum_region) == 0:
            return None
        
        # 找到septum区域的中心
        septum_points = np.where(septum_region)
        if len(septum_points[0]) == 0:
            return None
        
        septum_center_y = np.mean(septum_points[0])
        septum_center_x = np.mean(septum_points[1])
        septum_center = np.array([septum_center_x, septum_center_y])
        
        print(f"距离变换方法找到室间隔中心: {septum_center}")
        return septum_center
        
    except Exception as e:
        print(f"距离变换方法失败: {e}")
        return None

def find_ventricular_axis(mask, septum_center):
    """
    找到心室的轴线
    """
    try:
        # 获取左右心室血腔掩码
        lv_blood_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        rv_blood_mask = (mask == RV_BLOOD_POOL_ID).astype(np.uint8)
        
        # 找到两个心室的中心
        lv_points = np.where(lv_blood_mask > 0)
        rv_points = np.where(rv_blood_mask > 0)
        
        if len(lv_points[0]) == 0 or len(rv_points[0]) == 0:
            return None
        
        lv_center = np.array([np.mean(lv_points[1]), np.mean(lv_points[0])])
        rv_center = np.array([np.mean(rv_points[1]), np.mean(rv_points[0])])
        
        print(f"左心室中心: {lv_center}")
        print(f"右心室中心: {rv_center}")
        
        # 计算左右心室中心连线的方向
        axis_direction = rv_center - lv_center
        axis_norm = np.linalg.norm(axis_direction)
        
        if axis_norm > 0:
            axis_direction = axis_direction / axis_norm
        else:
            axis_direction = np.array([1.0, 0.0])  # 默认方向
        
        print(f"心室轴线方向: {axis_direction}")
        
        return axis_direction
        
    except Exception as e:
        print(f"寻找心室轴线失败: {e}")
        return None

def measure_ventricle_diameter_along_axis(start_point, direction, blood_mask, spacing_xy, ventricle_name):
    """
    测量沿轴线方向心室血腔的完整长度
    """
    try:
        # 找到轴线与血腔的两个交点
        boundaries = find_ventricle_boundaries_along_axis(start_point, direction, blood_mask, ventricle_name)
        
        if boundaries is None:
            print(f"{ventricle_name}边界查找失败，使用备用方法")
            # 备用方法：使用血腔边界框估算
            points = np.where(blood_mask > 0)
            if len(points[0]) > 0:
                # 计算血腔在轴线方向上的投影长度
                coords = np.column_stack((points[1], points[0]))  # (x, y)
                center = np.mean(coords, axis=0)
                
                # 计算点到轴线的距离
                axis_perp = np.array([-direction[1], direction[0]])
                projections = np.dot(coords - center, direction)
                
                if len(projections) > 0:
                    length = (np.max(projections) - np.min(projections)) * spacing_xy[0]
                    return max(5.0, min(length, 80.0))  # 合理范围
            return 30.0  # 默认值
        
        # 计算两个边界点之间的距离
        near_boundary, far_boundary = boundaries
        dx = (far_boundary[0] - near_boundary[0]) * spacing_xy[0]
        dy = (far_boundary[1] - near_boundary[1]) * spacing_xy[1]
        diameter = np.sqrt(dx*dx + dy*dy)
        
        # 确保数值合理
        diameter = max(5.0, min(diameter, 80.0))
        
        print(f"{ventricle_name}内径测量: 近端{np.array(near_boundary).astype(int)}, 远端{np.array(far_boundary).astype(int)}, 长度{diameter:.1f}mm")
        
        return diameter
        
    except Exception as e:
        print(f"测量{ventricle_name}内径失败: {e}")
        return 30.0  # 默认值

def find_ventricle_boundaries_along_axis(start_point, direction, blood_mask, ventricle_name):
    """
    找到轴线与心室血腔的两个边界交点
    """
    try:
        step_size = 1.0
        max_steps = 300
        
        # 向前搜索找到第一个进入点
        first_entry = None
        was_outside = blood_mask[int(round(start_point[1])), int(round(start_point[0]))] == 0
        
        for step in range(max_steps):
            test_point = start_point + direction * step * step_size
            x, y = int(round(test_point[0])), int(round(test_point[1]))
            
            if not (0 <= x < blood_mask.shape[1] and 0 <= y < blood_mask.shape[0]):
                break
            
            is_inside = blood_mask[y, x] > 0
            
            if was_outside and is_inside:
                first_entry = test_point - direction * (step_size / 2)
                break
            
            was_outside = not is_inside
        
        # 向后搜索找到第一个进入点（如果向前没找到）
        if first_entry is None:
            was_outside = blood_mask[int(round(start_point[1])), int(round(start_point[0]))] == 0
            
            for step in range(1, max_steps):
                test_point = start_point - direction * step * step_size
                x, y = int(round(test_point[0])), int(round(test_point[1]))
                
                if not (0 <= x < blood_mask.shape[1] and 0 <= y < blood_mask.shape[0]):
                    break
                
                is_inside = blood_mask[y, x] > 0
                
                if was_outside and is_inside:
                    first_entry = test_point + direction * (step_size / 2)
                    break
                
                was_outside = not is_inside
        
        if first_entry is None:
            print(f"未找到{ventricle_name}血腔进入点")
            return None
        
        # 从第一个进入点开始，继续向前找到退出点
        first_exit = None
        was_inside = True
        
        for step in range(max_steps):
            test_point = first_entry + direction * step * step_size
            x, y = int(round(test_point[0])), int(round(test_point[1]))
            
            if not (0 <= x < blood_mask.shape[1] and 0 <= y < blood_mask.shape[0]):
                first_exit = test_point - direction * step_size
                break
            
            is_inside = blood_mask[y, x] > 0
            
            if was_inside and not is_inside:
                first_exit = test_point - direction * (step_size / 2)
                break
            
            was_inside = is_inside
        
        if first_exit is None:
            print(f"未找到{ventricle_name}血腔退出点")
            return None
        
        # 从第一个退出点继续，寻找可能的第二个血腔区域（对于不规则形状）
        second_entry = None
        second_exit = None
        was_outside = True
        
        for step in range(max_steps):
            test_point = first_exit + direction * step * step_size
            x, y = int(round(test_point[0])), int(round(test_point[1]))
            
            if not (0 <= x < blood_mask.shape[1] and 0 <= y < blood_mask.shape[0]):
                break
            
            is_inside = blood_mask[y, x] > 0
            
            if was_outside and is_inside:
                second_entry = test_point - direction * (step_size / 2)
                # 继续找第二个退出点
                for step2 in range(step + 1, max_steps):
                    test_point2 = first_exit + direction * step2 * step_size
                    x2, y2 = int(round(test_point2[0])), int(round(test_point2[1]))
                    
                    if not (0 <= x2 < blood_mask.shape[1] and 0 <= y2 < blood_mask.shape[0]):
                        second_exit = test_point2 - direction * step_size
                        break
                    
                    is_inside2 = blood_mask[y2, x2] > 0
                    
                    if not is_inside2:
                        second_exit = test_point2 - direction * (step_size / 2)
                        break
                break
            
            was_outside = not is_inside
        
        # 选择最长的血腔段
        if second_entry is not None and second_exit is not None:
            first_length = np.linalg.norm(first_exit - first_entry)
            second_length = np.linalg.norm(second_exit - second_entry)
            
            if second_length > first_length:
                print(f"{ventricle_name}使用第二段血腔，长度{second_length:.1f} > 第一段{first_length:.1f}")
                return (second_entry, second_exit)
        
        return (first_entry, first_exit)
        
    except Exception as e:
        print(f"查找{ventricle_name}边界失败: {e}")
        return None

def find_left_ventricle_far_boundary(septum_center, axis_direction, blood_mask, myo_mask, max_steps=300):
    """
    找到左心室血腔最远端交点（用于可视化）
    """
    try:
        # 测量完整的左心室内径
        lv_diameter = measure_ventricle_diameter_along_axis(
            septum_center, -axis_direction, blood_mask, (1.0, 1.0), "左心室")
        
        # 返回远端边界点用于可视化
        boundaries = find_ventricle_boundaries_along_axis(
            septum_center, -axis_direction, blood_mask, "左心室")
        
        if boundaries is not None:
            return boundaries[1]  # 返回远端边界点
        else:
            # 备用方法
            return septum_center - axis_direction * lv_diameter
        
    except Exception as e:
        print(f"寻找左心室远端边界失败: {e}")
        return septum_center - axis_direction * 20

def find_right_ventricle_far_boundary(septum_center, axis_direction, blood_mask, max_steps=300):
    """
    找到右心室血腔最远端交点（用于可视化）
    """
    try:
        # 测量完整的右心室内径
        rv_diameter = measure_ventricle_diameter_along_axis(
            septum_center, axis_direction, blood_mask, (1.0, 1.0), "右心室")
        
        # 返回远端边界点用于可视化
        boundaries = find_ventricle_boundaries_along_axis(
            septum_center, axis_direction, blood_mask, "右心室")
        
        if boundaries is not None:
            return boundaries[1]  # 返回远端边界点
        else:
            # 备用方法
            return septum_center + axis_direction * rv_diameter
        
    except Exception as e:
        print(f"寻找右心室远端边界失败: {e}")
        return septum_center + axis_direction * 20


def calculate_angle_with_xaxis(point1, point2):
    """
    计算两点构成的线段与x轴（向上为正）的夹角（单位：度）
    Args:
        point1: 线段起点坐标 (x1, y1)
        point2: 线段终点坐标 (x2, y2)
    Returns:
        angle: 与y轴的夹角（0°~180°）
    """
    # 计算线段向量 (dx, dy)
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    
    # x轴参考向量（向上为正）
    y_axis_vec = np.array([1, 0])
    line_vec = np.array([dx, dy])
    
    # 计算向量长度（避免零向量报错）
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return 0.0  # 线段过短，返回0°
    
    # 点积计算夹角余弦值（修正数值误差，确保在[-1,1]范围内）
    dot_product = np.dot(line_vec, y_axis_vec)
    cos_theta = np.clip(dot_product / line_len, -1.0, 1.0)
    
    # 计算弧度并转换为角度（0°~180°）
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def create_visualization(mask, septum_center, lv_boundary, rv_boundary, lv_diameter, rv_diameter, axis_direction):
    """
    创建可视化图像
    """
    try:
        # 创建RGB图像
        if len(mask.shape) == 2:
            vis_image = np.stack([mask, mask, mask], axis=-1).astype(np.uint8) * 50
        else:
            vis_image = mask.astype(np.uint8)
        
        vis_image = np.ascontiguousarray(vis_image)
        
        # 获取掩码用于显示血腔轮廓
        lv_blood_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        rv_blood_mask = (mask == RV_BLOOD_POOL_ID).astype(np.uint8)
        
        # 添加血腔轮廓
        lv_contours, _ = cv2.findContours(lv_blood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rv_contours, _ = cv2.findContours(rv_blood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(vis_image, lv_contours, -1, (0, 255, 0), 1)
        cv2.drawContours(vis_image, rv_contours, -1, (255, 0, 0), 1)
        
        # 转换点坐标为整数
        septum_x, septum_y = int(round(septum_center[0])), int(round(septum_center[1]))
        lv_x, lv_y = int(round(lv_boundary[0])), int(round(lv_boundary[1]))
        rv_x, rv_y = int(round(rv_boundary[0])), int(round(rv_boundary[1]))
        
        # 画轴线
        line_length = 200
        axis_start = septum_center - axis_direction * line_length
        axis_end = septum_center + axis_direction * line_length
        
        axis_start_x, axis_start_y = int(round(axis_start[0])), int(round(axis_start[1]))
        axis_end_x, axis_end_y = int(round(axis_end[0])), int(round(axis_end[1]))
        
        cv2.line(vis_image, (axis_start_x, axis_start_y), (axis_end_x, axis_end_y), 
                (128, 128, 128), 1)
        
        # 标记测量点
        cv2.circle(vis_image, (septum_x, septum_y), 6, (0, 255, 255), -1)  # 黄色 - 室间隔中心
        cv2.circle(vis_image, (lv_x, lv_y), 5, (0, 255, 0), -1)  # 绿色 - 左心室内径边界
        cv2.circle(vis_image, (rv_x, rv_y), 5, (255, 0, 0), -1)  # 蓝色 - 右心室内径边界
        
        # 画内径测量线
        cv2.line(vis_image, (septum_x, septum_y), (lv_x, lv_y), (0, 255, 0), 2)  # 左心室内径
        cv2.line(vis_image, (septum_x, septum_y), (rv_x, rv_y), (255, 0, 0), 2)  # 右心室内径
        
        # 添加测量文本
        cv2.putText(vis_image, f'LV: {lv_diameter:.1f}mm', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_image, f'RV: {rv_diameter:.1f}mm', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 添加点标签
        cv2.putText(vis_image, 'Septum', (septum_x+5, septum_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis_image, 'LV Far', (lv_x+5, lv_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_image, 'RV Far', (rv_x+5, rv_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        lv_angle = calculate_angle_with_xaxis((septum_x, septum_y), (lv_x, lv_y))
        
        return vis_image, lv_angle
        
    except Exception as e:
        print(f"创建可视化失败: {e}")
        return None, None

def visualize_results(mask, results, visualization):
    """
    可视化结果
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 1. 原始分割
    axes[0].imshow(mask, cmap='tab10')
    axes[0].set_title('心脏分割掩码')
    axes[0].axis('off')
    
    # 2. 测量结果
    if visualization is not None:
        axes[1].imshow(visualization)
        axes[1].set_title('心室内径测量')
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, '测量失败', ha='center', va='center', 
                    transform=axes[1].transAxes)
        axes[1].set_title('心室内径测量')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def calculate_ventricular_diameters_robust(mask, spacing_xy):
    """
    修正版心室内径计算 - 测量轴线穿过整个心室血腔的长度
    """
    try:
        # 找到室间隔中心点
        septum_center = find_interventricular_septum_center_robust(mask)
        if septum_center is None:
            print("所有方法都未能找到室间隔中心点，使用备用方法")
            septum_center = np.array([mask.shape[1] // 2, mask.shape[0] // 2])
            print(f"使用图像中心作为室间隔中心: {septum_center}")
        
        print(f"室间隔中心点: {septum_center}")
        
        # 找到心室轴线
        axis_direction = find_ventricular_axis(mask, septum_center)
        if axis_direction is None:
            print("未找到心室轴线，使用默认方向")
            axis_direction = np.array([1.0, 0.0])
        
        print(f"心室轴线方向: {axis_direction}")
        
        # 获取掩码
        lv_blood_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        rv_blood_mask = (mask == RV_BLOOD_POOL_ID).astype(np.uint8)
        lv_myo_mask = (mask == LV_MYOCARDIUM_ID).astype(np.uint8)
        
        # 测量左心室内径 - 整个轴线穿过左心室血腔的长度
        lv_diameter = measure_ventricle_diameter_along_axis(
            septum_center, -axis_direction, lv_blood_mask, spacing_xy, "左心室")
        
        # 测量右心室内径 - 整个轴线穿过右心室血腔的长度  
        rv_diameter = measure_ventricle_diameter_along_axis(
            septum_center, axis_direction, rv_blood_mask, spacing_xy, "右心室")
        
        # 找到边界点用于可视化
        lv_far_boundary = find_left_ventricle_far_boundary(
            septum_center, axis_direction, lv_blood_mask, lv_myo_mask)
        
        rv_far_boundary = find_right_ventricle_far_boundary(
            septum_center, axis_direction, rv_blood_mask)
        
        print(f"左心室内径: {lv_diameter:.2f}mm")
        print(f"右心室内径: {rv_diameter:.2f}mm")
        
        # 创建可视化
        visualization, lv_angle = create_visualization(
            mask, septum_center, lv_far_boundary, rv_far_boundary, lv_diameter, rv_diameter, axis_direction)
        
        results = {
            'left_ventricle': {
                'transverse_diameter_mm': float(lv_diameter)
            },
            'right_ventricle': {
                'transverse_diameter_mm': float(rv_diameter)
            },
            'lv_angle': {
                'lv_angle': float(lv_angle)
            }
        }
        
        return results, visualization
        
    except Exception as e:
        print(f"计算心室内径失败: {e}")
        import traceback
        traceback.print_exc()
        
        default_results = {
            'left_ventricle': {'transverse_diameter_mm': 0.0},
            'right_ventricle': {'transverse_diameter_mm': 0.0}
        }
        return default_results, None

def analyze_ventricular_dimensions(mask, spacing_xy):
    """
    健壮的心室内径分析 - 确保总是返回有效结果
    """
    print("=== 心室内径测量（修正版本）===")
    print(f"掩码形状: {mask.shape}")
    print(f"掩码数据类型: {mask.dtype}")
    print(f"唯一值: {np.unique(mask)}")
    
    results, visualization = calculate_ventricular_diameters_robust(mask, spacing_xy)
    
    # 可视化结果
    # if visualization is not None:
    #     visualize_results(mask, results, visualization)
    
    if results and 'left_ventricle' in results and 'right_ventricle' in results:
        print("\n=== 最终测量结果 ===")
        print(f"左心室内径: {results['left_ventricle']['transverse_diameter_mm']:.1f}mm")
        print(f"右心室内径: {results['right_ventricle']['transverse_diameter_mm']:.1f}mm")
        print(f"lv_angle: {results['lv_angle']['lv_angle']:.1f}mm")

    else:
        print("测量失败，返回默认值")
        # 确保结果结构完整
        if 'left_ventricle' not in results:
            results = {
                'left_ventricle': {'transverse_diameter_mm': 0.0},
                'right_ventricle': {'transverse_diameter_mm': 0.0}
            }
    
    return results
#-------上面为计算左右心室横径---------

def compute_lv_volume_only(blk_norm, spacing=(1.0, 1.0, 1.0)):
    """只计算LV体积用于找ED块"""
    total_lv = 0.0
    for si in range(blk_norm.shape[2]):
        sl = blk_norm[:, :, si]
        lv_vol = np.sum(sl == LV_BLOOD_POOL_ID) * spacing[0] * spacing[1] * spacing[2] / 1000.0
        total_lv += lv_vol
    return total_lv

def compute_block_totals(blk_norm, img_norm, spacing=(1.0, 1.0, 1.0)):
    """计算块的总容积"""
    total_lv = 0.0
    total_lv_myo = 0.0
    total_rv = 0.0
    total_rv_myo = 0.0
    for si in range(blk_norm.shape[2]):
        sl = blk_norm[:, :, si]
        lv_vol = np.sum(sl == LV_BLOOD_POOL_ID) * spacing[0] * spacing[1] * spacing[2] / 1000.0
        lv_myo_vol = np.sum(sl == LV_MYOCARDIUM_ID) * spacing[0] * spacing[1] * spacing[2] / 1000.0
        rv_vol = np.sum(sl == RV_BLOOD_POOL_ID) * spacing[0] * spacing[1] * spacing[2] / 1000.0
        rv_myo_vol = np.sum(sl == RV_MYOCARDIUM_ID) * spacing[0] * spacing[1] * spacing[2] / 1000.0
        
        total_lv += lv_vol
        total_lv_myo += lv_myo_vol
        total_rv += rv_vol
        total_rv_myo += rv_myo_vol
    
    return total_lv, total_lv_myo, total_rv, total_rv_myo

def process_block(blk, img_blk=None, block_type=""):
    """处理单个块，返回处理后的数据和间距"""
    if blk is None:
        return None, None, None, None
        
    # 记录原始形状
    original_shape = blk.shape
    logging.info(f"    {block_type}块原始形状: {original_shape}")
    
    processed_blk = blk.copy()
    processed_img = img_blk.copy() if img_blk is not None else None
    
    # 步骤1: 裁剪 (Crop)
    non_zero = np.where(processed_blk > 0)
    if len(non_zero[0]) == 0:
        logging.warning(f"    {block_type}块没有非零像素，跳过")
        return None, None, None, None
        
    # 计算裁剪边界
    y_min = max(0, np.min(non_zero[0]) - CROP_MARGIN)
    y_max = min(processed_blk.shape[0] - 1, np.max(non_zero[0]) + CROP_MARGIN)
    x_min = max(0, np.min(non_zero[1]) - CROP_MARGIN) 
    x_max = min(processed_blk.shape[1] - 1, np.max(non_zero[1]) + CROP_MARGIN)
    
    # 执行裁剪
    processed_blk = processed_blk[y_min:y_max+1, x_min:x_max+1, :]
    if processed_img is not None:
        processed_img = processed_img[y_min:y_max+1, x_min:x_max+1, :]
    
    crop_shape = processed_blk.shape
    logging.info(f"    {block_type}块裁剪后形状: {crop_shape}")
    
    # 步骤2: Resize到64层，spacing变为(1,1,1)
    target_shape = (crop_shape[0], crop_shape[1], 64)  # 保持XY尺寸，Z设为64
    
    scale_factors = [
        target_shape[0] / crop_shape[0],  # y方向缩放因子
        target_shape[1] / crop_shape[1],  # x方向缩放因子  
        target_shape[2] / crop_shape[2]   # z方向缩放因子到64层
    ]
    
    # 执行resize
    processed_blk = zoom(processed_blk, scale_factors, order=0)  # mask用最近邻
    if processed_img is not None:
        processed_img = zoom(processed_img, scale_factors, order=1)  # image用线性插值
    
    final_shape = processed_blk.shape
    final_spacing = (1.0, 1.0, 1.0)  # 最终spacing为(1,1,1)
    
    logging.info(f"    {block_type}块Resize后形状: {final_shape}, 最终spacing: {final_spacing}")
    
    return processed_blk, processed_img, final_spacing, {
        'original_shape': original_shape,
        'crop_shape': crop_shape, 
        'final_shape': final_shape,
        'scale_factors': scale_factors
    }

def calculate_metrics_with_layers(ed_block, es_block, spacing=(1.0, 1.0, 1.0)):
    """计算心脏功能指标"""
    if ed_block is None or es_block is None:
        return None
    
    # 计算总体积
    ed_lv_vol_total, ed_lv_myo_vol_total, ed_rv_vol_total, ed_rv_myo_vol_total = compute_block_totals(ed_block, None, spacing)
    es_lv_vol_total, es_lv_myo_vol_total, es_rv_vol_total, es_rv_myo_vol_total = compute_block_totals(es_block, None, spacing)
    
    # 计算左心室指标
    lv_sv = ed_lv_vol_total - es_lv_vol_total
    lv_ef = (lv_sv / ed_lv_vol_total * 100) if ed_lv_vol_total > 0 else 0
    lv_co = lv_sv * ASSUMED_HEART_RATE / 1000.0
    
    # 计算右心室指标
    rv_sv = ed_rv_vol_total - es_rv_vol_total
    rv_ef = (rv_sv / ed_rv_vol_total * 100) if ed_rv_vol_total > 0 else 0
    rv_co = rv_sv * ASSUMED_HEART_RATE / 1000.0
    
    total_metrics = {
        'LV_EDV': ed_lv_vol_total,
        'LV_ESV': es_lv_vol_total, 
        'LV_SV': lv_sv,
        'LV_EF': lv_ef,
        'LV_CO': lv_co,
        'LV_Mass': ed_lv_myo_vol_total * MYOCARDIUM_DENSITY,
        'RV_EDV': ed_rv_vol_total,
        'RV_ESV': es_rv_vol_total,
        'RV_SV': rv_sv,
        'RV_EF': rv_ef,
        'RV_CO': rv_co,
        'RV_Mass': ed_rv_myo_vol_total * MYOCARDIUM_DENSITY
    }
    
    return total_metrics

def calculate_cine_sa_metrics(cine_sa_mask_path, slice_num):
    """
    计算SA图像的关键指标（修复空间校准问题）
    """
    try:
        # 加载数据
        pred_img = nib.load(cine_sa_mask_path)
        pred_data = np.round(pred_img.get_fdata()).astype(np.int16)
        pred_data = np.flip(pred_data, axis=1)
        # 修复1: 获取原始spacing
        original_spacing = pred_img.header.get_zooms()  # 获取(x, y, z) spacing
        logging.info(f"原始图像spacing: {original_spacing}")
        
        # if pred_data.shape[2] % 30 == 0:
        #     BLOCK_SIZES = 30
        # elif pred_data.shape[2] % 25 == 0:
        #     BLOCK_SIZES = 25
        # else:
        #     BLOCK_SIZES = 30
        BLOCK_SIZES = slice_num
        # 创建块
        blocks, original_blocks = create_3d_blocks(pred_data, BLOCK_SIZES)
        
        if blocks is None or original_blocks is None:
            return None
        
        valid_z_lengths = []
        for i, block in enumerate(blocks):
            if block is None:
                continue
            # 判断每个Z切片是否有LV血池
            has_lv_in_slice = np.any(block == LV_BLOOD_POOL_ID, axis=(0, 1))
            valid_z_indices = np.where(has_lv_in_slice)[0]
            if len(valid_z_indices) > 0:  # 只收集有有效切片的长度
                valid_z_lengths.append(len(valid_z_indices))

        # 计算众数（处理无有效切片的极端情况）
        if len(valid_z_lengths) == 0:
            # 所有block都无有效切片，众数设为0（后续所有block都append空）
            target_z_length = 0
        else:
            # 用scipy的mode计算众数（返回值格式：ModeResult(mode=array([x]), count=array([y]))）
            target_z_length = mode(valid_z_lengths, keepdims=False).mode
            # 若没有scipy，可手动实现众数（注释上面一行，取消下面注释）
            # from collections import Counter
            # count_dict = Counter(valid_z_lengths)
            # target_z_length = max(count_dict, key=count_dict.get)

        print(f"有效切片数的众数：{target_z_length}")
        block_volumes = []
        for i, block in enumerate(blocks):
            if block is None:
                # block_volumes.append((i, None))  # block为空，append空 
                continue
            
            # 判断每个Z切片是否有LV血池 
            has_lv_in_slice = np.any(block == LV_BLOOD_POOL_ID, axis=(0, 1))
            valid_z_indices = np.where(has_lv_in_slice)[0]
            current_z_length = len(valid_z_indices)
            
            # 仅当有效切片数等于众数时，计算容积；否则append空 
            if current_z_length != target_z_length:
                # block_volumes.append((i, None)) 
                continue
            
            # 以下是原计算逻辑（仅对符合众数条件的block执行）
            valid_block = block[..., valid_z_indices]
            lv_voxels = np.sum(valid_block == LV_BLOOD_POOL_ID)
            
            actual_z_spacing = original_spacing[2] * (pred_data.shape[2] / current_z_length) if current_z_length > 0 else original_spacing[2]
            single_phase_z_spacing = original_spacing[2] * (BLOCK_SIZES) if current_z_length > 0 else original_spacing[2]
            # 计算容积（ml）
            lv_vol = lv_voxels * original_spacing[0] * original_spacing[1] * actual_z_spacing / 1000.0
            block_volumes.append((i, lv_vol, valid_z_indices))
            
            # logging.info(f"块 {i}: 使用{min_valid_layers}层(索引{selected_layer_indices}), LV体素数={total_lv_voxels}, 容积={lv_vol:.2f}ml")
                
        if not block_volumes:
            return None
            
        # 找到ED和ES块
        block_volumes.sort(key=lambda x: x[1], reverse=True)
        ed_idx = block_volumes[0][0]  # 最大体积的块为ED
        es_idx = block_volumes[-1][0]  # 最小体积的块为ES

        ed_idx_slice = block_volumes[0][2]  # 最大体积的块为ED
        es_idx_slice = block_volumes[-1][2]  # 最小体积的块为ES
        
        logging.info(f"ED块索引: {ed_idx}, ES块索引: {es_idx}")
        
        metrics = {}
        
        # 计算ED期容积
        ed_block_original = original_blocks[ed_idx][..., ed_idx_slice]
        es_block_original = original_blocks[es_idx][..., es_idx_slice]
        ed_block_original_nocrop = original_blocks[ed_idx]
        
        # ed_block_original_cal_mass = original_blocks[ed_idx][..., ed_idx_slice]

        ed_lv_voxels = np.sum(ed_block_original == LV_BLOOD_POOL_ID)
        ed_lv_myo_voxels = np.sum(ed_block_original == LV_MYOCARDIUM_ID)
        ed_rv_voxels = np.sum(ed_block_original == RV_BLOOD_POOL_ID)
        
        es_lv_voxels = np.sum(es_block_original == LV_BLOOD_POOL_ID)
        es_rv_voxels = np.sum(es_block_original == RV_BLOOD_POOL_ID)
        
        voxel_volume_ml = original_spacing[0] * original_spacing[1] * actual_z_spacing / 1000.0
        voxel_volume_ml_mass = original_spacing[0] * original_spacing[1] * single_phase_z_spacing / 1000.0
        metrics['LV_EDV'] = ed_lv_voxels * voxel_volume_ml # * 0.92
        metrics['LV_ESV'] = es_lv_voxels * voxel_volume_ml # * 0.88
        metrics['LV_SV'] = metrics['LV_EDV'] - metrics['LV_ESV']
        metrics['LV_EF'] = (metrics['LV_SV'] / metrics['LV_EDV'] * 100) if metrics['LV_EDV'] > 0 else 0
        metrics['LV_CO'] = metrics['LV_SV'] * ASSUMED_HEART_RATE / 1000.0
        # print(ed_lv_myo_voxels, voxel_volume_ml, actual_z_spacing, es_block_original.shape,block_lvm_size, block_volumes[0][1],block_volumes[-1][1])
        metrics['LV_Mass'] = ed_lv_myo_voxels * voxel_volume_ml_mass * MYOCARDIUM_DENSITY
        
        metrics['RV_EDV'] = ed_rv_voxels * voxel_volume_ml
        metrics['RV_ESV'] = es_rv_voxels * voxel_volume_ml
        metrics['RV_SV'] = metrics['RV_EDV'] - metrics['RV_ESV']
        metrics['RV_EF'] = (metrics['RV_SV'] / metrics['RV_EDV'] * 100) if metrics['RV_EDV'] > 0 else 0
        metrics['RV_CO'] = metrics['RV_SV'] * ASSUMED_HEART_RATE / 1000.0
        
        # 长短径计算
        if ed_block_original is not None:
            target_slice = ed_block_original.shape[2] // 2
            ed_target_slice = ed_block_original[:, :, target_slice]
            results = analyze_ventricular_dimensions(ed_target_slice, original_spacing[:2])
            print(f"左心室横径: {results['left_ventricle']['transverse_diameter_mm']:.1f}mm")
            print(f"右心室横径: {results['right_ventricle']['transverse_diameter_mm']:.1f}mm")
            metrics['LV_ED_Long_Diameter'] = results['left_ventricle']['transverse_diameter_mm']
            metrics['RV_ED_Long_Diameter'] = results['right_ventricle']['transverse_diameter_mm']
            lv_angle = results['lv_angle']['lv_angle']
        has_lv_in_slice = np.any(ed_block_original_nocrop == LV_MYOCARDIUM_ID, axis=(0, 1))
        valid_z_indices = np.where(has_lv_in_slice)[0]
        z1 = valid_z_indices[0]
        if z1 > 3:
            z2 = 2
        z2 = (valid_z_indices[0] + valid_z_indices[-1]) // 2
        if z2 > 5:
            z2 = 5
        z3 = valid_z_indices[-1] - 1
        # if z3 > 8:
        #     z3 = 7
        #     z2 = 4
        SEGMENTATION_SLICE_INDICES = [z1, z2, z3]
        print(SEGMENTATION_SLICE_INDICES)
        # SEGMENTATION_SLICE_INDICES = [1, (valid_z_indices[1] + valid_z_indices[-1]) // 2 - 1, valid_z_indices[-1] - 3]
        # 心肌厚度和切片等分分析
        for slice_idx, num_divisions in zip(SEGMENTATION_SLICE_INDICES, SEGMENTATION_DIVISIONS):
            # 确定起始角度：第2张切片（索引1）从45度开始，第4张和第6张（索引3、5）从0度开始
            start_angle = 45 + lv_angle if slice_idx < 3 else 0 + lv_angle
            # 角度修复
            # ED块分析
            if ed_block_original_nocrop is not None and ed_block_original_nocrop.shape[2] > slice_idx:
                ed_slice = ed_block_original_nocrop[:, :, slice_idx]
                
                # 计算心肌厚度
                ed_thickness_max, ed_thickness_mean, ed_thickness_min, _, _ = test_thickness_calculation(ed_slice, original_spacing[:2])
                
                # 创建等分
                ed_segmented_mask, ed_center = create_slice_segmentation(ed_slice, num_divisions, start_angle)
                
                # 分析等分区域的心肌厚度
                if ed_segmented_mask is not None:
                    ed_segment_stats = analyze_slice_segments_for_thickness(ed_slice, ed_segmented_mask, original_spacing[:2], num_divisions)
                    
                    slice_num = slice_idx # + 1  # 转换为切片编号（1-based）
                    
                    # 添加整体心肌厚度
                    metrics[f'ED_Slice_{slice_num}_Thickness_max'] = ed_thickness_max
                    metrics[f'ED_Slice_{slice_num}_Thickness_mean'] = ed_thickness_mean
                    metrics[f'ED_Slice_{slice_num}_Thickness_min'] = ed_thickness_min
                    
                    # 添加各分区的心肌厚度（使用标准心脏分区命名）
                    TARGET_SEGMENT_IDS = {1, 6, 7, 12}

                    for div_id, div_stats in ed_segment_stats.items():
                        if slice_num in SEGMENT_NAMES and div_id in SEGMENT_NAMES[slice_num]:
                            segment_info = SEGMENT_NAMES[slice_num][div_id]
                            segment_id = segment_info['id']
                            segment_name = segment_info['name']
                            
                            # 获取原始壁厚值
                            thickness_max = div_stats['thickness_mm_max']
                            thickness_mean = div_stats['thickness_mm_mean']
                            thickness_min = div_stats['thickness_mm_min']
                            
                            if segment_id in TARGET_SEGMENT_IDS:
                                # 增加数值校验，避免非数字值报错  间隔壁挨着右心室误差小
                                if isinstance(thickness_max, (int, float)):
                                    thickness_max += original_spacing[0] * original_spacing[1]
                                if isinstance(thickness_mean, (int, float)):
                                    thickness_mean += original_spacing[0] * original_spacing[1]
                                if isinstance(thickness_min, (int, float)):
                                    thickness_min += original_spacing[0] * original_spacing[1]
                            
                            # 使用标准分区ID和名称存储（修改后的值）
                            metrics[f'ED_Segment_{segment_id:02d}_{segment_name}_Thickness_max'] = thickness_max
                            metrics[f'ED_Segment_{segment_id:02d}_{segment_name}_Thickness_mean'] = thickness_mean
                            metrics[f'ED_Segment_{segment_id:02d}_{segment_name}_Thickness_min'] = thickness_min
        
        return metrics
        
    except Exception as e:
        logging.error(f"计算cine SA图像的关键指标失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 设置日志级别以便查看详细信息
    logging.basicConfig(level=logging.INFO)
    
    cine_sa_mask_path = "/Users/zhanglantian/Documents/BAAI/Code/code/measure/data/0000375_sa_pred.nii.gz"
    metrics = calculate_cine_sa_metrics(cine_sa_mask_path)
    print("计算完成，结果:")
    print(metrics)