import os
import nibabel as nib
import numpy as np
import pandas as pd
import warnings
import re
import math
import cv2
import logging
from scipy.ndimage import zoom
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any

# --- 核心参数配置 ---
# 裁剪控制
CROP_MARGIN = 5     # 裁剪时在边界外保留的像素余量

# 医学参数定义 - 4ch图像标签
BACKGROUND_ID = 0       # 背景
LV_BLOOD_POOL_ID = 1    # 左心室血腔
LV_MYOCARDIUM_ID = 2    # 左心室心肌
# 3 不使用
RV_BLOOD_POOL_ID = 3    # 右心室血腔
RV_MYOCARDIUM_ID = 4    # 右心室心肌
RA_BLOOD_POOL_ID = 5    # 右心房
LA_BLOOD_POOL_ID = 6    # 左心房

# 分块参数
# BLOCK_SIZES = [30]

# 核心参数
MAX_SLICES_PER_BLOCK = 3  # 在每块中选择前N层切片来计算指标
SKIP_HEAD_SLICES_PER_BLOCK = 0  # 每块前面跳过的切片数
SKIP_TAIL_SLICES_PER_BLOCK = 0  # 每块后面跳过的切片数
TARGET_SLICE_INDEX = 1  # 用于计算心房径线的切片索引（第2张，索引从0开始）
RV_WALL_THICKNESS_DIVISIONS = 3  # 右心室室壁厚度3等分
APEX_SLICE_INDEX = 1  # 心尖位置的切片索引

# 医学参数
ASSUMED_HEART_RATE = 70
MYOCARDIUM_DENSITY = 1.05

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

def calculate_rv_wall_thickness_segmented(mask, spacing_xy, num_divisions=3):
    """
    计算右心室心肌厚度的3等分（右心室室壁厚度）
    
    Args:
        mask: 分割掩码
        spacing_xy: XY方向的像素间距
        num_divisions: 等分数量（默认3）
    
    Returns:
        segment_thickness_stats: 各分区的厚度统计
        segmentation_info: 分割信息（用于可视化）
    """
    try:
        # 只使用右心室心肌标签进行分析
        rv_myo_mask = (mask == RV_MYOCARDIUM_ID).astype(np.uint8)
        
        if np.sum(rv_myo_mask) == 0:
            logging.warning("没有找到右心室心肌区域")
            return {}, None
            
        # 获取右心室血腔作为参考
        rv_blood_mask = (mask == RV_BLOOD_POOL_ID).astype(np.uint8)
        
        # 创建弧形等分而非圆形等分
        segmented_mask, center, arc_info = create_arc_segmentation(rv_myo_mask, rv_blood_mask, num_divisions)
        
        if segmented_mask is None:
            return {}, None
            
        segment_thickness_stats = {}
        
        # 对每个分区计算厚度
        for div in range(1, num_divisions + 1):
            segment_area_mask = (segmented_mask == div)
            
            # 在该分区内计算厚度
            segment_mask = mask.copy()
            segment_mask[~segment_area_mask] = 0
            
            # 计算该分区的右心室心肌厚度
            thickness = calculate_rv_myocardium_thickness_in_segment(segment_mask, center, spacing_xy)
            
            segment_thickness_stats[div] = {
                'thickness_mm': thickness
            }
        
        # 构建分割信息用于可视化
        segmentation_info = {
            'segmented_mask': segmented_mask,
            'center': center,
            'rv_myo_mask': rv_myo_mask,
            'rv_blood_mask': rv_blood_mask,
            'arc_info': arc_info,
            'num_divisions': num_divisions
        }
        
        return segment_thickness_stats, segmentation_info
        
    except Exception as e:
        logging.error(f"计算右心室心肌厚度分区失败: {e}")
        return {}, None

def create_arc_segmentation(rv_myo_mask, rv_blood_mask, num_divisions=3):
    """
    创建基于弧形的等分（而非圆形）
    
    Args:
        rv_myo_mask: 右心室心肌掩码
        rv_blood_mask: 右心室血腔掩码
        num_divisions: 等分数量
    
    Returns:
        segmented_mask: 分割掩码
        center: 中心点
        arc_info: 弧形信息
    """
    try:
        # 找到右心室血腔的中心作为参考点
        if np.sum(rv_blood_mask) > 0:
            moments = cv2.moments(rv_blood_mask)
            if moments['m00'] > 0:
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
            else:
                # 如果血腔中心计算失败，使用心肌中心
                moments = cv2.moments(rv_myo_mask)
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
        else:
            # 使用心肌中心
            moments = cv2.moments(rv_myo_mask)
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
        
        center = (center_x, center_y)
        
        # 获取右心室心肌的轮廓
        contours, _ = cv2.findContours(rv_myo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None, None, None
            
        main_contour = max(contours, key=cv2.contourArea)
        contour_points = np.squeeze(main_contour)
        
        if len(contour_points.shape) == 1:
            contour_points = contour_points.reshape(1, -1)
        
        # 计算每个轮廓点相对于中心的角度
        angles = []
        for point in contour_points:
            dx = point[0] - center_x
            dy = point[1] - center_y
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        angles = np.array(angles)
        
        # 找到角度范围（弧形的起始和结束角度）
        min_angle = np.min(angles)
        max_angle = np.max(angles)
        
        # 处理角度跨越-π到π的情况
        if max_angle - min_angle > np.pi:
            # 重新计算角度，避免跨越边界
            adjusted_angles = []
            for angle in angles:
                if angle < 0:
                    adjusted_angles.append(angle + 2*np.pi)
                else:
                    adjusted_angles.append(angle)
            adjusted_angles = np.array(adjusted_angles)
            min_angle = np.min(adjusted_angles)
            max_angle = np.max(adjusted_angles)
            angles = adjusted_angles
        
        # 创建分割掩码
        segmented_mask = np.zeros(rv_myo_mask.shape, dtype=np.int32)
        segmented_mask = np.ascontiguousarray(segmented_mask)
        h, w = rv_myo_mask.shape
        
        # 计算每个像素的角度并分配到对应的弧形分区
        y_coords, x_coords = np.ogrid[:h, :w]
        dx = x_coords - center_x
        dy = y_coords - center_y
        pixel_angles = np.arctan2(dy, dx)
        
        # 处理跨越边界的情况
        if max_angle > np.pi:
            pixel_angles = np.where(pixel_angles < 0, pixel_angles + 2*np.pi, pixel_angles)
        
        # 计算弧形范围内的等分
        arc_range = max_angle - min_angle
        division_size = arc_range / num_divisions
        
        for div in range(num_divisions):
            div_start = min_angle + div * division_size
            div_end = min_angle + (div + 1) * division_size
            
            # 找到在这个角度范围内且在右心室心肌区域内的像素
            in_range = (pixel_angles >= div_start) & (pixel_angles < div_end) & (rv_myo_mask > 0)
            segmented_mask[in_range] = div + 1
        
        arc_info = {
            'min_angle': min_angle,
            'max_angle': max_angle,
            'arc_range': arc_range,
            'division_size': division_size
        }
        
        return segmented_mask, center, arc_info
        
    except Exception as e:
        logging.error(f"创建弧形分割失败: {e}")
        return None, None, None

def calculate_rv_myocardium_thickness_in_segment(mask, center, spacing_xy):
    """
    计算分区内右心室心肌的真实厚度（垂直距离）
    """
    try:
        # 只处理右心室心肌区域
        rv_myo_pixels = np.where(mask == RV_MYOCARDIUM_ID)
        
        if len(rv_myo_pixels[0]) == 0:
            return 2.0
            
        # 创建右心室心肌掩码用于形态学操作
        rv_myo_mask = (mask == RV_MYOCARDIUM_ID).astype(np.uint8)
        
        # 获取心肌的内外边界
        # 使用形态学操作找到心肌的内表面和外表面
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # 心肌的外边界（外表面）
        outer_boundary = cv2.morphologyEx(rv_myo_mask, cv2.MORPH_GRADIENT, kernel)
        
        # 心肌的内边界：通过收缩后再做边界检测
        eroded = cv2.erode(rv_myo_mask, kernel, iterations=2)
        inner_boundary = cv2.morphologyEx(eroded, cv2.MORPH_GRADIENT, kernel)
        
        # 如果内边界为空，使用收缩后的掩码中心线
        if np.sum(inner_boundary) == 0:
            # 计算心肌区域的骨架作为内边界
            from skimage.morphology import skeletonize
            skeleton = skeletonize(rv_myo_mask > 0)
            inner_boundary = skeleton.astype(np.uint8)
        
        if np.sum(outer_boundary) == 0 or np.sum(inner_boundary) == 0:
            # 退回到简单的径向距离方法
            distances = []
            for i in range(len(rv_myo_pixels[0])):
                y, x = rv_myo_pixels[0][i], rv_myo_pixels[1][i]
                dx = (x - center[0]) * spacing_xy[0]
                dy = (y - center[1]) * spacing_xy[1]
                dist = np.sqrt(dx*dx + dy*dy)
                distances.append(dist)
            
            if distances:
                return np.max(distances) - np.min(distances) if len(distances) > 1 else 2.0
            else:
                return 2.0
        
        # 计算内外边界之间的最短距离作为厚度
        outer_points = np.where(outer_boundary > 0)
        inner_points = np.where(inner_boundary > 0)
        
        if len(outer_points[0]) == 0 or len(inner_points[0]) == 0:
            return 2.0
        
        # 计算每个外边界点到内边界的最短距离
        min_thickness = float('inf')
        max_thickness = 0.0
        total_thickness = 0.0
        count = 0
        
        for i in range(len(outer_points[0])):
            outer_y, outer_x = outer_points[0][i], outer_points[1][i]
            
            min_dist = float('inf')
            for j in range(len(inner_points[0])):
                inner_y, inner_x = inner_points[0][j], inner_points[1][j]
                
                dx = (outer_x - inner_x) * spacing_xy[0]
                dy = (outer_y - inner_y) * spacing_xy[1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < min_dist:
                    min_dist = dist
            
            if min_dist < float('inf'):
                min_thickness = min(min_thickness, min_dist)
                max_thickness = max(max_thickness, min_dist)
                total_thickness += min_dist
                count += 1
        
        if count > 0:
            # 返回平均厚度
            avg_thickness = total_thickness / count
            return avg_thickness
        else:
            return 2.0
        
    except Exception as e:
        logging.error(f"计算分区右心室心肌厚度失败: {e}")
        return 2.0

def calculate_apex_thickness_fallback(lv_blood_mask, lv_myocardium_mask, spacing_xy):
    """
    备用心尖厚度计算方法：基于径向距离
    """
    try:
        # 找到血腔质心
        moments = cv2.moments(lv_blood_mask)
        if moments['m00'] == 0:
            return 3.0, None
        center_x = moments['m10'] / moments['m00']
        center_y = moments['m01'] / moments['m00']
        center = (center_x, center_y)
        
        # 找到心肌轮廓
        contours, _ = cv2.findContours(lv_myocardium_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 3.0, None
        
        main_contour = max(contours, key=cv2.contourArea)
        contour_points = np.squeeze(main_contour)
        if contour_points.ndim == 1:
            contour_points = contour_points.reshape(1, -1)
        
        # 计算轮廓上每个点到中心的距离
        distances = []
        for point in contour_points:
            dx = (point[0] - center_x) * spacing_xy[0]
            dy = (point[1] - center_y) * spacing_xy[1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances.append(dist)
        
        if not distances:
            return 3.0, None
        
        # 心尖厚度 = 最大距离 - 最小距离
        max_dist = np.max(distances)
        min_dist = np.min(distances)
        apex_thickness = max_dist - min_dist
        # 创建简单的可视化掩码
        apex_segment_mask = np.zeros(lv_myocardium_mask.shape, dtype=np.uint8)
        apex_segment_mask = np.ascontiguousarray(apex_segment_mask)
        cv2.circle(apex_segment_mask, (int(center_x), int(center_y)), int(max_dist), 1, 2)
        
        apex_info = {
            'method': 'radial_distance_fallback',
            'lv_blood_mask': lv_blood_mask,
            'lv_myocardium_mask': lv_myocardium_mask,
            'center': center,
            'max_radius': max_dist,
            'min_radius': min_dist,
            'apex_segment_mask': apex_segment_mask,
            'apex_segment_bounds': (0, lv_myocardium_mask.shape[0], 0, lv_myocardium_mask.shape[1]),
            'apex_segment_center': center,
            'segment_height': lv_myocardium_mask.shape[0] * spacing_xy[1],
            'segment_width': lv_myocardium_mask.shape[1] * spacing_xy[0],
            'apex_segment_pixels': int(np.sum(apex_segment_mask > 0))
        }
        
        logging.info(f"备用方法心尖厚度: {apex_thickness:.2f}mm (径向距离: {min_dist:.2f} - {max_dist:.2f})")
        return apex_thickness, apex_info
        
    except Exception as e:
        logging.error(f"备用心尖厚度计算失败: {e}")
        return 3.0, None

def pick_nearest_thickness(inner_hits, outer_hits, apex_endpoint, spacing_xy):
    inner_pt = pick_nearest(inner_hits, apex_endpoint)
    outer_pt = pick_nearest(outer_hits, apex_endpoint)
    dx = (outer_pt[0] - inner_pt[0]) * spacing_xy[0]
    dy = (outer_pt[1] - inner_pt[1]) * spacing_xy[1]
    return np.sqrt(dx * dx + dy * dy)

# 筛选心尖端邻域内的交点（核心：定义“附近”范围）
def filter_apex_neighborhood(hits, ref, radius=30):
    ref_arr = np.array(ref, dtype=np.float32)
    neighborhood_hits = []
    hit_distances = []  # 存储邻域内交点到心尖端的距离（可选）
    
    for h in hits:
        h_arr = np.array(h, dtype=np.float32)
        dist = np.linalg.norm(h_arr - ref_arr)
        if dist <= radius:
            neighborhood_hits.append(h_arr)
            hit_distances.append(dist)
    
    return np.array(neighborhood_hits), hit_distances

# 计算心尖端附近内、外交点对的厚度统计（最大/平均/最小）
def calculate_apex_thickness_stats(inner_hits, outer_hits, apex_endpoint, spacing_xy):
    # 筛选心尖端邻域内的内、外交点
    inner_neighborhood, _ = filter_apex_neighborhood(inner_hits, apex_endpoint)
    outer_neighborhood, _ = filter_apex_neighborhood(outer_hits, apex_endpoint)
    
    if len(inner_neighborhood) == 0 or len(outer_neighborhood) == 0:
        return 3.0, 3.0, 3.0  # 邻域内无足够交点
    
    # 计算邻域内所有内-外交点对的厚度（确保一一对应或全组合，根据你的交点匹配逻辑）
    thicknesses = []
    # 场景1：内、外交点是按射线配对的（如同一射线的内/外点）
    if len(inner_neighborhood) == len(outer_neighborhood):
        for inner_pt, outer_pt in zip(inner_neighborhood, outer_neighborhood):
            dx = (outer_pt[0] - inner_pt[0]) * spacing_xy[0]
            dy = (outer_pt[1] - inner_pt[1]) * spacing_xy[1]
            thickness = np.sqrt(dx * dx + dy * dy)
            thicknesses.append(thickness)
    # 场景2：内、外交点无配对关系，计算所有组合（可选，根据你的数据逻辑）
    else:
        for inner_pt in inner_neighborhood:
            for outer_pt in outer_neighborhood:
                dx = (outer_pt[0] - inner_pt[0]) * spacing_xy[0]
                dy = (outer_pt[1] - inner_pt[1]) * spacing_xy[1]
                thickness = np.sqrt(dx * dx + dy * dy)
                thicknesses.append(thickness)
    if not thicknesses:
        return 3.0, 3.0, 3.0
    
    # 计算统计值
    min_thickness = float(np.min(thicknesses))
    if min_thickness > 2 * spacing_xy[0] * spacing_xy[1] + 2:   #目前的预测mask稍微厚一点
        min_thickness = float(np.min(thicknesses)) - 2 * spacing_xy[0] * spacing_xy[1]
        max_thickness = float(np.mean(thicknesses)) - 2 * spacing_xy[0] * spacing_xy[1] 
        # 按照最小处理
        mean_thickness = float(np.min(thicknesses)) - 2 * spacing_xy[0] * spacing_xy[1]
    else:
        max_thickness = float(np.min(thicknesses))
        # 按照最小处理
        mean_thickness = float(np.min(thicknesses))

    
    return min_thickness, max_thickness, mean_thickness

# 选择距离心尖端最近的交点
def pick_nearest(hits, ref):
    ref_arr = np.array(ref, dtype=np.float32)
    dists = [np.linalg.norm(np.array(h, dtype=np.float32) - ref_arr) for h in hits]
    return np.array(hits[int(np.argmin(dists))], dtype=np.float32)

import matplotlib.pyplot as plt

def visualize_line_contour(long_line, inner_contour_points, inner_hits, 
                           apex_endpoint=None, title="直线与内轮廓位置关系"):
    """
    可视化直线、内轮廓及交点
    
    参数：
        long_line: 直线对象（需包含a, b, c属性，满足ax+by+c=0）
        inner_contour_points: 内轮廓点数组，shape=(N,2)
        inner_hits: 直线与内轮廓的交点列表，shape=(M,2)
        apex_endpoint: 心尖端点（可选，用于标记参考点）
        title: 图表标题
    """
    # 创建画布
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    # 1. 绘制内轮廓（多边形+散点）
    if len(inner_contour_points) > 0:
        # 绘制轮廓多边形（闭合）
        contour_poly = plt.Polygon(inner_contour_points, 
                                   fill=False, color='blue', linewidth=2, 
                                   label='内轮廓')
        ax.add_patch(contour_poly)
        # 绘制轮廓点（散点）
        ax.scatter(inner_contour_points[:, 0], inner_contour_points[:, 1], 
                   color='blue', s=10, alpha=0.5)
    
    # 2. 绘制直线（计算显示范围）
    # 获取轮廓的边界范围，确定直线的显示区间
    x_min, x_max = np.min(inner_contour_points[:, 0]) - 10, np.max(inner_contour_points[:, 0]) + 10
    y_min, y_max = np.min(inner_contour_points[:, 1]) - 10, np.max(inner_contour_points[:, 1]) + 10
    
    # 根据直线方程ax+by+c=0生成线上的点
    if abs(long_line.b) > 1e-10:  # 直线非垂直，用x求y
        x_line = np.linspace(x_min, x_max, 100)
        y_line = -(long_line.a * x_line + long_line.c) / long_line.b
    else:  # 直线垂直（b=0），用y求x
        y_line = np.linspace(y_min, y_max, 100)
        x_line = -long_line.c / long_line.a * np.ones_like(y_line)
    
    ax.plot(x_line, y_line, color='red', linewidth=2, label='长轴直线', linestyle='--')
    
    # 3. 绘制交点（红色大圆点）
    if inner_hits:
        hits_arr = np.array(inner_hits)
        ax.scatter(hits_arr[:, 0], hits_arr[:, 1], 
                   color='red', s=50, marker='o', 
                   label='交点', zorder=5)
    
    # 4. 标记心尖端点（可选）
    if apex_endpoint is not None:
        ax.scatter(apex_endpoint[0], apex_endpoint[1], 
                   color='green', s=80, marker='*', 
                   label='心尖端点', zorder=6)
    
    # 5. 图表美化
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X 坐标（像素）')
    ax.set_ylabel('Y 坐标（像素）')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')  # 等比例显示，避免图形变形
    
    plt.show()


def calculate_apex_thickness(mask, spacing_xy):
    """
    计算左心室心尖厚度（新方法：长径与心肌内/外轮廓的交点距离）
    
    Args:
        mask: 分割掩码
        spacing_xy: XY方向的像素间距
    
    Returns:
        apex_thickness: 心尖厚度 (mm)
        apex_info: 信息字典
    """
    try:
        lv_blood_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        lv_myocardium_mask = (mask == LV_MYOCARDIUM_ID).astype(np.uint8)
        
        logging.info(f"心尖厚度计算开始 - LV血腔像素数: {np.sum(lv_blood_mask)}, LV心肌像素数: {np.sum(lv_myocardium_mask)}")
        
        if np.sum(lv_blood_mask) == 0 or np.sum(lv_myocardium_mask) == 0:
            logging.warning("没有找到左心室血腔或心肌区域")
            return 3.0, 3.0, 3.0, None
            
        # 1) 获取心肌内/外边界轮廓（改进方法）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        myo_edge = cv2.morphologyEx(lv_myocardium_mask, cv2.MORPH_GRADIENT, kernel)
        
        # 方法1：基于接触关系
        blood_dil = cv2.dilate(lv_blood_mask, kernel, iterations=2)  # 增加膨胀次数
        inner_edge_mask = ((myo_edge > 0) & (blood_dil > 0)).astype(np.uint8)
        outer_edge_mask = ((myo_edge > 0) & (blood_dil == 0)).astype(np.uint8)

        # 提取轮廓
        inner_contours, _ = cv2.findContours(inner_edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        outer_contours, _ = cv2.findContours(outer_edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # 如果接触关系方法失败，使用距离变换方法
        if np.sum(inner_edge_mask) == 0 or np.sum(outer_edge_mask) == 0 or not inner_contours or not outer_contours:
            logging.info("接触关系方法失败，尝试距离变换方法")
            # 计算心肌到血腔的距离
            dist_transform = cv2.distanceTransform(lv_myocardium_mask, cv2.DIST_L2, 5)
            # 内边界：距离血腔较近的边界
            inner_edge_mask = ((myo_edge > 0) & (dist_transform < 5)).astype(np.uint8)
            # 外边界：距离血腔较远的边界
            outer_edge_mask = ((myo_edge > 0) & (dist_transform >= 5)).astype(np.uint8)

            inner_contours, _ = cv2.findContours(inner_edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            outer_contours, _ = cv2.findContours(outer_edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        logging.info(f"找到内轮廓数: {len(inner_contours)}, 外轮廓数: {len(outer_contours)}")
        if not inner_contours or not outer_contours:
            logging.warning("未找到内/外边界轮廓")
            return 3.0, 3.0, 3.0, None

        inner_contour_points = np.squeeze(max(inner_contours, key=cv2.contourArea)).astype(np.int32)
        outer_contour_points = np.squeeze(max(outer_contours, key=cv2.contourArea)).astype(np.int32)
        if inner_contour_points.ndim == 1:
            inner_contour_points = inner_contour_points.reshape(1, -1)
        if outer_contour_points.ndim == 1:
            outer_contour_points = outer_contour_points.reshape(1, -1)

        # 2) 计算LV血腔长轴
        blood_contours, _ = cv2.findContours(lv_blood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        logging.info(f"找到LV血腔轮廓数: {len(blood_contours)}")
        if not blood_contours:
            logging.warning("未找到LV血腔轮廓")
            return 3.0, None
        blood_pts = np.squeeze(max(blood_contours, key=cv2.contourArea))
        if blood_pts.ndim == 1:
            blood_pts = blood_pts.reshape(1, -1)
        long_st, long_ed, long_ok = get_long_diameter_start_end_index(blood_pts)
        logging.info(f"LV长轴计算: 起始点索引={long_st}, 结束点索引={long_ed}, 成功={long_ok}")
        if not long_ok:
            logging.warning("无法计算LV长轴")
            return 3.0, None
        long_p1 = blood_pts[long_st]
        long_p2 = blood_pts[long_ed]
        long_line = build_line2D(long_p1, long_p2)

        # 判定心尖端（屏幕坐标Y更大者）
        apex_endpoint = long_p1 if long_p1[1] > long_p2[1] else long_p2
        base_endpoint = long_p2 if apex_endpoint is long_p1 else long_p1

        # 延长长轴：从心尖端向基底方向延长，确保能穿过心肌
        long_vec = np.array([long_p2[0] - long_p1[0], long_p2[1] - long_p1[1]], dtype=np.float32)
        long_len = np.linalg.norm(long_vec)
        if long_len > 0:
            long_dir = long_vec / long_len
            # 向心尖端方向延长2倍长度（增加延长距离）
            extended_apex = apex_endpoint + long_dir * long_len * 2.0
            # 向基底方向延长2倍长度  
            extended_base = base_endpoint - long_dir * long_len * 2.0
            # 重新构建延长后的长轴
            long_line = build_line2D(extended_apex, extended_base)
            logging.info(f"延长长轴: 心尖端{apex_endpoint} -> {extended_apex}, 基底端{base_endpoint} -> {extended_base}")
            logging.info(f"长轴方向向量: {long_dir}, 长度: {long_len:.2f}")
        else:
            logging.warning("长轴长度为0，无法延长")
            extended_apex = apex_endpoint
            extended_base = base_endpoint

        # 3) 计算直线与多边形轮廓的交点（改进版）
        def intersect_line_with_contour(line, contour_points, contour_name="轮廓"):
            hits = []
            n = len(contour_points)
            logging.info(f"{contour_name}点数: {n}")
            logging.info(f"直线方程: {line.a:.3f}x + {line.b:.3f}y + {line.c:.3f} = 0")
            
            # 检查轮廓边界框
            if n > 0:
                min_x, max_x = np.min(contour_points[:, 0]), np.max(contour_points[:, 0])
                min_y, max_y = np.min(contour_points[:, 1]), np.max(contour_points[:, 1])
                logging.info(f"{contour_name}边界框: x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]")
            
            for i in range(n):
                p = contour_points[i]
                q = contour_points[(i + 1) % n]
                
                # 直接计算直线与线段的交点
                # 线段参数方程: P = p + t*(q-p), t in [0,1]
                # 直线方程: ax + by + c = 0
                # 代入: a*(px + t*(qx-px)) + b*(py + t*(qy-py)) + c = 0
                # 解得: t = -(a*px + b*py + c) / (a*(qx-px) + b*(qy-py))
                
                denom = line.a * (q[0] - p[0]) + line.b * (q[1] - p[1])
                if abs(denom) < 1e-10:  # 平行线
                    continue
                    
                t = -(line.a * p[0] + line.b * p[1] + line.c) / denom
                
                if 0 <= t <= 1:  # 交点在线段上
                    x = p[0] + t * (q[0] - p[0])
                    y = p[1] + t * (q[1] - p[1])
                    hits.append([x, y])
                    logging.info(f"找到{contour_name}交点: 线段{i}-{i+1}, t={t:.3f}, 坐标=({x:.2f}, {y:.2f})")
            
            logging.info(f"{contour_name}总交点数: {len(hits)}")
            return hits

        inner_hits = intersect_line_with_contour(long_line, inner_contour_points, "内轮廓")
        outer_hits = intersect_line_with_contour(long_line, outer_contour_points, "外轮廓")
        logging.info(f"长轴与内轮廓交点数: {len(inner_hits)}, 与外轮廓交点数: {len(outer_hits)}")


        def get_bbox_intersection(line, contour_points):
            """计算直线与轮廓边界框的交点（替代无交点的情况）"""
            if len(contour_points) == 0:
                return []
            min_x, max_x = np.min(contour_points[:,0]), np.max(contour_points[:,0])
            min_y, max_y = np.min(contour_points[:,1]), np.max(contour_points[:,1])
            
            # 边界框的四条边
            bbox_edges = [
                [(min_x, min_y), (max_x, min_y)],  # 下
                [(max_x, min_y), (max_x, max_y)],  # 右
                [(max_x, max_y), (min_x, max_y)],  # 上
                [(min_x, max_y), (min_x, min_y)]   # 左
            ]
            
            hits = []
            for (p, q) in bbox_edges:
                denom = line.a * (q[0]-p[0]) + line.b * (q[1]-p[1])
                if abs(denom) < 1e-10:
                    continue
                t = -(line.a*p[0] + line.b*p[1] + line.c) / denom
                if 0 <= t <= 1:
                    x = p[0] + t*(q[0]-p[0])
                    y = p[1] + t*(q[1]-p[1])
                    hits.append([x, y])
            return hits
        

        # visualize_line_contour(
        #     long_line=long_line, 
        #     inner_contour_points=inner_contour_points, 
        #     inner_hits=inner_hits, 
        #     apex_endpoint=apex_endpoint
        # )
        if len(inner_hits) == 0 or len(outer_hits) == 0: # 说明这块很薄
            4.0, 3.0, 2.0, None



        inner_pt = pick_nearest(inner_hits, apex_endpoint)
        outer_pt = pick_nearest(outer_hits, apex_endpoint)

        # 4) 计算两交点之间的物理距离（厚度）
        dx = (outer_pt[0] - inner_pt[0]) * spacing_xy[0]
        dy = (outer_pt[1] - inner_pt[1]) * spacing_xy[1]
        apex_thickness = float(np.sqrt(dx * dx + dy * dy))

        # 计算心尖端附近的统计厚度（新增）
        apex_thickness_min, apex_thickness_max, apex_thickness_mean = calculate_apex_thickness_stats(
            inner_hits, outer_hits, apex_endpoint, spacing_xy)

        # 5) 构建可视化掩码（绘制延长长轴、交点和连线）
        apex_segment_mask = np.zeros(lv_myocardium_mask.shape, dtype=np.uint8)
        # 确保数组是连续的，OpenCV要求
        apex_segment_mask = np.ascontiguousarray(apex_segment_mask)
        
        # 绘制延长后的长轴
        extended_apex_int = (int(round(extended_apex[0])), int(round(extended_apex[1])))
        extended_base_int = (int(round(extended_base[0])), int(round(extended_base[1])))
        cv2.line(apex_segment_mask, extended_apex_int, extended_base_int, 2, 2)
        
        # 绘制原始长轴
        original_apex_int = (int(round(apex_endpoint[0])), int(round(apex_endpoint[1])))
        original_base_int = (int(round(base_endpoint[0])), int(round(base_endpoint[1])))
        cv2.line(apex_segment_mask, original_apex_int, original_base_int, 3, 1)
        
        # 绘制内外轮廓交点
        p1_i = (int(round(inner_pt[0])), int(round(inner_pt[1])))
        p2_o = (int(round(outer_pt[0])), int(round(outer_pt[1])))
        cv2.circle(apex_segment_mask, p1_i, 3, 4, -1)  # 内轮廓交点
        cv2.circle(apex_segment_mask, p2_o, 3, 5, -1)  # 外轮廓交点
        
        # 绘制厚度连线
        cv2.line(apex_segment_mask, p1_i, p2_o, 1, 2)

        ys = [p1_i[1], p2_o[1]]
        xs = [p1_i[0], p2_o[0]]
        seg_min_y = max(0, min(ys) - 5)
        seg_max_y = min(apex_segment_mask.shape[0], max(ys) + 6)
        seg_min_x = max(0, min(xs) - 5)
        seg_max_x = min(apex_segment_mask.shape[1], max(xs) + 6)

        seg_h_mm = max(1.0, (seg_max_y - seg_min_y) * spacing_xy[1])
        seg_w_mm = max(1.0, (seg_max_x - seg_min_x) * spacing_xy[0])
        center_x = (inner_pt[0] + outer_pt[0]) / 2.0
        center_y = (inner_pt[1] + outer_pt[1]) / 2.0

        apex_info = {
            'method': 'long_axis_intersection',
            'lv_blood_mask': lv_blood_mask,
            'lv_myocardium_mask': lv_myocardium_mask,
            'long_diameter_points': (np.array(long_p1).astype(np.int32), np.array(long_p2).astype(np.int32)),
            'extended_long_axis_points': (extended_apex.astype(np.float32), extended_base.astype(np.float32)),
            'inner_intersection_point': inner_pt.astype(np.float32),
            'outer_intersection_point': outer_pt.astype(np.float32),
            'inner_contour_points': inner_contour_points,
            'outer_contour_points': outer_contour_points,
            'inner_hits': inner_hits,
            'outer_hits': outer_hits,
            'apex_segment_mask': apex_segment_mask,
            'apex_segment_bounds': (seg_min_y, seg_max_y, seg_min_x, seg_max_x),
            'apex_segment_center': (center_x, center_y),
            'segment_height': seg_h_mm,
            'segment_width': seg_w_mm,
            'apex_segment_pixels': int(np.sum(apex_segment_mask > 0))
        }

        # logging.info(f"基于长轴交点的心尖厚度: {apex_thickness:.2f}mm")
        return apex_thickness_max, apex_thickness_mean, apex_thickness_min, apex_info
        
    except Exception as e:
        logging.error(f"计算心尖厚度失败: {e}")
        return 3.0, 3.0, 3.0, None

# def calculate_diameter_from_mask(mask, spacing_xy, target_id=LA_BLOOD_POOL_ID):
#     """
#     计算掩码的长短径
#     Args:
#         mask: 分割掩码
#         spacing_xy: XY方向的像素间距
#         target_id: 要计算直径的目标ID（默认为左心房血腔）
#     Returns:
#         long_diameter, short_diameter: 长径和短径
#     """
#     # 只处理目标ID对应的区域
#     binary_mask = (mask == target_id).astype(np.uint8)
    
#     # 找到轮廓
#     contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
#     if not contours:
#         logging.warning("未找到轮廓")
#         return -1.0, -1.0
    
#     # 找到最大的轮廓
#     out_cont = contours[0]
#     if len(contours) > 1:
#         area = cv2.contourArea(contours[0])
#         for cont in contours[1:]:  
#             cont_area = cv2.contourArea(cont)
#             if cont_area > area:
#                 area = cont_area
#                 out_cont = cont
    
#     if len(out_cont) == 0:
#         logging.warning("轮廓为空")
#         return 0.0, 0.0
    
#     # 处理特殊情况
#     x, y = np.unique(out_cont[:, :, 0]), np.unique(out_cont[:, :, 1])
#     if len(x) == 1 or len(y) == 1:
#         l = (max(x) - min(x) + 1) * spacing_xy[0]  # noqa: E741
#         r = (max(y) - min(y) + 1) * spacing_xy[1]
#         return max(l, r), min(l, r)
    
#     # 计算长径
#     cont = np.squeeze(out_cont)
#     st_idx, ed_idx, ok = get_long_diameter_start_end_index(cont)
#     if not ok:
#         logging.warning("无法找到长径")
#         return -1.0, -1.0
    
#     p1 = cont[st_idx]
#     p2 = cont[ed_idx]
#     long_diameter = build_diameter(p1, p2, spacing_xy)
    
#     # 计算短径
#     st_idx, ed_idx, ok = get_short_diameter_start_end_index(cont, st_idx, ed_idx)
#     if not ok:
#         logging.warning("无法找到短径")
#         return -1.0, -1.0
    
#     p1 = cont[st_idx]
#     p2 = cont[ed_idx]
#     short_diameter = build_diameter(p1, p2, spacing_xy)
    
#     return long_diameter, short_diameter

#-------心房前后计算----------------
def create_base_visualization(mask: np.ndarray) -> np.ndarray:
    """
    创建基础可视化图像
    """
    try:
        # 创建彩色图像 - 确保正确的数据类型和布局
        if len(mask.shape) == 2:
            vis_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            # 设置背景为深灰色
            vis_image[:, :] = (30, 30, 30)
        else:
            vis_image = mask.astype(np.uint8)
            # 如果已经是彩色图像，确保是3通道
            if vis_image.shape[-1] != 3:
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # 确保图像是连续的并且是BGR格式
        vis_image = np.ascontiguousarray(vis_image)
        
        # 定义颜色映射
        colors = {
            LV_BLOOD_POOL_ID: (0, 255, 0),    # 绿色 - 左心室
            RV_BLOOD_POOL_ID: (255, 0, 0),    # 蓝色 - 右心室  
            LA_BLOOD_POOL_ID: (0, 255, 255),  # 黄色 - 左心房
            RA_BLOOD_POOL_ID: (255, 255, 0)   # 青色 - 右心房
        }
        
        for region_id, color in colors.items():
            region_mask = (mask == region_id).astype(np.uint8)
            if np.sum(region_mask) > 0:
                # 找到轮廓
                contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # 绘制轮廓
                    cv2.drawContours(vis_image, contours, -1, color, 2)
        
        return vis_image
        
    except Exception as e:
        logging.error(f"创建基础可视化失败: {e}")
        # 返回一个简单的备用图像
        h, w = mask.shape[:2]
        backup_image = np.zeros((h, w, 3), dtype=np.uint8)
        backup_image[:, :] = (30, 30, 30)
        return backup_image

def find_approximate_junction_region(atrium_mask: np.ndarray, ventricle_mask: np.ndarray) -> Optional[np.ndarray]:
    """
    增强版本：使用心房下边界的轮廓信息
    """
    try:
        # 找到心房轮廓
        contours, _ = cv2.findContours(atrium_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # 找到最大的轮廓（心房）
        atrium_contour = max(contours, key=cv2.contourArea)
        
        # 找到轮廓中最下方的点
        bottom_points = []
        max_y = np.max(atrium_contour[:, 0, 1])
        
        for point in atrium_contour:
            x, y = point[0]
            if y == max_y:
                bottom_points.append((x, y))
        
        if not bottom_points:
            return None
        
        # 对底部点按x坐标排序
        bottom_points.sort(key=lambda p: p[0])
        
        # 计算中心点
        center_x = (bottom_points[0][0] + bottom_points[-1][0]) // 2
        center_y = max_y
        
        # 创建交界区域 - 在中心点周围创建一个区域
        junction_region = np.zeros_like(atrium_mask)
        
        # 根据底部宽度调整区域大小
        bottom_width = bottom_points[-1][0] - bottom_points[0][0]
        region_height = 4
        region_width = max(6, bottom_width // 8)  # 宽度至少为6，或者底部宽度的1/8
        
        y_start = max(0, center_y)
        y_end = min(atrium_mask.shape[0], center_y + region_height)
        x_start = max(0, center_x - region_width // 2)
        x_end = min(atrium_mask.shape[1], center_x + region_width // 2)
        
        if x_start < x_end and y_start < y_end:
            junction_region[y_start:y_end, x_start:x_end] = 1
        
        return junction_region
        
    except Exception as e:
        logging.error(f"寻找增强近似交界区域失败: {e}")
        return None

def fit_line_ransac(points: np.ndarray, max_iterations: int = 100, threshold: float = 2.0) -> Optional[Tuple[float, float, float]]:
    """
    使用RANSAC算法拟合直线
    """
    try:
        if len(points) < 2:
            return None
        
        best_line = None
        best_inliers = 0
        
        for _ in range(max_iterations):
            # 随机选择两个点
            idx = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]
            
            # 计算直线参数
            if p2[0] - p1[0] == 0:  # 垂直线
                A, B, C = 1.0, 0.0, -p1[0]
            else:
                slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                A = -slope
                B = 1.0
                C = slope * p1[0] - p1[1]
            
            # 归一化
            norm = np.sqrt(A*A + B*B)
            if norm > 0:
                A, B, C = A/norm, B/norm, C/norm
            
            # 计算内点
            distances = np.abs(A * points[:, 0] + B * points[:, 1] + C)
            inliers = np.sum(distances < threshold)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_line = (A, B, C)
        
        return best_line if best_inliers >= len(points) * 0.3 else None
        
    except Exception as e:
        logging.error(f"RANSAC拟合失败: {e}")
        return None

def fit_line_least_squares(points: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    使用最小二乘法拟合直线
    """
    try:
        if len(points) < 2:
            return None
        
        x = points[:, 0]
        y = points[:, 1]
        
        # 最小二乘法拟合直线 y = mx + b
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # 转换为标准形式 Ax + By + C = 0
        A_param = -m
        B_param = 1.0
        C_param = -b
        
        # 归一化
        norm = np.sqrt(A_param*A_param + B_param*B_param)
        if norm > 0:
            A_param, B_param, C_param = A_param/norm, B_param/norm, C_param/norm
        
        return (A_param, B_param, C_param)
        
    except Exception as e:
        logging.error(f"最小二乘法拟合失败: {e}")
        return None

def find_atrioventricular_junction(atrium_mask: np.ndarray, ventricle_mask: np.ndarray) -> Tuple[Optional[Tuple[float, float, float]], Optional[np.ndarray]]:
    """
    找到房室交界线 - 使用心房下边界中心点
    """
    try:
        # 创建房室接触区域
        kernel = np.ones((3, 3), np.uint8)
        atrium_dilated = cv2.dilate(atrium_mask, kernel, iterations=2)
        ventricle_dilated = cv2.dilate(ventricle_mask, kernel, iterations=2)
        
        # 找到接触区域
        contact_region = atrium_dilated & ventricle_dilated
        
        if np.sum(contact_region) == 0:
            # 如果没有直接接触，使用心房下边界中心点
            contact_region = find_approximate_junction_region(atrium_mask, ventricle_mask)
        
        if np.sum(contact_region) == 0:
            return None, None
        
        # 找到接触区域的点
        contact_points = np.where(contact_region > 0)
        if len(contact_points[0]) < 2:
            return None, None
        
        # 将点坐标转换为 (x, y) 格式
        points = np.column_stack((contact_points[1], contact_points[0]))
        
        # 使用RANSAC拟合直线
        line_params = fit_line_ransac(points)
        if line_params is None:
            # 如果RANSAC失败，使用最小二乘法
            line_params = fit_line_least_squares(points)
        
        return line_params, points
        
    except Exception as e:
        logging.error(f"寻找房室交界线失败: {e}")
        return None, None

def calculate_diameter_perpendicular_to_line(mask: np.ndarray, line_params: Tuple[float, float, float], spacing_xy: Tuple[float, float]) -> Tuple[float, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    计算垂直于给定直线的直径
    Returns:
        diameter: 直径长度
        diameter_points: (p1, p2) 直径端点
    """
    try:
        A, B, C = line_params
        
        points = np.where(mask > 0)
        if len(points[0]) == 0:
            return -1.0, None
        
        coords = np.column_stack((points[1], points[0]))
        
        # 计算每个点到直线的距离（带符号）
        distances = A * coords[:, 0] + B * coords[:, 1] + C
        
        # 找到最小和最大距离的点
        min_idx = np.argmin(distances)
        max_idx = np.argmax(distances)
        
        p1 = coords[min_idx]
        p2 = coords[max_idx]
        
        dx = (p2[0] - p1[0]) * spacing_xy[0]
        dy = (p2[1] - p1[1]) * spacing_xy[1]
        diameter = np.sqrt(dx*dx + dy*dy)
        
        return diameter, (p1, p2)
        
    except Exception as e:
        logging.error(f"计算垂直直径失败: {e}")
        return -1.0, None

def calculate_diameter_parallel_to_line(mask: np.ndarray, line_params: Tuple[float, float, float], spacing_xy: Tuple[float, float]) -> Tuple[float, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    计算平行于给定直线的直径
    Returns:
        diameter: 直径长度
        diameter_points: (p1, p2) 直径端点
    """
    try:
        A, B, C = line_params
        
        # 获取心房掩码中的所有点
        points = np.where(mask > 0)
        if len(points[0]) == 0:
            return -1.0, None
        
        coords = np.column_stack((points[1], points[0]))
        
        # 计算每个点到直线的距离（带符号）
        distances = A * coords[:, 0] + B * coords[:, 1] + C
        
        # 找到距离直线最近和最远的点
        min_idx = np.argmin(np.abs(distances))
        max_idx = np.argmax(np.abs(distances))
        
        p1 = coords[min_idx]
        p2 = coords[max_idx]
        
        dx = (p2[0] - p1[0]) * spacing_xy[0]
        dy = (p2[1] - p1[1]) * spacing_xy[1]
        diameter = np.sqrt(dx*dx + dy*dy)
        
        return diameter, (p1, p2)
        
    except Exception as e:
        logging.error(f"计算平行直径失败: {e}")
        return -1.0, None

def calculate_parallel2_diameter(mask: np.ndarray, perpendicular_points: Tuple[np.ndarray, np.ndarray], spacing_xy: Tuple[float, float]) -> Tuple[float, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    计算parallel2直径：严格过垂直径中心点、与垂直径夹角90度的直线在心房内的交线长度
    核心逻辑：
    1. 计算垂直径的中心点（严格均分）
    2. 计算与垂直径严格垂直的方向向量（确保90度夹角）
    3. 生成过中心点且沿垂直方向的直线（延伸至图像边界）
    4. 找到该直线与心房掩码的所有交点
    5. 取交点中距离最远的两个点，计算其距离（即交线长度）
    """
    try:
        p1_perp, p2_perp = perpendicular_points
        
        # 1. 计算垂直径的中心点（严格均分）
        center_x = (p1_perp[0] + p2_perp[0]) / 2.0
        center_y = (p1_perp[1] + p2_perp[1]) / 2.0
        center = np.array([center_x, center_y], dtype=np.float32)
        
        # 2. 计算垂直径的方向向量
        dir_perp_x = p2_perp[0] - p1_perp[0]
        dir_perp_y = p2_perp[1] - p1_perp[1]
        
        # 计算与垂直径严格垂直的方向向量（顺时针旋转90度）
        # 垂直向量公式：(x, y) → (y, -x)（确保夹角90度）
        dir_parallel2_x = dir_perp_y
        dir_parallel2_y = -dir_perp_x
        
        # 归一化垂直方向向量（避免方向向量长度影响）
        norm = np.sqrt(dir_parallel2_x**2 + dir_parallel2_y**2)
        if norm < 1e-6:  # 避免除以零（垂直径为单点的极端情况）
            logging.warning("垂直径方向向量为零，无法计算parallel2")
            return -1.0, None
        dir_parallel2 = np.array([dir_parallel2_x, dir_parallel2_y]) / norm
        
        # 3. 生成过中心点且沿垂直方向的直线（延伸至图像边界）
        h, w = mask.shape[:2]
        
        # 计算直线与图像边界的交点，确定直线的覆盖范围
        t_min = -1e6  # 负方向延伸足够远
        t_max = 1e6   # 正方向延伸足够远
        
        # 计算直线与图像四条边界的交点对应的t值，限制直线在图像内
        t_values = []
        # 左边界 x=0
        if abs(dir_parallel2[0]) > 1e-6:
            t = (0 - center[0]) / dir_parallel2[0]
            y = center[1] + t * dir_parallel2[1]
            if 0 <= y < h:
                t_values.append(t)
        # 右边界 x=w-1
        if abs(dir_parallel2[0]) > 1e-6:
            t = (w-1 - center[0]) / dir_parallel2[0]
            y = center[1] + t * dir_parallel2[1]
            if 0 <= y < h:
                t_values.append(t)
        # 上边界 y=0
        if abs(dir_parallel2[1]) > 1e-6:
            t = (0 - center[1]) / dir_parallel2[1]
            x = center[0] + t * dir_parallel2[0]
            if 0 <= x < w:
                t_values.append(t)
        # 下边界 y=h-1
        if abs(dir_parallel2[1]) > 1e-6:
            t = (h-1 - center[1]) / dir_parallel2[1]
            x = center[0] + t * dir_parallel2[0]
            if 0 <= x < w:
                t_values.append(t)
        
        # 确定直线在图像内的t范围
        if t_values:
            t_min = min(t_values)
            t_max = max(t_values)
        
        # 4. 生成直线上的所有像素点（沿t范围均匀采样）
        num_samples = max(int(t_max - t_min) + 1, 100)  # 确保足够的采样点
        t_range = np.linspace(t_min, t_max, num_samples)
        line_points = []
        
        for t in t_range:
            x = center[0] + t * dir_parallel2[0]
            y = center[1] + t * dir_parallel2[1]
            x_int = int(round(x))
            y_int = int(round(y))
            
            # 只保留图像内的点
            if 0 <= x_int < w and 0 <= y_int < h:
                line_points.append((x_int, y_int))
        
        if not line_points:
            logging.warning("直线上没有图像内的点")
            return -1.0, None
        
        # 5. 找到直线与心房掩码的交集（即直线上属于心房的点）
        atrial_line_points = []
        for (x, y) in line_points:
            if mask[y, x] > 0:  # 该点在心房内
                atrial_line_points.append((x, y))
        
        if len(atrial_line_points) < 2:
            logging.warning(f"直线与心房的交点不足2个（仅找到{len(atrial_line_points)}个）")
            return -1.0, None
        
        # 6. 计算交线的长度（取交集中距离最远的两个点）
        atrial_line_points = np.array(atrial_line_points, dtype=np.float32)
        
        # 计算所有点对之间的距离，找到最远的一对
        max_dist = 0.0
        best_p1 = None
        best_p2 = None
        
        for i in range(len(atrial_line_points)):
            for j in range(i+1, len(atrial_line_points)):
                dx = (atrial_line_points[j][0] - atrial_line_points[i][0]) * spacing_xy[0]
                dy = (atrial_line_points[j][1] - atrial_line_points[i][1]) * spacing_xy[1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist > max_dist:
                    max_dist = dist
                    best_p1 = atrial_line_points[i]
                    best_p2 = atrial_line_points[j]
        
        if best_p1 is None or best_p2 is None:
            logging.warning("无法找到交线上的有效端点")
            return -1.0, None
        
        return max_dist, (best_p1, best_p2)
        
    except Exception as e:
        logging.error(f"计算parallel2直径失败: {e}")
        return -1.0, None

def draw_atrial_measurement(vis_image: np.ndarray, av_junction_line: Tuple[float, float, float], 
                          junction_points: Optional[np.ndarray], diameter_points: Optional[Tuple[np.ndarray, np.ndarray]], 
                          diameter: float, atrium_id: int, color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    在图像上绘制心房测量结果
    """
    try:
        # 确保图像是BGR格式且连续
        if vis_image is None:
            return None
            
        vis_image = np.ascontiguousarray(vis_image)
        
        if color is None:
            # 根据心房ID选择颜色
            color = (0, 255, 255) if atrium_id == LA_BLOOD_POOL_ID else (255, 255, 0)
        
        A, B, C = av_junction_line
        
        # 绘制房室交界点
        if junction_points is not None:
            for point in junction_points[:50]:  # 只绘制前50个点避免太密集
                x, y = int(point[0]), int(point[1])
                if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                    cv2.circle(vis_image, (x, y), 2, (255, 255, 255), -1)
        
        # 绘制房室交界线
        h, w = vis_image.shape[:2]
        points_on_line = []
        
        # 在图像边界上找到直线的两个点
        for x in [0, w-1]:
            if B != 0:
                y = int((-C - A * x) / B)
                if 0 <= y < h:
                    points_on_line.append((x, y))
        
        for y in [0, h-1]:
            if A != 0:
                x = int((-C - B * y) / A)
                if 0 <= x < w:
                    points_on_line.append((x, y))
        
        if len(points_on_line) >= 2:
            # 取距离最远的两个点来画线
            if len(points_on_line) > 2:
                max_distance = 0
                best_pair = (points_on_line[0], points_on_line[1])
                for i in range(len(points_on_line)):
                    for j in range(i+1, len(points_on_line)):
                        dist = np.sqrt((points_on_line[i][0]-points_on_line[j][0])**2 + 
                                     (points_on_line[i][1]-points_on_line[j][1])**2)
                        if dist > max_distance:
                            max_distance = dist
                            best_pair = (points_on_line[i], points_on_line[j])
                points_on_line = [best_pair[0], best_pair[1]]
            
            pt1, pt2 = points_on_line[0], points_on_line[1]
            cv2.line(vis_image, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 绘制横径测量线
        if diameter_points is not None:
            p1, p2 = diameter_points
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            
            # 检查点是否在图像范围内
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                
                # 绘制测量线
                cv2.line(vis_image, pt1, pt2, color, 3, cv2.LINE_AA)
                
                # 绘制端点
                cv2.circle(vis_image, pt1, 5, color, -1)
                cv2.circle(vis_image, pt2, 5, color, -1)
        
        # 添加图例
        atrium_name = "Left Atrium" if atrium_id == LA_BLOOD_POOL_ID else "Right Atrium"
        legend_text = f"{atrium_name}: {diameter:.1f}mm"
        
        # 确保文本位置在图像范围内
        y_position = 30 if atrium_id == LA_BLOOD_POOL_ID else 60
        if y_position < vis_image.shape[0]:
            cv2.putText(vis_image, legend_text, (10, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return vis_image
        
    except Exception as e:
        logging.error(f"绘制心房测量失败: {e}")
        return vis_image

def calculate_atrial_diameters_with_visualization(mask: np.ndarray, spacing_xy: Tuple[float, float]) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    计算左右心房的三个直径并返回可视化：
    - perpendicular_diameter: 垂直于房室交界线
    - parallel_diameter: 平行于房室交界线
    - parallel2_diameter: 垂直于perpendicular、穿过其中心点（严格90度）
    Args:
        mask: 分割掩码
        spacing_xy: XY方向的像素间距
    Returns:
        dict: 包含左右心房三个直径的字典
        vis_image: 可视化图像
    """
    try:
        # 创建基础可视化图像
        vis_image = create_base_visualization(mask)
        
        # 计算左心房直径（包含parallel2）
        la_vis_image, la_perpendicular_diameter, la_perpendicular_points, la_parallel_diameter, la_parallel_points, la_parallel2_diameter, la_parallel2_points = calculate_atrial_three_diameters(
            mask, spacing_xy, LA_BLOOD_POOL_ID, LV_BLOOD_POOL_ID, vis_image)
        
        # 计算右心房直径（包含parallel2）
        ra_vis_image, ra_perpendicular_diameter, ra_perpendicular_points, ra_parallel_diameter, ra_parallel_points, ra_parallel2_diameter, ra_parallel2_points = calculate_atrial_three_diameters(
            mask, spacing_xy, RA_BLOOD_POOL_ID, RV_BLOOD_POOL_ID, la_vis_image)
        
        results = {
            'left_atrium': {
                'perpendicular_diameter_mm': float(la_perpendicular_diameter) if la_perpendicular_diameter > 0 else 0.0,
                'parallel_diameter_mm': float(la_parallel_diameter) if la_parallel_diameter > 0 else 0.0,
                'parallel2_diameter_mm': float(la_parallel2_diameter) if la_parallel2_diameter > 0 else 0.0
            },
            'right_atrium': {
                'perpendicular_diameter_mm': float(ra_perpendicular_diameter) if ra_perpendicular_diameter > 0 else 0.0,
                'parallel_diameter_mm': float(ra_parallel_diameter) if ra_parallel_diameter > 0 else 0.0,
                'parallel2_diameter_mm': float(ra_parallel2_diameter) if ra_parallel2_diameter > 0 else 0.0
            }
        }
        
        # 添加标题
        cv2.putText(ra_vis_image, "Atrial Diameter Measurements (3 Directions)", 
                   (10, ra_vis_image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        logging.info(f"左心房 - 垂直径: {la_perpendicular_diameter:.1f}mm, 平行径: {la_parallel_diameter:.1f}mm, Parallel2: {la_parallel2_diameter:.1f}mm")
        logging.info(f"右心房 - 垂直径: {ra_perpendicular_diameter:.1f}mm, 平行径: {ra_parallel_diameter:.1f}mm, Parallel2: {ra_parallel2_diameter:.1f}mm")
        
        return results, ra_vis_image
        
    except Exception as e:
        logging.error(f"计算心房直径失败: {e}")
        default_results = {
            'left_atrium': {
                'perpendicular_diameter_mm': 0.0,
                'parallel_diameter_mm': 0.0,
                'parallel2_diameter_mm': 0.0
            },
            'right_atrium': {
                'perpendicular_diameter_mm': 0.0,
                'parallel_diameter_mm': 0.0,
                'parallel2_diameter_mm': 0.0
            }
        }
        return default_results, create_base_visualization(mask)

def calculate_atrial_three_diameters(mask: np.ndarray, spacing_xy: Tuple[float, float], 
                                   atrium_id: int, ventricle_id: int, 
                                   visualization_image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, Optional[Tuple[np.ndarray, np.ndarray]], float, Optional[Tuple[np.ndarray, np.ndarray]], float, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    计算心房三个直径：
    - perpendicular_diameter: 垂直于房室交界线
    - parallel_diameter: 平行于房室交界线
    - parallel2_diameter: 垂直于perpendicular、穿过其中心点（严格90度）
    Returns:
        vis_image: 可视化图像
        perpendicular_diameter: 垂直径长度
        perpendicular_points: 垂直径端点
        parallel_diameter: 平行径长度
        parallel_points: 平行径端点
        parallel2_diameter: parallel2直径长度
        parallel2_points: parallel2直径端点
    """
    try:
        # 获取心房和心室掩码
        atrium_mask = (mask == atrium_id).astype(np.uint8)
        ventricle_mask = (mask == ventricle_id).astype(np.uint8)
        
        # 创建可视化图像
        if visualization_image is None:
            vis_image = create_base_visualization(mask)
        else:
            vis_image = visualization_image.copy()
        
        if np.sum(atrium_mask) == 0 or np.sum(ventricle_mask) == 0:
            logging.warning(f"心房或心室掩码为空: 心房{atrium_id}, 心室{ventricle_id}")
            return vis_image, -1.0, None, -1.0, None, -1.0, None
        
        # 找到房室交界线
        av_junction_line, junction_points = find_atrioventricular_junction(atrium_mask, ventricle_mask)
        if av_junction_line is None:
            logging.warning("无法找到房室交界线")
            return vis_image, -1.0, None, -1.0, None, -1.0, None
        
        # 计算垂直于交界线的直径
        perpendicular_diameter, perpendicular_points = calculate_diameter_perpendicular_to_line(
            atrium_mask, av_junction_line, spacing_xy)
        
        # 计算平行于交界线的直径
        parallel_diameter, parallel_points = calculate_diameter_parallel_to_line(
            atrium_mask, av_junction_line, spacing_xy)
        
        # 计算parallel2直径（严格过中心点、90度垂直）
        parallel2_diameter = -1.0
        parallel2_points = None
        if perpendicular_points is not None:
            parallel2_diameter, parallel2_points = calculate_parallel2_diameter(
                atrium_mask, perpendicular_points, spacing_xy)
        
        # 在可视化图像上绘制三个直径
        vis_image = draw_atrial_three_measurements(vis_image, av_junction_line, junction_points,
                                                 perpendicular_points, perpendicular_diameter,
                                                 parallel_points, parallel_diameter,
                                                 parallel2_points, parallel2_diameter,
                                                 atrium_id)
        
        return vis_image, perpendicular_diameter, perpendicular_points, parallel_diameter, parallel_points, parallel2_diameter, parallel2_points
        
    except Exception as e:
        logging.error(f"计算心房三个直径失败: {e}")
        return visualization_image if visualization_image is not None else create_base_visualization(mask), -1.0, None, -1.0, None, -1.0, None

def draw_atrial_three_measurements(vis_image: np.ndarray, av_junction_line: Tuple[float, float, float], 
                                 junction_points: Optional[np.ndarray], 
                                 perpendicular_points: Optional[Tuple[np.ndarray, np.ndarray]],
                                 perpendicular_diameter: float,
                                 parallel_points: Optional[Tuple[np.ndarray, np.ndarray]],
                                 parallel_diameter: float,
                                 parallel2_points: Optional[Tuple[np.ndarray, np.ndarray]],
                                 parallel2_diameter: float,
                                 atrium_id: int) -> np.ndarray:
    """
    在图像上绘制心房三个直径的测量结果（突出显示parallel2的90度特性）
    """
    try:
        # 确保图像是BGR格式且连续
        if vis_image is None:
            return None
            
        vis_image = np.ascontiguousarray(vis_image)
        
        # 根据心房ID选择颜色（确保三条线颜色区分明显）
        if atrium_id == LA_BLOOD_POOL_ID:
            perpendicular_color = (0, 200, 255)  # 橙色 - 左心房垂直径
            parallel_color = (0, 150, 255)       # 红色 - 左心房平行径
            parallel2_color = (255, 0, 255)      # 紫色 - 左心房parallel2（90度垂直）
        else:
            perpendicular_color = (255, 200, 0)  # 蓝色 - 右心房垂直径
            parallel_color = (255, 150, 0)       # 青色 - 右心房平行径
            parallel2_color = (0, 255, 150)      # 青绿色 - 右心房parallel2（90度垂直）
        
        A, B, C = av_junction_line
        h, w = vis_image.shape[:2]
        
        # 绘制房室交界点
        if junction_points is not None:
            for point in junction_points[:50]:  # 只绘制前50个点避免太密集
                x, y = int(point[0]), int(point[1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(vis_image, (x, y), 2, (255, 255, 255), -1)
        
        # 绘制房室交界线
        points_on_line = []
        for x in [0, w-1]:
            if B != 0:
                y = int((-C - A * x) / B)
                if 0 <= y < h:
                    points_on_line.append((x, y))
        
        for y in [0, h-1]:
            if A != 0:
                x = int((-C - B * y) / A)
                if 0 <= x < w:
                    points_on_line.append((x, y))
        
        if len(points_on_line) >= 2:
            # 取距离最远的两个点来画线
            if len(points_on_line) > 2:
                max_distance = 0
                best_pair = (points_on_line[0], points_on_line[1])
                for i in range(len(points_on_line)):
                    for j in range(i+1, len(points_on_line)):
                        dist = np.sqrt((points_on_line[i][0]-points_on_line[j][0])**2 + 
                                     (points_on_line[i][1]-points_on_line[j][1])**2)
                        if dist > max_distance:
                            max_distance = dist
                            best_pair = (points_on_line[i], points_on_line[j])
                points_on_line = [best_pair[0], best_pair[1]]
            
            pt1, pt2 = points_on_line[0], points_on_line[1]
            cv2.line(vis_image, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 绘制垂直径测量线 + 中心点（parallel2的起点）
        if perpendicular_points is not None:
            p1, p2 = perpendicular_points
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                
                # 绘制垂直径线
                cv2.line(vis_image, pt1, pt2, perpendicular_color, 2, cv2.LINE_AA)
                
                # 绘制垂直径端点
                cv2.circle(vis_image, pt1, 4, perpendicular_color, -1)
                cv2.circle(vis_image, pt2, 4, perpendicular_color, -1)
                
                # 绘制中心点（parallel2的穿过点）- 突出显示
                center_x = (pt1[0] + pt2[0]) // 2
                center_y = (pt1[1] + pt2[1]) // 2
                cv2.circle(vis_image, (center_x, center_y), 5, (255, 255, 255), -1)  # 白色中心点
                cv2.circle(vis_image, (center_x, center_y), 7, parallel2_color, 2)   # 紫色边框
        
        # 绘制平行径测量线
        if parallel_points is not None:
            p1, p2 = parallel_points
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                
                cv2.line(vis_image, pt1, pt2, parallel_color, 2, cv2.LINE_AA)
                cv2.circle(vis_image, pt1, 4, parallel_color, -1)
                cv2.circle(vis_image, pt2, 4, parallel_color, -1)
        
        # 绘制parallel2测量线（严格90度垂直、过中心点）
        if parallel2_points is not None:
            p1, p2 = parallel2_points
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                
                # 绘制虚线样式，突出与其他线的区别
                cv2.line(vis_image, pt1, pt2, parallel2_color, 2, cv2.LINE_AA)
                
                # 绘制端点（带边框，更醒目）
                cv2.circle(vis_image, pt1, 4, parallel2_color, -1)
                cv2.circle(vis_image, pt1, 6, (255, 255, 255), 1)
                cv2.circle(vis_image, pt2, 4, parallel2_color, -1)
                cv2.circle(vis_image, pt2, 6, (255, 255, 255), 1)
        
        # 添加图例（清晰说明三条线的关系）
        atrium_name = "Left Atrium" if atrium_id == LA_BLOOD_POOL_ID else "Right Atrium"
        y_offset = 30 if atrium_id == LA_BLOOD_POOL_ID else 90
        
        # 垂直径图例
        cv2.putText(vis_image, f"{atrium_name} - Perpendicular (橙/蓝): {perpendicular_diameter:.1f}mm", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, perpendicular_color, 2)
        # 平行径图例
        cv2.putText(vis_image, f"Parallel (红/青): {parallel_diameter:.1f}mm", 
                   (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, parallel_color, 2)
        # Parallel2图例（强调90度和中心点）
        cv2.putText(vis_image, f"Parallel2 (90度垂直/过中心点): {parallel2_diameter:.1f}mm", 
                   (10, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, parallel2_color, 2)
        
        return vis_image
        
    except Exception as e:
        logging.error(f"绘制心房三个直径测量失败: {e}")
        return vis_image

def visualize_atrial_measurements(mask: np.ndarray, results: Dict[str, Any], visualization: np.ndarray) -> None:
    """
    可视化心房测量结果（显示三个直径，突出parallel2的特性）
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 原始分割
        axes[0].imshow(mask, cmap='tab10')
        axes[0].set_title('Cardiac Segmentation Mask')
        axes[0].axis('off')
        
        # 添加图例
        unique_ids = np.unique(mask)
        legend_labels = {
            LV_BLOOD_POOL_ID: 'LV Blood',
            RV_BLOOD_POOL_ID: 'RV Blood', 
            LA_BLOOD_POOL_ID: 'LA',
            RA_BLOOD_POOL_ID: 'RA'
        }
        
        legend_elements = []
        legend_labels_list = []
        for id_val in unique_ids:
            if id_val in legend_labels and id_val > 0:  # 跳过背景
                color = plt.cm.tab10((id_val % 10) / 10)
                legend_elements.append(plt.Rectangle((0,0),1,1, fc=color))
                legend_labels_list.append(legend_labels[id_val])
        
        if legend_elements:
            axes[0].legend(legend_elements, legend_labels_list,
                          loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        # 2. 测量结果可视化
        if visualization is not None:
            # 确保图像是RGB格式用于matplotlib显示
            if len(visualization.shape) == 3 and visualization.shape[2] == 3:
                display_image = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
            else:
                display_image = visualization
            
            axes[1].imshow(display_image)
            axes[1].set_title('Atrial Diameter Measurements (3 Directions)')
            
            # 显示详细测量结果
            la_data = results.get('left_atrium', {})
            ra_data = results.get('right_atrium', {})
            
            measurement_text = (
                f'Left Atrium:\n'
                f'  Perpendicular: {la_data.get("perpendicular_diameter_mm", 0):.1f}mm\n'
                f'  Parallel: {la_data.get("parallel_diameter_mm", 0):.1f}mm\n'
                f'  Parallel2 (90°垂直/过中心点): {la_data.get("parallel2_diameter_mm", 0):.1f}mm\n\n'
                f'Right Atrium:\n'
                f'  Perpendicular: {ra_data.get("perpendicular_diameter_mm", 0):.1f}mm\n'
                f'  Parallel: {ra_data.get("parallel_diameter_mm", 0):.1f}mm\n'
                f'  Parallel2 (90°垂直/过中心点): {ra_data.get("parallel2_diameter_mm", 0):.1f}mm'
            )
            
            axes[1].text(0.02, 0.98, measurement_text, transform=axes[1].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=10, fontweight='bold')
            
            axes[1].axis('off')
        else:
            axes[1].text(0.5, 0.5, 'Measurement Failed', ha='center', va='center', 
                        transform=axes[1].transAxes, fontsize=14)
            axes[1].set_title('Atrial Diameter Measurements')
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logging.error(f"可视化失败: {e}")

def analyze_cardiac_chambers_with_visualization(mask: np.ndarray, spacing_xy: Tuple[float, float] = (1.0, 1.0)) -> Dict[str, Any]:
    """
    综合分析所有心腔的直径并可视化（包含三个直径）
    """
    print("=== 心房直径测量分析（三个方向）===")
    print(f"掩码形状: {mask.shape}")
    print(f"掩码数据类型: {mask.dtype}")
    print(f"唯一值: {np.unique(mask)}")
    
    # 心房直径测量
    atrial_results, atrial_vis = calculate_atrial_diameters_with_visualization(mask, spacing_xy)
    
    # 可视化结果
    # visualize_atrial_measurements(mask, atrial_results, atrial_vis)
    
    # 打印详细结果
    print("\n=== 心房测量结果 ===")
    la = atrial_results['left_atrium']
    ra = atrial_results['right_atrium']
    print(f"左心房:")
    print(f"  - 垂直径（垂直于房室交界线）: {la['perpendicular_diameter_mm']:.1f}mm")
    print(f"  - 平行径（平行于房室交界线）: {la['parallel_diameter_mm']:.1f}mm")
    print(f"  - Parallel2（垂直于垂直径+过中心点）: {la['parallel2_diameter_mm']:.1f}mm")
    print(f"右心房:")
    print(f"  - 垂直径（垂直于房室交界线）: {ra['perpendicular_diameter_mm']:.1f}mm")
    print(f"  - 平行径（平行于房室交界线）: {ra['parallel_diameter_mm']:.1f}mm")
    print(f"  - Parallel2（垂直于垂直径+过中心点）: {ra['parallel2_diameter_mm']:.1f}mm")
    
    return atrial_results


def draw_atrial_measurement(vis_image: np.ndarray, av_junction_line: Tuple[float, float, float], 
                          junction_points: Optional[np.ndarray], diameter_points: Optional[Tuple[np.ndarray, np.ndarray]], 
                          diameter: float, atrium_id: int, color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    在图像上绘制心房测量结果
    """
    try:
        # 确保图像是BGR格式且连续
        if vis_image is None:
            return None
            
        vis_image = np.ascontiguousarray(vis_image)
        
        if color is None:
            # 根据心房ID选择颜色
            color = (0, 255, 255) if atrium_id == LA_BLOOD_POOL_ID else (255, 255, 0)
        
        A, B, C = av_junction_line
        
        # 绘制房室交界点
        if junction_points is not None:
            for point in junction_points[:50]:  # 只绘制前50个点避免太密集
                x, y = int(point[0]), int(point[1])
                if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                    cv2.circle(vis_image, (x, y), 2, (255, 255, 255), -1)
        
        # 绘制房室交界线
        h, w = vis_image.shape[:2]
        points_on_line = []
        
        # 在图像边界上找到直线的两个点
        for x in [0, w-1]:
            if B != 0:
                y = int((-C - A * x) / B)
                if 0 <= y < h:
                    points_on_line.append((x, y))
        
        for y in [0, h-1]:
            if A != 0:
                x = int((-C - B * y) / A)
                if 0 <= x < w:
                    points_on_line.append((x, y))
        
        if len(points_on_line) >= 2:
            # 取距离最远的两个点来画线
            if len(points_on_line) > 2:
                # 计算所有点对之间的距离，选择最远的
                max_distance = 0
                best_pair = (points_on_line[0], points_on_line[1])
                for i in range(len(points_on_line)):
                    for j in range(i+1, len(points_on_line)):
                        dist = np.sqrt((points_on_line[i][0]-points_on_line[j][0])**2 + 
                                     (points_on_line[i][1]-points_on_line[j][1])**2)
                        if dist > max_distance:
                            max_distance = dist
                            best_pair = (points_on_line[i], points_on_line[j])
                points_on_line = [best_pair[0], best_pair[1]]
            
            pt1, pt2 = points_on_line[0], points_on_line[1]
            cv2.line(vis_image, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 绘制横径测量线
        if diameter_points is not None:
            p1, p2 = diameter_points
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            
            # 检查点是否在图像范围内
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                
                # 绘制测量线
                cv2.line(vis_image, pt1, pt2, color, 3, cv2.LINE_AA)
                
                # 绘制端点
                cv2.circle(vis_image, pt1, 5, color, -1)
                cv2.circle(vis_image, pt2, 5, color, -1)
        
        # 添加图例
        atrium_name = "Left Atrium" if atrium_id == LA_BLOOD_POOL_ID else "Right Atrium"
        legend_text = f"{atrium_name}: {diameter:.1f}mm"
        
        # 确保文本位置在图像范围内
        y_position = 30 if atrium_id == LA_BLOOD_POOL_ID else 60
        if y_position < vis_image.shape[0]:
            cv2.putText(vis_image, legend_text, (10, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return vis_image
        
    except Exception as e:
        logging.error(f"绘制心房测量失败: {e}")
        return vis_image

def calculate_atrial_diameters_with_visualization(mask: np.ndarray, spacing_xy: Tuple[float, float]) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    计算左右心房的三个直径并返回可视化：
    - perpendicular_diameter: 垂直于房室交界线
    - parallel_diameter: 平行于房室交界线
    - parallel2_diameter: 垂直于perpendicular、穿过其中心点
    Args:
        mask: 分割掩码
        spacing_xy: XY方向的像素间距
    Returns:
        dict: 包含左右心房三个直径的字典
        vis_image: 可视化图像
    """
    try:
        # 创建基础可视化图像
        vis_image = create_base_visualization(mask)
        
        # 计算左心房直径（新增parallel2）
        la_vis_image, la_perpendicular_diameter, la_perpendicular_points, la_parallel_diameter, la_parallel_points, la_parallel2_diameter, la_parallel2_points = calculate_atrial_three_diameters(
            mask, spacing_xy, LA_BLOOD_POOL_ID, LV_BLOOD_POOL_ID, vis_image)
        
        # 计算右心房直径（新增parallel2）
        ra_vis_image, ra_perpendicular_diameter, ra_perpendicular_points, ra_parallel_diameter, ra_parallel_points, ra_parallel2_diameter, ra_parallel2_points = calculate_atrial_three_diameters(
            mask, spacing_xy, RA_BLOOD_POOL_ID, RV_BLOOD_POOL_ID, la_vis_image)
        
        results = {
            'left_atrium': {
                'perpendicular_diameter_mm': float(la_perpendicular_diameter) if la_perpendicular_diameter > 0 else 0.0,
                'parallel_diameter_mm': float(la_parallel_diameter) if la_parallel_diameter > 0 else 0.0,
                'parallel2_diameter_mm': float(la_parallel2_diameter) if la_parallel2_diameter > 0 else 0.0  # 新增
            },
            'right_atrium': {
                'perpendicular_diameter_mm': float(ra_perpendicular_diameter) if ra_perpendicular_diameter > 0 else 0.0,
                'parallel_diameter_mm': float(ra_parallel_diameter) if ra_parallel_diameter > 0 else 0.0,
                'parallel2_diameter_mm': float(ra_parallel2_diameter) if ra_parallel2_diameter > 0 else 0.0  # 新增
            }
        }
        
        # 添加标题
        cv2.putText(ra_vis_image, "Atrial Diameter Measurements (3 Directions)", 
                   (10, ra_vis_image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        logging.info(f"左心房 - 垂直径: {la_perpendicular_diameter:.1f}mm, 平行径: {la_parallel_diameter:.1f}mm, Parallel2: {la_parallel2_diameter:.1f}mm")
        logging.info(f"右心房 - 垂直径: {ra_perpendicular_diameter:.1f}mm, 平行径: {ra_parallel_diameter:.1f}mm, Parallel2: {ra_parallel2_diameter:.1f}mm")
        
        return results, ra_vis_image
        
    except Exception as e:
        logging.error(f"计算心房直径失败: {e}")
        default_results = {
            'left_atrium': {
                'perpendicular_diameter_mm': 0.0,
                'parallel_diameter_mm': 0.0,
                'parallel2_diameter_mm': 0.0  # 新增
            },
            'right_atrium': {
                'perpendicular_diameter_mm': 0.0,
                'parallel_diameter_mm': 0.0,
                'parallel2_diameter_mm': 0.0  # 新增
            }
        }
        return default_results, create_base_visualization(mask)

def calculate_atrial_three_diameters(mask: np.ndarray, spacing_xy: Tuple[float, float], 
                                   atrium_id: int, ventricle_id: int, 
                                   visualization_image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, Optional[Tuple[np.ndarray, np.ndarray]], float, Optional[Tuple[np.ndarray, np.ndarray]], float, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    计算心房三个直径：
    - perpendicular_diameter: 垂直于房室交界线
    - parallel_diameter: 平行于房室交界线
    - parallel2_diameter: 垂直于perpendicular、穿过其中心点
    Returns:
        vis_image: 可视化图像
        perpendicular_diameter: 垂直径长度
        perpendicular_points: 垂直径端点
        parallel_diameter: 平行径长度
        parallel_points: 平行径端点
        parallel2_diameter: parallel2直径长度（新增）
        parallel2_points: parallel2直径端点（新增）
    """
    try:
        # 获取心房和心室掩码
        atrium_mask = (mask == atrium_id).astype(np.uint8)
        ventricle_mask = (mask == ventricle_id).astype(np.uint8)
        
        # 创建可视化图像
        if visualization_image is None:
            vis_image = create_base_visualization(mask)
        else:
            vis_image = visualization_image.copy()
        
        if np.sum(atrium_mask) == 0 or np.sum(ventricle_mask) == 0:
            logging.warning(f"心房或心室掩码为空: 心房{atrium_id}, 心室{ventricle_id}")
            return vis_image, -1.0, None, -1.0, None, -1.0, None
        
        # 找到房室交界线
        av_junction_line, junction_points = find_atrioventricular_junction(atrium_mask, ventricle_mask)
        if av_junction_line is None:
            logging.warning("无法找到房室交界线")
            return vis_image, -1.0, None, -1.0, None, -1.0, None
        
        # 计算垂直于交界线的直径
        perpendicular_diameter, perpendicular_points = calculate_diameter_perpendicular_to_line(
            atrium_mask, av_junction_line, spacing_xy)
        
        # 计算平行于交界线的直径
        parallel_diameter, parallel_points = calculate_diameter_parallel_to_line(
            atrium_mask, av_junction_line, spacing_xy)
        
        # 计算parallel2直径（新增）
        parallel2_diameter = -1.0
        parallel2_points = None
        if perpendicular_points is not None:
            parallel2_diameter, parallel2_points = calculate_parallel2_diameter(
                atrium_mask, perpendicular_points, spacing_xy)
        
        # 在可视化图像上绘制三个直径（修改后的绘制函数）
        vis_image = draw_atrial_three_measurements(vis_image, av_junction_line, junction_points,
                                                 perpendicular_points, perpendicular_diameter,
                                                 parallel_points, parallel_diameter,
                                                 parallel2_points, parallel2_diameter,
                                                 atrium_id)
        
        return vis_image, perpendicular_diameter, perpendicular_points, parallel_diameter, parallel_points, parallel2_diameter, parallel2_points
        
    except Exception as e:
        logging.error(f"计算心房三个直径失败: {e}")
        return visualization_image if visualization_image is not None else create_base_visualization(mask), -1.0, None, -1.0, None, -1.0, None

def draw_atrial_three_measurements(vis_image: np.ndarray, av_junction_line: Tuple[float, float, float], 
                                 junction_points: Optional[np.ndarray], 
                                 perpendicular_points: Optional[Tuple[np.ndarray, np.ndarray]],
                                 perpendicular_diameter: float,
                                 parallel_points: Optional[Tuple[np.ndarray, np.ndarray]],
                                 parallel_diameter: float,
                                 parallel2_points: Optional[Tuple[np.ndarray, np.ndarray]],  # 新增
                                 parallel2_diameter: float,  # 新增
                                 atrium_id: int) -> np.ndarray:
    """
    在图像上绘制心房三个直径的测量结果（新增parallel2）
    """
    try:
        # 确保图像是BGR格式且连续
        if vis_image is None:
            return None
            
        vis_image = np.ascontiguousarray(vis_image)
        
        # 根据心房ID选择颜色
        if atrium_id == LA_BLOOD_POOL_ID:
            perpendicular_color = (0, 200, 255)  # 橙色 - 左心房垂直径（垂直于交界线）
            parallel_color = (0, 150, 255)       # 红色 - 左心房平行径（平行于交界线）
            parallel2_color = (255, 0, 255)      # 紫色 - 左心房parallel2（垂直于perpendicular）
        else:
            perpendicular_color = (255, 200, 0)  # 蓝色 - 右心房垂直径（垂直于交界线）
            parallel_color = (255, 150, 0)       # 青色 - 右心房平行径（平行于交界线）
            parallel2_color = (0, 255, 150)      # 青绿色 - 右心房parallel2（垂直于perpendicular）
        
        A, B, C = av_junction_line
        h, w = vis_image.shape[:2]
        
        # 绘制房室交界点
        if junction_points is not None:
            for point in junction_points[:50]:  # 只绘制前50个点避免太密集
                x, y = int(point[0]), int(point[1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(vis_image, (x, y), 2, (255, 255, 255), -1)
        
        # 绘制房室交界线
        points_on_line = []
        for x in [0, w-1]:
            if B != 0:
                y = int((-C - A * x) / B)
                if 0 <= y < h:
                    points_on_line.append((x, y))
        
        for y in [0, h-1]:
            if A != 0:
                x = int((-C - B * y) / A)
                if 0 <= x < w:
                    points_on_line.append((x, y))
        
        if len(points_on_line) >= 2:
            # 取距离最远的两个点来画线
            if len(points_on_line) > 2:
                max_distance = 0
                best_pair = (points_on_line[0], points_on_line[1])
                for i in range(len(points_on_line)):
                    for j in range(i+1, len(points_on_line)):
                        dist = np.sqrt((points_on_line[i][0]-points_on_line[j][0])**2 + 
                                     (points_on_line[i][1]-points_on_line[j][1])**2)
                        if dist > max_distance:
                            max_distance = dist
                            best_pair = (points_on_line[i], points_on_line[j])
                points_on_line = [best_pair[0], best_pair[1]]
            
            pt1, pt2 = points_on_line[0], points_on_line[1]
            cv2.line(vis_image, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 绘制垂直径测量线（垂直于交界线）
        if perpendicular_points is not None:
            p1, p2 = perpendicular_points
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                
                # 绘制测量线
                cv2.line(vis_image, pt1, pt2, perpendicular_color, 2, cv2.LINE_AA)
                
                # 绘制端点
                cv2.circle(vis_image, pt1, 4, perpendicular_color, -1)
                cv2.circle(vis_image, pt2, 4, perpendicular_color, -1)
                
                # 绘制中心点（用于parallel2的参考）
                center_x = (pt1[0] + pt2[0]) // 2
                center_y = (pt1[1] + pt2[1]) // 2
                cv2.circle(vis_image, (center_x, center_y), 3, (255, 255, 255), -1)  # 白色中心点
        
        # 绘制平行径测量线（平行于交界线）
        if parallel_points is not None:
            p1, p2 = parallel_points
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                
                # 绘制测量线
                cv2.line(vis_image, pt1, pt2, parallel_color, 2, cv2.LINE_AA)
                
                # 绘制端点
                cv2.circle(vis_image, pt1, 4, parallel_color, -1)
                cv2.circle(vis_image, pt2, 4, parallel_color, -1)
        
        # 绘制parallel2测量线（垂直于perpendicular、穿过中心点）- 新增
        if parallel2_points is not None:
            p1, p2 = parallel2_points
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                
                # 绘制测量线（虚线样式，与其他两条线区分）
                cv2.line(vis_image, pt1, pt2, parallel2_color, 2, cv2.LINE_AA)
                
                # 绘制端点
                cv2.circle(vis_image, pt1, 4, parallel2_color, -1)
                cv2.circle(vis_image, pt2, 4, parallel2_color, -1)
        
        # 添加图例
        atrium_name = "Left Atrium" if atrium_id == LA_BLOOD_POOL_ID else "Right Atrium"
        
        # 确定文本位置
        if atrium_id == LA_BLOOD_POOL_ID:
            y_pos1 = 30
            y_pos2 = 60
        else:
            y_pos1 = 90
            y_pos2 = 120
        
        # 绘制三条直径的图例
        legend_text1 = f"{atrium_name}: Perp={perpendicular_diameter:.1f}mm (橙/蓝)"
        legend_text2 = f"Parallel={parallel_diameter:.1f}mm (红/青)"
        legend_text3 = f"Parallel2={parallel2_diameter:.1f}mm (紫/青绿)"
        
        # if y_pos1 < h:
        #     cv2.putText(vis_image, legend_text1, (10, y_pos1),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, perpendicular_color, 2)
        # if y_pos2 < h:
        #     cv2.putText(vis_image, legend_text2, (10, y_pos1 + 20),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, parallel_color, 2)
        #     cv2.putText(vis_image, legend_text3, (10, y_pos1 + 40),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, parallel2_color, 2)
        
        return vis_image
        
    except Exception as e:
        logging.error(f"绘制心房三个直径测量失败: {e}")
        return vis_image

def visualize_atrial_measurements(mask: np.ndarray, results: Dict[str, Any], visualization: np.ndarray) -> None:
    """
    可视化心房测量结果（更新为显示三个直径）
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 原始分割
        axes[0].imshow(mask, cmap='tab10')
        axes[0].set_title('Cardiac Segmentation Mask')
        axes[0].axis('off')
        
        # 添加图例
        unique_ids = np.unique(mask)
        legend_labels = {
            LV_BLOOD_POOL_ID: 'LV Blood',
            RV_BLOOD_POOL_ID: 'RV Blood', 
            LA_BLOOD_POOL_ID: 'LA',
            RA_BLOOD_POOL_ID: 'RA'
        }
        
        legend_elements = []
        legend_labels_list = []
        for id_val in unique_ids:
            if id_val in legend_labels and id_val > 0:  # 跳过背景
                color = plt.cm.tab10((id_val % 10) / 10)
                legend_elements.append(plt.Rectangle((0,0),1,1, fc=color))
                legend_labels_list.append(legend_labels[id_val])
        
        if legend_elements:
            axes[0].legend(legend_elements, legend_labels_list,
                          loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        # 2. 测量结果可视化
        if visualization is not None:
            # 确保图像是RGB格式用于matplotlib显示
            if len(visualization.shape) == 3 and visualization.shape[2] == 3:
                # OpenCV是BGR，matplotlib需要RGB
                display_image = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
            else:
                display_image = visualization
            
            axes[1].imshow(display_image)
            axes[1].set_title('Atrial Diameter Measurements (3 Directions)')
            
            # 在图上添加测量结果文本（包含parallel2）
            la_perpendicular = results.get('left_atrium', {}).get('perpendicular_diameter_mm', 0)
            la_parallel = results.get('left_atrium', {}).get('parallel_diameter_mm', 0)
            la_parallel2 = results.get('left_atrium', {}).get('parallel2_diameter_mm', 0)  # 新增
            ra_perpendicular = results.get('right_atrium', {}).get('perpendicular_diameter_mm', 0)
            ra_parallel = results.get('right_atrium', {}).get('parallel_diameter_mm', 0)
            ra_parallel2 = results.get('right_atrium', {}).get('parallel2_diameter_mm', 0)  # 新增
            
            measurement_text = f'Left Atrium:\n  Perpendicular: {la_perpendicular:.1f}mm\n  Parallel: {la_parallel:.1f}mm\n  Parallel2: {la_parallel2:.1f}mm\n\nRight Atrium:\n  Perpendicular: {ra_perpendicular:.1f}mm\n  Parallel: {ra_parallel:.1f}mm\n  Parallel2: {ra_parallel2:.1f}mm'
            axes[1].text(0.02, 0.98, measurement_text, transform=axes[1].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=10, fontweight='bold')
            
            axes[1].axis('off')
        else:
            axes[1].text(0.5, 0.5, 'Measurement Failed', ha='center', va='center', 
                        transform=axes[1].transAxes, fontsize=14)
            axes[1].set_title('Atrial Diameter Measurements')
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logging.error(f"可视化失败: {e}")

def analyze_cardiac_chambers_with_visualization(mask: np.ndarray, spacing_xy: Tuple[float, float] = (1.0, 1.0)) -> Dict[str, Any]:
    """
    综合分析所有心腔的直径并可视化（包含三个直径）
    """
    print("=== 心房直径测量分析（三个方向）===")
    print(f"掩码形状: {mask.shape}")
    print(f"掩码数据类型: {mask.dtype}")
    print(f"唯一值: {np.unique(mask)}")
    
    # 心房直径测量（更新为三个直径）
    atrial_results, atrial_vis = calculate_atrial_diameters_with_visualization(mask, spacing_xy)
    
    # 可视化结果
    # visualize_atrial_measurements(mask, atrial_results, atrial_vis)
    
    # 打印结果（包含parallel2）
    print("\n=== 心房测量结果 ===")
    print(f"左心房 - 垂直径: {atrial_results['left_atrium']['perpendicular_diameter_mm']:.1f}mm")
    # print(f"左心房 - 平行径: {atrial_results['left_atrium']['parallel_diameter_mm']:.1f}mm")
    print(f"左心房 - Parallel2: {atrial_results['left_atrium']['parallel2_diameter_mm']:.1f}mm")
    print(f"右心房 - 垂直径: {atrial_results['right_atrium']['perpendicular_diameter_mm']:.1f}mm")
    # print(f"右心房 - 平行径: {atrial_results['right_atrium']['parallel_diameter_mm']:.1f}mm")
    print(f"右心房 - Parallel2: {atrial_results['right_atrium']['parallel2_diameter_mm']:.1f}mm")
    
    return atrial_results
# ---------------------------------

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
        
        # 应用跳过切片逻辑，选择指定数量的切片
        effective_slice_indices = slice_indices[SKIP_HEAD_SLICES_PER_BLOCK:]
        
        if SKIP_TAIL_SLICES_PER_BLOCK > 0 and len(effective_slice_indices) > SKIP_TAIL_SLICES_PER_BLOCK:
            effective_slice_indices = effective_slice_indices[:-SKIP_TAIL_SLICES_PER_BLOCK]
        
        # 从有效切片中选择最多指定数量的切片
        if len(effective_slice_indices) > MAX_SLICES_PER_BLOCK:
            effective_slice_indices = effective_slice_indices[:MAX_SLICES_PER_BLOCK]
        
        if len(effective_slice_indices) == 0:
            logging.warning(f"    块 {block_idx} 跳过切片后没有有效切片，跳过")
            blocks.append(None)
            continue
            
        # 获取实际的切片数据索引
        effective_indices_in_data = effective_slice_indices
        
        if len(effective_indices_in_data) == 0:
            logging.warning(f"    块 {block_idx} 有效切片索引为空，跳过")
            blocks.append(None)
            continue
            
        block_data = data[:, :, effective_indices_in_data]
        blocks.append(block_data)
        
        logging.info(f"      块 {block_idx}: 原始切片 {slice_indices[:5]}{'...' if len(slice_indices) > 5 else ''}")
        logging.info(f"              有效切片 {effective_indices_in_data}, 最终形状: {block_data.shape}")
    
    logging.info(f"    成功创建 {len(blocks)} 个块（包含 {len([b for b in blocks if b is not None])} 个有效块）")
    return blocks, original_blocks

def compute_volume_only_with_original_spacing(blk, original_spacing, BLOOD_POOL_ID):
    """使用原始spacing计算LV体积用于找ED块"""
    total_lv = 0.0
    for si in range(blk.shape[2]):
        sl = blk[:, :, si]
        lv_vol = np.sum(sl == BLOOD_POOL_ID) * original_spacing[0] * original_spacing[1] * original_spacing[2] / 1000.0
        total_lv += lv_vol
    return total_lv

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

def calculate_cine_4ch_metrics(cine_4ch_mask_path, slice_num):
    """
    计算4ch图像的关键指标（修复空间校准问题）
    """
    try:
        # 加载数据
        pred_img = nib.load(cine_4ch_mask_path)
        pred_data = np.round(pred_img.get_fdata()).astype(np.int16)
        pred_data = np.flip(pred_data, axis=1)
        # 修复1: 获取原始spacing
        original_spacing = pred_img.header.get_zooms()  # 获取(x, y, z) spacing
        logging.info(f"原始图像spacing: {original_spacing}, 形状: {pred_data.shape}")
        BLOCK_SIZES = slice_num
        # if pred_data.shape[2] % 30 == 0:
        #     BLOCK_SIZES = 30
        # elif pred_data.shape[2] % 25 == 0:
        #     BLOCK_SIZES = 25
        # else:
        #     BLOCK_SIZES = 30
        # 创建块
        blocks, original_blocks = create_3d_blocks(pred_data, BLOCK_SIZES)
        
        if blocks is None or original_blocks is None:
            return None
            
        # 修复2: 使用原始spacing计算容积来识别ED块
        block_volumes_lv = []
        block_volumes_la = []
        for i, block in enumerate(blocks):
            if block is None:
                continue
                
            # 使用原始spacing计算LV/LA容积
            lv_vol = compute_volume_only_with_original_spacing(block, original_spacing, LV_BLOOD_POOL_ID)
            block_volumes_lv.append((i, lv_vol))
            la_vol = compute_volume_only_with_original_spacing(block, original_spacing, LA_BLOOD_POOL_ID)
            block_volumes_la.append((i, la_vol))
        
        if not block_volumes_lv:
            return None

        if not block_volumes_la:
            return None
            
        # 找到ED块（最大体积）
        block_volumes_lv.sort(key=lambda x: x[1], reverse=True)
        ed_idx_lv = block_volumes_lv[0][0]

        block_volumes_la.sort(key=lambda x: x[1], reverse=True)
        ed_idx_la = block_volumes_la[0][0]
        
        logging.info(f"ED块索引: {ed_idx_lv}, LV容积: {block_volumes_lv[0][1]:.2f}ml")
        logging.info(f"ED块索引: {ed_idx_la}, LA容积: {block_volumes_la[0][1]:.2f}ml")
        
        # 获取ED原始块
        ed_block_original = original_blocks[ed_idx_lv]
        ed_block_original_la = original_blocks[ed_idx_la]
        
        # 初始化结果字典
        result = {}
        
        # 确保原始块中有足够的切片
        if ed_block_original_la is not None and ed_block_original_la.shape[2] > TARGET_SLICE_INDEX:
            # 获取ED块的目标切片（第2张，索引1）
            ed_target_slice = ed_block_original_la[:, :, TARGET_SLICE_INDEX]
            
            # 1. 计算心房长短径
            atrial_results = analyze_cardiac_chambers_with_visualization(
                ed_target_slice, original_spacing[:2]
            )
            result['LA_ED_Long_Diameter'] = atrial_results['left_atrium']['parallel2_diameter_mm'] if atrial_results['left_atrium']['parallel2_diameter_mm'] > 0 else None
            result['RA_ED_Long_Diameter'] = atrial_results['right_atrium']['parallel2_diameter_mm'] if atrial_results['right_atrium']['parallel2_diameter_mm'] > 0 else None
        # 2. 计算右心室壁厚度（三段）
        if ed_block_original is not None and ed_block_original.shape[2] > TARGET_SLICE_INDEX:
            ed_target_slice = ed_block_original[:, :, TARGET_SLICE_INDEX]
            rv_wall_thickness, _ = calculate_rv_wall_thickness_segmented(
                ed_target_slice, original_spacing[:2], RV_WALL_THICKNESS_DIVISIONS
            )
            
            for div_id, div_stats in rv_wall_thickness.items():
                result[f'ED_RV_Wall_Thickness_Div_{div_id}'] = div_stats['thickness_mm']
        
        # 4. 计算左心室心尖厚度（使用第2张切片，索引1）
        if ed_block_original is not None and ed_block_original.shape[2] > APEX_SLICE_INDEX:
            ed_apex_slice = ed_block_original[:, :, APEX_SLICE_INDEX]
            apex_thickness_max, apex_thickness_mean, apex_thickness_min, _ = calculate_apex_thickness(ed_apex_slice, original_spacing[:2])
            result['ED_LV_Apex_Thickness_max'] = apex_thickness_max
            result['ED_LV_Apex_Thickness_mean'] = apex_thickness_mean
            result['ED_LV_Apex_Thickness_min'] = apex_thickness_min
        
        return result
        
    except Exception as e:
        logging.error(f"计算cine 4ch图像的关键指标失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 设置日志级别以便查看详细信息
    logging.basicConfig(level=logging.INFO)
    
    cine_4ch_mask_path = "/Users/zhanglantian/Documents/BAAI/Code/code/measure/data_4ch/00101_pred.nii.gz"
    metrics = calculate_cine_4ch_metrics(cine_4ch_mask_path)
    print("计算完成，结果:")
    print(metrics)