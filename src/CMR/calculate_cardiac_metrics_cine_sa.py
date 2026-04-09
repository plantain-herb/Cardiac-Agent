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

                
      
CROP_MARGIN = 5                     
from scipy.stats import mode 
        
BACKGROUND_ID = 0         
LV_MYOCARDIUM_ID = 1         
LV_BLOOD_POOL_ID = 2         
RV_BLOOD_POOL_ID = 3         
RV_MYOCARDIUM_ID = 4         
MYOCARDIUM_DENSITY = 1.05
ASSUMED_HEART_RATE = 70

      
# BLOCK_SIZES = [30]

      
MAX_SLICES_PER_BLOCK = 8                    
SKIP_HEAD_SLICES_PER_BLOCK = 1              
SKIP_TAIL_SLICES_PER_BLOCK = 2              
TARGET_SLICE_INDEX = 3                           

        
SEGMENTATION_SLICE_INDICES = [1, 3, 5]                       
SEGMENTATION_DIVISIONS = [4, 6, 6]                       

                      
SEGMENT_NAMES = {
                                    
    1: {
        1: {'id': 13, 'name': 'Apical_anterior'},
        2: {'id': 14, 'name': 'Apical_lateral'}, 
        3: {'id': 15, 'name': 'Apical_inferior'},
        4: {'id': 16, 'name': 'Apical_septal'}
    },
    2: {
        1: {'id': 13, 'name': 'Apical_anterior'},
        2: {'id': 14, 'name': 'Apical_lateral'}, 
        3: {'id': 15, 'name': 'Apical_inferior'},
        4: {'id': 16, 'name': 'Apical_septal'}
    },
    3: {
        1: {'id': 13, 'name': 'Apical_anterior'},
        2: {'id': 14, 'name': 'Apical_lateral'}, 
        3: {'id': 15, 'name': 'Apical_inferior'},
        4: {'id': 16, 'name': 'Apical_septal'}
    },
                                  
    4: {
        1: {'id': 7, 'name': 'Mid_anteroseptal'},
        2: {'id': 8, 'name': 'Mid_anterior'},
        3: {'id': 9, 'name': 'Mid_lateral'},
        4: {'id': 10, 'name': 'Mid_posterior'},
        5: {'id': 11, 'name': 'Mid_inferior'},
        6: {'id': 12, 'name': 'Mid_inferoseptal'}
    },
    5: {
        1: {'id': 7, 'name': 'Mid_anteroseptal'},
        2: {'id': 8, 'name': 'Mid_anterior'},
        3: {'id': 9, 'name': 'Mid_lateral'},
        4: {'id': 10, 'name': 'Mid_posterior'},
        5: {'id': 11, 'name': 'Mid_inferior'},
        6: {'id': 12, 'name': 'Mid_inferoseptal'}
    },
                                  
    6: {
        1: {'id': 1, 'name': 'Basal_anteroseptal'},
        2: {'id': 2, 'name': 'Basal_anterior'},
        3: {'id': 3, 'name': 'Basal_lateral'},
        4: {'id': 4, 'name': 'Basal_posterior'},
        5: {'id': 5, 'name': 'Basal_inferior'},
        6: {'id': 6, 'name': 'Basal_inferoseptal'}
    },
    7: {
        1: {'id': 1, 'name': 'Basal_anteroseptal'},
        2: {'id': 2, 'name': 'Basal_anterior'},
        3: {'id': 3, 'name': 'Basal_lateral'},
        4: {'id': 4, 'name': 'Basal_posterior'},
        5: {'id': 5, 'name': 'Basal_inferior'},
        6: {'id': 6, 'name': 'Basal_inferoseptal'}
    },
    8: {
        1: {'id': 1, 'name': 'Basal_anteroseptal'},
        2: {'id': 2, 'name': 'Basal_anterior'},
        3: {'id': 3, 'name': 'Basal_lateral'},
        4: {'id': 4, 'name': 'Basal_posterior'},
        5: {'id': 5, 'name': 'Basal_inferior'},
        6: {'id': 6, 'name': 'Basal_inferoseptal'}
    },
    9: {
        1: {'id': 1, 'name': 'Basal_anteroseptal'},
        2: {'id': 2, 'name': 'Basal_anterior'},
        3: {'id': 3, 'name': 'Basal_lateral'},
        4: {'id': 4, 'name': 'Basal_posterior'},
        5: {'id': 5, 'name': 'Basal_inferior'},
        6: {'id': 6, 'name': 'Basal_inferoseptal'}
    },
    10: {
        1: {'id': 1, 'name': 'Basal_anteroseptal'},
        2: {'id': 2, 'name': 'Basal_anterior'},
        3: {'id': 3, 'name': 'Basal_lateral'},
        4: {'id': 4, 'name': 'Basal_posterior'},
        5: {'id': 5, 'name': 'Basal_inferior'},
        6: {'id': 6, 'name': 'Basal_inferoseptal'}
    },
    11: {
        1: {'id': 1, 'name': 'Basal_anteroseptal'},
        2: {'id': 2, 'name': 'Basal_anterior'},
        3: {'id': 3, 'name': 'Basal_lateral'},
        4: {'id': 4, 'name': 'Basal_posterior'},
        5: {'id': 5, 'name': 'Basal_inferior'},
        6: {'id': 6, 'name': 'Basal_inferoseptal'}
    }
}

warnings.filterwarnings('ignore')

                                   
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
        logging.error(f"getlong-axis diameterfailed: {e}")
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


                                                                                                                
def test_thickness_calculation(ed_slice, original_spacing):
    """fixedthicknesscompute - radialmeasure"""
    global LV_BLOOD_POOL_ID, LV_MYOCARDIUM_ID
    
    LV_BLOOD_POOL_ID = 2         
    LV_MYOCARDIUM_ID = 1         
    
                  
    thickness_max, thickness_mean, thickness_min, thickness_map, message, _ = calculate_thickness_radial_accurate(ed_slice, original_spacing)
    
    
    return thickness_max, thickness_mean, thickness_min, thickness_map, message

def calculate_thickness_radial_accurate(mask, spacing_xy, num_angles=36):
    """radialthicknessmeasuremethod thickness myocardium"""
    try:
        blood_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        myo_mask = (mask == LV_MYOCARDIUM_ID).astype(np.uint8)
        
                
        blood_points = np.where(blood_mask > 0)
        if len(blood_points[0]) == 0:
            return 8.0, 8.0, 8.0, None, "not foundblood pool", []
        
        center_y = np.mean(blood_points[0])
        center_x = np.mean(blood_points[1])
        center = np.array([center_x, center_y])
        
               
        measurement_lines = []
        thickness_values = []
        
                       
        thickness_map = np.zeros_like(mask, dtype=np.float32)
        
                  
        for angle_idx in range(num_angles):
            angle = angle_idx * (360 / num_angles)
            rad = np.radians(angle)
            direction = np.array([np.cos(rad), np.sin(rad)])
            
                              
            endo_point = find_boundary_along_ray_accurate(center, direction, blood_mask)
            if endo_point is None:
                continue
            
                                   
            epi_point = find_boundary_along_ray_accurate(endo_point, direction, myo_mask)
            if epi_point is None:
                continue
            
                    
            dx = (epi_point[0] - endo_point[0]) * spacing_xy[0]
            dy = (epi_point[1] - endo_point[1]) * spacing_xy[1]
            thickness = np.sqrt(dx*dx + dy*dy)
            if 1 <= thickness <= 40.0:           
                measurement_lines.append({
                    'angle': angle,
                    'endo_point': endo_point,
                    'epi_point': epi_point,
                    'thickness': thickness,
                    'center': center
                })
                thickness_values.append(thickness)
                
                                    
                fill_thickness_line(thickness_map, endo_point, epi_point, thickness, spacing_xy)
        
        if not thickness_values:
            return 8.0, 8.0, 8.0, None, None, []
        q1 = np.percentile(thickness_values, 25)
        q3 = np.percentile(thickness_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr                       
        upper_bound = q3 + 1.5 * iqr      
        filtered_thickness = [t for t in thickness_values if lower_bound <= t <= upper_bound]
        if len(filtered_thickness) < 3:                  
            filtered_thickness = thickness_values.copy()

        if len(measurement_lines) >= 5:              
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

                                                                 
        #     max_t = np.max(filtered_thickness)
                                                    
                                                                                     
        #     filtered_thickness = [t for t in filtered_thickness if t >= max_t * 0.75]

                                     
        if len(filtered_thickness) < 3:
                                                   
                                              
            if 'valid_ts' in locals() and len(valid_ts) >= 3:
                filtered_thickness = valid_ts
            elif len([t for t in thickness_values if lower_bound <= t <= upper_bound]) >= 3:
                filtered_thickness = [t for t in thickness_values if lower_bound <= t <= upper_bound]
            else:
                                           
                filtered_thickness = thickness_values.copy()
                              
        if len(filtered_thickness) >= 1:
            if np.min(filtered_thickness) > 2 + 2 * spacing_xy[0] * spacing_xy[1]:
                max_thickness = np.max(filtered_thickness) - 2 * spacing_xy[0] * spacing_xy[1]             
                mean_thickness = np.mean(filtered_thickness) - 2 * spacing_xy[0] * spacing_xy[1] 
                min_thickness = np.min(filtered_thickness) - 2 * spacing_xy[0] * spacing_xy[1] 
        else:
            max_thickness = None
            mean_thickness = None
            min_thickness = None

                               
                                        
        # if max_thickness > 5:
                                                                  
        
        message = f"radialmeasure: {len(thickness_values)} direction, thickness {max_thickness:.2f}mm"
        
        return float(max_thickness), float(mean_thickness), float(min_thickness), thickness_map, message, measurement_lines
        
    except Exception as e:
        return 8.0, 8.0, 8.0, None, f"radialmeasurefailed: {str(e)}", []

def find_boundary_along_ray_accurate(start_point, direction, mask, max_steps=100):
    """boundary - direction boundary"""
    step_size = 1.0
    current_point = start_point.copy()
    
                   
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
        
                   
        if was_inside and not is_inside:
                                       
            boundary_point = current_point - direction * (step_size / 2)
            return boundary_point
        
        was_inside = is_inside
    
    return None

def fill_thickness_line(thickness_map, start_point, end_point, thickness, spacing_xy):
    """thickness"""
             
    length_pixels = np.linalg.norm(end_point - start_point)
    num_points = max(2, int(length_pixels))
    
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        point = start_point + t * (end_point - start_point)
        x, y = int(round(point[0])), int(round(point[1]))
        
        if 0 <= y < thickness_map.shape[0] and 0 <= x < thickness_map.shape[1]:
            thickness_map[y, x] = thickness

def calculate_myocardium_thickness_simple(mask, spacing_xy):
    """thicknesscomputemethod - radialmeasure method"""
    try:
        thickness_max, thickness_mean, thickness_min, thickness_map, message, _ = calculate_thickness_radial_accurate(mask, spacing_xy)
        return thickness_max, thickness_mean, thickness_min, thickness_map, message
    except Exception as e:
        return 0.0, 0.0, 0.0, None, f"computefailed: {str(e)}"
# --------------------------------------------------------------------------------------------------------------------
def create_slice_segmentation(mask, num_divisions=6, start_angle_degrees=0):
    """slice sector split Args: mask: 2Dmask num_divisions: sector split start_angle_degrees: （ ），0 X direction Returns: segmented_mask: sector split mask， center: centroid"""
    try:
                  
        lv_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        
        if np.sum(lv_mask) == 0:
            return None, None
            
              
        moments = cv2.moments(lv_mask)
        if moments['m00'] == 0:
            return None, None
            
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
        center = (center_x, center_y)
        
                
        segmented_mask = np.zeros_like(mask, dtype=np.int32)
        h, w = mask.shape
        
                            
        y_coords, x_coords = np.ogrid[:h, :w]
        
                        
        dx = x_coords - center_x
        dy = y_coords - center_y
        angles = np.arctan2(dy, dx)              
        
                   
        start_angle_radians = np.radians(start_angle_degrees)
        
                     
        angles = angles - start_angle_radians
        angles = np.where(angles < -np.pi, angles + 2*np.pi, angles)
        angles = np.where(angles > np.pi, angles - 2*np.pi, angles)
        
                                   
        angles = (angles + np.pi) / (2 * np.pi) * num_divisions
        angles = np.floor(angles).astype(np.int32)
        angles = np.clip(angles, 0, num_divisions - 1)
        
                            
        for div in range(num_divisions):
            division_mask = (angles == div) & (mask > 0)
            segmented_mask[division_mask] = div + 1          
            
        return segmented_mask, center
        
    except Exception as e:
        logging.error(f"slicesector splitfailed: {e}")
        return None, None

def analyze_slice_segments_for_thickness(mask, segmented_mask, spacing_xy, num_divisions):
    """computemyocardiumthickness Args: mask: originalmask segmented_mask: sector split mask spacing_xy: pixel spacing num_divisions: sector split Returns: segment_thickness_stats: myocardiumthickness"""
    try:
        segment_thickness_stats = {}
        
        for div in range(1, num_divisions + 1):
            segment_area_mask = (segmented_mask == div)
            
                      
            segment_mask = mask.copy()
            segment_mask[~segment_area_mask] = 0              
            
                        
            thickness_max, thickness_mean, thickness_min, thickness_map, _ = test_thickness_calculation(segment_mask, spacing_xy)
            
            segment_thickness_stats[div] = {
                'thickness_mm_max': thickness_max,
                'thickness_mm_mean': thickness_mean,
                'thickness_mm_min': thickness_min,
            }
            
        return segment_thickness_stats
        
    except Exception as e:
        logging.error(f"myocardiumthicknessfailed: {e}")
        return {}

def calculate_diameter_from_mask(mask, spacing_xy, target_id=LV_BLOOD_POOL_ID):
    """computemask short-axis diameter Args: mask: mask spacing_xy: XYdirection pixel spacing target_id: compute ID（ ventricleblood pool） Returns: long_diameter, short_diameter: long-axis diameter short-axis diameter"""
                  
    binary_mask = (mask == target_id).astype(np.uint8)
    
          
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        logging.warning("No contour found")
        return -1.0, -1.0
    
             
    out_cont = contours[0]
    if len(contours) > 1:
        area = cv2.contourArea(contours[0])
        for cont in contours[1:]:  
            cont_area = cv2.contourArea(cont)
            if cont_area > area:
                area = cont_area
                out_cont = cont
    
    if len(out_cont) == 0:
        logging.warning("Empty contour")
        return 0.0, 0.0
    
            
    x, y = np.unique(out_cont[:, :, 0]), np.unique(out_cont[:, :, 1])
    if len(x) == 1 or len(y) == 1:
        l = (max(x) - min(x) + 1) * spacing_xy[0]  # noqa: E741
        r = (max(y) - min(y) + 1) * spacing_xy[1]
        return max(l, r), min(l, r)
    
          
    cont = np.squeeze(out_cont)
    st_idx, ed_idx, ok = get_long_diameter_start_end_index(cont)
    if not ok:
        logging.warning("Unable to find long-axis diameter")
        return -1.0, -1.0
    
    p1 = cont[st_idx]
    p2 = cont[ed_idx]
    long_diameter = build_diameter(p1, p2, spacing_xy)
    
          
    st_idx, ed_idx, ok = get_short_diameter_start_end_index(cont, st_idx, ed_idx)
    if not ok:
        logging.warning("Unable to find short-axis diameter")
        return -1.0, -1.0
    
    p1 = cont[st_idx]
    p2 = cont[ed_idx]
    short_diameter = build_diameter(p1, p2, spacing_xy)
    
    return long_diameter, short_diameter

def create_3d_blocks(data, num_blocks):
    """3D block block， block septal slice : data: 3D numpy ，shape (H, W, D) num_blocks: block (30 25) : blocks: ， 3D numpy original_blocks: originalblock （ ）"""
    total_slices = data.shape[2]
    
    if total_slices < num_blocks:
        logging.warning(f"slice {total_slices} block {num_blocks}， block")
        return None, None
    
    slices_per_block = total_slices // num_blocks
    if slices_per_block == 0:
        logging.warning(f"blockslice 0， block")
        return None, None
    
    logging.info(f"slice : {total_slices}, {num_blocks} block, block {slices_per_block}")
    
    blocks = []
    original_blocks = []           
    
    for block_idx in range(num_blocks):
        slice_indices = []
        for i in range(slices_per_block):
            slice_idx = block_idx + i * num_blocks
            if slice_idx < total_slices:
                slice_indices.append(slice_idx)
        
        if len(slice_indices) == 0:
            logging.warning(f"block {block_idx} validslice，skip")
            continue
            
                 
        original_block_data = data[:, :, slice_indices]
        original_blocks.append(original_block_data)
        
        # print(original_block_data.shape[2])

        if original_block_data.shape[2] <= 10:
            effective_slice_indices = slice_indices
                         

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
                      
        # if len(effective_slice_indices) > MAX_SLICES_PER_BLOCK:
        #     effective_slice_indices = effective_slice_indices[:MAX_SLICES_PER_BLOCK]
        
        if len(effective_slice_indices) == 0:
            logging.warning(f"block {block_idx} skipslice validslice，skip")
            blocks.append(None)
            continue
            
                    
        # effective_indices_in_data = [slice_indices[i] for i in range(len(slice_indices)) 
        #                            if i >= SKIP_HEAD_SLICES_PER_BLOCK and 
        #                               i < len(slice_indices) - SKIP_TAIL_SLICES_PER_BLOCK and
        #                               i - SKIP_HEAD_SLICES_PER_BLOCK < MAX_SLICES_PER_BLOCK]
        
        if len(effective_slice_indices) == 0:
            logging.warning(f"block {block_idx} validslice ，skip")
            blocks.append(None)
            continue
            
        block_data = data[:, :, effective_slice_indices]
        blocks.append(block_data)
        
        logging.info(f"      block {block_idx}: originalslice {slice_indices[:5]}{'...' if len(slice_indices) > 5 else ''}")
        logging.info(f"validslice {effective_slice_indices}, shape: {block_data.shape}")
    
    logging.info(f"{len(blocks)} block（ {len([b for b in blocks if b is not None])} validblock）")
    
    # import numpy as np
    # import matplotlib.pyplot as plt
    # plt.subplot(1,1,1)
    # plt.imshow(blocks[0][:,:,7])
    # plt.show()

    return blocks, original_blocks

                            
def find_interventricular_septum_center_robust(mask):
    """septumcenter method"""
    try:
                    
        lv_blood_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        rv_blood_mask = (mask == RV_BLOOD_POOL_ID).astype(np.uint8)
        
        print(f"ventricleblood pool : {np.sum(lv_blood_mask)}")
        print(f"ventricleblood pool : {np.sum(rv_blood_mask)}")
        
        if np.sum(lv_blood_mask) == 0 or np.sum(rv_blood_mask) == 0:
            print("ventricle ventricleblood pool")
            return None
        
                       
        septum_center = find_septum_by_centroid_line(lv_blood_mask, rv_blood_mask)
        if septum_center is not None:
            return septum_center
        
                       
        septum_center = find_septum_by_distance_transform(lv_blood_mask, rv_blood_mask)
        if septum_center is not None:
            return septum_center
        
        print("All methods failed")
        return None
        
    except Exception as e:
        print(f"septumcenterfailed: {e}")
        return None

def find_septum_by_centroid_line(lv_mask, rv_mask):
    """centroid method septumcenter"""
    try:
                   
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
        
        print(f"ventriclecentroid: {lv_center}")
        print(f"ventriclecentroid: {rv_center}")
        
                   
        centers_midpoint = (lv_center + rv_center) / 2
        
                
        direction = rv_center - lv_center
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
        
                       
        contact_points = []
        search_range = 15        
        step_size = 2
        
              
        perpendicular = np.array([-direction[1], direction[0]])
        
        for offset in range(-search_range, search_range + 1, step_size):
            search_point = centers_midpoint + perpendicular * offset
            
                         
            lv_boundary = find_closest_boundary(search_point, -direction, lv_mask, 30)
            rv_boundary = find_closest_boundary(search_point, direction, rv_mask, 30)
            
            if lv_boundary is not None and rv_boundary is not None:
                      
                contact_midpoint = (lv_boundary + rv_boundary) / 2
                contact_points.append(contact_midpoint)
        
        if contact_points:
                         
            contact_array = np.array(contact_points)
            septum_center = np.mean(contact_array, axis=0)
            print(f"centroid method septumcenter: {septum_center} ( {len(contact_points)} )")
            return septum_center
        
                          
        print(f"centroid septumcenter: {centers_midpoint}")
        return centers_midpoint
        
    except Exception as e:
        print(f"centroid methodfailed: {e}")
        return None

def find_closest_boundary(start_point, direction, mask, max_steps=50):
    """direction boundary"""
    try:
        step_size = 1.0
        
                       
        x, y = int(round(start_point[0])), int(round(start_point[1]))
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
            return None
        
        is_inside = mask[y, x] > 0
        
                  
        for step in range(max_steps):
            test_point = start_point + direction * step * step_size
            x, y = int(round(test_point[0])), int(round(test_point[1]))
            
            if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
                return None
            
            current_inside = mask[y, x] > 0
            
                  
            if is_inside != current_inside:
                boundary_point = test_point - direction * (step_size / 2)
                return boundary_point
        
        return None
        
    except Exception as e:
        return None

def find_septum_by_distance_transform(lv_mask, rv_mask):
    """method septumcenter"""
    try:
                
        combined_mask = lv_mask | rv_mask
        
                
        dist_transform = cv2.distanceTransform(combined_mask, cv2.DIST_L2, 5)
        
                            
        inverted_dist = cv2.distanceTransform(255 - combined_mask * 255, cv2.DIST_L2, 5)
        
                        
        septum_region = (dist_transform > 5) & (inverted_dist > 5)
        
        if np.sum(septum_region) == 0:
            return None
        
                       
        septum_points = np.where(septum_region)
        if len(septum_points[0]) == 0:
            return None
        
        septum_center_y = np.mean(septum_points[0])
        septum_center_x = np.mean(septum_points[1])
        septum_center = np.array([septum_center_x, septum_center_y])
        
        print(f"method septumcenter: {septum_center}")
        return septum_center
        
    except Exception as e:
        print(f"methodfailed: {e}")
        return None

def find_ventricular_axis(mask, septum_center):
    """ventricle axis"""
    try:
                    
        lv_blood_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        rv_blood_mask = (mask == RV_BLOOD_POOL_ID).astype(np.uint8)
        
                   
        lv_points = np.where(lv_blood_mask > 0)
        rv_points = np.where(rv_blood_mask > 0)
        
        if len(lv_points[0]) == 0 or len(rv_points[0]) == 0:
            return None
        
        lv_center = np.array([np.mean(lv_points[1]), np.mean(lv_points[0])])
        rv_center = np.array([np.mean(rv_points[1]), np.mean(rv_points[0])])
        
        print(f"ventriclecenter: {lv_center}")
        print(f"ventriclecenter: {rv_center}")
        
                       
        axis_direction = rv_center - lv_center
        axis_norm = np.linalg.norm(axis_direction)
        
        if axis_norm > 0:
            axis_direction = axis_direction / axis_norm
        else:
            axis_direction = np.array([1.0, 0.0])        
        
        print(f"ventricleaxisdirection: {axis_direction}")
        
        return axis_direction
        
    except Exception as e:
        print(f"ventricleaxisfailed: {e}")
        return None

def measure_ventricle_diameter_along_axis(start_point, direction, blood_mask, spacing_xy, ventricle_name):
    """measure axisdirectionventricleblood pool"""
    try:
                      
        boundaries = find_ventricle_boundaries_along_axis(start_point, direction, blood_mask, ventricle_name)
        
        if boundaries is None:
            print(f"{ventricle_name}boundary failed， fallbackmethod")
                            
            points = np.where(blood_mask > 0)
            if len(points[0]) > 0:
                                 
                coords = np.column_stack((points[1], points[0]))  # (x, y)
                center = np.mean(coords, axis=0)
                
                           
                axis_perp = np.array([-direction[1], direction[0]])
                projections = np.dot(coords - center, direction)
                
                if len(projections) > 0:
                    length = (np.max(projections) - np.min(projections)) * spacing_xy[0]
                    return max(5.0, min(length, 80.0))        
            return 30.0       
        
                      
        near_boundary, far_boundary = boundaries
        dx = (far_boundary[0] - near_boundary[0]) * spacing_xy[0]
        dy = (far_boundary[1] - near_boundary[1]) * spacing_xy[1]
        diameter = np.sqrt(dx*dx + dy*dy)
        
                
        diameter = max(5.0, min(diameter, 80.0))
        
        print(f"{ventricle_name}inner diametermeasure: {np.array(near_boundary).astype(int)}, {np.array(far_boundary).astype(int)}, {diameter:.1f}mm")
        
        return diameter
        
    except Exception as e:
        print(f"measure{ventricle_name}inner diameterfailed: {e}")
        return 30.0       

def find_ventricle_boundaries_along_axis(start_point, direction, blood_mask, ventricle_name):
    """axis ventricleblood pool boundary"""
    try:
        step_size = 1.0
        max_steps = 300
        
                      
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
            print(f"not found{ventricle_name}blood pool")
            return None
        
                             
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
            print(f"not found{ventricle_name}blood pool")
            return None
        
                                         
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
        
                  
        if second_entry is not None and second_exit is not None:
            first_length = np.linalg.norm(first_exit - first_entry)
            second_length = np.linalg.norm(second_exit - second_entry)
            
            if second_length > first_length:
                print(f"{ventricle_name} blood pool， {second_length:.1f} > {first_length:.1f}")
                return (second_entry, second_exit)
        
        return (first_entry, first_exit)
        
    except Exception as e:
        print(f"{ventricle_name}boundaryfailed: {e}")
        return None

def find_left_ventricle_far_boundary(septum_center, axis_direction, blood_mask, myo_mask, max_steps=300):
    """ventricleblood pool （ ）"""
    try:
                    
        lv_diameter = measure_ventricle_diameter_along_axis(
            septum_center, -axis_direction, blood_mask, (1.0, 1.0), "ventricle")
        
                      
        boundaries = find_ventricle_boundaries_along_axis(
            septum_center, -axis_direction, blood_mask, "ventricle")
        
        if boundaries is not None:
            return boundaries[1]           
        else:
                  
            return septum_center - axis_direction * lv_diameter
        
    except Exception as e:
        print(f"ventricle boundaryfailed: {e}")
        return septum_center - axis_direction * 20

def find_right_ventricle_far_boundary(septum_center, axis_direction, blood_mask, max_steps=300):
    """ventricleblood pool （ ）"""
    try:
                    
        rv_diameter = measure_ventricle_diameter_along_axis(
            septum_center, axis_direction, blood_mask, (1.0, 1.0), "ventricle")
        
                      
        boundaries = find_ventricle_boundaries_along_axis(
            septum_center, axis_direction, blood_mask, "ventricle")
        
        if boundaries is not None:
            return boundaries[1]           
        else:
                  
            return septum_center + axis_direction * rv_diameter
        
    except Exception as e:
        print(f"ventricle boundaryfailed: {e}")
        return septum_center + axis_direction * 20


def calculate_angle_with_xaxis(point1, point2):
    """compute x （ ） （ ： ） Args: point1: (x1, y1) point2: (x2, y2) Returns: angle: y （0°~180°）"""
                     
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    
                  
    y_axis_vec = np.array([1, 0])
    line_vec = np.array([dx, dy])
    
                     
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return 0.0             
    
                                    
    dot_product = np.dot(line_vec, y_axis_vec)
    cos_theta = np.clip(dot_product / line_len, -1.0, 1.0)
    
                         
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def create_visualization(mask, septum_center, lv_boundary, rv_boundary, lv_diameter, rv_diameter, axis_direction):
    """UNTRANSLATED"""
    try:
                 
        if len(mask.shape) == 2:
            vis_image = np.stack([mask, mask, mask], axis=-1).astype(np.uint8) * 50
        else:
            vis_image = mask.astype(np.uint8)
        
        vis_image = np.ascontiguousarray(vis_image)
        
                      
        lv_blood_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        rv_blood_mask = (mask == RV_BLOOD_POOL_ID).astype(np.uint8)
        
                
        lv_contours, _ = cv2.findContours(lv_blood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rv_contours, _ = cv2.findContours(rv_blood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(vis_image, lv_contours, -1, (0, 255, 0), 1)
        cv2.drawContours(vis_image, rv_contours, -1, (255, 0, 0), 1)
        
                  
        septum_x, septum_y = int(round(septum_center[0])), int(round(septum_center[1]))
        lv_x, lv_y = int(round(lv_boundary[0])), int(round(lv_boundary[1]))
        rv_x, rv_y = int(round(rv_boundary[0])), int(round(rv_boundary[1]))
        
             
        line_length = 200
        axis_start = septum_center - axis_direction * line_length
        axis_end = septum_center + axis_direction * line_length
        
        axis_start_x, axis_start_y = int(round(axis_start[0])), int(round(axis_start[1]))
        axis_end_x, axis_end_y = int(round(axis_end[0])), int(round(axis_end[1]))
        
        cv2.line(vis_image, (axis_start_x, axis_start_y), (axis_end_x, axis_end_y), 
                (128, 128, 128), 1)
        
               
        cv2.circle(vis_image, (septum_x, septum_y), 6, (0, 255, 255), -1)              
        cv2.circle(vis_image, (lv_x, lv_y), 5, (0, 255, 0), -1)                
        cv2.circle(vis_image, (rv_x, rv_y), 5, (255, 0, 0), -1)                
        
                
        cv2.line(vis_image, (septum_x, septum_y), (lv_x, lv_y), (0, 255, 0), 2)         
        cv2.line(vis_image, (septum_x, septum_y), (rv_x, rv_y), (255, 0, 0), 2)         
        
                
        cv2.putText(vis_image, f'LV: {lv_diameter:.1f}mm', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_image, f'RV: {rv_diameter:.1f}mm', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
               
        cv2.putText(vis_image, 'Septum', (septum_x+5, septum_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis_image, 'LV Far', (lv_x+5, lv_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_image, 'RV Far', (rv_x+5, rv_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        lv_angle = calculate_angle_with_xaxis((septum_x, septum_y), (lv_x, lv_y))
        
        return vis_image, lv_angle
        
    except Exception as e:
        print(f"failed: {e}")
        return None, None

def visualize_results(mask, results, visualization):
    """results"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
             
    axes[0].imshow(mask, cmap='tab10')
    axes[0].set_title('Heart segmentation mask')
    axes[0].axis('off')
    
             
    if visualization is not None:
        axes[1].imshow(visualization)
        axes[1].set_title('Ventricular inner diameter')
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'Measurement failed', ha='center', va='center', 
                    transform=axes[1].transAxes)
        axes[1].set_title('Ventricular inner diameter')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def calculate_ventricular_diameters_robust(mask, spacing_xy):
    """fixedventricleinner diametercompute - measureaxis ventricleblood pool"""
    try:
                  
        septum_center = find_interventricular_septum_center_robust(mask)
        if septum_center is None:
            print("method septumcenter ， fallbackmethod")
            septum_center = np.array([mask.shape[1] // 2, mask.shape[0] // 2])
            print(f"center septumcenter: {septum_center}")
        
        print(f"septumcenter : {septum_center}")
        
                
        axis_direction = find_ventricular_axis(mask, septum_center)
        if axis_direction is None:
            print("not foundventricleaxis， direction")
            axis_direction = np.array([1.0, 0.0])
        
        print(f"ventricleaxisdirection: {axis_direction}")
        
              
        lv_blood_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        rv_blood_mask = (mask == RV_BLOOD_POOL_ID).astype(np.uint8)
        lv_myo_mask = (mask == LV_MYOCARDIUM_ID).astype(np.uint8)
        
                                  
        lv_diameter = measure_ventricle_diameter_along_axis(
            septum_center, -axis_direction, lv_blood_mask, spacing_xy, "ventricle")
        
                                    
        rv_diameter = measure_ventricle_diameter_along_axis(
            septum_center, axis_direction, rv_blood_mask, spacing_xy, "ventricle")
        
                    
        lv_far_boundary = find_left_ventricle_far_boundary(
            septum_center, axis_direction, lv_blood_mask, lv_myo_mask)
        
        rv_far_boundary = find_right_ventricle_far_boundary(
            septum_center, axis_direction, rv_blood_mask)
        
        print(f"ventricleinner diameter: {lv_diameter:.2f}mm")
        print(f"ventricleinner diameter: {rv_diameter:.2f}mm")
        
               
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
        print(f"computeventricleinner diameterfailed: {e}")
        import traceback
        traceback.print_exc()
        
        default_results = {
            'left_ventricle': {'transverse_diameter_mm': 0.0},
            'right_ventricle': {'transverse_diameter_mm': 0.0}
        }
        return default_results, None

def analyze_ventricular_dimensions(mask, spacing_xy):
    """ventricleinner diameter - validresults"""
    print("=== ventricleinner diametermeasure（fixed version）===")
    print(f"maskshape: {mask.shape}")
    print(f"maskdtype: {mask.dtype}")
    print(f"unique values: {np.unique(mask)}")
    
    results, visualization = calculate_ventricular_diameters_robust(mask, spacing_xy)
    
           
    # if visualization is not None:
    #     visualize_results(mask, results, visualization)
    
    if results and 'left_ventricle' in results and 'right_ventricle' in results:
        print("\n=== measureresults ===")
        print(f"ventricleinner diameter: {results['left_ventricle']['transverse_diameter_mm']:.1f}mm")
        print(f"ventricleinner diameter: {results['right_ventricle']['transverse_diameter_mm']:.1f}mm")
        print(f"lv_angle: {results['lv_angle']['lv_angle']:.1f}mm")

    else:
        print("measurefailed， default value")
                  
        if 'left_ventricle' not in results:
            results = {
                'left_ventricle': {'transverse_diameter_mm': 0.0},
                'right_ventricle': {'transverse_diameter_mm': 0.0}
            }
    
    return results
                            

def compute_lv_volume_only(blk_norm, spacing=(1.0, 1.0, 1.0)):
    """computeLV EDblock"""
    total_lv = 0.0
    for si in range(blk_norm.shape[2]):
        sl = blk_norm[:, :, si]
        lv_vol = np.sum(sl == LV_BLOOD_POOL_ID) * spacing[0] * spacing[1] * spacing[2] / 1000.0
        total_lv += lv_vol
    return total_lv

def compute_block_totals(blk_norm, img_norm, spacing=(1.0, 1.0, 1.0)):
    """computeblock"""
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
    """block，"""
    if blk is None:
        return None, None, None, None
        
            
    original_shape = blk.shape
    logging.info(f"    {block_type}blockoriginalshape: {original_shape}")
    
    processed_blk = blk.copy()
    processed_img = img_blk.copy() if img_blk is not None else None
    
                    
    non_zero = np.where(processed_blk > 0)
    if len(non_zero[0]) == 0:
        logging.warning(f"{block_type}block ，skip")
        return None, None, None, None
        
            
    y_min = max(0, np.min(non_zero[0]) - CROP_MARGIN)
    y_max = min(processed_blk.shape[0] - 1, np.max(non_zero[0]) + CROP_MARGIN)
    x_min = max(0, np.min(non_zero[1]) - CROP_MARGIN) 
    x_max = min(processed_blk.shape[1] - 1, np.max(non_zero[1]) + CROP_MARGIN)
    
          
    processed_blk = processed_blk[y_min:y_max+1, x_min:x_max+1, :]
    if processed_img is not None:
        processed_img = processed_img[y_min:y_max+1, x_min:x_max+1, :]
    
    crop_shape = processed_blk.shape
    logging.info(f"{block_type}block shape: {crop_shape}")
    
                                      
    target_shape = (crop_shape[0], crop_shape[1], 64)                
    
    scale_factors = [
        target_shape[0] / crop_shape[0],           
        target_shape[1] / crop_shape[1],             
        target_shape[2] / crop_shape[2]                
    ]
    
              
    processed_blk = zoom(processed_blk, scale_factors, order=0)            
    if processed_img is not None:
        processed_img = zoom(processed_img, scale_factors, order=1)              
    
    final_shape = processed_blk.shape
    final_spacing = (1.0, 1.0, 1.0)                     
    
    logging.info(f"{block_type}blockResize shape: {final_shape}, spacing: {final_spacing}")
    
    return processed_blk, processed_img, final_spacing, {
        'original_shape': original_shape,
        'crop_shape': crop_shape, 
        'final_shape': final_shape,
        'scale_factors': scale_factors
    }

def calculate_metrics_with_layers(ed_block, es_block, spacing=(1.0, 1.0, 1.0)):
    """computeheart"""
    if ed_block is None or es_block is None:
        return None
    
           
    ed_lv_vol_total, ed_lv_myo_vol_total, ed_rv_vol_total, ed_rv_myo_vol_total = compute_block_totals(ed_block, None, spacing)
    es_lv_vol_total, es_lv_myo_vol_total, es_rv_vol_total, es_rv_myo_vol_total = compute_block_totals(es_block, None, spacing)
    
             
    lv_sv = ed_lv_vol_total - es_lv_vol_total
    lv_ef = (lv_sv / ed_lv_vol_total * 100) if ed_lv_vol_total > 0 else 0
    lv_co = lv_sv * ASSUMED_HEART_RATE / 1000.0
    
             
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
    """computeSA （ ）"""
    try:
              
        pred_img = nib.load(cine_sa_mask_path)
        pred_data = np.round(pred_img.get_fdata()).astype(np.int16)
        pred_data = np.flip(pred_data, axis=1)
                          
        original_spacing = pred_img.header.get_zooms()                       
        logging.info(f"original spacing: {original_spacing}")
        
        # if pred_data.shape[2] % 30 == 0:
        #     BLOCK_SIZES = 30
        # elif pred_data.shape[2] % 25 == 0:
        #     BLOCK_SIZES = 25
        # else:
        #     BLOCK_SIZES = 30
        BLOCK_SIZES = slice_num
             
        blocks, original_blocks = create_3d_blocks(pred_data, BLOCK_SIZES)
        
        if blocks is None or original_blocks is None:
            return None
        
        valid_z_lengths = []
        for i, block in enumerate(blocks):
            if block is None:
                continue
                            
            has_lv_in_slice = np.any(block == LV_BLOOD_POOL_ID, axis=(0, 1))
            valid_z_indices = np.where(has_lv_in_slice)[0]
            if len(valid_z_indices) > 0:               
                valid_z_lengths.append(len(valid_z_indices))

                            
        if len(valid_z_lengths) == 0:
                                                    
            target_z_length = 0
        else:
                                                                                  
            target_z_length = mode(valid_z_lengths, keepdims=False).mode
                                             
            # from collections import Counter
            # count_dict = Counter(valid_z_lengths)
            # target_z_length = max(count_dict, key=count_dict.get)

        print(f"validslice ：{target_z_length}")
        block_volumes = []
        for i, block in enumerate(blocks):
            if block is None:
                                                                     
                continue
            
                             
            has_lv_in_slice = np.any(block == LV_BLOOD_POOL_ID, axis=(0, 1))
            valid_z_indices = np.where(has_lv_in_slice)[0]
            current_z_length = len(valid_z_indices)
            
                                          
            if current_z_length != target_z_length:
                # block_volumes.append((i, None)) 
                continue
            
                                        
            valid_block = block[..., valid_z_indices]
            lv_voxels = np.sum(valid_block == LV_BLOOD_POOL_ID)
            
            actual_z_spacing = original_spacing[2] * (pred_data.shape[2] / current_z_length) if current_z_length > 0 else original_spacing[2]
            single_phase_z_spacing = original_spacing[2] * (BLOCK_SIZES) if current_z_length > 0 else original_spacing[2]
                      
            lv_vol = lv_voxels * original_spacing[0] * original_spacing[1] * actual_z_spacing / 1000.0
            block_volumes.append((i, lv_vol, valid_z_indices))
            
                                                                                                                                   
                
        if not block_volumes:
            return None
            
                  
        block_volumes.sort(key=lambda x: x[1], reverse=True)
        ed_idx = block_volumes[0][0]             
        es_idx = block_volumes[-1][0]             

        ed_idx_slice = block_volumes[0][2]             
        es_idx_slice = block_volumes[-1][2]             
        
        logging.info(f"EDblock : {ed_idx}, ESblock : {es_idx}")
        
        metrics = {}
        
                 
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
        
               
        if ed_block_original is not None:
            target_slice = ed_block_original.shape[2] // 2
            ed_target_slice = ed_block_original[:, :, target_slice]
            results = analyze_ventricular_dimensions(ed_target_slice, original_spacing[:2])
            print(f"ventricle : {results['left_ventricle']['transverse_diameter_mm']:.1f}mm")
            print(f"ventricle : {results['right_ventricle']['transverse_diameter_mm']:.1f}mm")
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
                     
        for slice_idx, num_divisions in zip(SEGMENTATION_SLICE_INDICES, SEGMENTATION_DIVISIONS):
                                                         
            start_angle = 45 + lv_angle if slice_idx < 3 else 0 + lv_angle
                  
                   
            if ed_block_original_nocrop is not None and ed_block_original_nocrop.shape[2] > slice_idx:
                ed_slice = ed_block_original_nocrop[:, :, slice_idx]
                
                        
                ed_thickness_max, ed_thickness_mean, ed_thickness_min, _, _ = test_thickness_calculation(ed_slice, original_spacing[:2])
                
                      
                ed_segmented_mask, ed_center = create_slice_segmentation(ed_slice, num_divisions, start_angle)
                
                             
                if ed_segmented_mask is not None:
                    ed_segment_stats = analyze_slice_segments_for_thickness(ed_slice, ed_segmented_mask, original_spacing[:2], num_divisions)
                    
                    slice_num = slice_idx                          
                    
                              
                    metrics[f'ED_Slice_{slice_num}_Thickness_max'] = ed_thickness_max
                    metrics[f'ED_Slice_{slice_num}_Thickness_mean'] = ed_thickness_mean
                    metrics[f'ED_Slice_{slice_num}_Thickness_min'] = ed_thickness_min
                    
                                            
                    TARGET_SEGMENT_IDS = {1, 6, 7, 12}

                    for div_id, div_stats in ed_segment_stats.items():
                        if slice_num in SEGMENT_NAMES and div_id in SEGMENT_NAMES[slice_num]:
                            segment_info = SEGMENT_NAMES[slice_num][div_id]
                            segment_id = segment_info['id']
                            segment_name = segment_info['name']
                            
                                     
                            thickness_max = div_stats['thickness_mm_max']
                            thickness_mean = div_stats['thickness_mm_mean']
                            thickness_min = div_stats['thickness_mm_min']
                            
                            if segment_id in TARGET_SEGMENT_IDS:
                                                              
                                if isinstance(thickness_max, (int, float)):
                                    thickness_max += original_spacing[0] * original_spacing[1]
                                if isinstance(thickness_mean, (int, float)):
                                    thickness_mean += original_spacing[0] * original_spacing[1]
                                if isinstance(thickness_min, (int, float)):
                                    thickness_min += original_spacing[0] * original_spacing[1]
                            
                                                  
                            metrics[f'ED_Segment_{segment_id:02d}_{segment_name}_Thickness_max'] = thickness_max
                            metrics[f'ED_Segment_{segment_id:02d}_{segment_name}_Thickness_mean'] = thickness_mean
                            metrics[f'ED_Segment_{segment_id:02d}_{segment_name}_Thickness_min'] = thickness_min
        
        return metrics
        
    except Exception as e:
        logging.error(f"computecine SA failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
                    
    logging.basicConfig(level=logging.INFO)
    
    cine_sa_mask_path = "0000375_sa_pred.nii.gz"
    metrics = calculate_cine_sa_metrics(cine_sa_mask_path)
    print("Computation finished. Results:")
    print(metrics)