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

                
      
CROP_MARGIN = 5                     

                  
BACKGROUND_ID = 0           
LV_BLOOD_POOL_ID = 1           
LV_MYOCARDIUM_ID = 2           
       
RV_BLOOD_POOL_ID = 3           
RV_MYOCARDIUM_ID = 4           
RA_BLOOD_POOL_ID = 5         
LA_BLOOD_POOL_ID = 6         

      
# BLOCK_SIZES = [30]

      
MAX_SLICES_PER_BLOCK = 3                    
SKIP_HEAD_SLICES_PER_BLOCK = 0              
SKIP_TAIL_SLICES_PER_BLOCK = 0              
TARGET_SLICE_INDEX = 1                             
RV_WALL_THICKNESS_DIVISIONS = 3              
APEX_SLICE_INDEX = 1             

      
ASSUMED_HEART_RATE = 70
MYOCARDIUM_DENSITY = 1.05

                                   
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

def calculate_rv_wall_thickness_segmented(mask, spacing_xy, num_divisions=3):
    """compute ventriclemyocardiumthickness 3sector split（ ventricle thickness） Args: mask: mask spacing_xy: XYdirection pixel spacing num_divisions: sector split （ 3） Returns: segment_thickness_stats: thickness segmentation_info: （ ）"""
    try:
                        
        rv_myo_mask = (mask == RV_MYOCARDIUM_ID).astype(np.uint8)
        
        if np.sum(rv_myo_mask) == 0:
            logging.warning("ventriclemyocardium")
            return {}, None
            
                     
        rv_blood_mask = (mask == RV_BLOOD_POOL_ID).astype(np.uint8)
        
                      
        segmented_mask, center, arc_info = create_arc_segmentation(rv_myo_mask, rv_blood_mask, num_divisions)
        
        if segmented_mask is None:
            return {}, None
            
        segment_thickness_stats = {}
        
                   
        for div in range(1, num_divisions + 1):
            segment_area_mask = (segmented_mask == div)
            
                       
            segment_mask = mask.copy()
            segment_mask[~segment_area_mask] = 0
            
                           
            thickness = calculate_rv_myocardium_thickness_in_segment(segment_mask, center, spacing_xy)
            
            segment_thickness_stats[div] = {
                'thickness_mm': thickness
            }
        
                     
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
        logging.error(f"compute ventriclemyocardiumthickness failed: {e}")
        return {}, None

def create_arc_segmentation(rv_myo_mask, rv_blood_mask, num_divisions=3):
    """sector split（ ） Args: rv_myo_mask: ventriclemyocardiummask rv_blood_mask: ventricleblood poolmask num_divisions: sector split Returns: segmented_mask: mask center: center arc_info:"""
    try:
                         
        if np.sum(rv_blood_mask) > 0:
            moments = cv2.moments(rv_blood_mask)
            if moments['m00'] > 0:
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
            else:
                                   
                moments = cv2.moments(rv_myo_mask)
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
        else:
                    
            moments = cv2.moments(rv_myo_mask)
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
        
        center = (center_x, center_y)
        
                    
        contours, _ = cv2.findContours(rv_myo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None, None, None
            
        main_contour = max(contours, key=cv2.contourArea)
        contour_points = np.squeeze(main_contour)
        
        if len(contour_points.shape) == 1:
            contour_points = contour_points.reshape(1, -1)
        
                         
        angles = []
        for point in contour_points:
            dx = point[0] - center_x
            dy = point[1] - center_y
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        angles = np.array(angles)
        
                            
        min_angle = np.min(angles)
        max_angle = np.max(angles)
        
                       
        if max_angle - min_angle > np.pi:
                           
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
        
                
        segmented_mask = np.zeros(rv_myo_mask.shape, dtype=np.int32)
        segmented_mask = np.ascontiguousarray(segmented_mask)
        h, w = rv_myo_mask.shape
        
                              
        y_coords, x_coords = np.ogrid[:h, :w]
        dx = x_coords - center_x
        dy = y_coords - center_y
        pixel_angles = np.arctan2(dy, dx)
        
                   
        if max_angle > np.pi:
            pixel_angles = np.where(pixel_angles < 0, pixel_angles + 2*np.pi, pixel_angles)
        
                    
        arc_range = max_angle - min_angle
        division_size = arc_range / num_divisions
        
        for div in range(num_divisions):
            div_start = min_angle + div * division_size
            div_end = min_angle + (div + 1) * division_size
            
                                     
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
        logging.error(f"failed: {e}")
        return None, None, None

def calculate_rv_myocardium_thickness_in_segment(mask, center, spacing_xy):
    """compute ventriclemyocardium thickness（ ）"""
    try:
                    
        rv_myo_pixels = np.where(mask == RV_MYOCARDIUM_ID)
        
        if len(rv_myo_pixels[0]) == 0:
            return 2.0
            
                          
        rv_myo_mask = (mask == RV_MYOCARDIUM_ID).astype(np.uint8)
        
                   
                             
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
                     
        outer_boundary = cv2.morphologyEx(rv_myo_mask, cv2.MORPH_GRADIENT, kernel)
        
                            
        eroded = cv2.erode(rv_myo_mask, kernel, iterations=2)
        inner_boundary = cv2.morphologyEx(eroded, cv2.MORPH_GRADIENT, kernel)
        
                             
        if np.sum(inner_boundary) == 0:
                            
            from skimage.morphology import skeletonize
            skeleton = skeletonize(rv_myo_mask > 0)
            inner_boundary = skeleton.astype(np.uint8)
        
        if np.sum(outer_boundary) == 0 or np.sum(inner_boundary) == 0:
                          
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
        
                           
        outer_points = np.where(outer_boundary > 0)
        inner_points = np.where(inner_boundary > 0)
        
        if len(outer_points[0]) == 0 or len(inner_points[0]) == 0:
            return 2.0
        
                           
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
                    
            avg_thickness = total_thickness / count
            return avg_thickness
        else:
            return 2.0
        
    except Exception as e:
        logging.error(f"compute ventriclemyocardiumthicknessfailed: {e}")
        return 2.0

def calculate_apex_thickness_fallback(lv_blood_mask, lv_myocardium_mask, spacing_xy):
    """fallbackapicalthicknesscomputemethod： radial"""
    try:
                
        moments = cv2.moments(lv_blood_mask)
        if moments['m00'] == 0:
            return 3.0, None
        center_x = moments['m10'] / moments['m00']
        center_y = moments['m01'] / moments['m00']
        center = (center_x, center_y)
        
                
        contours, _ = cv2.findContours(lv_myocardium_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 3.0, None
        
        main_contour = max(contours, key=cv2.contourArea)
        contour_points = np.squeeze(main_contour)
        if contour_points.ndim == 1:
            contour_points = contour_points.reshape(1, -1)
        
                        
        distances = []
        for point in contour_points:
            dx = (point[0] - center_x) * spacing_xy[0]
            dy = (point[1] - center_y) * spacing_xy[1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances.append(dist)
        
        if not distances:
            return 3.0, None
        
                            
        max_dist = np.max(distances)
        min_dist = np.min(distances)
        apex_thickness = max_dist - min_dist
                    
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
        
        logging.info(f"fallbackmethodapicalthickness: {apex_thickness:.2f}mm (radial : {min_dist:.2f} - {max_dist:.2f})")
        return apex_thickness, apex_info
        
    except Exception as e:
        logging.error(f"fallbackapicalthicknesscomputefailed: {e}")
        return 3.0, None

def pick_nearest_thickness(inner_hits, outer_hits, apex_endpoint, spacing_xy):
    inner_pt = pick_nearest(inner_hits, apex_endpoint)
    outer_pt = pick_nearest(outer_hits, apex_endpoint)
    dx = (outer_pt[0] - inner_pt[0]) * spacing_xy[0]
    dy = (outer_pt[1] - inner_pt[1]) * spacing_xy[1]
    return np.sqrt(dx * dx + dy * dy)

                          
def filter_apex_neighborhood(hits, ref, radius=30):
    ref_arr = np.array(ref, dtype=np.float32)
    neighborhood_hits = []
    hit_distances = []                      
    
    for h in hits:
        h_arr = np.array(h, dtype=np.float32)
        dist = np.linalg.norm(h_arr - ref_arr)
        if dist <= radius:
            neighborhood_hits.append(h_arr)
            hit_distances.append(dist)
    
    return np.array(neighborhood_hits), hit_distances

                              
def calculate_apex_thickness_stats(inner_hits, outer_hits, apex_endpoint, spacing_xy):
                    
    inner_neighborhood, _ = filter_apex_neighborhood(inner_hits, apex_endpoint)
    outer_neighborhood, _ = filter_apex_neighborhood(outer_hits, apex_endpoint)
    
    if len(inner_neighborhood) == 0 or len(outer_neighborhood) == 0:
        return 3.0, 3.0, 3.0            
    
                                             
    thicknesses = []
                                  
    if len(inner_neighborhood) == len(outer_neighborhood):
        for inner_pt, outer_pt in zip(inner_neighborhood, outer_neighborhood):
            dx = (outer_pt[0] - inner_pt[0]) * spacing_xy[0]
            dy = (outer_pt[1] - inner_pt[1]) * spacing_xy[1]
            thickness = np.sqrt(dx * dx + dy * dy)
            thicknesses.append(thickness)
                                        
    else:
        for inner_pt in inner_neighborhood:
            for outer_pt in outer_neighborhood:
                dx = (outer_pt[0] - inner_pt[0]) * spacing_xy[0]
                dy = (outer_pt[1] - inner_pt[1]) * spacing_xy[1]
                thickness = np.sqrt(dx * dx + dy * dy)
                thicknesses.append(thickness)
    if not thicknesses:
        return 3.0, 3.0, 3.0
    
           
    min_thickness = float(np.min(thicknesses))
    if min_thickness > 2 * spacing_xy[0] * spacing_xy[1] + 2:                  
        min_thickness = float(np.min(thicknesses)) - 2 * spacing_xy[0] * spacing_xy[1]
        max_thickness = float(np.mean(thicknesses)) - 2 * spacing_xy[0] * spacing_xy[1] 
                
        mean_thickness = float(np.min(thicknesses)) - 2 * spacing_xy[0] * spacing_xy[1]
    else:
        max_thickness = float(np.min(thicknesses))
                
        mean_thickness = float(np.min(thicknesses))

    
    return min_thickness, max_thickness, mean_thickness

              
def pick_nearest(hits, ref):
    ref_arr = np.array(ref, dtype=np.float32)
    dists = [np.linalg.norm(np.array(h, dtype=np.float32) - ref_arr) for h in hits]
    return np.array(hits[int(np.argmin(dists))], dtype=np.float32)

import matplotlib.pyplot as plt

def visualize_line_contour(long_line, inner_contour_points, inner_hits, 
                           apex_endpoint=None, title="contour"):
    """、 contour ： long_line: （ a, b, c ， ax+by+c=0） inner_contour_points: contour ，shape=(N,2) inner_hits: contour ，shape=(M,2) apex_endpoint: apical （ ， ） title:"""
          
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
                      
    if len(inner_contour_points) > 0:
                     
        contour_poly = plt.Polygon(inner_contour_points, 
                                   fill=False, color='blue', linewidth=2, 
                                   label='contour')
        ax.add_patch(contour_poly)
                   
        ax.scatter(inner_contour_points[:, 0], inner_contour_points[:, 1], 
                   color='blue', s=10, alpha=0.5)
    
                     
                         
    x_min, x_max = np.min(inner_contour_points[:, 0]) - 10, np.max(inner_contour_points[:, 0]) + 10
    y_min, y_max = np.min(inner_contour_points[:, 1]) - 10, np.max(inner_contour_points[:, 1]) + 10
    
                           
    if abs(long_line.b) > 1e-10:              
        x_line = np.linspace(x_min, x_max, 100)
        y_line = -(long_line.a * x_line + long_line.c) / long_line.b
    else:                  
        y_line = np.linspace(y_min, y_max, 100)
        x_line = -long_line.c / long_line.a * np.ones_like(y_line)
    
    ax.plot(x_line, y_line, color='red', linewidth=2, label='UNTRANSLATED', linestyle='--')
    
                    
    if inner_hits:
        hits_arr = np.array(inner_hits)
        ax.scatter(hits_arr[:, 0], hits_arr[:, 1], 
                   color='red', s=50, marker='o', 
                   label='UNTRANSLATED', zorder=5)
    
                   
    if apex_endpoint is not None:
        ax.scatter(apex_endpoint[0], apex_endpoint[1], 
                   color='green', s=80, marker='*', 
                   label='apical', zorder=6)
    
             
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X （ ）')
    ax.set_ylabel('Y （ ）')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')                
    
    plt.show()


def calculate_apex_thickness(mask, spacing_xy):
    """compute ventricleapicalthickness（ method：long-axis diameter myocardium / contour ） Args: mask: mask spacing_xy: XYdirection pixel spacing Returns: apex_thickness: apicalthickness (mm) apex_info:"""
    try:
        lv_blood_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        lv_myocardium_mask = (mask == LV_MYOCARDIUM_ID).astype(np.uint8)
        
        logging.info(f"apicalthicknesscompute - LVblood pool : {np.sum(lv_blood_mask)}, LVmyocardium : {np.sum(lv_myocardium_mask)}")
        
        if np.sum(lv_blood_mask) == 0 or np.sum(lv_myocardium_mask) == 0:
            logging.warning("ventricleblood pool myocardium")
            return 3.0, 3.0, 3.0, None
            
                              
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        myo_edge = cv2.morphologyEx(lv_myocardium_mask, cv2.MORPH_GRADIENT, kernel)
        
                    
        blood_dil = cv2.dilate(lv_blood_mask, kernel, iterations=2)          
        inner_edge_mask = ((myo_edge > 0) & (blood_dil > 0)).astype(np.uint8)
        outer_edge_mask = ((myo_edge > 0) & (blood_dil == 0)).astype(np.uint8)

              
        inner_contours, _ = cv2.findContours(inner_edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        outer_contours, _ = cv2.findContours(outer_edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
                             
        if np.sum(inner_edge_mask) == 0 or np.sum(outer_edge_mask) == 0 or not inner_contours or not outer_contours:
            logging.info("methodfailed， method")
                        
            dist_transform = cv2.distanceTransform(lv_myocardium_mask, cv2.DIST_L2, 5)
                           
            inner_edge_mask = ((myo_edge > 0) & (dist_transform < 5)).astype(np.uint8)
                           
            outer_edge_mask = ((myo_edge > 0) & (dist_transform >= 5)).astype(np.uint8)

            inner_contours, _ = cv2.findContours(inner_edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            outer_contours, _ = cv2.findContours(outer_edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        logging.info(f"contour : {len(inner_contours)}, contour : {len(outer_contours)}")
        if not inner_contours or not outer_contours:
            logging.warning("not found / boundarycontour")
            return 3.0, 3.0, 3.0, None

        inner_contour_points = np.squeeze(max(inner_contours, key=cv2.contourArea)).astype(np.int32)
        outer_contour_points = np.squeeze(max(outer_contours, key=cv2.contourArea)).astype(np.int32)
        if inner_contour_points.ndim == 1:
            inner_contour_points = inner_contour_points.reshape(1, -1)
        if outer_contour_points.ndim == 1:
            outer_contour_points = outer_contour_points.reshape(1, -1)

                     
        blood_contours, _ = cv2.findContours(lv_blood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        logging.info(f"LVblood poolcontour : {len(blood_contours)}")
        if not blood_contours:
            logging.warning("not foundLVblood poolcontour")
            return 3.0, None
        blood_pts = np.squeeze(max(blood_contours, key=cv2.contourArea))
        if blood_pts.ndim == 1:
            blood_pts = blood_pts.reshape(1, -1)
        long_st, long_ed, long_ok = get_long_diameter_start_end_index(blood_pts)
        logging.info(f"LV compute: ={long_st}, ={long_ed}, ={long_ok}")
        if not long_ok:
            logging.warning("computeLV")
            return 3.0, None
        long_p1 = blood_pts[long_st]
        long_p2 = blood_pts[long_ed]
        long_line = build_line2D(long_p1, long_p2)

                         
        apex_endpoint = long_p1 if long_p1[1] > long_p2[1] else long_p2
        base_endpoint = long_p2 if apex_endpoint is long_p1 else long_p1

                                  
        long_vec = np.array([long_p2[0] - long_p1[0], long_p2[1] - long_p1[1]], dtype=np.float32)
        long_len = np.linalg.norm(long_vec)
        if long_len > 0:
            long_dir = long_vec / long_len
                                  
            extended_apex = apex_endpoint + long_dir * long_len * 2.0
                           
            extended_base = base_endpoint - long_dir * long_len * 2.0
                        
            long_line = build_line2D(extended_apex, extended_base)
            logging.info(f": apical {apex_endpoint} -> {extended_apex}, basal {base_endpoint} -> {extended_base}")
            logging.info(f"direction : {long_dir}, : {long_len:.2f}")
        else:
            logging.warning("0，")
            extended_apex = apex_endpoint
            extended_base = base_endpoint

                               
        def intersect_line_with_contour(line, contour_points, contour_name="contour"):
            hits = []
            n = len(contour_points)
            logging.info(f"{contour_name} : {n}")
            logging.info(f": {line.a:.3f}x + {line.b:.3f}y + {line.c:.3f} = 0")
            
                     
            if n > 0:
                min_x, max_x = np.min(contour_points[:, 0]), np.max(contour_points[:, 0])
                min_y, max_y = np.min(contour_points[:, 1]), np.max(contour_points[:, 1])
                logging.info(f"{contour_name}boundary : x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]")
            
            for i in range(n):
                p = contour_points[i]
                q = contour_points[(i + 1) % n]
                
                              
                                                     
                                       
                                                                     
                                                                      
                
                denom = line.a * (q[0] - p[0]) + line.b * (q[1] - p[1])
                if abs(denom) < 1e-10:       
                    continue
                    
                t = -(line.a * p[0] + line.b * p[1] + line.c) / denom
                
                if 0 <= t <= 1:          
                    x = p[0] + t * (q[0] - p[0])
                    y = p[1] + t * (q[1] - p[1])
                    hits.append([x, y])
                    logging.info(f"{contour_name} : {i}-{i+1}, t={t:.3f}, =({x:.2f}, {y:.2f})")
            
            logging.info(f"{contour_name} : {len(hits)}")
            return hits

        inner_hits = intersect_line_with_contour(long_line, inner_contour_points, "contour")
        outer_hits = intersect_line_with_contour(long_line, outer_contour_points, "contour")
        logging.info(f"contour : {len(inner_hits)}, contour : {len(outer_hits)}")


        def get_bbox_intersection(line, contour_points):
            """compute contourboundary （ ）"""
            if len(contour_points) == 0:
                return []
            min_x, max_x = np.min(contour_points[:,0]), np.max(contour_points[:,0])
            min_y, max_y = np.min(contour_points[:,1]), np.max(contour_points[:,1])
            
                     
            bbox_edges = [
                [(min_x, min_y), (max_x, min_y)],     
                [(max_x, min_y), (max_x, max_y)],     
                [(max_x, max_y), (min_x, max_y)],     
                [(min_x, max_y), (min_x, min_y)]      
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
        if len(inner_hits) == 0 or len(outer_hits) == 0:         
            4.0, 3.0, 2.0, None



        inner_pt = pick_nearest(inner_hits, apex_endpoint)
        outer_pt = pick_nearest(outer_hits, apex_endpoint)

                             
        dx = (outer_pt[0] - inner_pt[0]) * spacing_xy[0]
        dy = (outer_pt[1] - inner_pt[1]) * spacing_xy[1]
        apex_thickness = float(np.sqrt(dx * dx + dy * dy))

                          
        apex_thickness_min, apex_thickness_max, apex_thickness_mean = calculate_apex_thickness_stats(
            inner_hits, outer_hits, apex_endpoint, spacing_xy)

                                  
        apex_segment_mask = np.zeros(lv_myocardium_mask.shape, dtype=np.uint8)
                           
        apex_segment_mask = np.ascontiguousarray(apex_segment_mask)
        
                  
        extended_apex_int = (int(round(extended_apex[0])), int(round(extended_apex[1])))
        extended_base_int = (int(round(extended_base[0])), int(round(extended_base[1])))
        cv2.line(apex_segment_mask, extended_apex_int, extended_base_int, 2, 2)
        
                
        original_apex_int = (int(round(apex_endpoint[0])), int(round(apex_endpoint[1])))
        original_base_int = (int(round(base_endpoint[0])), int(round(base_endpoint[1])))
        cv2.line(apex_segment_mask, original_apex_int, original_base_int, 3, 1)
        
                  
        p1_i = (int(round(inner_pt[0])), int(round(inner_pt[1])))
        p2_o = (int(round(outer_pt[0])), int(round(outer_pt[1])))
        cv2.circle(apex_segment_mask, p1_i, 3, 4, -1)         
        cv2.circle(apex_segment_mask, p2_o, 3, 5, -1)         
        
                
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

                                                              
        return apex_thickness_max, apex_thickness_mean, apex_thickness_min, apex_info
        
    except Exception as e:
        logging.error(f"computeapicalthicknessfailed: {e}")
        return 3.0, 3.0, 3.0, None

# def calculate_diameter_from_mask(mask, spacing_xy, target_id=LA_BLOOD_POOL_ID):
#     """
              
#     Args:
                    
                               
                                         
#     Returns:
                                              
#     """
                    
#     binary_mask = (mask == target_id).astype(np.uint8)
    
            
#     contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
#     if not contours:
                                  
#         return -1.0, -1.0
    
               
#     out_cont = contours[0]
#     if len(contours) > 1:
#         area = cv2.contourArea(contours[0])
#         for cont in contours[1:]:  
#             cont_area = cv2.contourArea(cont)
#             if cont_area > area:
#                 area = cont_area
#                 out_cont = cont
    
#     if len(out_cont) == 0:
                                 
#         return 0.0, 0.0
    
              
#     x, y = np.unique(out_cont[:, :, 0]), np.unique(out_cont[:, :, 1])
#     if len(x) == 1 or len(y) == 1:
#         l = (max(x) - min(x) + 1) * spacing_xy[0]  # noqa: E741
#         r = (max(y) - min(y) + 1) * spacing_xy[1]
#         return max(l, r), min(l, r)
    
            
#     cont = np.squeeze(out_cont)
#     st_idx, ed_idx, ok = get_long_diameter_start_end_index(cont)
#     if not ok:
                                   
#         return -1.0, -1.0
    
#     p1 = cont[st_idx]
#     p2 = cont[ed_idx]
#     long_diameter = build_diameter(p1, p2, spacing_xy)
    
            
#     st_idx, ed_idx, ok = get_short_diameter_start_end_index(cont, st_idx, ed_idx)
#     if not ok:
                                   
#         return -1.0, -1.0
    
#     p1 = cont[st_idx]
#     p2 = cont[ed_idx]
#     short_diameter = build_diameter(p1, p2, spacing_xy)
    
#     return long_diameter, short_diameter

                              
def create_base_visualization(mask: np.ndarray) -> np.ndarray:
    """UNTRANSLATED"""
    try:
                               
        if len(mask.shape) == 2:
            vis_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                      
            vis_image[:, :] = (30, 30, 30)
        else:
            vis_image = mask.astype(np.uint8)
                              
            if vis_image.shape[-1] != 3:
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
                          
        vis_image = np.ascontiguousarray(vis_image)
        
                
        colors = {
            LV_BLOOD_POOL_ID: (0, 255, 0),              
            RV_BLOOD_POOL_ID: (255, 0, 0),                
            LA_BLOOD_POOL_ID: (0, 255, 255),            
            RA_BLOOD_POOL_ID: (255, 255, 0)             
        }
        
        for region_id, color in colors.items():
            region_mask = (mask == region_id).astype(np.uint8)
            if np.sum(region_mask) > 0:
                      
                contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                          
                    cv2.drawContours(vis_image, contours, -1, color, 2)
        
        return vis_image
        
    except Exception as e:
        logging.error(f"failed: {e}")
                     
        h, w = mask.shape[:2]
        backup_image = np.zeros((h, w, 3), dtype=np.uint8)
        backup_image[:, :] = (30, 30, 30)
        return backup_image

def find_approximate_junction_region(atrium_mask: np.ndarray, ventricle_mask: np.ndarray) -> Optional[np.ndarray]:
    """： boundary contour"""
    try:
                
        contours, _ = cv2.findContours(atrium_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
                     
        atrium_contour = max(contours, key=cv2.contourArea)
        
                    
        bottom_points = []
        max_y = np.max(atrium_contour[:, 0, 1])
        
        for point in atrium_contour:
            x, y = point[0]
            if y == max_y:
                bottom_points.append((x, y))
        
        if not bottom_points:
            return None
        
                    
        bottom_points.sort(key=lambda p: p[0])
        
               
        center_x = (bottom_points[0][0] + bottom_points[-1][0]) // 2
        center_y = max_y
        
                               
        junction_region = np.zeros_like(atrium_mask)
        
                      
        bottom_width = bottom_points[-1][0] - bottom_points[0][0]
        region_height = 4
        region_width = max(6, bottom_width // 8)                     
        
        y_start = max(0, center_y)
        y_end = min(atrium_mask.shape[0], center_y + region_height)
        x_start = max(0, center_x - region_width // 2)
        x_end = min(atrium_mask.shape[1], center_x + region_width // 2)
        
        if x_start < x_end and y_start < y_end:
            junction_region[y_start:y_end, x_start:x_end] = 1
        
        return junction_region
        
    except Exception as e:
        logging.error(f"failed: {e}")
        return None

def fit_line_ransac(points: np.ndarray, max_iterations: int = 100, threshold: float = 2.0) -> Optional[Tuple[float, float, float]]:
    """RANSAC"""
    try:
        if len(points) < 2:
            return None
        
        best_line = None
        best_inliers = 0
        
        for _ in range(max_iterations):
                     
            idx = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]
            
                    
            if p2[0] - p1[0] == 0:       
                A, B, C = 1.0, 0.0, -p1[0]
            else:
                slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                A = -slope
                B = 1.0
                C = slope * p1[0] - p1[1]
            
                 
            norm = np.sqrt(A*A + B*B)
            if norm > 0:
                A, B, C = A/norm, B/norm, C/norm
            
                  
            distances = np.abs(A * points[:, 0] + B * points[:, 1] + C)
            inliers = np.sum(distances < threshold)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_line = (A, B, C)
        
        return best_line if best_inliers >= len(points) * 0.3 else None
        
    except Exception as e:
        logging.error(f"RANSAC failed: {e}")
        return None

def fit_line_least_squares(points: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """UNTRANSLATED"""
    try:
        if len(points) < 2:
            return None
        
        x = points[:, 0]
        y = points[:, 1]
        
                              
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        
                                 
        A_param = -m
        B_param = 1.0
        C_param = -b
        
             
        norm = np.sqrt(A_param*A_param + B_param*B_param)
        if norm > 0:
            A_param, B_param, C_param = A_param/norm, B_param/norm, C_param/norm
        
        return (A_param, B_param, C_param)
        
    except Exception as e:
        logging.error(f"failed: {e}")
        return None

def find_atrioventricular_junction(atrium_mask: np.ndarray, ventricle_mask: np.ndarray) -> Tuple[Optional[Tuple[float, float, float]], Optional[np.ndarray]]:
    """- boundarycenter"""
    try:
                  
        kernel = np.ones((3, 3), np.uint8)
        atrium_dilated = cv2.dilate(atrium_mask, kernel, iterations=2)
        ventricle_dilated = cv2.dilate(ventricle_mask, kernel, iterations=2)
        
                
        contact_region = atrium_dilated & ventricle_dilated
        
        if np.sum(contact_region) == 0:
                                 
            contact_region = find_approximate_junction_region(atrium_mask, ventricle_mask)
        
        if np.sum(contact_region) == 0:
            return None, None
        
                  
        contact_points = np.where(contact_region > 0)
        if len(contact_points[0]) < 2:
            return None, None
        
                           
        points = np.column_stack((contact_points[1], contact_points[0]))
        
                      
        line_params = fit_line_ransac(points)
        if line_params is None:
                                
            line_params = fit_line_least_squares(points)
        
        return line_params, points
        
    except Exception as e:
        logging.error(f"failed: {e}")
        return None, None

def calculate_diameter_perpendicular_to_line(mask: np.ndarray, line_params: Tuple[float, float, float], spacing_xy: Tuple[float, float]) -> Tuple[float, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """compute Returns: diameter: diameter_points: (p1, p2)"""
    try:
        A, B, C = line_params
        
        points = np.where(mask > 0)
        if len(points[0]) == 0:
            return -1.0, None
        
        coords = np.column_stack((points[1], points[0]))
        
                          
        distances = A * coords[:, 0] + B * coords[:, 1] + C
        
                     
        min_idx = np.argmin(distances)
        max_idx = np.argmax(distances)
        
        p1 = coords[min_idx]
        p2 = coords[max_idx]
        
        dx = (p2[0] - p1[0]) * spacing_xy[0]
        dy = (p2[1] - p1[1]) * spacing_xy[1]
        diameter = np.sqrt(dx*dx + dy*dy)
        
        return diameter, (p1, p2)
        
    except Exception as e:
        logging.error(f"compute failed: {e}")
        return -1.0, None

def calculate_diameter_parallel_to_line(mask: np.ndarray, line_params: Tuple[float, float, float], spacing_xy: Tuple[float, float]) -> Tuple[float, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """compute Returns: diameter: diameter_points: (p1, p2)"""
    try:
        A, B, C = line_params
        
                     
        points = np.where(mask > 0)
        if len(points[0]) == 0:
            return -1.0, None
        
        coords = np.column_stack((points[1], points[0]))
        
                          
        distances = A * coords[:, 0] + B * coords[:, 1] + C
        
                       
        min_idx = np.argmin(np.abs(distances))
        max_idx = np.argmax(np.abs(distances))
        
        p1 = coords[min_idx]
        p2 = coords[max_idx]
        
        dx = (p2[0] - p1[0]) * spacing_xy[0]
        dy = (p2[1] - p1[1]) * spacing_xy[1]
        diameter = np.sqrt(dx*dx + dy*dy)
        
        return diameter, (p1, p2)
        
    except Exception as e:
        logging.error(f"compute failed: {e}")
        return -1.0, None

def calculate_parallel2_diameter(mask: np.ndarray, perpendicular_points: Tuple[np.ndarray, np.ndarray], spacing_xy: Tuple[float, float]) -> Tuple[float, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """computeparallel2 ： center 、 90 ： 1. compute center （ ） 2. compute direction （ 90 ） 3. center direction （ boundary） 4. mask 5. ，compute （ ）"""
    try:
        p1_perp, p2_perp = perpendicular_points
        
                            
        center_x = (p1_perp[0] + p2_perp[0]) / 2.0
        center_y = (p1_perp[1] + p2_perp[1]) / 2.0
        center = np.array([center_x, center_y], dtype=np.float32)
        
                       
        dir_perp_x = p2_perp[0] - p1_perp[0]
        dir_perp_y = p2_perp[1] - p1_perp[1]
        
                                   
                                          
        dir_parallel2_x = dir_perp_y
        dir_parallel2_y = -dir_perp_x
        
                               
        norm = np.sqrt(dir_parallel2_x**2 + dir_parallel2_y**2)
        if norm < 1e-6:                      
            logging.warning("direction ， computeparallel2")
            return -1.0, None
        dir_parallel2 = np.array([dir_parallel2_x, dir_parallel2_y]) / norm
        
                                     
        h, w = mask.shape[:2]
        
                                
        t_min = -1e6            
        t_max = 1e6             
        
                                      
        t_values = []
                 
        if abs(dir_parallel2[0]) > 1e-6:
            t = (0 - center[0]) / dir_parallel2[0]
            y = center[1] + t * dir_parallel2[1]
            if 0 <= y < h:
                t_values.append(t)
                   
        if abs(dir_parallel2[0]) > 1e-6:
            t = (w-1 - center[0]) / dir_parallel2[0]
            y = center[1] + t * dir_parallel2[1]
            if 0 <= y < h:
                t_values.append(t)
                 
        if abs(dir_parallel2[1]) > 1e-6:
            t = (0 - center[1]) / dir_parallel2[1]
            x = center[0] + t * dir_parallel2[0]
            if 0 <= x < w:
                t_values.append(t)
                   
        if abs(dir_parallel2[1]) > 1e-6:
            t = (h-1 - center[1]) / dir_parallel2[1]
            x = center[0] + t * dir_parallel2[0]
            if 0 <= x < w:
                t_values.append(t)
        
                      
        if t_values:
            t_min = min(t_values)
            t_max = max(t_values)
        
                                  
        num_samples = max(int(t_max - t_min) + 1, 100)            
        t_range = np.linspace(t_min, t_max, num_samples)
        line_points = []
        
        for t in t_range:
            x = center[0] + t * dir_parallel2[0]
            y = center[1] + t * dir_parallel2[1]
            x_int = int(round(x))
            y_int = int(round(y))
            
                      
            if 0 <= x_int < w and 0 <= y_int < h:
                line_points.append((x_int, y_int))
        
        if not line_points:
            logging.warning("UNTRANSLATED")
            return -1.0, None
        
                                     
        atrial_line_points = []
        for (x, y) in line_points:
            if mask[y, x] > 0:          
                atrial_line_points.append((x, y))
        
        if len(atrial_line_points) < 2:
            logging.warning(f"2 （ {len(atrial_line_points)} ）")
            return -1.0, None
        
                                  
        atrial_line_points = np.array(atrial_line_points, dtype=np.float32)
        
                             
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
            logging.warning("unable to find valid")
            return -1.0, None
        
        return max_dist, (best_p1, best_p2)
        
    except Exception as e:
        logging.error(f"computeparallel2 failed: {e}")
        return -1.0, None

def draw_atrial_measurement(vis_image: np.ndarray, av_junction_line: Tuple[float, float, float], 
                          junction_points: Optional[np.ndarray], diameter_points: Optional[Tuple[np.ndarray, np.ndarray]], 
                          diameter: float, atrium_id: int, color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """measureresults"""
    try:
                       
        if vis_image is None:
            return None
            
        vis_image = np.ascontiguousarray(vis_image)
        
        if color is None:
                        
            color = (0, 255, 255) if atrium_id == LA_BLOOD_POOL_ID else (255, 255, 0)
        
        A, B, C = av_junction_line
        
                 
        if junction_points is not None:
            for point in junction_points[:50]:                 
                x, y = int(point[0]), int(point[1])
                if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                    cv2.circle(vis_image, (x, y), 2, (255, 255, 255), -1)
        
                 
        h, w = vis_image.shape[:2]
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
        
                 
        if diameter_points is not None:
            p1, p2 = diameter_points
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            
                         
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                
                       
                cv2.line(vis_image, pt1, pt2, color, 3, cv2.LINE_AA)
                
                      
                cv2.circle(vis_image, pt1, 5, color, -1)
                cv2.circle(vis_image, pt2, 5, color, -1)
        
              
        atrium_name = "Left Atrium" if atrium_id == LA_BLOOD_POOL_ID else "Right Atrium"
        legend_text = f"{atrium_name}: {diameter:.1f}mm"
        
                      
        y_position = 30 if atrium_id == LA_BLOOD_POOL_ID else 60
        if y_position < vis_image.shape[0]:
            cv2.putText(vis_image, legend_text, (10, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return vis_image
        
    except Exception as e:
        logging.error(f"measurefailed: {e}")
        return vis_image

def calculate_atrial_diameters_with_visualization(mask: np.ndarray, spacing_xy: Tuple[float, float]) -> Tuple[Dict[str, Any], np.ndarray]:
    """compute ： - perpendicular_diameter: - parallel_diameter: - parallel2_diameter: perpendicular、 center （ 90 ） Args: mask: mask spacing_xy: XYdirection pixel spacing Returns: dict: vis_image:"""
    try:
                   
        vis_image = create_base_visualization(mask)
        
                              
        la_vis_image, la_perpendicular_diameter, la_perpendicular_points, la_parallel_diameter, la_parallel_points, la_parallel2_diameter, la_parallel2_points = calculate_atrial_three_diameters(
            mask, spacing_xy, LA_BLOOD_POOL_ID, LV_BLOOD_POOL_ID, vis_image)
        
                              
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
        
              
        cv2.putText(ra_vis_image, "Atrial Diameter Measurements (3 Directions)", 
                   (10, ra_vis_image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        logging.info(f"- : {la_perpendicular_diameter:.1f}mm, : {la_parallel_diameter:.1f}mm, Parallel2: {la_parallel2_diameter:.1f}mm")
        logging.info(f"- : {ra_perpendicular_diameter:.1f}mm, : {ra_parallel_diameter:.1f}mm, Parallel2: {ra_parallel2_diameter:.1f}mm")
        
        return results, ra_vis_image
        
    except Exception as e:
        logging.error(f"compute failed: {e}")
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
    """compute ： - perpendicular_diameter: - parallel_diameter: - parallel2_diameter: perpendicular、 center （ 90 ） Returns: vis_image: perpendicular_diameter: perpendicular_points: parallel_diameter: parallel_points: parallel2_diameter: parallel2 parallel2_points: parallel2"""
    try:
                   
        atrium_mask = (mask == atrium_id).astype(np.uint8)
        ventricle_mask = (mask == ventricle_id).astype(np.uint8)
        
                 
        if visualization_image is None:
            vis_image = create_base_visualization(mask)
        else:
            vis_image = visualization_image.copy()
        
        if np.sum(atrium_mask) == 0 or np.sum(ventricle_mask) == 0:
            logging.warning(f"ventriclemask : {atrium_id}, ventricle{ventricle_id}")
            return vis_image, -1.0, None, -1.0, None, -1.0, None
        
                 
        av_junction_line, junction_points = find_atrioventricular_junction(atrium_mask, ventricle_mask)
        if av_junction_line is None:
            logging.warning("unable to find")
            return vis_image, -1.0, None, -1.0, None, -1.0, None
        
                     
        perpendicular_diameter, perpendicular_points = calculate_diameter_perpendicular_to_line(
            atrium_mask, av_junction_line, spacing_xy)
        
                     
        parallel_diameter, parallel_points = calculate_diameter_parallel_to_line(
            atrium_mask, av_junction_line, spacing_xy)
        
                                     
        parallel2_diameter = -1.0
        parallel2_points = None
        if perpendicular_points is not None:
            parallel2_diameter, parallel2_points = calculate_parallel2_diameter(
                atrium_mask, perpendicular_points, spacing_xy)
        
                       
        vis_image = draw_atrial_three_measurements(vis_image, av_junction_line, junction_points,
                                                 perpendicular_points, perpendicular_diameter,
                                                 parallel_points, parallel_diameter,
                                                 parallel2_points, parallel2_diameter,
                                                 atrium_id)
        
        return vis_image, perpendicular_diameter, perpendicular_points, parallel_diameter, parallel_points, parallel2_diameter, parallel2_points
        
    except Exception as e:
        logging.error(f"compute failed: {e}")
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
    """measureresults（ parallel2 90 ）"""
    try:
                       
        if vis_image is None:
            return None
            
        vis_image = np.ascontiguousarray(vis_image)
        
                                 
        if atrium_id == LA_BLOOD_POOL_ID:
            perpendicular_color = (0, 200, 255)               
            parallel_color = (0, 150, 255)                    
            parallel2_color = (255, 0, 255)                                
        else:
            perpendicular_color = (255, 200, 0)               
            parallel_color = (255, 150, 0)                    
            parallel2_color = (0, 255, 150)                                 
        
        A, B, C = av_junction_line
        h, w = vis_image.shape[:2]
        
                 
        if junction_points is not None:
            for point in junction_points[:50]:                 
                x, y = int(point[0]), int(point[1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(vis_image, (x, y), 2, (255, 255, 255), -1)
        
                 
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
        
                                      
        if perpendicular_points is not None:
            p1, p2 = perpendicular_points
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                
                        
                cv2.line(vis_image, pt1, pt2, perpendicular_color, 2, cv2.LINE_AA)
                
                         
                cv2.circle(vis_image, pt1, 4, perpendicular_color, -1)
                cv2.circle(vis_image, pt2, 4, perpendicular_color, -1)
                
                                            
                center_x = (pt1[0] + pt2[0]) // 2
                center_y = (pt1[1] + pt2[1]) // 2
                cv2.circle(vis_image, (center_x, center_y), 5, (255, 255, 255), -1)         
                cv2.circle(vis_image, (center_x, center_y), 7, parallel2_color, 2)         
        
                  
        if parallel_points is not None:
            p1, p2 = parallel_points
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                
                cv2.line(vis_image, pt1, pt2, parallel_color, 2, cv2.LINE_AA)
                cv2.circle(vis_image, pt1, 4, parallel_color, -1)
                cv2.circle(vis_image, pt2, 4, parallel_color, -1)
        
                                      
        if parallel2_points is not None:
            p1, p2 = parallel2_points
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                
                                  
                cv2.line(vis_image, pt1, pt2, parallel2_color, 2, cv2.LINE_AA)
                
                               
                cv2.circle(vis_image, pt1, 4, parallel2_color, -1)
                cv2.circle(vis_image, pt1, 6, (255, 255, 255), 1)
                cv2.circle(vis_image, pt2, 4, parallel2_color, -1)
                cv2.circle(vis_image, pt2, 6, (255, 255, 255), 1)
        
                          
        atrium_name = "Left Atrium" if atrium_id == LA_BLOOD_POOL_ID else "Right Atrium"
        y_offset = 30 if atrium_id == LA_BLOOD_POOL_ID else 90
        
               
        cv2.putText(vis_image, f"{atrium_name} - Perpendicular ( / ): {perpendicular_diameter:.1f}mm", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, perpendicular_color, 2)
               
        cv2.putText(vis_image, f"Parallel ( / ): {parallel_diameter:.1f}mm", 
                   (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, parallel_color, 2)
                                
        cv2.putText(vis_image, f"Parallel2 (90 / center ): {parallel2_diameter:.1f}mm", 
                   (10, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, parallel2_color, 2)
        
        return vis_image
        
    except Exception as e:
        logging.error(f"measurefailed: {e}")
        return vis_image

def visualize_atrial_measurements(mask: np.ndarray, results: Dict[str, Any], visualization: np.ndarray) -> None:
    """measureresults（ ， parallel2 ）"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
                 
        axes[0].imshow(mask, cmap='tab10')
        axes[0].set_title('Cardiac Segmentation Mask')
        axes[0].axis('off')
        
              
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
            if id_val in legend_labels and id_val > 0:        
                color = plt.cm.tab10((id_val % 10) / 10)
                legend_elements.append(plt.Rectangle((0,0),1,1, fc=color))
                legend_labels_list.append(legend_labels[id_val])
        
        if legend_elements:
            axes[0].legend(legend_elements, legend_labels_list,
                          loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
                    
        if visualization is not None:
                                      
            if len(visualization.shape) == 3 and visualization.shape[2] == 3:
                display_image = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
            else:
                display_image = visualization
            
            axes[1].imshow(display_image)
            axes[1].set_title('Atrial Diameter Measurements (3 Directions)')
            
                      
            la_data = results.get('left_atrium', {})
            ra_data = results.get('right_atrium', {})
            
            measurement_text = (
                f'Left Atrium:\n'
                f'  Perpendicular: {la_data.get("perpendicular_diameter_mm", 0):.1f}mm\n'
                f'  Parallel: {la_data.get("parallel_diameter_mm", 0):.1f}mm\n'
                f'Parallel2 (90° / center ): {la_data.get("parallel2_diameter_mm", 0):.1f}mm\n\n'
                f'Right Atrium:\n'
                f'  Perpendicular: {ra_data.get("perpendicular_diameter_mm", 0):.1f}mm\n'
                f'  Parallel: {ra_data.get("parallel_diameter_mm", 0):.1f}mm\n'
                f'Parallel2 (90° / center ): {ra_data.get("parallel2_diameter_mm", 0):.1f}mm'
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
        logging.error(f"failed: {e}")

def analyze_cardiac_chambers_with_visualization(mask: np.ndarray, spacing_xy: Tuple[float, float] = (1.0, 1.0)) -> Dict[str, Any]:
    """（ ）"""
    print("=== measure （ direction）===")
    print(f"maskshape: {mask.shape}")
    print(f"maskdtype: {mask.dtype}")
    print(f"unique values: {np.unique(mask)}")
    
            
    atrial_results, atrial_vis = calculate_atrial_diameters_with_visualization(mask, spacing_xy)
    
           
    # visualize_atrial_measurements(mask, atrial_results, atrial_vis)
    
            
    print("\n=== measureresults ===")
    la = atrial_results['left_atrium']
    ra = atrial_results['right_atrium']
    print(f":")
    print(f"- （ ）: {la['perpendicular_diameter_mm']:.1f}mm")
    print(f"- （ ）: {la['parallel_diameter_mm']:.1f}mm")
    print(f"- Parallel2（ + center ）: {la['parallel2_diameter_mm']:.1f}mm")
    print(f":")
    print(f"- （ ）: {ra['perpendicular_diameter_mm']:.1f}mm")
    print(f"- （ ）: {ra['parallel_diameter_mm']:.1f}mm")
    print(f"- Parallel2（ + center ）: {ra['parallel2_diameter_mm']:.1f}mm")
    
    return atrial_results


def draw_atrial_measurement(vis_image: np.ndarray, av_junction_line: Tuple[float, float, float], 
                          junction_points: Optional[np.ndarray], diameter_points: Optional[Tuple[np.ndarray, np.ndarray]], 
                          diameter: float, atrium_id: int, color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """measureresults"""
    try:
                       
        if vis_image is None:
            return None
            
        vis_image = np.ascontiguousarray(vis_image)
        
        if color is None:
                        
            color = (0, 255, 255) if atrium_id == LA_BLOOD_POOL_ID else (255, 255, 0)
        
        A, B, C = av_junction_line
        
                 
        if junction_points is not None:
            for point in junction_points[:50]:                 
                x, y = int(point[0]), int(point[1])
                if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                    cv2.circle(vis_image, (x, y), 2, (255, 255, 255), -1)
        
                 
        h, w = vis_image.shape[:2]
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
        
                 
        if diameter_points is not None:
            p1, p2 = diameter_points
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            
                         
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                
                       
                cv2.line(vis_image, pt1, pt2, color, 3, cv2.LINE_AA)
                
                      
                cv2.circle(vis_image, pt1, 5, color, -1)
                cv2.circle(vis_image, pt2, 5, color, -1)
        
              
        atrium_name = "Left Atrium" if atrium_id == LA_BLOOD_POOL_ID else "Right Atrium"
        legend_text = f"{atrium_name}: {diameter:.1f}mm"
        
                      
        y_position = 30 if atrium_id == LA_BLOOD_POOL_ID else 60
        if y_position < vis_image.shape[0]:
            cv2.putText(vis_image, legend_text, (10, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return vis_image
        
    except Exception as e:
        logging.error(f"measurefailed: {e}")
        return vis_image

def calculate_atrial_diameters_with_visualization(mask: np.ndarray, spacing_xy: Tuple[float, float]) -> Tuple[Dict[str, Any], np.ndarray]:
    """compute ： - perpendicular_diameter: - parallel_diameter: - parallel2_diameter: perpendicular、 center Args: mask: mask spacing_xy: XYdirection pixel spacing Returns: dict: vis_image:"""
    try:
                   
        vis_image = create_base_visualization(mask)
        
                              
        la_vis_image, la_perpendicular_diameter, la_perpendicular_points, la_parallel_diameter, la_parallel_points, la_parallel2_diameter, la_parallel2_points = calculate_atrial_three_diameters(
            mask, spacing_xy, LA_BLOOD_POOL_ID, LV_BLOOD_POOL_ID, vis_image)
        
                              
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
        
              
        cv2.putText(ra_vis_image, "Atrial Diameter Measurements (3 Directions)", 
                   (10, ra_vis_image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        logging.info(f"- : {la_perpendicular_diameter:.1f}mm, : {la_parallel_diameter:.1f}mm, Parallel2: {la_parallel2_diameter:.1f}mm")
        logging.info(f"- : {ra_perpendicular_diameter:.1f}mm, : {ra_parallel_diameter:.1f}mm, Parallel2: {ra_parallel2_diameter:.1f}mm")
        
        return results, ra_vis_image
        
    except Exception as e:
        logging.error(f"compute failed: {e}")
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
    """compute ： - perpendicular_diameter: - parallel_diameter: - parallel2_diameter: perpendicular、 center Returns: vis_image: perpendicular_diameter: perpendicular_points: parallel_diameter: parallel_points: parallel2_diameter: parallel2 （ ） parallel2_points: parallel2 （ ）"""
    try:
                   
        atrium_mask = (mask == atrium_id).astype(np.uint8)
        ventricle_mask = (mask == ventricle_id).astype(np.uint8)
        
                 
        if visualization_image is None:
            vis_image = create_base_visualization(mask)
        else:
            vis_image = visualization_image.copy()
        
        if np.sum(atrium_mask) == 0 or np.sum(ventricle_mask) == 0:
            logging.warning(f"ventriclemask : {atrium_id}, ventricle{ventricle_id}")
            return vis_image, -1.0, None, -1.0, None, -1.0, None
        
                 
        av_junction_line, junction_points = find_atrioventricular_junction(atrium_mask, ventricle_mask)
        if av_junction_line is None:
            logging.warning("unable to find")
            return vis_image, -1.0, None, -1.0, None, -1.0, None
        
                     
        perpendicular_diameter, perpendicular_points = calculate_diameter_perpendicular_to_line(
            atrium_mask, av_junction_line, spacing_xy)
        
                     
        parallel_diameter, parallel_points = calculate_diameter_parallel_to_line(
            atrium_mask, av_junction_line, spacing_xy)
        
                           
        parallel2_diameter = -1.0
        parallel2_points = None
        if perpendicular_points is not None:
            parallel2_diameter, parallel2_points = calculate_parallel2_diameter(
                atrium_mask, perpendicular_points, spacing_xy)
        
                                 
        vis_image = draw_atrial_three_measurements(vis_image, av_junction_line, junction_points,
                                                 perpendicular_points, perpendicular_diameter,
                                                 parallel_points, parallel_diameter,
                                                 parallel2_points, parallel2_diameter,
                                                 atrium_id)
        
        return vis_image, perpendicular_diameter, perpendicular_points, parallel_diameter, parallel_points, parallel2_diameter, parallel2_points
        
    except Exception as e:
        logging.error(f"compute failed: {e}")
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
    """measureresults（ parallel2）"""
    try:
                       
        if vis_image is None:
            return None
            
        vis_image = np.ascontiguousarray(vis_image)
        
                    
        if atrium_id == LA_BLOOD_POOL_ID:
            perpendicular_color = (0, 200, 255)                       
            parallel_color = (0, 150, 255)                            
            parallel2_color = (255, 0, 255)                                           
        else:
            perpendicular_color = (255, 200, 0)                       
            parallel_color = (255, 150, 0)                            
            parallel2_color = (0, 255, 150)                                            
        
        A, B, C = av_junction_line
        h, w = vis_image.shape[:2]
        
                 
        if junction_points is not None:
            for point in junction_points[:50]:                 
                x, y = int(point[0]), int(point[1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(vis_image, (x, y), 2, (255, 255, 255), -1)
        
                 
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
        
                          
        if perpendicular_points is not None:
            p1, p2 = perpendicular_points
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                
                       
                cv2.line(vis_image, pt1, pt2, perpendicular_color, 2, cv2.LINE_AA)
                
                      
                cv2.circle(vis_image, pt1, 4, perpendicular_color, -1)
                cv2.circle(vis_image, pt2, 4, perpendicular_color, -1)
                
                                       
                center_x = (pt1[0] + pt2[0]) // 2
                center_y = (pt1[1] + pt2[1]) // 2
                cv2.circle(vis_image, (center_x, center_y), 3, (255, 255, 255), -1)         
        
                          
        if parallel_points is not None:
            p1, p2 = parallel_points
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                
                       
                cv2.line(vis_image, pt1, pt2, parallel_color, 2, cv2.LINE_AA)
                
                      
                cv2.circle(vis_image, pt1, 4, parallel_color, -1)
                cv2.circle(vis_image, pt2, 4, parallel_color, -1)
        
                                                    
        if parallel2_points is not None:
            p1, p2 = parallel2_points
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                
                                      
                cv2.line(vis_image, pt1, pt2, parallel2_color, 2, cv2.LINE_AA)
                
                      
                cv2.circle(vis_image, pt1, 4, parallel2_color, -1)
                cv2.circle(vis_image, pt2, 4, parallel2_color, -1)
        
              
        atrium_name = "Left Atrium" if atrium_id == LA_BLOOD_POOL_ID else "Right Atrium"
        
                
        if atrium_id == LA_BLOOD_POOL_ID:
            y_pos1 = 30
            y_pos2 = 60
        else:
            y_pos1 = 90
            y_pos2 = 120
        
                   
        legend_text1 = f"{atrium_name}: Perp={perpendicular_diameter:.1f}mm ( / )"
        legend_text2 = f"Parallel={parallel_diameter:.1f}mm ( / )"
        legend_text3 = f"Parallel2={parallel2_diameter:.1f}mm ( / )"
        
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
        logging.error(f"measurefailed: {e}")
        return vis_image

def visualize_atrial_measurements(mask: np.ndarray, results: Dict[str, Any], visualization: np.ndarray) -> None:
    """measureresults（ ）"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
                 
        axes[0].imshow(mask, cmap='tab10')
        axes[0].set_title('Cardiac Segmentation Mask')
        axes[0].axis('off')
        
              
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
            if id_val in legend_labels and id_val > 0:        
                color = plt.cm.tab10((id_val % 10) / 10)
                legend_elements.append(plt.Rectangle((0,0),1,1, fc=color))
                legend_labels_list.append(legend_labels[id_val])
        
        if legend_elements:
            axes[0].legend(legend_elements, legend_labels_list,
                          loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
                    
        if visualization is not None:
                                      
            if len(visualization.shape) == 3 and visualization.shape[2] == 3:
                                            
                display_image = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
            else:
                display_image = visualization
            
            axes[1].imshow(display_image)
            axes[1].set_title('Atrial Diameter Measurements (3 Directions)')
            
                                      
            la_perpendicular = results.get('left_atrium', {}).get('perpendicular_diameter_mm', 0)
            la_parallel = results.get('left_atrium', {}).get('parallel_diameter_mm', 0)
            la_parallel2 = results.get('left_atrium', {}).get('parallel2_diameter_mm', 0)      
            ra_perpendicular = results.get('right_atrium', {}).get('perpendicular_diameter_mm', 0)
            ra_parallel = results.get('right_atrium', {}).get('parallel_diameter_mm', 0)
            ra_parallel2 = results.get('right_atrium', {}).get('parallel2_diameter_mm', 0)      
            
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
        logging.error(f"failed: {e}")

def analyze_cardiac_chambers_with_visualization(mask: np.ndarray, spacing_xy: Tuple[float, float] = (1.0, 1.0)) -> Dict[str, Any]:
    """（ ）"""
    print("=== measure （ direction）===")
    print(f"maskshape: {mask.shape}")
    print(f"maskdtype: {mask.dtype}")
    print(f"unique values: {np.unique(mask)}")
    
                     
    atrial_results, atrial_vis = calculate_atrial_diameters_with_visualization(mask, spacing_xy)
    
           
    # visualize_atrial_measurements(mask, atrial_results, atrial_vis)
    
                       
    print("\n=== measureresults ===")
    print(f"- : {atrial_results['left_atrium']['perpendicular_diameter_mm']:.1f}mm")
                                                                                        
    print(f"- Parallel2: {atrial_results['left_atrium']['parallel2_diameter_mm']:.1f}mm")
    print(f"- : {atrial_results['right_atrium']['perpendicular_diameter_mm']:.1f}mm")
                                                                                         
    print(f"- Parallel2: {atrial_results['right_atrium']['parallel2_diameter_mm']:.1f}mm")
    
    return atrial_results
# ---------------------------------

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
        
                            
        effective_slice_indices = slice_indices[SKIP_HEAD_SLICES_PER_BLOCK:]
        
        if SKIP_TAIL_SLICES_PER_BLOCK > 0 and len(effective_slice_indices) > SKIP_TAIL_SLICES_PER_BLOCK:
            effective_slice_indices = effective_slice_indices[:-SKIP_TAIL_SLICES_PER_BLOCK]
        
                           
        if len(effective_slice_indices) > MAX_SLICES_PER_BLOCK:
            effective_slice_indices = effective_slice_indices[:MAX_SLICES_PER_BLOCK]
        
        if len(effective_slice_indices) == 0:
            logging.warning(f"block {block_idx} skipslice validslice，skip")
            blocks.append(None)
            continue
            
                     
        effective_indices_in_data = effective_slice_indices
        
        if len(effective_indices_in_data) == 0:
            logging.warning(f"block {block_idx} validslice ，skip")
            blocks.append(None)
            continue
            
        block_data = data[:, :, effective_indices_in_data]
        blocks.append(block_data)
        
        logging.info(f"      block {block_idx}: originalslice {slice_indices[:5]}{'...' if len(slice_indices) > 5 else ''}")
        logging.info(f"validslice {effective_indices_in_data}, shape: {block_data.shape}")
    
    logging.info(f"{len(blocks)} block（ {len([b for b in blocks if b is not None])} validblock）")
    return blocks, original_blocks

def compute_volume_only_with_original_spacing(blk, original_spacing, BLOOD_POOL_ID):
    """originalspacingcomputeLV EDblock"""
    total_lv = 0.0
    for si in range(blk.shape[2]):
        sl = blk[:, :, si]
        lv_vol = np.sum(sl == BLOOD_POOL_ID) * original_spacing[0] * original_spacing[1] * original_spacing[2] / 1000.0
        total_lv += lv_vol
    return total_lv

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

def calculate_cine_4ch_metrics(cine_4ch_mask_path, slice_num):
    """compute4ch （ ）"""
    try:
              
        pred_img = nib.load(cine_4ch_mask_path)
        pred_data = np.round(pred_img.get_fdata()).astype(np.int16)
        pred_data = np.flip(pred_data, axis=1)
                          
        original_spacing = pred_img.header.get_zooms()                       
        logging.info(f"original spacing: {original_spacing}, shape: {pred_data.shape}")
        BLOCK_SIZES = slice_num
        # if pred_data.shape[2] % 30 == 0:
        #     BLOCK_SIZES = 30
        # elif pred_data.shape[2] % 25 == 0:
        #     BLOCK_SIZES = 25
        # else:
        #     BLOCK_SIZES = 30
             
        blocks, original_blocks = create_3d_blocks(pred_data, BLOCK_SIZES)
        
        if blocks is None or original_blocks is None:
            return None
            
                                    
        block_volumes_lv = []
        block_volumes_la = []
        for i, block in enumerate(blocks):
            if block is None:
                continue
                
                                  
            lv_vol = compute_volume_only_with_original_spacing(block, original_spacing, LV_BLOOD_POOL_ID)
            block_volumes_lv.append((i, lv_vol))
            la_vol = compute_volume_only_with_original_spacing(block, original_spacing, LA_BLOOD_POOL_ID)
            block_volumes_la.append((i, la_vol))
        
        if not block_volumes_lv:
            return None

        if not block_volumes_la:
            return None
            
                     
        block_volumes_lv.sort(key=lambda x: x[1], reverse=True)
        ed_idx_lv = block_volumes_lv[0][0]

        block_volumes_la.sort(key=lambda x: x[1], reverse=True)
        ed_idx_la = block_volumes_la[0][0]
        
        logging.info(f"EDblock : {ed_idx_lv}, LV : {block_volumes_lv[0][1]:.2f}ml")
        logging.info(f"EDblock : {ed_idx_la}, LA : {block_volumes_la[0][1]:.2f}ml")
        
                 
        ed_block_original = original_blocks[ed_idx_lv]
        ed_block_original_la = original_blocks[ed_idx_la]
        
                 
        result = {}
        
                      
        if ed_block_original_la is not None and ed_block_original_la.shape[2] > TARGET_SLICE_INDEX:
                                 
            ed_target_slice = ed_block_original_la[:, :, TARGET_SLICE_INDEX]
            
                        
            atrial_results = analyze_cardiac_chambers_with_visualization(
                ed_target_slice, original_spacing[:2]
            )
            result['LA_ED_Long_Diameter'] = atrial_results['left_atrium']['parallel2_diameter_mm'] if atrial_results['left_atrium']['parallel2_diameter_mm'] > 0 else None
            result['RA_ED_Long_Diameter'] = atrial_results['right_atrium']['parallel2_diameter_mm'] if atrial_results['right_atrium']['parallel2_diameter_mm'] > 0 else None
                         
        if ed_block_original is not None and ed_block_original.shape[2] > TARGET_SLICE_INDEX:
            ed_target_slice = ed_block_original[:, :, TARGET_SLICE_INDEX]
            rv_wall_thickness, _ = calculate_rv_wall_thickness_segmented(
                ed_target_slice, original_spacing[:2], RV_WALL_THICKNESS_DIVISIONS
            )
            
            for div_id, div_stats in rv_wall_thickness.items():
                result[f'ED_RV_Wall_Thickness_Div_{div_id}'] = div_stats['thickness_mm']
        
                                   
        if ed_block_original is not None and ed_block_original.shape[2] > APEX_SLICE_INDEX:
            ed_apex_slice = ed_block_original[:, :, APEX_SLICE_INDEX]
            apex_thickness_max, apex_thickness_mean, apex_thickness_min, _ = calculate_apex_thickness(ed_apex_slice, original_spacing[:2])
            result['ED_LV_Apex_Thickness_max'] = apex_thickness_max
            result['ED_LV_Apex_Thickness_mean'] = apex_thickness_mean
            result['ED_LV_Apex_Thickness_min'] = apex_thickness_min
        
        return result
        
    except Exception as e:
        logging.error(f"computecine 4ch failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
                    
    logging.basicConfig(level=logging.INFO)
    
    cine_4ch_mask_path = "00101_pred.nii.gz"
    metrics = calculate_cine_4ch_metrics(cine_4ch_mask_path)
    print("Computation finished. Results:")
    print(metrics)