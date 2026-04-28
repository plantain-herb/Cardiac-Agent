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

BACKGROUND_ID = 0
LV_MYOCARDIUM_ID = 1
LV_BLOOD_POOL_ID = 2
RV_BLOOD_POOL_ID = 3
RV_MYOCARDIUM_ID = 4
MYOCARDIUM_DENSITY = 1.05
ASSUMED_HEART_RATE = 70

MAX_SLICES_PER_BLOCK = 8
SKIP_HEAD_SLICES_PER_BLOCK = 1
SKIP_TAIL_SLICES_PER_BLOCK = 2
TARGET_SLICE_INDEX = 3

SEGMENTATION_DIVISIONS = {'apex': 4, 'mid': 6, 'base': 6}

SEGMENT_NAMES = {
    'apex': {
        1: {'id': 13, 'name': 'Apical_anterior'},
        2: {'id': 14, 'name': 'Apical_lateral'},
        3: {'id': 15, 'name': 'Apical_inferior'},
        4: {'id': 16, 'name': 'Apical_septal'},
    },
    'mid': {
        1: {'id': 7,  'name': 'Mid_anteroseptal'},
        2: {'id': 8,  'name': 'Mid_anterior'},
        3: {'id': 9,  'name': 'Mid_lateral'},
        4: {'id': 10, 'name': 'Mid_posterior'},
        5: {'id': 11, 'name': 'Mid_inferior'},
        6: {'id': 12, 'name': 'Mid_inferoseptal'},
    },
    'base': {
        1: {'id': 1, 'name': 'Basal_anteroseptal'},
        2: {'id': 2, 'name': 'Basal_anterior'},
        3: {'id': 3, 'name': 'Basal_lateral'},
        4: {'id': 4, 'name': 'Basal_posterior'},
        5: {'id': 5, 'name': 'Basal_inferior'},
        6: {'id': 6, 'name': 'Basal_inferoseptal'},
    },
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

def line2d_normal_to_point(l, p):
    return Line2D(-l.b, l.a, l.b * p[0] - l.a * p[1])

def line_intersection(l1, l2):
    if l1.a * l2.b - l2.a * l1.b == 0:
        return [0, 0], False

    x = (l2.c * l1.b - l1.c * l2.b) / (l1.a * l2.b - l2.a * l1.b)
    y = (l2.c * l1.a - l1.c * l2.a) / (l1.b * l2.a - l2.b * l1.a)

    return [x, y], True

def line2d_project_to_point(l, p):
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
        logging.error(f"Failed to obtain long diameter: {e}")
        return -1, -1, False

def get_short_diameter_start_end_index(contour, long_diameter_point_idx1, long_diameter_point_idx2):
    idx1, idx2 = -1, -1
    if len(contour) < 4:
        return idx1, idx2, False
    if (
        long_diameter_point_idx1 < 0
        or long_diameter_point_idx1 >= len(contour)
        or long_diameter_point_idx2 < 0
        or long_diameter_point_idx2 >= len(contour)
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

    global LV_BLOOD_POOL_ID, LV_MYOCARDIUM_ID

    LV_BLOOD_POOL_ID = 2
    LV_MYOCARDIUM_ID = 1

    thickness_max, thickness_mean, thickness_min, thickness_map, message, _ = calculate_thickness_radial_accurate(ed_slice, original_spacing)

    return thickness_max, thickness_mean, thickness_min, thickness_map, message

def calculate_thickness_radial_accurate(mask, spacing_xy, num_angles=36, myo_shrink_pixels=1):

    try:
        blood_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        myo_mask = (mask == LV_MYOCARDIUM_ID).astype(np.uint8)

        myo_mask_for_epi = myo_mask
        if myo_shrink_pixels and myo_shrink_pixels > 0 and np.any(myo_mask) and np.any(blood_mask):
            kernel = np.ones((3, 3), np.uint8)
            heart_disk = ((blood_mask | myo_mask) > 0).astype(np.uint8)
            heart_disk_eroded = cv2.erode(heart_disk, kernel, iterations=int(myo_shrink_pixels))
            candidate = (heart_disk_eroded & (1 - blood_mask)).astype(np.uint8)

            if np.sum(candidate) >= max(8, 0.3 * np.sum(myo_mask)):
                myo_mask_for_epi = candidate

        blood_points = np.where(blood_mask > 0)
        if len(blood_points[0]) == 0:
            return 8.0, 8.0, 8.0, None, "Blood pool not found", []

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

            epi_point = find_boundary_along_ray_accurate(endo_point, direction, myo_mask_for_epi)
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

        if len(filtered_thickness) < 3:

            if 'valid_ts' in locals() and len(valid_ts) >= 3:
                filtered_thickness = valid_ts
            elif len([t for t in thickness_values if lower_bound <= t <= upper_bound]) >= 3:
                filtered_thickness = [t for t in thickness_values if lower_bound <= t <= upper_bound]
            else:

                filtered_thickness = thickness_values.copy()

        if len(filtered_thickness) >= 1:
            max_thickness = float(np.max(filtered_thickness))
            mean_thickness = float(np.mean(filtered_thickness))
            min_thickness = float(np.min(filtered_thickness))
        else:
            max_thickness = None
            mean_thickness = None
            min_thickness = None

        message = f"Radial measurement: {len(thickness_values)}directions, thickness {max_thickness:.2f}mm"

        return float(max_thickness), float(mean_thickness), float(min_thickness), thickness_map, message, measurement_lines

    except Exception as e:
        return 8.0, 8.0, 8.0, None, f"Radial measurement failed: {str(e)}", []

def find_boundary_along_ray_accurate(start_point, direction, mask, max_steps=100):

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

    length_pixels = np.linalg.norm(end_point - start_point)
    num_points = max(2, int(length_pixels))

    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        point = start_point + t * (end_point - start_point)
        x, y = int(round(point[0])), int(round(point[1]))

        if 0 <= y < thickness_map.shape[0] and 0 <= x < thickness_map.shape[1]:
            thickness_map[y, x] = thickness

def calculate_myocardium_thickness_simple(mask, spacing_xy):

    try:
        thickness_max, thickness_mean, thickness_min, thickness_map, message, _ = calculate_thickness_radial_accurate(mask, spacing_xy)
        return thickness_max, thickness_mean, thickness_min, thickness_map, message
    except Exception as e:
        return 0.0, 0.0, 0.0, None, f"Calculation failed: {str(e)}"

def create_slice_segmentation(mask, num_divisions=6, start_angle_degrees=0):

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

        lv_region_mask = (mask == LV_BLOOD_POOL_ID) | (mask == LV_MYOCARDIUM_ID)
        for div in range(num_divisions):
            division_mask = (angles == div) & lv_region_mask
            segmented_mask[division_mask] = div + 1

        return segmented_mask, center

    except Exception as e:
        logging.error(f"Slice equal-division failed: {e}")
        return None, None

def find_rv_insertion_points(mask, lv_centroid):

    try:
        lv_myo = (mask == LV_MYOCARDIUM_ID).astype(np.uint8)
        rv_blood = (mask == RV_BLOOD_POOL_ID).astype(np.uint8)
        if lv_myo.sum() == 0 or rv_blood.sum() < 5:
            return None, None

        kernel = np.ones((3, 3), dtype=np.uint8)
        lv_myo_dil = cv2.dilate(lv_myo, kernel, iterations=1)
        contact = (lv_myo_dil & rv_blood).astype(np.uint8)
        if contact.sum() < 3:
            return None, None

        ys, xs = np.where(contact > 0)
        cx = float(lv_centroid[0])
        cy = float(lv_centroid[1])
        angs = np.arctan2(ys - cy, xs - cx)

        order = np.argsort(angs)
        sorted_angs = angs[order]
        diffs = np.diff(sorted_angs)
        wrap_diff = 2 * np.pi - (sorted_angs[-1] - sorted_angs[0])
        all_diffs = np.concatenate([diffs, [wrap_diff]])
        idx_gap = int(np.argmax(all_diffs))
        if idx_gap == len(sorted_angs) - 1:
            end1_ang = sorted_angs[-1]
            end2_ang = sorted_angs[0]
        else:
            end1_ang = sorted_angs[idx_gap]
            end2_ang = sorted_angs[idx_gap + 1]

        i1 = int(np.argmin(np.abs(angs - end1_ang)))
        i2 = int(np.argmin(np.abs(angs - end2_ang)))
        p1 = (int(xs[i1]), int(ys[i1]))
        p2 = (int(xs[i2]), int(ys[i2]))

        y_tol = 5
        if abs(p1[1] - p2[1]) > y_tol:
            p_ant, p_inf = (p1, p2) if p1[1] < p2[1] else (p2, p1)
        else:
            p_ant, p_inf = (p1, p2) if p1[0] < p2[0] else (p2, p1)

        return p_ant, p_inf
    except Exception as e:
        logging.warning(f"find_rv_insertion_points failed: {e}")
        return None, None

def create_slice_segmentation_aha(mask, lv_centroid, p_ant, p_inf, num_divisions):

    try:
        if p_ant is None or p_inf is None:
            return None, None
        if num_divisions not in (4, 6):
            return None, None

        lv_mask = (mask == LV_BLOOD_POOL_ID) | (mask == LV_MYOCARDIUM_ID)
        if not lv_mask.any():
            return None, None
        rv_blood = (mask == RV_BLOOD_POOL_ID)
        if not rv_blood.any():
            return None, None

        h, w = mask.shape
        cx = float(lv_centroid[0])
        cy = float(lv_centroid[1])

        rv_ys, rv_xs = np.where(rv_blood)
        rv_cx = float(rv_xs.mean())
        rv_cy = float(rv_ys.mean())

        th_ant = math.atan2(p_ant[1] - cy, p_ant[0] - cx)
        th_inf = math.atan2(p_inf[1] - cy, p_inf[0] - cx)
        th_rv  = math.atan2(rv_cy - cy, rv_cx - cx)

        two_pi = 2 * math.pi

        def unwrap_plus(x):
            return (x - th_ant) % two_pi

        u_inf_plus = unwrap_plus(th_inf)
        u_rv_plus  = unwrap_plus(th_rv)

        if u_rv_plus < u_inf_plus:
            sign = -1.0
            u_inf = two_pi - u_inf_plus
        else:
            sign = 1.0
            u_inf = u_inf_plus

        if u_inf < 1e-3 or u_inf > two_pi - 1e-3:
            return None, None

        ys_all, xs_all = np.ogrid[:h, :w]
        pixel_ang = np.arctan2(ys_all - cy, xs_all - cx)
        u_px = (sign * (pixel_ang - th_ant)) % two_pi

        segmented = np.zeros((h, w), dtype=np.int32)
        contact_span = two_pi - u_inf

        if num_divisions == 6:

            free_edges = np.linspace(0.0, u_inf, 5)
            for i in range(4):
                lo, hi = free_edges[i], free_edges[i + 1]
                region = (u_px >= lo) & (u_px < hi) & lv_mask
                segmented[region] = i + 2

            cont_edges = np.linspace(u_inf, two_pi, 3)
            for i in range(2):
                lo, hi = cont_edges[i], cont_edges[i + 1]
                region = (u_px >= lo) & (u_px < hi) & lv_mask
                segmented[region] = 6 if i == 0 else 1
        else:

            free_edges = np.linspace(0.0, u_inf, 4)
            for i in range(3):
                lo, hi = free_edges[i], free_edges[i + 1]
                region = (u_px >= lo) & (u_px < hi) & lv_mask
                segmented[region] = i + 1

            region = (u_px >= u_inf) & lv_mask
            segmented[region] = 4

        return segmented, (int(round(cx)), int(round(cy)))
    except Exception as e:
        logging.error(f"create_slice_segmentation_aha failed: {e}")
        return None, None

def analyze_slice_segments_for_thickness(mask, segmented_mask, spacing_xy, num_divisions):

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
        logging.error(f"Failed to analyze segmented myocardial thicknessfailed: {e}")
        return {}

def save_segment_qc_figure(ed_slice, segmented_mask, lv_centroid, lv_angle_deg,
                           region, num_divisions, segment_name_map, save_path,
                           title_extra: str = "",
                           p_ant=None, p_inf=None, mode_label: str = ""):

    try:
        if ed_slice is None or segmented_mask is None:
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        h, w = ed_slice.shape
        fig, ax = plt.subplots(figsize=(6, 6))

        bg = np.zeros((h, w, 3), dtype=np.float32)
        bg[ed_slice == LV_MYOCARDIUM_ID] = (0.30, 0.80, 0.30)
        bg[ed_slice == LV_BLOOD_POOL_ID] = (0.90, 0.30, 0.30)
        bg[ed_slice == RV_BLOOD_POOL_ID] = (0.30, 0.50, 0.90)
        bg[ed_slice == RV_MYOCARDIUM_ID] = (0.30, 0.75, 0.75)
        ax.imshow(bg)

        try:
            cmap = plt.get_cmap('tab10', num_divisions)
        except Exception:
            cmap = plt.cm.get_cmap('tab10', num_divisions)
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        for div in range(1, num_divisions + 1):
            m = (segmented_mask == div)
            if not m.any():
                continue
            r, g, b, _ = cmap(div - 1)
            overlay[m] = (r, g, b, 0.45)
        ax.imshow(overlay)

        for div in range(1, num_divisions + 1):
            m = (segmented_mask == div)
            if not m.any():
                continue
            ys, xs = np.where(m)
            cy, cx = float(ys.mean()), float(xs.mean())
            info = segment_name_map.get(div, {'id': '?', 'name': f'div{div}'})
            ax.text(
                cx, cy,
                f"div{div}\n#{info['id']}\n{info['name']}",
                color='white', fontsize=7, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.55, boxstyle='round,pad=0.2'),
            )

        lv_cx, lv_cy = float(lv_centroid[0]), float(lv_centroid[1])
        L = 0.45 * min(h, w)

        ang = math.radians(lv_angle_deg)
        end_x = lv_cx + L * math.cos(ang)
        end_y = lv_cy + L * math.sin(ang)
        ax.annotate('', xy=(end_x, end_y), xytext=(lv_cx, lv_cy),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=1.8))
        ax.text(end_x, end_y, 'lv_angle (lateral)',
                color='yellow', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.15'))
        end_x2 = lv_cx - L * math.cos(ang)
        end_y2 = lv_cy - L * math.sin(ang)
        ax.annotate('', xy=(end_x2, end_y2), xytext=(lv_cx, lv_cy),
                    arrowprops=dict(arrowstyle='->', color='cyan', lw=1.2))
        ax.text(end_x2, end_y2, '180° (septal)',
                color='cyan', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.15'))

        if p_ant is not None:
            ax.plot([lv_cx, p_ant[0]], [lv_cy, p_ant[1]],
                    color='lime', lw=2.2, ls='-')
            ax.plot(p_ant[0], p_ant[1], marker='o', color='lime',
                    markersize=11, mec='black', mew=1.3)
            ax.text(p_ant[0] + 3, p_ant[1] - 3, 'ant_ins',
                    color='lime', fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.55,
                              boxstyle='round,pad=0.15'))
        if p_inf is not None:
            ax.plot([lv_cx, p_inf[0]], [lv_cy, p_inf[1]],
                    color='red', lw=2.2, ls='-')
            ax.plot(p_inf[0], p_inf[1], marker='o', color='red',
                    markersize=11, mec='black', mew=1.3)
            ax.text(p_inf[0] + 3, p_inf[1] + 10, 'inf_ins',
                    color='red', fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.55,
                              boxstyle='round,pad=0.15'))

        ax.plot(lv_cx, lv_cy, marker='+', color='white', markersize=12, mew=2)

        mode_str = f" | mode={mode_label}" if mode_label else ""
        title = (f"{region} | num_div={num_divisions} | "
                 f"lv_angle={lv_angle_deg:.1f}°{mode_str}")
        if title_extra:
            title = f"{title_extra}\n{title}"
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        logging.warning(f"Failed to save 17-segment QC figure({save_path}): {e}")
        try:
            plt.close('all')
        except Exception:
            pass

def calculate_diameter_from_mask(mask, spacing_xy, target_id=LV_BLOOD_POOL_ID):

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
        logging.warning("Contour is empty")
        return 0.0, 0.0

    x, y = np.unique(out_cont[:, :, 0]), np.unique(out_cont[:, :, 1])
    if len(x) == 1 or len(y) == 1:
        l = (max(x) - min(x) + 1) * spacing_xy[0]
        r = (max(y) - min(y) + 1) * spacing_xy[1]
        return max(l, r), min(l, r)

    cont = np.squeeze(out_cont)
    st_idx, ed_idx, ok = get_long_diameter_start_end_index(cont)
    if not ok:
        logging.warning("Unable to find long diameter")
        return -1.0, -1.0

    p1 = cont[st_idx]
    p2 = cont[ed_idx]
    long_diameter = build_diameter(p1, p2, spacing_xy)

    st_idx, ed_idx, ok = get_short_diameter_start_end_index(cont, st_idx, ed_idx)
    if not ok:
        logging.warning("Unable to find short diameter")
        return -1.0, -1.0

    p1 = cont[st_idx]
    p2 = cont[ed_idx]
    short_diameter = build_diameter(p1, p2, spacing_xy)

    return long_diameter, short_diameter

def create_3d_blocks(data, num_blocks):

    total_slices = data.shape[2]

    if total_slices < num_blocks:
        logging.warning(f"    Total slices {total_slices} is less than required number of blocks {num_blocks}，cannot split into blocks")
        return None, None

    slices_per_block = total_slices // num_blocks
    if slices_per_block == 0:
        logging.warning(f"    Slices per block is 0，cannot split into blocks")
        return None, None

    logging.info(f"    Total slices: {total_slices}, split into {num_blocks} blocks, each block {slices_per_block} slices")

    blocks = []
    original_blocks = []

    for block_idx in range(num_blocks):
        slice_indices = []
        for i in range(slices_per_block):
            slice_idx = block_idx + i * num_blocks
            if slice_idx < total_slices:
                slice_indices.append(slice_idx)

        if len(slice_indices) == 0:
            logging.warning(f"    blocks {block_idx} has no valid slices, skipped")
            continue

        original_block_data = data[:, :, slice_indices]
        original_blocks.append(original_block_data)

        blocks.append(original_block_data)

        logging.info(
            f"      block {block_idx}: slice_indices={slice_indices[:5]}"
            f"{'...' if len(slice_indices) > 5 else ''}, "
            f"shape={original_block_data.shape}"
        )

    logging.info(f"    Successfully created {len(blocks)}  blocks (including  {len([b for b in blocks if b is not None])} valid blocks)")

    return blocks, original_blocks

def find_interventricular_septum_center_robust(mask):
    try:

        lv_blood_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        rv_blood_mask = (mask == RV_BLOOD_POOL_ID).astype(np.uint8)

        print(f"Left ventricleblood-pool pixels: {np.sum(lv_blood_mask)}")
        print(f"Right ventricleblood-pool pixels: {np.sum(rv_blood_mask)}")

        if np.sum(lv_blood_mask) == 0 or np.sum(rv_blood_mask) == 0:
            print("LV or RV blood pool is empty")
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
        print(f"Failed to find septum center: {e}")
        return None

def find_septum_by_centroid_line(lv_mask, rv_mask):

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

        print(f"LV centroid: {lv_center}")
        print(f"RV centroid: {rv_center}")

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
            print(f"Centroid-line method found septum center: {septum_center} (based on{len(contact_points)}points)")
            return septum_center

        print(f"Using centroid midpoint as septum center: {centers_midpoint}")
        return centers_midpoint

    except Exception as e:
        print(f"Centroid-line method failed: {e}")
        return None

def find_closest_boundary(start_point, direction, mask, max_steps=50):

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

        print(f"Distance-transform method found septum center: {septum_center}")
        return septum_center

    except Exception as e:
        print(f"Distance-transform method failed: {e}")
        return None

def find_ventricular_axis(mask, septum_center):

    try:

        lv_blood_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        rv_blood_mask = (mask == RV_BLOOD_POOL_ID).astype(np.uint8)

        lv_points = np.where(lv_blood_mask > 0)
        rv_points = np.where(rv_blood_mask > 0)

        if len(lv_points[0]) == 0 or len(rv_points[0]) == 0:
            return None

        lv_center = np.array([np.mean(lv_points[1]), np.mean(lv_points[0])])
        rv_center = np.array([np.mean(rv_points[1]), np.mean(rv_points[0])])

        print(f"LV center: {lv_center}")
        print(f"RV center: {rv_center}")

        axis_direction = rv_center - lv_center
        axis_norm = np.linalg.norm(axis_direction)

        if axis_norm > 0:
            axis_direction = axis_direction / axis_norm
        else:
            axis_direction = np.array([1.0, 0.0])

        print(f"Ventricular axis direction: {axis_direction}")

        return axis_direction

    except Exception as e:
        print(f"Failed to find ventricular axis: {e}")
        return None

def measure_ventricle_diameter_along_axis(start_point, direction, blood_mask, spacing_xy, ventricle_name):

    try:

        boundaries = find_ventricle_boundaries_along_axis(start_point, direction, blood_mask, ventricle_name)

        if boundaries is None:
            print(f"{ventricle_name}Boundary search failed, using fallback method")

            points = np.where(blood_mask > 0)
            if len(points[0]) > 0:

                coords = np.column_stack((points[1], points[0]))
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

        print(f"{ventricle_name}inner-diameter measurement: proximal{np.array(near_boundary).astype(int)}, distal{np.array(far_boundary).astype(int)}, length{diameter:.1f}mm")

        return diameter

    except Exception as e:
        print(f"measurement{ventricle_name}inner diameter failed: {e}")
        return 30.0

def find_ventricle_boundaries_along_axis(start_point, direction, blood_mask, ventricle_name):

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
            print(f"Not found{ventricle_name}blood-pool entry point")
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
            print(f"Not found{ventricle_name}blood-pool exit point")
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
                print(f"{ventricle_name}Using second blood-pool segment，length{second_length:.1f} > first segment{first_length:.1f}")
                return (second_entry, second_exit)

        return (first_entry, first_exit)

    except Exception as e:
        print(f"search{ventricle_name}boundary failed: {e}")
        return None

def find_left_ventricle_far_boundary(septum_center, axis_direction, blood_mask, myo_mask, max_steps=300):
    try:

        lv_diameter = measure_ventricle_diameter_along_axis(
            septum_center, -axis_direction, blood_mask, (1.0, 1.0), "Left ventricle")

        boundaries = find_ventricle_boundaries_along_axis(
            septum_center, -axis_direction, blood_mask, "Left ventricle")

        if boundaries is not None:
            return boundaries[1]
        else:

            return septum_center - axis_direction * lv_diameter

    except Exception as e:
        print(f"Failed to find LV distal boundary: {e}")
        return septum_center - axis_direction * 20

def find_right_ventricle_far_boundary(septum_center, axis_direction, blood_mask, max_steps=300):

    try:

        rv_diameter = measure_ventricle_diameter_along_axis(
            septum_center, axis_direction, blood_mask, (1.0, 1.0), "Right ventricle")

        boundaries = find_ventricle_boundaries_along_axis(
            septum_center, axis_direction, blood_mask, "Right ventricle")

        if boundaries is not None:
            return boundaries[1]
        else:

            return septum_center + axis_direction * rv_diameter

    except Exception as e:
        print(f"Failed to find RV distal boundary: {e}")
        return septum_center + axis_direction * 20

def calculate_angle_with_xaxis(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]

    line_len = math.sqrt(dx * dx + dy * dy)
    if line_len < 1e-6:
        return 0.0

    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)

def create_visualization(mask, septum_center, lv_boundary, rv_boundary, lv_diameter, rv_diameter, axis_direction):

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
        print(f"Failed to create visualization: {e}")
        return None, None

def visualize_results(mask, results, visualization):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(mask, cmap='tab10')
    axes[0].set_title('Cardiac segmentation mask')
    axes[0].axis('off')

    if visualization is not None:
        axes[1].imshow(visualization)
        axes[1].set_title('Ventricle inner-diameter measurement')
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'Measurement failed', ha='center', va='center',
                    transform=axes[1].transAxes)
        axes[1].set_title('Ventricle inner-diameter measurement')
        axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def calculate_ventricular_diameters_robust(mask, spacing_xy):
    try:

        septum_center = find_interventricular_septum_center_robust(mask)
        if septum_center is None:
            print("All methods failed to find septum center; using fallback method")
            septum_center = np.array([mask.shape[1] // 2, mask.shape[0] // 2])
            print(f"Using image center as septum center: {septum_center}")

        print(f"Septum center: {septum_center}")

        axis_direction = find_ventricular_axis(mask, septum_center)
        if axis_direction is None:
            print("Ventricular axis not found, using default direction")
            axis_direction = np.array([1.0, 0.0])

        print(f"Ventricular axis direction: {axis_direction}")

        lv_blood_mask = (mask == LV_BLOOD_POOL_ID).astype(np.uint8)
        rv_blood_mask = (mask == RV_BLOOD_POOL_ID).astype(np.uint8)
        lv_myo_mask = (mask == LV_MYOCARDIUM_ID).astype(np.uint8)

        lv_diameter = measure_ventricle_diameter_along_axis(
            septum_center, -axis_direction, lv_blood_mask, spacing_xy, "Left ventricle")

        rv_diameter = measure_ventricle_diameter_along_axis(
            septum_center, axis_direction, rv_blood_mask, spacing_xy, "Right ventricle")

        lv_far_boundary = find_left_ventricle_far_boundary(
            septum_center, axis_direction, lv_blood_mask, lv_myo_mask)

        rv_far_boundary = find_right_ventricle_far_boundary(
            septum_center, axis_direction, rv_blood_mask)

        print(f"LV inner diameter: {lv_diameter:.2f}mm")
        print(f"RV inner diameter: {rv_diameter:.2f}mm")

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
        print(f"Failed to calculate ventricular inner diameters: {e}")
        import traceback
        traceback.print_exc()

        default_results = {
            'left_ventricle': {'transverse_diameter_mm': 0.0},
            'right_ventricle': {'transverse_diameter_mm': 0.0}
        }
        return default_results, None

def analyze_ventricular_dimensions(mask, spacing_xy):
    print("=== Ventricular Inner-Diameter Measurement (Revised)===")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask data type: {mask.dtype}")
    print(f"Unique values: {np.unique(mask)}")

    results, visualization = calculate_ventricular_diameters_robust(mask, spacing_xy)

    if results and 'left_ventricle' in results and 'right_ventricle' in results:
        print("\n=== Final measurement results ===")
        print(f"LV inner diameter: {results['left_ventricle']['transverse_diameter_mm']:.1f}mm")
        print(f"RV inner diameter: {results['right_ventricle']['transverse_diameter_mm']:.1f}mm")
        print(f"lv_angle: {results['lv_angle']['lv_angle']:.1f}mm")

    else:
        print("Measurement failed, returning default values")

        if 'left_ventricle' not in results:
            results = {
                'left_ventricle': {'transverse_diameter_mm': 0.0},
                'right_ventricle': {'transverse_diameter_mm': 0.0}
            }

    return results

def process_block(blk, img_blk=None, block_type=""):

    if blk is None:
        return None, None, None, None

    original_shape = blk.shape
    logging.info(f"    {block_type}blocksoriginal shape: {original_shape}")

    processed_blk = blk.copy()
    processed_img = img_blk.copy() if img_blk is not None else None

    non_zero = np.where(processed_blk > 0)
    if len(non_zero[0]) == 0:
        logging.warning(f"    {block_type}blockshas no non-zero pixels, skipped")
        return None, None, None, None

    y_min = max(0, np.min(non_zero[0]) - CROP_MARGIN)
    y_max = min(processed_blk.shape[0] - 1, np.max(non_zero[0]) + CROP_MARGIN)
    x_min = max(0, np.min(non_zero[1]) - CROP_MARGIN)
    x_max = min(processed_blk.shape[1] - 1, np.max(non_zero[1]) + CROP_MARGIN)

    processed_blk = processed_blk[y_min:y_max+1, x_min:x_max+1, :]
    if processed_img is not None:
        processed_img = processed_img[y_min:y_max+1, x_min:x_max+1, :]

    crop_shape = processed_blk.shape
    logging.info(f"    {block_type}blockscropped shape: {crop_shape}")

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

    logging.info(f"    {block_type}blocksresized shape: {final_shape}, final spacing: {final_spacing}")

    return processed_blk, processed_img, final_spacing, {
        'original_shape': original_shape,
        'crop_shape': crop_shape,
        'final_shape': final_shape,
        'scale_factors': scale_factors
    }

def calculate_cine_sa_metrics(cine_sa_mask_path, slice_num, qc_save_dir=None):

    try:

        pred_img = nib.load(cine_sa_mask_path)
        pred_data = np.round(pred_img.get_fdata()).astype(np.int16)
        pred_data = np.flip(pred_data, axis=1)

        original_spacing = pred_img.header.get_zooms()
        logging.info(f"Original image spacing(zz compressed by phase): {original_spacing}")

        BLOCK_SIZES = slice_num

        is_z_compressed = (
            pred_data.shape[2] > BLOCK_SIZES
            and pred_data.shape[2] % BLOCK_SIZES == 0
            and original_spacing[2] < 2.0
        )
        if is_z_compressed:
            z_spacing_phys = original_spacing[2] * BLOCK_SIZES
            logging.info(
                f"Detected z dimension compressed by phase: header z-spacing={original_spacing[2]:.4f}mm, "
                f"Recovered single-phase slice spacing z_spacing_phys={z_spacing_phys:.4f}mm"
            )
        else:
            z_spacing_phys = original_spacing[2]
            logging.info(
                f"z dimension is not compressed, using header z-spacing={z_spacing_phys:.4f}mm"
            )

        blocks, original_blocks = create_3d_blocks(pred_data, BLOCK_SIZES)

        if blocks is None or original_blocks is None:
            return None
        lv_block_volumes = []
        rv_block_volumes = []
        voxel_volume_ml = original_spacing[0] * original_spacing[1] * z_spacing_phys / 1000.0

        for i, block in enumerate(blocks):
            if block is None:
                continue

            has_lv = np.any(block == LV_BLOOD_POOL_ID, axis=(0, 1))
            lv_valid_z = np.where(has_lv)[0]
            if len(lv_valid_z) > 0:
                lv_voxels = int(np.sum(block[..., lv_valid_z] == LV_BLOOD_POOL_ID))
                lv_vol = lv_voxels * voxel_volume_ml
                lv_block_volumes.append((i, lv_vol, lv_valid_z))

            has_rv = np.any(block == RV_BLOOD_POOL_ID, axis=(0, 1))
            rv_valid_z = np.where(has_rv)[0]
            if len(rv_valid_z) > 0:
                rv_voxels = int(np.sum(block[..., rv_valid_z] == RV_BLOOD_POOL_ID))
                rv_vol = rv_voxels * voxel_volume_ml
                rv_block_volumes.append((i, rv_vol, rv_valid_z))

        if not lv_block_volumes:
            logging.warning("All blocks have no LV blood pool; cannot compute LV metrics")
            return None

        lv_block_volumes.sort(key=lambda x: x[1], reverse=True)
        lv_ed_idx, lv_edv_raw, lv_ed_z = lv_block_volumes[0]
        lv_es_idx, lv_esv_raw, lv_es_z = lv_block_volumes[-1]

        if rv_block_volumes:
            rv_block_volumes.sort(key=lambda x: x[1], reverse=True)
            rv_ed_idx, rv_edv_raw, rv_ed_z = rv_block_volumes[0]
            rv_es_idx, rv_esv_raw, rv_es_z = rv_block_volumes[-1]
        else:
            logging.warning("All blocks have no RV blood pool; RV metrics will be set to null")
            rv_ed_idx = rv_es_idx = None
            rv_edv_raw = rv_esv_raw = 0.0
            rv_ed_z = rv_es_z = np.array([], dtype=int)

        print(
            f"[ED/ES Selection] LV: ED=block{lv_ed_idx}({lv_edv_raw:.2f}mL), "
            f"ES=block{lv_es_idx}({lv_esv_raw:.2f}mL) | "
            f"RV: ED=block{rv_ed_idx}({rv_edv_raw:.2f}mL), "
            f"ES=block{rv_es_idx}({rv_esv_raw:.2f}mL) | "
            f"total blocks={len(blocks)}, "
            f"per-block slices={original_blocks[0].shape[2] if original_blocks else '?'}"
        )

        print("    [blocks LV vol ml]: " + ", ".join(
            f"b{b[0]}={b[1]:.2f}" for b in sorted(lv_block_volumes)
        ))
        if rv_block_volumes:
            print("    [blocks RV vol ml]: " + ", ".join(
                f"b{b[0]}={b[1]:.2f}" for b in sorted(rv_block_volumes)
            ))

        metrics = {}

        lv_ed_block = original_blocks[lv_ed_idx][..., lv_ed_z]
        lv_es_block = original_blocks[lv_es_idx][..., lv_es_z]

        ed_block_original = lv_ed_block
        es_block_original = lv_es_block
        ed_block_original_nocrop = original_blocks[lv_ed_idx]

        ed_lv_voxels = int(np.sum(lv_ed_block == LV_BLOOD_POOL_ID))
        ed_lv_myo_voxels = int(np.sum(lv_ed_block == LV_MYOCARDIUM_ID))
        es_lv_voxels = int(np.sum(lv_es_block == LV_BLOOD_POOL_ID))

        if rv_ed_idx is not None:
            rv_ed_block = original_blocks[rv_ed_idx][..., rv_ed_z]
            rv_es_block = original_blocks[rv_es_idx][..., rv_es_z]
            ed_rv_voxels = int(np.sum(rv_ed_block == RV_BLOOD_POOL_ID))
            es_rv_voxels = int(np.sum(rv_es_block == RV_BLOOD_POOL_ID))
        else:
            ed_rv_voxels = 0
            es_rv_voxels = 0

        voxel_volume_ml_mass = voxel_volume_ml
        metrics['LV_EDV'] = ed_lv_voxels * voxel_volume_ml
        metrics['LV_ESV'] = es_lv_voxels * voxel_volume_ml
        metrics['LV_SV'] = metrics['LV_EDV'] - metrics['LV_ESV']
        metrics['LV_EF'] = (metrics['LV_SV'] / metrics['LV_EDV'] * 100) if metrics['LV_EDV'] > 0 else 0
        metrics['LV_CO'] = metrics['LV_SV'] * ASSUMED_HEART_RATE / 1000.0
        metrics['LV_Mass'] = ed_lv_myo_voxels * voxel_volume_ml_mass * MYOCARDIUM_DENSITY

        metrics['RV_EDV'] = ed_rv_voxels * voxel_volume_ml
        metrics['RV_ESV'] = es_rv_voxels * voxel_volume_ml
        metrics['RV_SV'] = metrics['RV_EDV'] - metrics['RV_ESV']
        metrics['RV_EF'] = (metrics['RV_SV'] / metrics['RV_EDV'] * 100) if metrics['RV_EDV'] > 0 else 0
        metrics['RV_CO'] = metrics['RV_SV'] * ASSUMED_HEART_RATE / 1000.0

        print(
            f"[Result] LV_EDV={metrics['LV_EDV']:.1f} LV_ESV={metrics['LV_ESV']:.1f} "
            f"LV_EF={metrics['LV_EF']:.1f}% LV_Mass={metrics['LV_Mass']:.1f}g | "
            f"RV_EDV={metrics['RV_EDV']:.1f} RV_ESV={metrics['RV_ESV']:.1f} "
            f"RV_EF={metrics['RV_EF']:.1f}%"
        )

        if ed_block_original is not None:
            target_slice = ed_block_original.shape[2] // 2
            ed_target_slice = ed_block_original[:, :, target_slice]
            results = analyze_ventricular_dimensions(ed_target_slice, original_spacing[:2])
            print(f"Left ventricletransverse diameter: {results['left_ventricle']['transverse_diameter_mm']:.1f}mm")
            print(f"Right ventricletransverse diameter: {results['right_ventricle']['transverse_diameter_mm']:.1f}mm")
            metrics['LV_ED_Long_Diameter'] = results['left_ventricle']['transverse_diameter_mm']
            metrics['RV_ED_Long_Diameter'] = results['right_ventricle']['transverse_diameter_mm']
            lv_angle = results['lv_angle']['lv_angle']

        has_lv_in_slice = np.any(ed_block_original_nocrop == LV_MYOCARDIUM_ID, axis=(0, 1))
        valid_z_indices = np.where(has_lv_in_slice)[0]
        n_valid = len(valid_z_indices)
        if n_valid >= 3:
            z_apex = int(valid_z_indices[int(round(0.15 * (n_valid - 1)))])
            z_mid  = int(valid_z_indices[(n_valid - 1) // 2])
            z_base = int(valid_z_indices[int(round(0.85 * (n_valid - 1)))])
            slice_plan = [('apex', z_apex), ('mid', z_mid), ('base', z_base)]
        elif n_valid > 0:

            slice_plan = [('mid', int(valid_z_indices[n_valid // 2]))]
        else:
            slice_plan = []
        logging.info(f"17-segment representative slices (slice_plan): {slice_plan}")

        SEPTAL_SEGMENT_IDS = {1, 6, 7, 12}
        septal_compensation_mm = float(original_spacing[0])

        _case_id = os.path.basename(cine_sa_mask_path)
        for _suffix in ('.nii.gz', '.nii'):
            if _case_id.endswith(_suffix):
                _case_id = _case_id[: -len(_suffix)]
                break

        for region, slice_idx in slice_plan:
            num_divisions = SEGMENTATION_DIVISIONS[region]

            if ed_block_original_nocrop is None or ed_block_original_nocrop.shape[2] <= slice_idx:
                continue
            ed_slice = ed_block_original_nocrop[:, :, slice_idx]

            ed_thickness_max, ed_thickness_mean, ed_thickness_min, _, _ = test_thickness_calculation(
                ed_slice, original_spacing[:2]
            )

            lv_blood_mask = (ed_slice == LV_BLOOD_POOL_ID).astype(np.uint8)
            lv_centroid_xy = None
            if lv_blood_mask.sum() > 0:
                M = cv2.moments(lv_blood_mask)
                if M['m00'] > 0:
                    lv_centroid_xy = (M['m10'] / M['m00'], M['m01'] / M['m00'])

            p_ant, p_inf = (None, None)
            ed_segmented_mask = None
            ed_center = None
            mode_label = 'fallback_lv_angle'

            if lv_centroid_xy is not None:
                p_ant, p_inf = find_rv_insertion_points(ed_slice, lv_centroid_xy)
                if p_ant is not None and p_inf is not None:
                    ed_segmented_mask, ed_center = create_slice_segmentation_aha(
                        ed_slice, lv_centroid_xy, p_ant, p_inf, num_divisions
                    )
                    if ed_segmented_mask is not None:
                        mode_label = 'aha'

            if ed_segmented_mask is None:
                start_angle = (45.0 if region == 'apex' else 0.0) + lv_angle
                ed_segmented_mask, ed_center = create_slice_segmentation(
                    ed_slice, num_divisions, start_angle
                )
                if ed_segmented_mask is None:
                    continue
            else:
                start_angle = None

            if qc_save_dir:
                qc_path = os.path.join(
                    qc_save_dir, f"{_case_id}_{region}_z{slice_idx}.png"
                )
                title_extra = f"{_case_id}"
                if start_angle is not None:
                    title_extra += f" | start_angle={start_angle:.1f}°"
                save_segment_qc_figure(
                    ed_slice=ed_slice,
                    segmented_mask=ed_segmented_mask,
                    lv_centroid=ed_center,
                    lv_angle_deg=float(lv_angle),
                    region=region,
                    num_divisions=num_divisions,
                    segment_name_map=SEGMENT_NAMES[region],
                    save_path=qc_path,
                    title_extra=title_extra,
                    p_ant=p_ant,
                    p_inf=p_inf,
                    mode_label=mode_label,
                )

            ed_segment_stats = analyze_slice_segments_for_thickness(
                ed_slice, ed_segmented_mask, original_spacing[:2], num_divisions
            )

            metrics[f'ED_Slice_{region}_Thickness_max']  = ed_thickness_max
            metrics[f'ED_Slice_{region}_Thickness_mean'] = ed_thickness_mean
            metrics[f'ED_Slice_{region}_Thickness_min']  = ed_thickness_min

            region_name_map = SEGMENT_NAMES[region]
            for div_id, div_stats in ed_segment_stats.items():
                segment_info = region_name_map.get(div_id)
                if segment_info is None:
                    continue
                segment_id = segment_info['id']
                segment_name = segment_info['name']

                thickness_max = div_stats['thickness_mm_max']
                thickness_mean = div_stats['thickness_mm_mean']
                thickness_min = div_stats['thickness_mm_min']

                if segment_id in SEPTAL_SEGMENT_IDS:
                    if isinstance(thickness_max, (int, float)):
                        thickness_max += septal_compensation_mm
                    if isinstance(thickness_mean, (int, float)):
                        thickness_mean += septal_compensation_mm
                    if isinstance(thickness_min, (int, float)):
                        thickness_min += septal_compensation_mm

                metrics[f'ED_Segment_{segment_id:02d}_{segment_name}_Thickness_max']  = thickness_max
                metrics[f'ED_Segment_{segment_id:02d}_{segment_name}_Thickness_mean'] = thickness_mean
                metrics[f'ED_Segment_{segment_id:02d}_{segment_name}_Thickness_min']  = thickness_min

        return metrics

    except Exception as e:
        logging.error(f"Failed to calculate key metrics for cine SA image: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    cine_sa_mask_path = "0000375_sa_pred.nii.gz"
    metrics = calculate_cine_sa_metrics(cine_sa_mask_path)
    print("Calculation completed, results:")
    print(metrics)
