"""DICOM / NIfTI 处理、抽帧、分割可视化工具"""

import os
import re
import shutil
import tempfile
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from PIL import Image

from app.config import (
    CACHE_FRAMES_DIR,
    CACHE_RESULTS_DIR,
    VERSION,
)


# ============ 基础编码 ============

def encode_image_to_base64(image_path: str) -> str:
    import base64
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============ DICOM 加载 ============

def load_scans(dcm_path: str) -> Tuple:
    """加载医学影像数据"""
    if dcm_path.endswith(".nii.gz") or dcm_path.endswith(".nii"):
        print(f"[load_scans] 加载 NIfTI 文件: {dcm_path}")
        sitk_img = sitk.ReadImage(dcm_path)
        slice_num = None
    else:
        print(f"[load_scans] 加载 DICOM 目录: {dcm_path}")

        if not os.path.isdir(dcm_path):
            raise ValueError(f"路径不是目录: {dcm_path}")

        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcm_path)

        if not series_ids:
            print(f"[load_scans] 目录内容:")
            for item in os.listdir(dcm_path)[:20]:
                item_path = os.path.join(dcm_path, item)
                if os.path.isfile(item_path):
                    print(f"  [FILE] {item} ({os.path.getsize(item_path)} bytes)")
                else:
                    print(f"  [DIR] {item}/")
            raise ValueError(f"在 {dcm_path} 中未找到 DICOM 文件")

        if len(series_ids) == 1:
            dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_path, series_ids[0])
        else:
            # 收集所有 series 的文件
            all_files_per_series = {
                sid: sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_path, sid)
                for sid in series_ids
            }
            max_files = max(len(v) for v in all_files_per_series.values())
            if max_files == 1:
                # 每个 Series 只有 1 个文件（每帧独立 Series UID），合并所有文件
                print(f"[load_scans] 检测到 {len(series_ids)} 个独立 Series（每 Series 1 个文件），合并所有文件")
                dicom_names = [f for files in all_files_per_series.values() for f in files]
            else:
                # 选取文件数最多的 series
                best_sid = max(all_files_per_series, key=lambda sid: len(all_files_per_series[sid]))
                dicom_names = all_files_per_series[best_sid]
                print(f"[load_scans] 多 Series，选取文件数最多的 Series ({len(dicom_names)} 个文件)")

        if not dicom_names:
            raise ValueError(f"在 {dcm_path} 中未找到 DICOM 文件")

        print(f"[load_scans] 找到 {len(dicom_names)} 个 DICOM 文件")

        dicom_files_with_position = []
        cardiac_number_of_images = None
        target_keys = ["0018|1090", "CardiacNumberOfImages", "Cardiac Nr of Images"]

        for dicom_name in dicom_names:
            img = sitk.ReadImage(dicom_name)
            try:
                position = img.GetMetaData('0020|0032').split('\\')[-1]
            except:
                position = "0"
            try:
                if cardiac_number_of_images is None:
                    for key in target_keys:
                        try:
                            cardiac_number_of_images = img.GetMetaData(key)
                            break
                        except:
                            continue
            except:
                pass
            dicom_files_with_position.append((float(position), dicom_name))

        dicom_files_with_position.sort(key=lambda x: x[0])
        sorted_dicom_names = [item[1] for item in dicom_files_with_position]

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sorted_dicom_names)
        sitk_img = reader.Execute()

        print(f"[load_scans] 图像尺寸: {sitk_img.GetSize()}")

        if cardiac_number_of_images is not None:
            try:
                slice_num = int(cardiac_number_of_images)
            except:
                slice_num = None
        else:
            slice_num = None

    return sitk_img, slice_num


def infer_slice_num_from_volume(volume: np.ndarray,
                                min_phase: int = 5,
                                max_phase: int = 50) -> Optional[int]:
    """
    从 3D volume 自动推断 slice_num（每个心动周期的时间帧数）。

    原理: cine MRI 的 z 轴按 [slice_0_t0 … slice_0_tN, slice_1_t0 … slice_1_tN, …]
    排列。同一 spatial slice 内相邻帧相似（MSE 低），不同 slice 之间跳变大（MSE 高）。

    两条推断路径:
      A) peaks >= 2 → 用峰值间距中位数得到 candidate
      B) peaks == 1 → 用 num_blocks = 2, slice_num = total / 2 或 peak_pos + 1

    最后通过周期优化 + 整除验证 确定最终 slice_num。

    Args:
        volume: 形状 [total_frames, H, W] 的 numpy 数组
        min_phase: 最小可接受的 phase 数
        max_phase: 最大可接受的 phase 数

    Returns:
        推断出的 slice_num，无法推断时返回 None
    """
    total = volume.shape[0]
    if total < min_phase:
        return None

    # 空间降采样加速（只需要检测跳变，不需要全分辨率）
    h, w = volume.shape[1], volume.shape[2]
    step = max(1, min(h, w) // 64)
    vol = volume[:, ::step, ::step].astype(np.float64)

    # --- Step 1: 计算相邻帧 MSE ---
    diffs = np.array([
        np.mean((vol[i] - vol[i + 1]) ** 2) for i in range(total - 1)
    ])

    if diffs.max() - diffs.min() < 1e-10:
        return None

    # --- Step 2: 检测峰值（空间跳变点） ---
    # 真实空间跳变通常比时间帧差异大 50-1000 倍，用对比度比值法检测
    max_diff = diffs.max()
    med_diff = np.median(diffs)
    contrast = max_diff / (med_diff + 1e-10)

    if contrast < 10:
        # 最大值和中位数差距不到 10 倍，没有明显空间跳变
        return None

    # 阈值取 max 的 10%：能过滤掉时间帧差异，只留真正的空间跳变
    threshold = max_diff * 0.1
    peaks = np.where(diffs > threshold)[0]

    if len(peaks) == 0:
        return None

    # --- Step 3: 由峰值得到候选 slice_num ---
    if len(peaks) >= 2:
        # 路径 A: 多个峰 → 用间距中位数
        intervals = np.diff(peaks)
        candidate = int(np.round(np.median(intervals)))
    else:
        # 路径 B: 单峰 → num_blocks = 2, slice_num ≈ peak_pos + 1
        candidate = peaks[0] + 1
        # 也可以用 total / 2 来交叉验证
        alt = total // 2
        if abs(alt - candidate) <= 2 and total % alt == 0:
            candidate = alt

    if candidate < min_phase or candidate > max_phase:
        return None

    # --- Step 4: 周期优化 ---
    # 在 candidate 附近搜索使 "跳变帧 MSE / 非跳变帧 MSE" 最大的周期
    best_period = candidate
    best_ratio = 0.0

    lo = max(min_phase, candidate - 3)
    hi = min(max_phase + 1, candidate + 4)
    for p in range(lo, hi):
        trans_idx = np.arange(p - 1, len(diffs), p)
        non_trans_mask = np.ones(len(diffs), dtype=bool)
        non_trans_mask[trans_idx] = False

        if len(trans_idx) < 1 or non_trans_mask.sum() < 1:
            continue

        ratio = np.mean(diffs[trans_idx]) / (np.mean(diffs[non_trans_mask]) + 1e-10)
        if ratio > best_ratio:
            best_ratio = ratio
            best_period = p

    if best_ratio < 2.0:
        return None

    # --- Step 5: 优先选择能整除 total_frames 的周期 ---
    if total % best_period != 0:
        for delta in range(1, 4):
            for p in [best_period + delta, best_period - delta]:
                if p < min_phase or p > max_phase:
                    continue
                if total % p == 0:
                    t_idx = np.arange(p - 1, len(diffs), p)
                    nt_mask = np.ones(len(diffs), dtype=bool)
                    nt_mask[t_idx] = False
                    if len(t_idx) >= 1 and nt_mask.sum() >= 1:
                        r = np.mean(diffs[t_idx]) / (np.mean(diffs[nt_mask]) + 1e-10)
                        if r > best_ratio * 0.8:
                            best_period = p
                            break
            else:
                continue
            break

    num_blocks = total // best_period if best_period > 0 else 0
    print(f"[infer_slice_num] total={total}, peaks={len(peaks)}, "
          f"candidate={candidate}, best_period={best_period}, "
          f"num_blocks={num_blocks}, ratio={best_ratio:.2f}")
    return best_period


def get_slice_num_from_path(image_path: str, default_slice_num: int = None) -> int:
    if image_path is None:
        return default_slice_num if default_slice_num is not None else 1

    if image_path.endswith(".nii.gz") or image_path.endswith(".nii"):
        # NII 文件没有 DICOM 元数据，尝试从 volume 数据自动推断
        total_frames = None
        try:
            sitk_img = sitk.ReadImage(image_path)
            volume = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
            total_frames = volume.shape[0]
            inferred = infer_slice_num_from_volume(volume)
            if inferred is not None:
                print(f"[get_slice_num_from_path] NII 自动推断 slice_num={inferred}: "
                      f"{os.path.basename(image_path)}")
                return inferred
        except Exception as e:
            print(f"[get_slice_num_from_path] 自动推断失败: {e}")

        # 推断失败 → fallback: 尝试用常见 phase 数（25, 30, 20）试除
        if total_frames is not None and total_frames > 1:
            common_phases = [25, 30, 20, 15, 35]
            for cp in common_phases:
                if total_frames % cp == 0 and total_frames // cp >= 2:
                    print(f"[get_slice_num_from_path] 跳变检测失败, "
                          f"fallback 试除: total={total_frames} / {cp} = "
                          f"{total_frames // cp} blocks: "
                          f"{os.path.basename(image_path)}")
                    return cp

            # 都不能整除 → 视为单 slice, slice_num = total_frames
            # phase_processing 的 total_layers = 1 → 自动跳过
            print(f"[get_slice_num_from_path] 跳变检测失败且无法试除, "
                  f"fallback slice_num=total_frames={total_frames}: "
                  f"{os.path.basename(image_path)}")
            return total_frames
        return default_slice_num if default_slice_num is not None else 1
    else:
        try:
            _, slice_num = load_scans(image_path)
            if slice_num is not None:
                return slice_num
        except:
            pass
        return default_slice_num or 1


def convert_dcm_to_nifti(dcm_path: str, output_dir: str = None, volume_name: str = None) -> str:
    """将 DICOM 目录转换为 NIfTI (.nii.gz) 格式"""
    print(f"[convert_dcm_to_nifti] 转换 DICOM 到 NIfTI: {dcm_path}")

    sitk_img, slice_num = load_scans(dcm_path)

    if output_dir is None:
        output_dir = os.path.dirname(dcm_path)
    os.makedirs(output_dir, exist_ok=True)

    if volume_name is None:
        volume_name = os.path.basename(dcm_path)

    output_path = os.path.join(output_dir, f"{volume_name}.nii.gz")
    sitk.WriteImage(sitk_img, output_path)
    print(f"[convert_dcm_to_nifti] 已保存: {output_path}")

    return output_path


# ============ 抽帧 ============

def extract_frames_from_volume(volume_path: str, frame_indices: List[int] = None,
                               slice_num: int = None, num_frames: int = 3,
                               session_id: str = None, save_to_cache: bool = False) -> Tuple[List[str], Dict]:
    """从 3D volume 中提取帧并保存为 PNG 图像"""
    import random

    if volume_path.endswith(".nii.gz") or volume_path.endswith(".nii"):
        sitk_img = sitk.ReadImage(volume_path)
    else:
        sitk_img, _ = load_scans(volume_path)

    volume = sitk.GetArrayFromImage(sitk_img)
    total_frames = volume.shape[0]

    if frame_indices is None:
        if slice_num is not None and slice_num > 0 and total_frames > slice_num:
            frames_per_group = total_frames // slice_num
            if frames_per_group > 0:
                group_idx = random.randint(0, slice_num - 1)
                group_frames = [group_idx + j * slice_num for j in range(frames_per_group)
                               if group_idx + j * slice_num < total_frames]
                if len(group_frames) >= num_frames:
                    frame_indices = sorted(random.sample(group_frames, num_frames))
                else:
                    frame_indices = group_frames
            else:
                frame_indices = [int(total_frames * i / (num_frames + 1))
                                for i in range(1, num_frames + 1)]
        else:
            frame_indices = [int(total_frames * i / (num_frames + 1))
                            for i in range(1, num_frames + 1)]
        frame_indices = [max(0, min(i, total_frames - 1)) for i in frame_indices]

    if save_to_cache and session_id:
        save_dir = os.path.join(CACHE_FRAMES_DIR, session_id)
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None

    temp_files = []
    frame_info = {
        "volume_path": volume_path,
        "volume_name": os.path.basename(volume_path),
        "total_frames": total_frames,
        "extracted_indices": frame_indices,
        "frame_files": [],
    }

    for i, idx in enumerate(frame_indices):
        frame = volume[idx]
        if len(frame.shape) == 3:
            frame = frame[frame.shape[0] // 2]

        frame = frame.astype(np.float32)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255
        frame = frame.astype(np.uint8)

        img = Image.fromarray(frame)
        img_rgb = img.convert('RGB')

        if save_dir:
            volume_name = os.path.basename(volume_path).replace('.', '_')
            filename = f"{volume_name}_frame{idx:04d}.png"
            save_path = os.path.join(save_dir, filename)
        else:
            save_path = tempfile.mktemp(suffix=".png")

        img_rgb.save(save_path)
        temp_files.append(save_path)
        frame_info["frame_files"].append({
            "path": save_path,
            "frame_index": idx,
            "filename": os.path.basename(save_path),
        })

    return temp_files, frame_info


def extract_frames_from_volume_simple(volume_path: str, frame_indices: List[int] = None,
                                      slice_num: int = None, num_frames: int = 3) -> List[str]:
    """简化版抽帧函数，兼容旧接口"""
    files, _ = extract_frames_from_volume(
        volume_path, frame_indices, slice_num, num_frames,
        session_id=None, save_to_cache=False
    )
    return files


def cleanup_temp_files(files: List[str]):
    """清理临时文件"""
    for f in files:
        if f and os.path.exists(f):
            try:
                os.unlink(f)
            except:
                pass


# ============ 分割可视化 ============

def save_segmentation_images(
    original_volume_path: str,
    seg_nii_path: str,
    session_id: str,
    frame_indices: List[int] = None,
    num_frames: int = 3
) -> dict:
    """保存分割结果的可视化图片"""
    if original_volume_path.endswith(".nii.gz") or original_volume_path.endswith(".nii"):
        sitk_img = sitk.ReadImage(original_volume_path)
    else:
        sitk_img, _ = load_scans(original_volume_path)

    original_volume = sitk.GetArrayFromImage(sitk_img)

    if not os.path.exists(seg_nii_path):
        return {"error": f"分割结果文件不存在: {seg_nii_path}"}

    seg_img = sitk.ReadImage(seg_nii_path)
    seg_volume = sitk.GetArrayFromImage(seg_img)

    total_frames = original_volume.shape[0]

    if frame_indices is None:
        frame_indices = [int(total_frames * i / (num_frames + 1))
                        for i in range(1, num_frames + 1)]
        frame_indices = [max(0, min(i, total_frames - 1)) for i in frame_indices]

    save_dir = os.path.join(CACHE_RESULTS_DIR, session_id, "segmentation")
    os.makedirs(save_dir, exist_ok=True)

    colors = {
        1: [255, 0, 0],
        2: [0, 255, 0],
        3: [0, 0, 255],
        4: [255, 255, 0],
        5: [255, 0, 255],
        6: [0, 255, 255],
    }
    alpha = 0.5

    volume_name = os.path.basename(original_volume_path).replace('.', '_')
    seg_files = []

    for i, idx in enumerate(frame_indices):
        orig_frame = original_volume[idx]
        if len(orig_frame.shape) == 3:
            orig_frame = orig_frame[orig_frame.shape[0] // 2]

        orig_frame = orig_frame.astype(np.float32)
        orig_frame = (orig_frame - orig_frame.min()) / (orig_frame.max() - orig_frame.min() + 1e-8) * 255
        orig_frame = orig_frame.astype(np.uint8)

        img_rgb = np.stack([orig_frame, orig_frame, orig_frame], axis=-1).astype(np.float32)

        if idx < seg_volume.shape[0]:
            seg_frame = seg_volume[idx]
            if len(seg_frame.shape) == 3:
                seg_frame = seg_frame[seg_frame.shape[0] // 2]

            for label, color in colors.items():
                mask = (seg_frame == label)
                if np.any(mask):
                    for c in range(3):
                        img_rgb[:, :, c][mask] = (
                            img_rgb[:, :, c][mask] * (1 - alpha) + color[c] * alpha
                        )

        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

        filename = f"{volume_name}_seg_frame{idx:04d}.png"
        save_path = os.path.join(save_dir, filename)

        img = Image.fromarray(img_rgb)
        img.save(save_path)

        seg_files.append({
            "path": save_path,
            "frame_index": idx,
            "filename": filename,
            "url": f"/cache/results/{VERSION}/{session_id}/segmentation/{filename}",
        })

    return {
        "seg_nii_path": seg_nii_path,
        "total_frames": total_frames,
        "extracted_indices": frame_indices,
        "seg_files": seg_files,
    }


# ============ ZIP / DICOM 目录查找 ============

def extract_zip_file(zip_path: str, extract_dir: str = None) -> str:
    """解压 zip 文件，返回包含 DICOM 文件的目录路径"""
    if extract_dir is None:
        extract_dir = tempfile.mkdtemp(prefix="dcm_")

    try:
        print(f"[extract_zip_file] 解压 {zip_path} 到 {extract_dir}")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print(f"[extract_zip_file] zip 包含 {len(file_list)} 个文件/目录")
            if len(file_list) <= 10:
                for f in file_list:
                    print(f"  - {f}")
            else:
                for f in file_list[:5]:
                    print(f"  - {f}")
                print(f"  ... 还有 {len(file_list) - 5} 个文件")

            zip_ref.extractall(extract_dir)

        print(f"[extract_zip_file] 解压完成，目录内容:")
        for item in os.listdir(extract_dir)[:10]:
            item_path = os.path.join(extract_dir, item)
            if os.path.isdir(item_path):
                sub_count = len(os.listdir(item_path))
                print(f"  [DIR] {item}/ ({sub_count} 项)")
            else:
                print(f"  [FILE] {item}")

        dcm_dir = find_dcm_directory(extract_dir)

        if dcm_dir:
            print(f"[extract_zip_file] 找到 DICOM 目录: {dcm_dir}")
        else:
            print(f"[extract_zip_file] 未找到 DICOM 目录，返回根目录: {extract_dir}")
            dcm_dir = extract_dir

        return dcm_dir

    except Exception as e:
        print(f"[extract_zip_file] 解压 zip 文件失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def find_dcm_directory(root_dir: str) -> Optional[str]:
    """递归查找包含 DICOM 序列的目录"""
    try:
        dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root_dir)
        if dicom_names and len(dicom_names) > 0:
            print(f"  [find_dcm_directory] 在 {root_dir} 找到 {len(dicom_names)} 个 DICOM 文件")
            return root_dir
    except:
        pass

    try:
        nii_files = [f for f in os.listdir(root_dir)
                     if f.endswith('.nii.gz') or f.endswith('.nii')]
        if nii_files:
            nii_path = os.path.join(root_dir, nii_files[0])
            print(f"  [find_dcm_directory] 找到 NIfTI 文件: {nii_path}")
            return nii_path
    except:
        pass

    try:
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path):
                if item.startswith('__') or item.startswith('.'):
                    continue
                result = find_dcm_directory(item_path)
                if result:
                    return result
    except Exception as e:
        print(f"  [find_dcm_directory] 遍历目录出错: {e}")

    return None


def cleanup_extracted_dir(dir_path: str):
    """清理解压的临时目录"""
    if dir_path and os.path.exists(dir_path):
        try:
            while dir_path and not dir_path.startswith(tempfile.gettempdir()):
                parent = os.path.dirname(dir_path)
                if parent == dir_path:
                    break
                if parent.startswith(tempfile.gettempdir()):
                    dir_path = parent
                    break
                dir_path = parent

            if dir_path.startswith(tempfile.gettempdir()):
                shutil.rmtree(dir_path, ignore_errors=True)
        except:
            pass


# ============ 文件名工具 ============

def get_clean_seg_name(original_name: str, suffix: str = "_seg") -> str:
    """根据原始上传文件名生成干净的分割 mask 文件名"""
    if not original_name:
        return f"unknown{suffix}.nii.gz"

    name = original_name
    for ext in [".nii.gz", ".nii", ".zip", ".dcm"]:
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break

    name = re.sub(r'[^\w\-.]', '_', name).strip('_')
    if not name:
        name = "unknown"

    return f"{name}{suffix}.nii.gz"


def enrich_metrics(merged: Dict, metrics_4ch_raw: Dict = None, metrics_sa_raw: Dict = None) -> Dict:
    """将 MRG Worker 合并后的 metrics 补充完整"""
    enriched = dict(merged)
    return enriched
