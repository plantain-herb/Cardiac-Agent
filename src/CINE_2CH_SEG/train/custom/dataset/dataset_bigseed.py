"""data loader."""

import os
import random
from typing import List, Union

import numpy as np
import SimpleITK as sitk
import torch
from custom.dataset.registry import DATASETS
from custom.dataset.utils import DefaultSampleDataset

def load_nii(nii_path):
    tmp_img = sitk.ReadImage(nii_path)
    data_np = sitk.GetArrayFromImage(tmp_img)
    return data_np

@DATASETS.register_module()
class HeartPyramid_Sample_Dataset(DefaultSampleDataset):
    def __init__(
        self,
        dst_list_file,
        data_root,
        patch_size,
        patch_size_inner,
        isotropy_spacing,
        rotation_prob,
        rot_range,
        spacing_range,
        shift_range,
        data_pyramid_level,
        data_pyramid_step,
        sample_frequent,
        constant_shift=15,
        bg_sample_ratio=0.05,
        whole_bright_aug=(0.5, 0.2, 0.2),
        local_tp_bright_aug=(0.5, -0.10, 0.30),
    ):
        # self.duplicate_list_file = duplicate_list_file
        self._sample_frequent = sample_frequent
        self._patch_size = np.array(patch_size)
        self._patch_size_inner = np.array(patch_size_inner)
        self._isotropy_spacing = np.array(isotropy_spacing)
        self._rotation_prob = rotation_prob
        self._rot_range = rot_range
        self._spacing_range = spacing_range
        self._shift_range = shift_range
        self._data_pyramid_level = data_pyramid_level
        self._data_pyramid_step = data_pyramid_step
        self._constant_shift = constant_shift
        self._bg_sample_ratio = bg_sample_ratio
        self._whole_bright_aug = whole_bright_aug
        self._local_tp_bright_aug = local_tp_bright_aug
        self._data_file_list = self._load_file_list(dst_list_file, data_root)

    def _load_file_list(self, dst_list_file, data_root):
        data_file_list = []
        with open(dst_list_file, "r") as f:
            for line in f:
                line = line.strip()
                file_name = line
                file_name = os.path.join(data_root, file_name)

                if not os.path.exists(file_name):
                    print(f"{line} not exist")
                    continue
                data_file_list.append(file_name)
        data_file_list = data_file_list * 2
        random.shuffle(data_file_list)
        assert len(data_file_list) != 0, "has no avilable file in dst_list_file"
        return data_file_list

    def _filter_points(self, points, mask):
        res = []
        for p in points:
            if mask[p[0], p[1], p[2]] > 0:
                res.append(p)
        return np.array(res)

    def _load_source_data(self, file_name):

        data = np.load(file_name, allow_pickle=True)
        result = {}
        with torch.no_grad():
            vol = data["vol"]
            shape = np.array(vol.shape)

            vol = torch.from_numpy(vol).float()[None, None]
            src_spacing = data["src_spacing"].copy()
            seg = torch.from_numpy(data["mask"]).float()[None, None]

            result['vol'] = vol.detach()
            result['seg'] = seg.detach()
            result['src_spacing'] = src_spacing
            result['points_tp'] = data["tp_points"].copy()
            result['src_shape'] = shape
            result['bbox'] = data['bbox'].copy()
            result['data_type'] = 1 - data["data_type"].copy()
            del data
            del seg

        return result

    def _sample_source_data(self, idx, source_data_info):
        info, _source_data = source_data_info
        vol, seg, src_spacing, src_shape, bbox, points_tp = (
            _source_data["vol"],
            _source_data["seg"],
            _source_data["src_spacing"],
            _source_data["src_shape"],
            _source_data["bbox"],
            _source_data["points_tp"],
        )
        c_t = self._get_random_crop_center(src_shape, src_spacing, bbox, points_tp)
        vol, seg = self._crop_data(vol, seg, c_t, src_shape, src_spacing)
        results = {
            "img": vol.detach(),
            "mask": seg.detach(),
        }
        return results

    def _bright_aug_param_generate(self):
        global_param = (0.0, 0.0)
        global_flag = False
        local_flag = False
        if random.random() <= self._whole_bright_aug[0]:
            global_param = (
                random.uniform(-self._whole_bright_aug[1], self._whole_bright_aug[1]),
                random.uniform(-self._whole_bright_aug[2], self._whole_bright_aug[2]),
            )
            global_flag = True

        local_param = 0.0
        if random.random() <= self._local_tp_bright_aug[0]:
            local_param = random.uniform(self._local_tp_bright_aug[1], self._local_tp_bright_aug[2])
            local_flag = True

        return global_flag, local_flag, global_param, local_param

    def _bright_aug_apply(self, vol, tp_seg, bright_param):
        global_flag, local_flag, global_param, local_param = bright_param
        if global_flag:
            vol = vol * (1 + global_param[0]) + global_param[1]
        if local_flag:
            vol = vol * (1 - tp_seg) + tp_seg * (vol + local_param)
        if global_flag or local_flag:
            vol = torch.clamp(vol, min=0.0, max=1.0)
        return vol

    def _get_random_crop_center(self, vol_shape, src_spacing, bbox, points_tp):
        # bbox = [zmin, zmax, ymin, ymax, xmin, xmax]
        if random.random() < 0.8:
            c_t = points_tp[np.random.choice(len(points_tp))]
            c_t = c_t + np.random.randint(-10, 10, size=(3,))
            c_t = c_t * src_spacing
        else:
            z_center = random.randint(int(bbox[0]), int(bbox[1]))
            y_center = random.randint(int(bbox[2]), int(bbox[3]))
            x_center = random.randint(int(bbox[4]), min(int(bbox[5]) + 40, vol_shape[2]))
            c_t = np.array([z_center, y_center, x_center])
            c_t = c_t * src_spacing
        return c_t

    def _get_rotate_mat(self, z_angle, y_angle, x_angle):
        def _create_matrix_rotation_z_3d(angle, matrix=None):
            rotation_x = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)],])
            if matrix is None:
                return rotation_x
            return np.dot(matrix, rotation_x)

        def _create_matrix_rotation_y_3d(angle, matrix=None):
            rotation_y = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)],])
            if matrix is None:
                return rotation_y

            return np.dot(matrix, rotation_y)

        def _create_matrix_rotation_x_3d(angle, matrix=None):
            rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1],])
            if matrix is None:
                return rotation_z

            return np.dot(matrix, rotation_z)

        rot_matrix = np.identity(3)
        rot_matrix = _create_matrix_rotation_z_3d(z_angle, rot_matrix)
        rot_matrix = _create_matrix_rotation_y_3d(y_angle, rot_matrix)
        rot_matrix = _create_matrix_rotation_x_3d(x_angle, rot_matrix)
        return rot_matrix

    def _rotate_aug(self, rot_range):
        z_angle = (np.random.random() * 2 - 1) * rot_range[0] * np.pi / 180.0
        y_angle = (np.random.random() * 2 - 1) * rot_range[1] * np.pi / 180.0
        x_angle = (np.random.random() * 2 - 1) * rot_range[2] * np.pi / 180.0

        rot_mat = self._get_rotate_mat(z_angle, y_angle, x_angle)
        return rot_mat

    def _get_sample_grid(
        self, center_point, half_patch_size, src_spacing, src_shape, tgt_spacing, rot_mat=None,
    ):
        grid = []
        for cent_px, ts, ps_half in zip(center_point, tgt_spacing, half_patch_size):
            p_s = cent_px - ps_half * ts
            p_e = cent_px + ps_half * ts - (ts / 2)
            grid.append(np.arange(p_s, p_e, ts))
        grid = np.meshgrid(*grid)
        grid = [g[:, :, :, None] for g in grid]  # shape (h,d,w,(zyx))
        grid = np.concatenate(grid, axis=-1)
        grid = np.transpose(grid, axes=(1, 0, 2, 3))  # shape (d,h,w,(zyx))

        if rot_mat is not None:
            grid -= center_point[None, None, None, :]
            grid = np.matmul(grid, np.linalg.inv(rot_mat))
            grid += center_point[None, None, None, :]
        grid *= 2
        grid /= src_spacing[None, None, None, :]
        grid /= (src_shape - 1)[None, None, None, :]
        grid -= 1
        # change z,y,x to x,y,z
        grid = np.array(grid[:, :, :, ::-1], dtype=np.float32)
        return torch.from_numpy(grid)[None]

    def _crop_data(self, vol, seg, c_t, src_shape, src_spacing):
        half_patch_size = [v // 2 for v in self._patch_size]

        rot_mat = None
        if random.random() <= self._rotation_prob:
            rot_mat = self._rotate_aug(self._rot_range)

        data_pyramid = []
        tgt_spacing = self._isotropy_spacing + (random.random() * 2 - 1) * self._spacing_range
        for level in range(self._data_pyramid_level):
            grid = self._get_sample_grid(c_t, half_patch_size, src_spacing, src_shape, tgt_spacing, rot_mat)
            vol_patch = torch.nn.functional.grid_sample(
                vol, grid, mode="bilinear", align_corners=True, padding_mode="border"
            )[0]
            vol_patch = self._normalization(vol_patch)
            seg_patch = torch.nn.functional.grid_sample(seg, grid, mode="nearest", padding_mode="border", align_corners=True)[0]
            data_pyramid.append((vol_patch, seg_patch))
            tgt_spacing = tgt_spacing * self._data_pyramid_step

        data_pyramid = data_pyramid[::-1]
        vol = [d[0] for d in data_pyramid]
        vol = torch.cat(vol, dim=0)

        seg = [d[1] for d in data_pyramid]
        seg = torch.cat(seg, dim=0)

        
        return vol, seg

    def _resize_torch(self, data, scale):
        return torch.nn.functional.interpolate(data, size=scale, mode="trilinear")

    def _window_array(self, vol):
        win = [
            self._win_level - self._win_width / 2,
            self._win_level + self._win_width / 2,
        ]
        vol = torch.clamp(vol, win[0], win[1])
        vol -= win[0]
        vol /= self._win_width
        return vol
    def _normalization(self, vol):
        hu_max = torch.max(vol)
        hu_min = torch.min(vol)
        vol_normalized = (vol - hu_min) / (hu_max - hu_min + 1e-8)
        return vol_normalized

    def __getitem__(self, idx):
        source_data = self._load_source_data(self._data_file_list[idx])
        return [None, source_data]

    @property
    def sampled_data_count(self):
        # TODO: sample后数据总数量
        return self._sample_frequent * self.source_data_count

    @property
    def source_data_count(self):
        # TODO: 原始数据总数量
        return len(self._data_file_list)

    def __len__(self):
        return self.source_data_count

    def sample_source_data(self, idx, source_data):
        sample = None
        if idx < self._sample_frequent:
            sample = self._sample_source_data(idx, source_data)
        return sample
