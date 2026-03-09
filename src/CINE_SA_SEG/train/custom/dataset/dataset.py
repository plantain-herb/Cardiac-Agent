"""data loader."""

import os
import random
import numpy as np
import SimpleITK as sitk
from custom.dataset.utils import build_pipelines
from custom.dataset.registry import DATASETS
from custom.dataset.utils import DefaultSampleDataset
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import torch

def load_nii(nii_path):
    tmp_img = sitk.ReadImage(nii_path)
    data_np = sitk.GetArrayFromImage(tmp_img)
    return data_np


# SampleDataset适用于3D分割场景，一般场景请使用CustomDataset
@DATASETS.register_module()
class Seg_Sample_Dataset(DefaultSampleDataset):
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
        sample_frequent,
    ):
        self._sample_frequent = sample_frequent
        self._patch_size = patch_size
        self._patch_size_inner = patch_size_inner
        self._isotropy_spacing = np.array([isotropy_spacing] * 3)
        self._rotation_prob = rotation_prob
        self._rot_range = rot_range
        self._spacing_range = spacing_range
        self._data_file_list = self._load_file_list(dst_list_file, data_root)

    def _load_file_list(self, dst_list_file, data_root):
        data_file_list = []
        with open(dst_list_file, "r") as f:
            for line in f:
                line = line.strip().split('/')[-1]
                path = os.path.join(data_root, line)
                if os.path.exists(path):
                    if 'CS020003-032537-136178-8' in path or 'CS020003-626829-116922-9' in path:
                        data_file_list.extend([path] * 3)
                    else:
                        data_file_list.append(path)
        random.shuffle(data_file_list)
        assert len(data_file_list) != 0, "has avilable file in dst_list_file"
        return data_file_list


    def _load_source_data(self, filename):
        data = np.load(filename, allow_pickle=True)
        result = {}

        vol = torch.from_numpy(data["vol"]).float()
        seg = torch.from_numpy(data["mask"]).float()
        spacing = data["src_spacing"].copy()

        heart_pos = np.argwhere(data["mask"])
        heart_pos_min = np.min(heart_pos, axis=0)
        heart_pos_max = np.max(heart_pos, axis=0)
        heart_box = np.concatenate([heart_pos_min, heart_pos_max], axis=0)

        result["vol"] = vol[None, None].detach()
        result["seg"] = seg[None, None].detach()
        result["spacing"] = spacing
        result["heart_box"] = heart_box
        result["vol_shape"] = np.array(vol.size())

        del data
        return result

    def _sample_source_data(self, idx, source_data_info):
        info, _source_data = source_data_info
        vol, seg, src_spacing, heart_box, vol_shape = (
            _source_data["vol"],
            _source_data["seg"],
            _source_data["spacing"],
            _source_data["heart_box"],
            _source_data["vol_shape"],
        )
        
        c_t = self._get_random_crop_center(vol_shape,  src_spacing)
        vol, seg = self._crop_data(vol, seg, c_t, vol_shape, src_spacing)
        # data_type = np.array([data_type])
        # data_type = torch.from_numpy(data_type).float()
        #print("============", vol.shape)  1 * 128 * 128 * 128
        # vol = torch.nn.functional.interpolate(vol, size=self._patch_size, mode="trilinear")[0]
        # vol = self._window_array(vol)
        # seg = torch.nn.functional.interpolate(seg, size=self._patch_size, mode="nearest")[0]
        # if True:
        #     import time
        #     s = str(time.time())
        #     os.makedirs('./debug', exist_ok=True)
        #     sitk.WriteImage(sitk.GetImageFromArray(vol[0].numpy()), './debug/' + s + '.nii.gz')
        #     sitk.WriteImage(sitk.GetImageFromArray(seg[0].numpy()), './debug/' + s + '-seg.nii.gz')

        results = {"img": vol.detach(), "mask": seg.detach(),}
        return results

    def _get_aug(self, vol, seg):
        mask = seg[0].clone()
        mask = binary_erosion(mask, iterations=2)
        mask = torch.from_numpy(mask)
        vol[0] = torch.where(mask.bool(), vol[0] - 80, vol[0])
        return vol

    def _get_random_crop_center(self, vol_shape, src_spacing):
        vol_phy_shape = vol_shape * src_spacing
        c_t = [random.random() * v for v in vol_phy_shape]
        c_t = np.array(c_t)
        return c_t

    def _crop_data(self, vol, seg, c_t, src_shape, src_spacing):
        half_patch_size = [v // 2 for v in self._patch_size]

        rot_mat = None
        if random.random() <= self._rotation_prob:
            rot_mat = self._rotate_aug(self._rot_range)

        # tgt_spacing = np.max((src_shape * src_spacing) / self._patch_size)
        # tgt_spacing = tgt_spacing + random.random() * self._spacing_range
        # tgt_spacing = np.array([tgt_spacing] * 3) 

        tgt_spacing = self._isotropy_spacing + (random.random() * 2 - 1) * self._spacing_range
        grid = self._get_sample_grid(c_t, half_patch_size, src_spacing, src_shape, tgt_spacing, rot_mat)
        vol = torch.nn.functional.grid_sample(vol, grid, mode="bilinear", align_corners=True, padding_mode="border")[0]
        # vol = torch.nn.functional.interpolate(vol, size=self._patch_size, mode="trilinear")[0]
        # seg = torch.nn.functional.interpolate(seg, size=self._patch_size, mode="nearest")[0]
        seg = torch.nn.functional.grid_sample(seg, grid, mode="nearest", align_corners=True)[0]
        #print("----------------", vol.shape, seg.shape)
        #vol = self._get_aug(vol, seg)
        # vol = self._window_array(vol)
        vol = self._normalization(vol)
        return vol, seg

    def _rotate_aug(self, rot_range):
        z_angle = (np.random.random() * 2 - 1) * rot_range[0] / 180.0 * np.pi
        y_angle = (np.random.random() * 2 - 1) * rot_range[1] / 180.0 * np.pi
        x_angle = (np.random.random() * 2 - 1) * rot_range[2] / 180.0 * np.pi

        rot_mat = self._get_rotate_mat(z_angle, y_angle, x_angle)
        return rot_mat

    def _get_rotate_mat(self, z_angle, y_angle, x_angle):
        def _create_matrix_rotation_z_3d(angle, matrix=None):
            rotation_x = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
            if matrix is None:
                return rotation_x
            return np.dot(matrix, rotation_x)

        def _create_matrix_rotation_y_3d(angle, matrix=None):
            rotation_y = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
            if matrix is None:
                return rotation_y

            return np.dot(matrix, rotation_y)

        def _create_matrix_rotation_x_3d(angle, matrix=None):
            rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            if matrix is None:
                return rotation_z

            return np.dot(matrix, rotation_z)

        rot_matrix = np.identity(3)
        rot_matrix = _create_matrix_rotation_z_3d(z_angle, rot_matrix)
        rot_matrix = _create_matrix_rotation_y_3d(y_angle, rot_matrix)
        rot_matrix = _create_matrix_rotation_x_3d(x_angle, rot_matrix)
        return rot_matrix

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
    # CT data
    # def _window_array(self, vol):
    #     win = [
    #         self._win_level - self._win_width / 2,
    #         self._win_level + self._win_width / 2,
    #     ]
    #     vol = torch.clamp(vol, win[0], win[1])
    #     vol -= win[0]
    #     vol /= self._win_width
    #     return vol

    # MR data
    def _normalization(self, vol):
        hu_max = torch.max(vol)
        hu_min = torch.min(vol)
        vol_normalized = (vol - hu_min) / (hu_max - hu_min + 1e-8)
        return vol_normalized

    @property
    def sampled_data_count(self):
        return self._sample_frequent * self.source_data_count

    @property
    def source_data_count(self):
        return len(self._data_file_list)

    def __getitem__(self, idx):
        source_data = self._load_source_data(self._data_file_list[idx])
        return [self._data_file_list[idx], source_data]

    def __len__(self):
        """get number of data."""
        return self.source_data_count

    def sample_source_data(self, idx, source_data):
        """get data for train."""
        sample = None
        if idx < self._sample_frequent:
            sample = self._sample_source_data(idx, source_data)
            # sample = self._pipelines(sample)
        return sample

    def _resize_torch(self, data, scale):
        return torch.nn.functional.interpolate(data, size=scale, mode="trilinear", align_corners=True)
