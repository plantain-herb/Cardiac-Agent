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
class DYPyramid_Sample_Dataset(DefaultSampleDataset):
    def __init__(
        self,
        dst_list_file,
        data_root,
        # win_level,
        # win_width,
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
        # self._win_level = win_level
        # self._win_width = win_width
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
            vol = data["vol"].astype(np.float32)
            shape = np.array(vol.shape)

            vol = torch.from_numpy(vol).float()[None, None]
            src_spacing = data["spacing"].copy()
            seg = torch.from_numpy(data["mask"].astype(np.float32)).float()[None, None]
            ## 
            seg1 = seg.clone()
            seg1[seg1 > 1] = 0
            seg2 = seg.clone()
            seg2[seg2 == 1] = 0
            seg2[seg2 == 3] = 0
            seg2[seg2 == 2] = 1
            seg3 = seg.clone()
            seg3[seg3 < 3] = 0
            seg3[seg3 == 3] = 1
            # seg3[seg3 == 4] = 1
            result['vol'] = vol.detach()
            result['seg1'] = seg1.detach()
            result['seg2'] = seg2.detach()
            result['seg3'] = seg3.detach()
            result['src_spacing'] = src_spacing
            result['points_tp'] = data["tp_points"].copy()
            result['src_shape'] = shape
            result['bbox'] = data['bbox'].copy()
            del data
            del seg
            del seg1
            del seg2
            del seg3

        return result

    def _sample_source_data(self, idx, source_data_info):
        info, _source_data = source_data_info
        vol, seg1, seg2, seg3, src_spacing, src_shape, bbox, points_tp = (
            _source_data["vol"],
            _source_data["seg1"],
            _source_data["seg2"],
            _source_data["seg3"],
            _source_data["src_spacing"],
            _source_data["src_shape"],
            _source_data["bbox"],
            _source_data["points_tp"],
        )
        c_t = self._get_random_crop_center(src_shape, src_spacing, bbox, points_tp)
        vol, seg1, seg2, seg3 = self._crop_data(vol, seg1, seg2, seg3, c_t, src_shape, src_spacing)

        # import time
        # s = str(time.time())
        # # # print(vol.shape)
        # sitk.WriteImage(sitk.GetImageFromArray(vol[0].cpu().float().numpy()), "./"+ s + ".nii.gz")

        results = {
            "img": vol.detach(),
            "mask1": seg1.detach(),
            "mask2": seg2.detach(),
            "mask3": seg3.detach(),
        }
        return results

    def _bright_aug_param_generate(self):
        global_param = (0.0, 0.0)
        global_flag = False
        if random.random() <= self._whole_bright_aug[0]:
            global_param = (
                random.uniform(-self._whole_bright_aug[1], self._whole_bright_aug[1]),
                random.uniform(-self._whole_bright_aug[2], self._whole_bright_aug[2]),
            )
            global_flag = True

        return global_flag, global_param

    def _bright_aug_apply(self, vol, tp_seg, bright_param):
        global_flag, global_param = bright_param
        if global_flag:
            vol = vol * (1 + global_param[0]) + global_param[1]
        # if global_flag or local_flag:
            vol = torch.clamp(vol, min=0.0, max=1.0)
        return vol

    def _get_random_crop_center(self, vol_shape, src_spacing, bbox, points_tp):
        # bbox = [zmin, zmax, ymin, ymax, xmin, xmax]
        if random.random() < 0.9:
            c_t = points_tp[np.random.choice(len(points_tp))]
            c_t = c_t + np.random.randint(-15, 15, size=(3,))
            c_t = c_t * src_spacing
        else:
            z_center = (int(bbox[0])+int(bbox[1]))//2 + np.random.randint(-10, 10)
            y_center = random.randint(int(bbox[2]), int(bbox[3]))
            x_center = random.randint(int(bbox[4]), int(bbox[5]))
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

    

    def _crop_data(self, vol, seg1, seg2, seg3, c_t, src_shape, src_spacing):
        half_patch_size = [v // 2 for v in self._patch_size]

        rot_mat = None
        if random.random() <= self._rotation_prob:
            rot_mat = self._rotate_aug(self._rot_range)

        # bright_param = self._bright_aug_param_generate()

        data_pyramid = []
        tgt_spacing = self._isotropy_spacing + (random.random() * 2 - 1) * self._spacing_range
        for level in range(self._data_pyramid_level):
            grid = self._get_sample_grid(c_t, half_patch_size, src_spacing, src_shape, tgt_spacing, rot_mat)
            vol_patch = torch.nn.functional.grid_sample(
                vol, grid, mode="bilinear", align_corners=True, padding_mode="border"
            )[0]
            vol_patch = self._normalization(vol_patch)
            if random.random() <= 0.5:
                rand_bright = torch.rand((vol_patch.shape[0],))
                rand_bright = (1 - 2 * rand_bright) * 0.05
                vol_patch += rand_bright[:, None, None, None]

            seg_patch1 = torch.nn.functional.grid_sample(seg1, grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
            seg_patch2 = torch.nn.functional.grid_sample(seg2, grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
            seg_patch3 = torch.nn.functional.grid_sample(seg3, grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
            # vol_patch = self._bright_aug_apply(vol_patch, seg_patch[1:2], bright_param)
            data_pyramid.append((vol_patch, seg_patch1, seg_patch2, seg_patch3))
            tgt_spacing = tgt_spacing * self._data_pyramid_step

        data_pyramid = data_pyramid[::-1]
        vol = [d[0] for d in data_pyramid]
        vol = torch.cat(vol, dim=0)

        seg1 = [d[1] for d in data_pyramid]
        seg1 = torch.cat(seg1, dim=0)

        seg2 = [d[2] for d in data_pyramid]
        seg2 = torch.cat(seg2, dim=0)

        seg3 = [d[3] for d in data_pyramid]
        seg3 = torch.cat(seg3, dim=0)

        return vol, seg1, seg2, seg3

    def _resize_torch(self, data, scale):
        return torch.nn.functional.interpolate(data, size=scale, mode="trilinear")

    # def _window_array(self, vol):
    #     win = [
    #         self._win_level - self._win_width / 2,
    #         self._win_level + self._win_width / 2,
    #     ]
    #     vol = torch.clamp(vol, win[0], win[1])
    #     vol -= win[0]
    #     vol /= self._win_width
    #     return vol
    def _normalization(self, vol):
        hu_max = torch.max(vol)
        hu_min = torch.min(vol)
        vol_normalized = (vol - hu_min) / (hu_max - hu_min + 1e-8)
        return vol_normalized
    
    def robust_scaler(self, X):
        median = torch.median(X)
        q1 = torch.quantile(X, 0.01) 
        q3 = torch.quantile(X, 0.99) 
        iqr = q3 - q1
        scaled_X = (X - median) / iqr
        # scaled_X_nor = self._normalization(scaled_X)
        return scaled_X

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
    


@DATASETS.register_module()
class DYPyramid_Sample_Val_Dataset(DefaultSampleDataset):
    def __init__(
        self,
        dst_list_file,
        dst_src_dir,
        win_level,
        win_width,
        patch_size,
        patch_size_inner,
        isotropy_spacing,
        data_pyramid_level,
        data_pyramid_step,
        constant_shift,
        sample_frequent,
    ):
        self._win_level = win_level
        self._win_width = win_width
        self._sample_frequent = sample_frequent
        self._patch_size = patch_size
        self._patch_size_inner = patch_size_inner
        self._isotropy_spacing = isotropy_spacing
        self._dst_src_dir = dst_src_dir
        self._data_file_list = self._load_files(dst_list_file)
        self._constant_shift = constant_shift
        self._data_pyramid_level = data_pyramid_level
        self._data_pyramid_step = data_pyramid_step

    def _load_file_list(self, dst_list_file):
        data_file_list = []
        with open(dst_list_file, "r") as f:
            for line in f:
                line = line.strip()
                line = os.path.join(self._dst_src_dir, line)
                if line and os.path.exists(line) and os.path.isfile(line):
                    data_file_list.append(line)
        assert len(data_file_list) != 0, "has avilable file in dst_list_file"
        return data_file_list

    def _load_source_data(self, file_name):
        data = np.load(file_name, allow_pickle=True)
        result = {}

        vol = torch.from_numpy(data["vol"]).float()
        seg = torch.from_numpy((data["mask"]).astype("uint8"))

        # np_kidney_bladder = data["mask_kidney_bladder"]
        # mask_kidney = torch.from_numpy(np_kidney_bladder).float()

        result["vol"] = vol[None].detach()
        result["seg"] = seg[None].detach()
        # result["mask_kidney"] = mask_kidney[None].detach()

        result["spacing"] = data["spacing"].copy()
        
        result["vol_shape"] = data["vol"].shape

        if data.get("sample_region") is None:
            loc = np.where(data["mask"] == 1)
            bbox = np.array(
                [
                    np.min(loc[0]),
                    np.max(loc[0]),
                    np.min(loc[1]),
                    np.max(loc[1]),
                    np.min(loc[2]),
                    np.max(loc[2]),
                ]
            )
        else:
            bbox = np.array(data["sample_region"])
        constant_shift_val = 10
        bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5] = (
            max(0, bbox[0] - constant_shift_val),
            max(0, bbox[2] - constant_shift_val),
            max(0, bbox[4] - constant_shift_val),
            min(result["vol_shape"][0], bbox[1] + constant_shift_val),
            min(result["vol_shape"][1], bbox[3] + constant_shift_val),
            min(result["vol_shape"][2], bbox[5] + constant_shift_val),
        )
        result["bbox"] = bbox

        del data

        return result

    def _sample_source_data(self, idx, source_data_info):
        info_idx, _source_data = source_data_info

        vol, seg, src_spacing, bbox = (
            _source_data["vol"],
            _source_data["seg"],
            _source_data["spacing"],
            _source_data["bbox"],
        )
        info = {
            "win_level": torch.tensor(self._win_level),
            "win_width": torch.tensor(self._win_width),
            "patch_size": torch.tensor(self._patch_size),
            "patch_size_inner": torch.tensor(self._patch_size_inner),
            "isotropy_spacing": torch.tensor(self._isotropy_spacing),
            "data_pyramid_level": torch.tensor(self._data_pyramid_level),
            "data_pyramid_step": torch.tensor(self._data_pyramid_step),
            "spacing_zyx": torch.tensor(src_spacing),
            "bbox": torch.tensor(bbox),
            "idx": torch.tensor(info_idx),
        }

        results = {"info": info, "img": vol.detach(), "mask": seg.detach()}
        return results

    def _get_rotate_mat(self, z_angle, y_angle, x_angle):
        def _create_matrix_rotation_z_3d(angle, matrix=None):
            rotation_x = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)],
                ]
            )
            if matrix is None:
                return rotation_x
            return np.dot(matrix, rotation_x)

        def _create_matrix_rotation_y_3d(angle, matrix=None):
            rotation_y = np.array(
                [
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)],
                ]
            )
            if matrix is None:
                return rotation_y

            return np.dot(matrix, rotation_y)

        def _create_matrix_rotation_x_3d(angle, matrix=None):
            rotation_z = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
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
        self,
        center_point,
        half_patch_size,
        src_spacing,
        src_shape,
        tgt_spacing,
        rot_mat=None,
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

    def _resize_torch(self, data, scale):
        return torch.nn.functional.interpolate(data, size=scale, mode="trilinear")

    # def _window_array(self, vol):
    #     win = [
    #         self._win_level - self._win_width / 2,
    #         self._win_level + self._win_width / 2,
    #     ]
    #     vol = torch.clamp(vol, win[0], win[1])
    #     vol -= win[0]
    #     vol /= self._win_width
    #     return vol

    def __getitem__(self, idx):
        source_data = self._load_source_data(self._data_file_list[idx])
        return [idx, source_data]

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

    def evaluate(self, results, logger=None):
        cl_dice_all = 0.0
        ce_loss_all = 0.0
        count_all = 0.0
        for result in results:
            idx, cl_dice, ce_loss = result
            cl_dice_all += float(cl_dice.detach().cpu())
            ce_loss_all += float(ce_loss.detach().cpu())
            count_all += 1.0
        return {
            "cldice": float(cl_dice_all / (count_all + 1e-5)),
            "ce": float(ce_loss_all / (count_all + 1e-5)),
        }