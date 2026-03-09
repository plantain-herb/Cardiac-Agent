"""data loader."""

import math
import os
import random

import numpy as np
import scipy
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
from custom.dataset.registry import DATASETS
from custom.dataset.utils import DefaultSampleDataset


@DATASETS.register_module()
class MsDataset(DefaultSampleDataset):

    def __init__(
        self,
        dst_list_file,
        data_root,
        # win_level,
        # win_width,
        patch_size_outer,
        patch_size_block,
        patch_size_block_inner,
        inner_sample_frequent,
        tgt_spacing_block,
        rotation_prob,
        rot_range,
        tp_prob,
        sample_frequent,
    ):
        # self._win_level = win_level
        # self._win_width = win_width
        self._sample_frequent = sample_frequent
        self._patch_size_outer = np.array(patch_size_outer)
        self._patch_size_block = np.array(patch_size_block)
        self._patch_size_block_inner = np.array(patch_size_block_inner)
        self._inner_sample_frequent = inner_sample_frequent
        self._tgt_spacing_block = np.array([tgt_spacing_block] * 3)
        self._rotation_prob = rotation_prob
        self._rot_range = rot_range
        self._tp_prob = tp_prob
        self._whole_bright_aug = [0.5, 0.1, 0.1]
        self._local_tp_bright_aug = [0.5, -0.2, 0.2]
        self._data_file_list = self._load_file_list(dst_list_file, data_root)
        self._base_grid_inner = None
        self._base_grid_outer = None
        self.draw_idx = 1

    def _load_file_list(self, dst_list_file, data_root):
        data_file_list = []
        #train_pred_seg_path = ""
        with open(dst_list_file, "r") as f:
            for line in f:
                line = line.strip()
                file_name = line.split('/')[-1]
                # train_name = file_name.replace('.npz', '-seg.nii.gz')
                # train_pred_name = os.path.join(train_pred_seg_path, train_name)
                file_name = os.path.join(data_root, file_name)
                
                # if not os.path.exists(file_name):
                #     print(f"{line} not exist")
                #     continue
                # data_file_list.append(file_name)
                if os.path.exists(file_name):
                    # if 'CS020003-032537-136178-8' in file_name or 'CS020003-626829-116922-9' in file_name:
                    #     data_file_list.extend([file_name] * 3)
                    #else:
                    data_file_list.append(file_name)
        data_file_list *= 3
        random.shuffle(data_file_list)
        assert len(data_file_list) != 0, "has no avilable file in dst_list_file"
        return data_file_list

    

    # def _dynamic_window_level(self, vol, indices, win_width=800):
    #     #print('-----------', vol.shape, indices.shape)
    #     points = vol[indices[:, 0], indices[:, 1], indices[:, 2]] 
    #     min_hu = np.min(points)
    #     max_hu = np.max(points)
    #     win_width = max_hu - min_hu
    #     bin_count = np.bincount(points - min_hu)
    #     #win_level = np.argmax(bin_count) + min_hu
    #     win_level = np.mean(np.argsort(bin_count)[-10:]) + min_hu
    #     # vol_ = _window_array(vol, win_level, win_width)
    #     #print(f"win_level: {win_level}, win_width: {win_width}")
    #     self._win_level = win_level
    #     self._win_width = win_width
    #     #return vol_

    def _load_source_data(self, file_name):

        data = np.load(file_name, allow_pickle=True)
        # first_seg = sitk.GetArrayFromImage(sitk.ReadImage(file_name[1]))
        # indices = np.argwhere(first_seg)
        # dcm = sitk.GetArrayFromImage(sitk.ReadImage(file_name[2]))
        # self._dynamic_window_level(dcm, indices)
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
            result['tp_points'] = data["border_points"].copy()
            result['src_shape'] = shape
            result['heart_range'] = data['heart_range'].copy()
            result['data_type'] = 1 - data["data_type"].copy()
            del data
            del seg

        return result
    
    def _get_outer_inputs(self, vol, seg, heart_range, src_shape, src_spacing, rot_mat):

        vessel_width = heart_range[3:] - heart_range[:3]
        vessel_center = (heart_range[3:] + heart_range[:3]) / 2

        outer_tgt_spacing = src_spacing * vessel_width / self._patch_size_outer
        outer_tgt_spacing = np.array([np.max(outer_tgt_spacing)] * 3)
        outer_ct = vessel_center * src_spacing
        outer_grid, start_points = self._get_sample_grid(outer_ct, self._patch_size_outer // 2, src_spacing, src_shape, outer_tgt_spacing, rot_mat=rot_mat)
        vol = F.grid_sample(vol, grid=outer_grid, align_corners=True)
        seg = F.grid_sample(seg, grid=outer_grid, mode='nearest', align_corners=True)

        # vol = [self._window_array(vol, win_level, win_width) for win_level, win_width in zip(self._win_level, self._win_width)]
        # vol = torch.cat(vol, dim=1)
        vol = self._normalization(vol)
        # # debug
        # import os
        # os.makedirs('./debug/', exist_ok=True)
        # import time
        # s = str(time.time())
        # sitk.WriteImage(sitk.GetImageFromArray(vol[0, 0]), './debug/' + s + ".nii.gz")
        # raise

        return vol[0], seg[0], outer_tgt_spacing, outer_grid, start_points, heart_range

    def _get_inner_inputs(self, vol, seg, c_t, src_spacing, outer_sp, outer_spacing, outer_shape, inner_spacing, rot_mat):
        src_shape = vol.shape[2:]

        # 获取crop grid
        src_shape = np.array(src_shape)
        half_patch_size = np.array([v // 2 for v in self._patch_size_block])

        inner_grid_t, inner_sp = self._get_sample_grid(c_t, half_patch_size, src_spacing, src_shape, inner_spacing, None)
        vol = F.grid_sample(vol, grid=inner_grid_t, align_corners=True)
        seg = F.grid_sample(seg, grid=inner_grid_t, mode='nearest', align_corners=True)

        # vol = [self._window_array(vol, win_level, win_width) for win_level, win_width in zip(self._win_level, self._win_width)]
        # vol = torch.cat(vol, dim=1)
        vol = self._normalization(vol)
        # 注意最后特征选取的维度是原来的一半,这里grid大小也为原来的一半
        c_t_inner_outer = np.array(c_t - outer_sp)
        inner_outer_grid, _ = self._get_sample_grid(c_t_inner_outer, half_patch_size // 2, outer_spacing, outer_shape, inner_spacing * 2, rot_mat)

        return inner_outer_grid, vol, seg

    def _sample_source_data(self, idx, source_data_info):
        info, _source_data = source_data_info
        vol, seg, src_spacing, tp_points, src_shape, heart_range, data_type = (
            _source_data['vol'],
            _source_data['seg'],
            _source_data['src_spacing'],
            _source_data['tp_points'].astype("float"),          ## 胰腺的表皮，然后取点
            _source_data['src_shape'],
            _source_data['heart_range'].astype("float"),    ## 就是seg中肝脏范围
            _source_data['data_type'].astype("float")
        )

        # bright_param = self._bright_aug_param_generate()

        if random.random() <= self._rotation_prob:
            center_point = (src_shape.astype("float") / 2)
            half_patch_size = src_shape.astype("float") // 2 + 1

            rot_mat = self._rotate_aug(self._rot_range)
            grid_vol, _ = self._get_sample_grid(center_point, half_patch_size, np.ones(3), src_shape, np.ones(3), rot_mat=rot_mat)
            vol = F.grid_sample(vol, grid_vol, align_corners=True)
            seg = F.grid_sample(seg, grid_vol, mode='nearest', align_corners=True)
            src_shape = np.array(vol.shape[2:])

            tp_points -= center_point
            tp_points = np.matmul(tp_points.astype("float"), rot_mat)
            tp_points += center_point

            tp_points = np.round(tp_points).astype("int")
            tp_points = tp_points[np.logical_not(np.any(tp_points < 0, axis=1))]
            tp_points = tp_points[np.logical_not(np.any(tp_points >= src_shape, axis=1))]

            zmin, ymin, xmin, zmax, ymax, xmax = heart_range
            heart_range_points = np.array(
                [
                    [zmin, ymin, xmin],
                    [zmin, ymin, xmax],
                    [zmin, ymax, xmin],
                    [zmin, ymax, xmax],
                    [zmax, ymin, xmin],
                    [zmax, ymin, xmax],
                    [zmax, ymax, xmin],
                    [zmax, ymax, xmax]
                ]
            )
            heart_range_points -= center_point
            heart_range_points = np.matmul(heart_range_points.astype("float"), rot_mat)
            heart_range_points += center_point
            heart_range_points = np.round(heart_range_points).astype("int")

            heart_range = np.concatenate([np.min(heart_range_points, axis=0), np.max(heart_range_points, axis=0)], axis=0)
            heart_range[:3] = np.array([max(0, i) for i in heart_range[:3]])
            heart_range[3:] = np.array([min(src_shape[idx], heart_range[3 + idx]) for idx in range(3)])

            # # debug
            # import time
            # s = str(time.time())
            # zmin, ymin, xmin, zmax, ymax, xmax = heart_range
            # sitk.WriteImage(sitk.GetImageFromArray(vol[0, 0].numpy()), s + ".nii.gz")
            # sitk.WriteImage(sitk.GetImageFromArray((seg[0, 0].numpy() > 0.5).astype("uint8")), s + "-seg.nii.gz")
            # sitk.WriteImage(sitk.GetImageFromArray((seg[0, 0, zmin: zmax, ymin: ymax, xmin: xmax].numpy() > 0.5).astype("uint8")), s + "_crop-seg.nii.gz")

            # mask_points = np.zeros(src_shape).astype("uint8")
            # for (z, y, x) in tp_points.astype("int"):
            #     mask_points[z-1: z+2, y-1: y+2, x-1: x+2] = 1
            # sitk.WriteImage(sitk.GetImageFromArray(mask_points), s + "_points-seg.nii.gz")
            # raise

        tp_points = tp_points.astype("int")
        heart_range = heart_range.astype("int").copy()
        
        with torch.no_grad():
            # 做扫描不全的augmentation
            # 获取outer输入
            if random.random() > 0.8:
                zmin, ymin, xmin = [max(0, v + np.random.randint(-30, 10)) for v in heart_range[:3]]
                zmax, ymax, xmax = [min(cs, v + np.random.randint(-10, 30)) for v, cs in zip(heart_range[3:], src_shape)]
            else:
                zmin, ymin, xmin = [max(0, v + np.random.randint(-50, 10)) for v in heart_range[:3]]
                zmax, ymax, xmax = [min(cs, v + np.random.randint(-10, 50)) for v, cs in zip(heart_range[3:], src_shape)]

            heart_range[:3] = np.array([zmin, ymin, xmin])
            heart_range[3:] = np.array([zmax, ymax, xmax])
            
            # 增加和减少整个case的明亮度
            # param = 0 
            # if random.random() < 0.3:
            #     if random.random() > 0.5:
            #         param =  random.randint(60, 80)
            #     else:
            #         param = random.randint(-80, -40)
            #vol += param
            
            aug_vol = torch.zeros(vol.shape)
            aug_vol[:, :, zmin: zmax, ymin: ymax, xmin: xmax] = vol[:, :, zmin: zmax, ymin: ymax, xmin: xmax].clone() 
            vol = aug_vol
            aug_seg = torch.zeros(seg.shape)
            aug_seg[:, :, zmin: zmax, ymin: ymax, xmin: xmax] = seg[:, :, zmin: zmax, ymin: ymax, xmin: xmax].clone() 
            seg = aug_seg

            del aug_seg
            del aug_vol
            
            bright_param = self._bright_aug_param_generate()
            vol = self._bright_aug_apply(vol[0][0], seg[0][0], bright_param)
            #print("ending*********", vol.shape, seg.shape)
            

        #     import time
        #     os.makedirs('./debug', exist_ok=True)
        #     s = str(time.time())
        #     sitk.WriteImage(sitk.GetImageFromArray(vol[0, 0].numpy()), './debug/' + s + ".nii.gz")
        #    # outer_inner_vol = F.grid_sample(outer_vol[None], grid=inner_grid, align_corners=True)
        #     sitk.WriteImage(sitk.GetImageFromArray(seg[0, 0].numpy()), './debug/' + s + "-seg.nii.gz")

            # 选取inner中心点并进行局部高斯亮度augmentation
            ct_list = []
            for idx in range(self._inner_sample_frequent):
                c_t = self._get_random_crop_center(tp_points, src_spacing, heart_range)
                ct_list.append(c_t)

            outer_vol, outer_seg, outer_tgt_spacing, outer_grid, outer_sp, heart_range = self._get_outer_inputs(vol, seg, heart_range.copy(), src_shape, src_spacing, None)
            outer_vol_shape = np.array(outer_vol.shape[1:])

            inner_img = []
            inner_mask = []
            inner_grids = []
            for idx in range(self._inner_sample_frequent):
                inner_grid, inner_vol, inner_seg = self._get_inner_inputs(vol, seg, ct_list[idx], src_spacing, outer_sp, outer_tgt_spacing, outer_vol_shape, self._tgt_spacing_block, None)
                
                # # debug
                # import time
                # s = str(time.time())
                # sitk.WriteImage(sitk.GetImageFromArray(inner_vol[0, 0].numpy()), s + "_vol.nii.gz")
                # outer_inner_vol = F.grid_sample(outer_vol[None], grid=inner_grid, align_corners=True)
                # sitk.WriteImage(sitk.GetImageFromArray(outer_inner_vol[0, 0].numpy()), s + "_vol1.nii.gz")

                inner_img.append(inner_vol)
                inner_mask.append(inner_seg)
                inner_grids.append(inner_grid)

            inner_img = torch.cat(inner_img, dim=0)    # 5维 _inner_sample_frequent * 2 * 96 * 96 * 96
            inner_mask = torch.cat(inner_mask, dim=0)
            inner_grids = torch.cat(inner_grids, dim=0)
            #print("---------", outer_vol.shape, outer_seg.shape, inner_img.shape, inner_mask)
        del _source_data
        return {"outer_img": outer_vol, "outer_mask": outer_seg, "inner_img": inner_img, "inner_mask": inner_mask, "inner_grids": inner_grids}

    def _bright_aug_param_generate(self):
        global_param = (0.0, 0.0)
        if random.random() <= self._whole_bright_aug[0]:
            global_param = (
                random.uniform(-self._whole_bright_aug[1], self._whole_bright_aug[1]),
                random.uniform(-self._whole_bright_aug[2], self._whole_bright_aug[2]),
            )

        local_param = 0.0
        if random.random() <= self._local_tp_bright_aug[0]:
            local_param = random.uniform(self._local_tp_bright_aug[1], self._local_tp_bright_aug[2])

        return global_param, local_param

    def _bright_aug_apply(self, vol, tp_seg, bright_param):
        global_param, local_param = bright_param

        vol = vol * (1 + global_param[0])
        vol = vol * (1 - tp_seg) + tp_seg * vol * (1 + local_param)

        # vol = torch.clamp(vol, min=0.0, max=1.0)

        return vol[None][None]


    def _get_random_crop_center(self, tp_points, src_spacing, heart_range):
        zmin, ymin, xmin, zmax, ymax, xmax = heart_range
        heart_range = ((zmin, zmax), (ymin, ymax), (xmin, xmax))
        half_patch_size_inner = [v // 2 for v in self._patch_size_block_inner]
        if random.random() < self._tp_prob:
            av_points = np.empty(shape=(0, 3))
            while av_points.shape[0] == 0:
                z_middle = zmin + (zmax - zmin) // 3
                if random.random() > 0.5:
                    z_rand_idx = np.random.randint(z_middle, zmax)
                else:
                    z_rand_idx = np.random.randint(zmin, z_middle)
                av_points = tp_points[:, 0] == z_rand_idx
                av_points = tp_points[av_points, :]

            c_t_idx = random.randint(0, av_points.shape[0] - 1)
            c_t = av_points[c_t_idx]
            c_t = c_t + np.random.randint(-20, 20, size=(3,))
            c_t = c_t * src_spacing
        else:
            c_t = [
                random.randint(hr_min, hr_max) * ss
                for (hr_min, hr_max), hpsi, ss in zip(heart_range, half_patch_size_inner, src_spacing)
            ]
            c_t = np.array(c_t)
        return c_t

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
        z_angle = (np.random.random() * 2 - 1) * rot_range[0] / 180.0 * np.pi
        y_angle = (np.random.random() * 2 - 1) * rot_range[1] / 180.0 * np.pi
        x_angle = (np.random.random() * 2 - 1) * rot_range[2] / 180.0 * np.pi

        rot_mat = self._get_rotate_mat(z_angle, y_angle, x_angle)
        return rot_mat.astype(np.float32)

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
        start_points = []
        for cent_px, ts, ps_half in zip(center_point, tgt_spacing, half_patch_size):
            p_s = cent_px - ps_half * ts
            start_points.append(p_s)
            p_e = cent_px + ps_half * ts - (ts / 2)
            grid.append(np.arange(p_s, p_e, ts))
        start_points = np.array(start_points)
        grid = np.meshgrid(*grid)
        grid = [g[:, :, :, None] for g in grid]  # shape (h,d,w,(zyx))
        grid = np.concatenate(grid, axis=-1)
        grid = np.transpose(grid, axes=(1, 0, 2, 3))  # shape (d,h,w,(zyx))

        if rot_mat is not None:
            grid -= center_point[None, None, None, :]
            grid = np.matmul(grid, np.linalg.inv(rot_mat))
            grid += center_point[None, None, None, :]

            # aug points
            start_points -= center_point
            start_points = np.matmul(start_points.astype("float"), rot_mat)
            start_points += center_point
        grid *= 2
        grid /= src_spacing[None, None, None, :]
        grid /= (src_shape - 1)[None, None, None, :]
        grid -= 1
        # change z,y,x to x,y,z
        grid = np.array(grid[:, :, :, ::-1], dtype=np.float32)
        return torch.from_numpy(grid)[None], start_points

    # def _window_array(self, vol, win_level, win_width):
    #     win = [win_level - win_width / 2, win_level + win_width / 2]
    #     vol = torch.clamp(vol, win[0], win[1])
    #     vol -= win[0]
    #     vol /= win_width
    #     return vol
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
        return self._sample_frequent * self.source_data_count

    @property
    def source_data_count(self):
        return len(self._data_file_list)

    def __len__(self):
        return self.source_data_count

    def sample_source_data(self, idx, source_data):
        sample = None
        if idx < self._sample_frequent:
            sample = self._sample_source_data(idx, source_data)
        return sample


@DATASETS.register_module()
class Ms_Val_Dataset(DefaultSampleDataset):

    def __init__(
        self,
        dst_list_file,
        data_root,
        sample_frequent,
    ):
        self._data_file_list = self._load_file_list(dst_list_file, data_root)

    def _load_file_list(self, dst_list_file, data_root):
        data_file_list = []
        #train_pred_seg_path = ""
        with open(dst_list_file, "r") as f:
            for line in f:
                line = line.strip()
                file_name = line.split('/')[-1]
                file_name = os.path.join(data_root, file_name)
                if os.path.exists(file_name):
                    data_file_list.append(file_name)
        assert len(data_file_list) != 0, "has no avilable file in dst_list_file"
        return data_file_list

    def __getitem__(self, idx):
        file_name = self._data_file_list[idx]
        data = np.load(file_name)
        
        with torch.no_grad():
            vol = data["vol"]
            shape = np.array(vol.shape)
            vol = torch.from_numpy(vol).float()[None]
            src_spacing = data["src_spacing"].copy()
            seg = torch.from_numpy(data["mask"]).float()[None]
        return {'img': vol, 'seg': seg, 'src_spacing': src_spacing}
        

    def __len__(self):
        return len(self._data_file_list)

    @property
    def sampled_data_count(self):
        return len(self._data_file_list)

    def sample_source_data(self, idx, source_data):
        sample = None
        if idx < 1:
            sample = source_data
        return sample

    def evaluate(self, results, logger=None):
        bce_loss = 0
        dice_loss = 0
        for bce, dice in results:
            bce_loss += bce
            dice_loss += dice
        N = len(results)
        return {'bce_loss': bce_loss / N, 'dice_loss': dice_loss / N}
