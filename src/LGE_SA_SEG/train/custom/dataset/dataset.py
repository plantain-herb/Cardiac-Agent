"""data loader."""

import os
import random
import numpy as np
import SimpleITK as sitk
from custom.dataset.utils import build_pipelines
from custom.dataset.registry import DATASETS
from custom.dataset.utils import DefaultSampleDataset, CustomDataset
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import torch
import cv2
import elasticdeform as ed
from scipy.ndimage.filters import gaussian_filter
from typing import Tuple, Union, Callable

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





@DATASETS.register_module()
class Seg_Sample_Dataset2d(DefaultSampleDataset):
    def __init__(
        self,
        dst_list_file,
        data_root,
        patch_size,
        patch_size_inner,
        rotation_prob,
        rot_range,
        noise_prob=0.1,
        color_prob=0.6,
        spacing_range=0.25,
        sample_frequent=10,
    ):
        self._sample_frequent = sample_frequent
        self._patch_size = patch_size
        self._patch_size_inner = patch_size_inner
        self._rotation_prob = rotation_prob
        self._rot_range = rot_range
        self._noise_prob = noise_prob,
        self._color_prob = color_prob,
        self._spacing_range = spacing_range
        self._data_file_list = self._load_file_list(dst_list_file, data_root)

    def _load_file_list(self, dst_list_file, data_root):
        data_file_list = []
        with open(dst_list_file, "r") as f:
            for line in f:
                line = line.strip().split('/')[-1]
                path = os.path.join(data_root, line)
                if os.path.exists(path):
                    data_file_list.append(path)
        random.shuffle(data_file_list)
        assert len(data_file_list) != 0, "has avilable file in dst_list_file"
        return data_file_list


    def _load_source_data(self, filename):
        data = np.load(filename, allow_pickle=True)
        
        vol = data['vol'].copy()        
        seg = data['mask'].copy()
        if vol.shape[0] == 1:
            idx_ = 0
        else:
            total_layers = vol.shape[0]
            last_idx = total_layers - 1
            if total_layers == 2:
                idx_ = random.choice([0, last_idx])
            else:
                weights = [0.15] + [0.7/(total_layers-2)]*(total_layers-2) + [0.15]
                indices = list(range(total_layers))
                idx_ = random.choices(indices, weights=weights, k=1)[0]
        img_data0 = []
        deg = np.random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D(((self._patch_size[2] - 1) / 2, (self._patch_size[1] - 1) / 2), deg, 1)
        M[0, 2] = M[0, 2] + (self._patch_size[2] - 1) / 2 - (self._patch_size[1] - 1) / 2
        M[1, 2] = M[1, 2] + (self._patch_size[2] - 1) / 2 - (self._patch_size[1] - 1) / 2

        z_range = 1
        pad_f = max(0, z_range - idx_)
        pad_p = max(0, z_range - (vol.shape[0] - idx_) + 1)

        if pad_f > 0 or pad_p > 0:
            pad_width = ((pad_f, pad_p), (0, 0), (0, 0))
            vol = np.pad(vol, pad_width, mode='constant', constant_values=0)
            idx_ += pad_f

        # 确保切片索引非负且不超出边界
        start_idx = max(0, idx_ - z_range)
        end_idx = min(vol.shape[0], idx_ + z_range + 1)
        vol = vol[start_idx:end_idx, :]

        for i in range(vol.shape[0]):
            img_tmp = vol[i, :]
            img_data0.append(cv2.warpAffine(img_tmp, M, (self._patch_size[2], self._patch_size[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
            # if i == idx_:  # center slice, label.
        lb_data = seg[idx_, :]
        lb_data = cv2.warpAffine(lb_data, M, (self._patch_size[2], self._patch_size[1]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        img_data0 = np.stack(img_data0)
        # deformation
        df_ny = max(int(round(self._patch_size[2] / 32) + 1), 3)
        df_nx = max(int(round(self._patch_size[1] / 32) + 1), 3)
        displacement = np.random.uniform(-5, 5, (
            2, df_ny, df_nx))  # ,np.random.uniform(-2,2,(1, self.df_ny, self.df_nx, self.df_nz))),axis=0)
        vol = ed.deform_grid(img_data0, displacement, order=1,
                                  mode='nearest', prefilter=False, axis=(1, 2))
        seg = (ed.deform_grid(lb_data, displacement, order=0,
                                  mode='nearest', prefilter=False)).astype(np.float32)
        
        # 随机决定是否水平翻转（50%概率）
        if random.random() < 0.5:
            vol = np.flip(vol, axis=-1)  
            seg = np.flip(seg, axis=-1)
        
        # 随机决定是否垂直翻转（50%概率）
        if random.random() < 0.5:
            vol = np.flip(vol, axis=-2) 
            seg = np.flip(seg, axis=-2)

        result = {}
        result['vol'] = torch.from_numpy(vol.astype(np.float32)).float()[None, None]
        result['seg'] = torch.from_numpy(seg).float()[None, None]
        return result

    def _sample_source_data(self, idx, source_data_info):
        info, _source_data = source_data_info
        vol, seg,  = (
            _source_data["vol"],
            _source_data["seg"],
        )
        
        aug_prob_color = random.random()
        aug_prob_noise = random.random()
        aug_prob = random.random()
        vol = self._normalization(vol)

        # vol = self._crop_data(vol, c_t, vol_shape, src_spacing, aug_prob=aug_prob, aug_prob_color=aug_prob_color, aug_prob_noise=aug_prob_noise)
        # vol = F.interpolate(vol, size=(31, 512, 512), mode="trilinear")[0]
        if aug_prob_color <= self._color_prob[0]:
            if aug_prob <= 0.55:
                vol = self.augment_gamma(vol)
            elif 0.55 < aug_prob <= 0.85:
                vol = self.augment_brightness_additive(vol)
            else:
                vol = self.augment_contrast(vol)

        if aug_prob_noise <= self._noise_prob[0]:
            if aug_prob <= 0.5:
                vol = self.augment_gaussian_noise(vol)
            elif 0.5 < aug_prob < 0.75:
                vol = self.augment_gaussian_blur(vol)
        
        # print(vol.shape, seg.shape)
        # import time
        # s = str(time.time())
        # sitk.WriteImage(sitk.GetImageFromArray(vol[0][0][1].cpu().float().numpy()), "./"+ s + ".nii.gz")
        # sitk.WriteImage(sitk.GetImageFromArray(seg[0][0].cpu().float().numpy()), "./"+ s + "-seg.nii.gz")

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

    def augment_gamma(self, data_sample, gamma_range=(0.75, 1.25), invert_image=False, epsilon=1e-7, per_channel=False,
                        retain_stats: Union[bool, Callable[[], bool]] = False):
            if invert_image:
                data_sample = - data_sample

            if not per_channel:
                retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
                if retain_stats_here:
                    mn = data_sample.mean()
                    sd = data_sample.std()
                if np.random.random() < 0.5 and gamma_range[0] < 1:
                    gamma = np.random.uniform(gamma_range[0], 1)
                else:
                    gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
                minm = data_sample.min()
                rnge = data_sample.max() - minm
                data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
                if retain_stats_here:
                    data_sample = data_sample - data_sample.mean()
                    data_sample = data_sample / (data_sample.std() + 1e-8) * sd
                    data_sample = data_sample + mn
            else:
                for c in range(data_sample.shape[0]):
                    retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
                    if retain_stats_here:
                        mn = data_sample[c].mean()
                        sd = data_sample[c].std()
                    if np.random.random() < 0.5 and gamma_range[0] < 1:
                        gamma = np.random.uniform(gamma_range[0], 1)
                    else:
                        gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
                    minm = data_sample[c].min()
                    rnge = data_sample[c].max() - minm
                    data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(
                        rnge + epsilon) + minm
                    if retain_stats_here:
                        data_sample[c] = data_sample[c] - data_sample[c].mean()
                        data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
                        data_sample[c] = data_sample[c] + mn
            if invert_image:
                data_sample = - data_sample
            return data_sample

    def augment_contrast(self, data_sample: np.ndarray,
                            contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),
                            preserve_range: bool = True,
                            per_channel: bool = True,
                            p_per_channel: float = 1) -> np.ndarray:
        if not per_channel:
            if callable(contrast_range):
                factor = contrast_range()
            else:
                if np.random.random() < 0.5 and contrast_range[0] < 1:
                    factor = np.random.uniform(contrast_range[0], 1)
                else:
                    factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])

            for c in range(data_sample.shape[0]):
                if np.random.uniform() < p_per_channel:
                    mn = data_sample[c].mean()
                    if preserve_range:
                        minm = data_sample[c].min()
                        maxm = data_sample[c].max()

                    data_sample[c] = (data_sample[c] - mn) * factor + mn

                    if preserve_range:
                        data_sample[c][data_sample[c] < minm] = minm
                        data_sample[c][data_sample[c] > maxm] = maxm
        else:
            for c in range(data_sample.shape[0]):
                if np.random.uniform() < p_per_channel:
                    if callable(contrast_range):
                        factor = contrast_range()
                    else:
                        if np.random.random() < 0.5 and contrast_range[0] < 1:
                            factor = np.random.uniform(contrast_range[0], 1)
                        else:
                            factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])

                    mn = data_sample[c].mean()
                    if preserve_range:
                        minm = data_sample[c].min()
                        maxm = data_sample[c].max()

                    data_sample[c] = (data_sample[c] - mn) * factor + mn

                    if preserve_range:
                        data_sample[c][data_sample[c] < minm] = minm
                        data_sample[c][data_sample[c] > maxm] = maxm
        return data_sample

    def augment_brightness_additive(self, data_sample, mu=2, sigma=0.2, per_channel: bool = True,
                                    p_per_channel: float = 1.):
        """
        data_sample must have shape (c, x, y(, z)))
        :param data_sample:
        :param mu:
        :param sigma:
        :param per_channel:
        :param p_per_channel:
        :return:
        """
        if not per_channel:
            rnd_nb = np.random.normal(mu, sigma)
            for c in range(data_sample.shape[0]):
                if np.random.uniform() <= p_per_channel:
                    data_sample[c] += rnd_nb
        else:
            for c in range(data_sample.shape[0]):
                if np.random.uniform() <= p_per_channel:
                    rnd_nb = np.random.normal(mu, sigma)
                    data_sample[c] += rnd_nb
        return data_sample

    def augment_gaussian_noise(self, data_sample: np.ndarray, noise_variance: Tuple[float, float] = (0, 0.1),
                                p_per_channel: float = 1, per_channel: bool = False) -> np.ndarray:
            if not per_channel:
                variance = noise_variance[0] if noise_variance[0] == noise_variance[1] else \
                    random.uniform(noise_variance[0], noise_variance[1])
            else:
                variance = None
            for c in range(data_sample.shape[0]):
                if np.random.uniform() < p_per_channel:
                    # lol good luck reading this
                    variance_here = variance if variance is not None else \
                        noise_variance[0] if noise_variance[0] == noise_variance[1] else \
                            random.uniform(noise_variance[0], noise_variance[1])
                    # bug fixed: https://github.com/MIC-DKFZ/batchgenerators/issues/86
                    data_sample[c] = data_sample[c] + np.random.normal(0.0, variance_here, size=data_sample[c].shape)
            return data_sample
    def augment_gaussian_blur(self, data_sample: np.ndarray, sigma_range: Tuple[float, float] = (0.5, 1.),
                              per_channel: bool = True,
                              p_per_channel: float = 1, different_sigma_per_axis: bool = False,
                              p_isotropic: float = 0) -> np.ndarray:
        if not per_channel:
            # Godzilla Had a Stroke Trying to Read This and F***ing Died
            # https://i.kym-cdn.com/entries/icons/original/000/034/623/Untitled-3.png
            sigma = self.get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                                        ((np.random.uniform() < p_isotropic) and
                                                         different_sigma_per_axis)) \
                else [self.get_range_val(sigma_range) for _ in data_sample.shape[1:]]
        else:
            sigma = None
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                if per_channel:
                    sigma = self.get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                                                ((np.random.uniform() < p_isotropic) and
                                                                 different_sigma_per_axis)) \
                        else [self.get_range_val(sigma_range) for _ in data_sample.shape[1:]]
                data_sample[c] = torch.from_numpy(gaussian_filter(data_sample[c], sigma, order=0))
        return data_sample
    
    def get_range_val(self, value, rnd_type="uniform"):
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 2:
                if value[0] == value[1]:
                    n_val = value[0]
                else:
                    orig_type = type(value[0])
                    if rnd_type == "uniform":
                        n_val = random.uniform(value[0], value[1])
                    elif rnd_type == "normal":
                        n_val = random.normalvariate(value[0], value[1])
                    n_val = orig_type(n_val)
            elif len(value) == 1:
                n_val = value[0]
            else:
                raise RuntimeError("value must be either a single value or a list/tuple of len 2")
            return n_val
        else:
            return value

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
    # def _normalization(self, vol):
    #     hu_max = torch.max(vol)
    #     hu_min = torch.min(vol)
    #     vol_normalized = (vol - hu_min) / (hu_max - hu_min + 1e-8)
    #     return vol_normalized

    def _normalization(self, vol):
        global_mean = 282.864980
        global_std = 273.596458
        vol_normalized = (vol - global_mean) / (global_std + 1e-8)
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