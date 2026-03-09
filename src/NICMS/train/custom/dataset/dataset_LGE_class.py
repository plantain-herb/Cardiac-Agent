"""data loader."""

import os
import random
from typing import List, Union
from scipy.ndimage.filters import gaussian_filter
from typing import Tuple, Union, Callable
import math
import numpy as np
import torch
import SimpleITK as sitk
from custom.dataset.registry import DATASETS
from custom.dataset.utils import DefaultSampleDataset
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms


@DATASETS.register_module()
class LGEClassificationPidReSampleDataset(DefaultSampleDataset):
    def __init__(
        self,
        root,
        dst_list_file,
        patch_size,
        rotation_prob,
        noise_prob,
        color_prob,
        rot_range,
        shift_range,
        sample_frequent,
        whole_bright_aug=(0.5, 0.2, 0.2),
    ):
        self._sample_frequent = sample_frequent
        self._patch_size = patch_size
        self._rotation_prob = rotation_prob
        self._noise_prob = noise_prob
        self._color_prob = color_prob
        self._rot_range = rot_range
        self._shift_range = shift_range
        self._data_file_list = self._load_file_list(dst_list_file, root)
        self._whole_bright_aug = whole_bright_aug
        self.draw_idx = 1
        self.transforms = self._transforms()

    def _load_file_list(self, dst_list_file, root):
        data_file_list = []
        with open(dst_list_file, "r") as f:
            for line in f:
                line = line.strip().split("/")[-1]
                path = os.path.join(root, line)
                if not os.path.exists(path):
                    print(f"{path} not exist")
                    continue
                data_file_list.extend([path] * 2)
        random.shuffle(data_file_list)
        assert len(data_file_list) != 0, "has no avilable file in dst_list_file"
        return data_file_list

    def _load_source_data(self, file_name):
        # print("file_name:"+file_name)
        data = np.load(file_name, allow_pickle=True)
        
        result = {}
        vol = torch.from_numpy(data["dcm"])
        gt = data["gt"]
        # time = data["time"]

        # gt_array = np.zeros(4, dtype=np.uint8)
        # gt_array[gt - 1] = 1
        # gt_array = torch.from_numpy(gt_array).long()

 
        result["vol"] = vol[None, None].detach()
        result["vol_shape"] = np.array(vol.size())
        result["gt"] = gt
        # result["time"] = time

        del data

        return result

    def _sample_source_data(self, idx, source_data_info):
        info, _source_data = source_data_info
        vol, vol_shape, gt = (
            _source_data["vol"],
            _source_data["vol_shape"],
            _source_data["gt"],
        )
        c_t = self._get_random_crop_center(vol_shape)
        # gt = np.array(gt - 1)
        # gt_tensor = torch.zeros(4)
        gt_tensor = torch.tensor(gt - 1)
        # print(time_tensor)
        vol = self._crop_data(vol, c_t, [self._patch_size[0], vol_shape[1], vol_shape[2]])
        # vol = torch.cat([vol, infar], dim=0)
        # print(time_tensor, gt_tensor)
        # import time
        # s = str(time.time())
        # print(vol.shape)
        # sitk.WriteImage(sitk.GetImageFromArray(vol[0].cpu().float().numpy()), "./"+ s + ".nii.gz")
        # raise
        results = {"img": vol.detach(), "gt": gt_tensor.detach()}
        return results

    def _get_random_crop_center(self, vol_shape):

        if vol_shape[0] == 1:
            z_center = 0
        else:
            z_center = random.randint(0, vol_shape[0]-1)
        y_center = vol_shape[1] // 2
        x_center = vol_shape[2] // 2
        c_t = np.array([z_center, y_center, x_center])

        return c_t
    
    def _transforms(self):
        train_transformer = transforms.Compose([
            transforms.Resize((256, 256)),  # (B, C, H, W) -> (B, C, 256, *)
            transforms.RandomRotation(30),           # 随机旋转 ±30°
            # transforms.RandomResizedCrop(224),       # 随机裁剪并缩放至 224x224
            transforms.RandomHorizontalFlip(),       # 随机水平翻转
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return train_transformer

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

    def _rotate_aug(self, rot_range):
        z_angle = (np.random.random() * 2 - 1) * rot_range[0] / 180.0 * np.pi
        y_angle = (np.random.random() * 2 - 1) * rot_range[1] / 180.0 * np.pi
        x_angle = (np.random.random() * 2 - 1) * rot_range[2] / 180.0 * np.pi

        rot_mat = self._get_rotate_mat(z_angle, y_angle, x_angle)
        return rot_mat


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

    def mask_random_square(self, img, square_size, n_val, channel_wise_n_val=False, square_pos=None):
        """Masks (sets = 0) a random square in an image"""

        img_h = img.shape[-2]
        img_w = img.shape[-1]

        # img = img.copy()

        if square_pos is None:
            w_start = np.random.randint(0, img_w - square_size)
            h_start = np.random.randint(0, img_h - square_size)
        else:
            pos_wh = square_pos[np.random.randint(0, len(square_pos))]
            w_start = pos_wh[0]
            h_start = pos_wh[1]

        if img.ndim == 2:
            rnd_n_val = self.get_range_val(n_val)
            img[h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val
        elif img.ndim == 3:
            if channel_wise_n_val:
                for i in range(img.shape[0]):
                    rnd_n_val = self.get_range_val(n_val)
                    img[i, h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val
            else:
                rnd_n_val = self.get_range_val(n_val)
                img[:, h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val
        elif img.ndim == 4:
            if channel_wise_n_val:
                for i in range(img.shape[0]):
                    rnd_n_val = self.get_range_val(n_val)
                    img[:, i, h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val
            else:
                rnd_n_val = self.get_range_val(n_val)
                img[:, :, h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val

        return img

    def mask_random_squares(self, img, square_size, n_squares, n_val, channel_wise_n_val=False, square_pos=None):
        """Masks a given number of squares in an image"""
        for i in range(n_squares):
            img = self.mask_random_square(img, square_size, n_val, channel_wise_n_val=channel_wise_n_val,
                                     square_pos=square_pos)
        return img

    def augment_blank_square_noise(self, data_sample, square_size=(10, 10), n_squares=4, noise_val=(0, 0),
                                   channel_wise_n_val=False,
                                   square_pos=None):
        # rnd_n_val = get_range_val(noise_val)
        rnd_square_size = self.get_range_val(square_size)
        rnd_n_squares = self.get_range_val(n_squares)

        data_sample = self.mask_random_squares(data_sample, square_size=rnd_square_size, n_squares=rnd_n_squares,
                                          n_val=noise_val, channel_wise_n_val=channel_wise_n_val,
                                          square_pos=square_pos)
        return data_sample

    def augment_gamma(self,  data_sample, gamma_range=(0.75, 1.25), invert_image=False, epsilon=1e-7, per_channel=False,
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

    def augment_brightness_additive(self, data_sample, mu=2, sigma=0.2, per_channel: bool = True, p_per_channel: float = 1.):
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

    def _crop_data(self, vol, c_t, target_shape):
        device = vol.device if hasattr(vol, 'device') else 'cpu'
        vol = torch.as_tensor(vol, device=device)
        c_t = torch.as_tensor(c_t, device=device)
        target_shape = torch.as_tensor(target_shape, device=device)
        
        input_shape = torch.tensor(vol.shape[-3:], device=device)
        
        start = (c_t - target_shape // 2).floor().long()
        end = start + target_shape
        
        output_shape = (*vol.shape[:-3], *target_shape)
        cropped_vol = torch.zeros(output_shape, dtype=vol.dtype, device=device)
        
        src_start = torch.maximum(start, torch.tensor([0,0,0], device=device))
        src_end = torch.minimum(end, input_shape)
        
        dst_start = (src_start - start).long()
        dst_end = dst_start + (src_end - src_start).long()
        
        cropped_vol[
            ...,
            dst_start[0]:dst_end[0],
            dst_start[1]:dst_end[1],
            dst_start[2]:dst_end[2]
        ] = vol[
            ...,
            src_start[0]:src_end[0],
            src_start[1]:src_end[1],
            src_start[2]:src_end[2]
        ]
        
        
        cropped_vol = self._normalization(cropped_vol)

        if random.random() <= self._color_prob:
            if random.random() <= 0.35:
                cropped_vol = self.augment_gamma(cropped_vol)
            elif 0.35 < random.random() <= 0.7:
                bright_param = self._bright_aug_param_generate()
                cropped_vol = self._bright_aug_apply(cropped_vol, bright_param)
            elif 0.7 < random.random() <= 0.85:
                cropped_vol = self.augment_brightness_additive(cropped_vol)
            else:
                cropped_vol = self.augment_contrast(cropped_vol)

        if random.random() <= self._noise_prob:
            if random.random() <= 0.5:
                cropped_vol = self.augment_gaussian_noise(cropped_vol)
            elif 0.5 < random.random() < 0.75:
                cropped_vol = self.augment_gaussian_blur(cropped_vol)
            else:
                cropped_vol = self.augment_blank_square_noise(cropped_vol)
            
        
        cropped_vol = self.transforms(cropped_vol)
        if cropped_vol.dim() == 5:
            cropped_vol = cropped_vol.squeeze(1)  
        # cropped_vol = cropped_vol.repeat(1, 3, 1, 1) 

        # import time
        # s = str(time.time())
        # sitk.WriteImage(sitk.GetImageFromArray(cropped_vol[0].cpu().float().numpy()), "./"+ s + ".nii.gz")

        # _, _, _, current_h, current_w = cropped_vol.shape
        # target_size=(1, 256, 256)
        # target_h, target_w = target_size[-2:]

        # if current_h == target_h and current_w == target_w:
        #     return cropped_vol

        # cropped_vol_resized = F.interpolate(
        #     cropped_vol,
        #     size=target_size,
        #     mode='trilinear',  
        #     align_corners=False
        # )
        # if cropped_vol_resized.dim() == 6:
        #     cropped_vol_resized = cropped_vol_resized.squeeze(2)  
        
        # sitk.WriteImage(sitk.GetImageFromArray(cropped_mask[0][0].cpu().float().numpy()), "./"+ s + "-mask.nii.gz")
        return cropped_vol
        

    def _bright_aug_param_generate(self):
        global_param = (0.0, 0.0)
        global_flag = False
        # local_flag = False
        if random.random() <= self._whole_bright_aug[0]:
            global_param = (
                random.uniform(-self._whole_bright_aug[1], self._whole_bright_aug[1]),
                random.uniform(-self._whole_bright_aug[2], self._whole_bright_aug[2]),
            )
            global_flag = True

        # local_param = 0.0
        # if random.random() <= self._local_tp_bright_aug[0]:
        #     local_param = random.uniform(self._local_tp_bright_aug[1], self._local_tp_bright_aug[2])
        #     local_flag = True

        return global_flag, global_param

    def _bright_aug_apply(self, vol, bright_param):
        global_flag, global_param= bright_param
        if global_flag:
            vol = vol * (1 + global_param[0]) + global_param[1]
        if global_flag:
            vol = torch.clamp(vol, min=0.0, max=1.0)
        return vol


    def _resize_torch(self, data, scale):
        return torch.nn.functional.interpolate(data, size=scale, mode="trilinear")

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



@DATASETS.register_module()
class LGE_Cls_ReSampleDataset_Val(DefaultSampleDataset):
    def __init__(
        self,
        root,
        dst_list_file,
        patch_size,
        rotation_prob,
        noise_prob,
        color_prob,
        rot_range,
        shift_range,
        sample_frequent,
    ):
        self._sample_frequent = sample_frequent
        self._patch_size = patch_size
        self._rotation_prob = rotation_prob
        self._noise_prob = noise_prob
        self._color_prob = color_prob
        self._rot_range = rot_range
        self._shift_range = shift_range
        self._data_file_list = self._load_file_list(dst_list_file, root)
        self.draw_idx = 1
        self.transforms = self._transforms()

    def _load_file_list(self, dst_list_file, root):
        data_file_list = []
        with open(dst_list_file, "r") as f:
            for line in f:
                line = line.strip().split("/")[-1]
                path = os.path.join(root, line)
                if not os.path.exists(path):
                    print(f"{path} not exist")
                    continue
                data_file_list.append(path)
        assert len(data_file_list) != 0, "has no avilable file in dst_list_file"
        return data_file_list

    def _load_source_data(self, file_name):
        data = np.load(file_name, allow_pickle=True)
        result = {}
        vol = torch.from_numpy(data["dcm"])
        gt = data["gt"]
        # time = data["time"]
        result["vol"] = vol[None, None].detach()
        result["vol_shape"] = np.array(vol.size())
        result["gt"] = gt
        # result["time"] = time

        del data
        return result

    def _sample_source_data(self, idx, source_data_info):
        info, _source_data = source_data_info
        vol, vol_shape, gt = (
            _source_data["vol"],
            _source_data["vol_shape"],
            _source_data["gt"],
        )

        c_t = self._get_crop_center(vol_shape)
        gt_tensor = torch.tensor(gt - 1)
        vol = self._crop_data(vol, c_t, [self._patch_size[0], vol_shape[1], vol_shape[2]])
        results = {"img": vol.detach(), "gt": gt_tensor.detach()}
        return results

    def _get_crop_center(self, vol_shape):
        
        if vol_shape[0] == 1:
            z_center = 0
        else:
            z_center = vol_shape[0] //2
        y_center = vol_shape[1] // 2
        x_center = vol_shape[2] // 2
        c_t = np.array([z_center, y_center, x_center])

        return c_t


    def _crop_data(self, vol, c_t, target_shape):
        device = vol.device if hasattr(vol, 'device') else 'cpu'
        vol = torch.as_tensor(vol, device=device)
        c_t = torch.as_tensor(c_t, device=device)
        target_shape = torch.as_tensor(target_shape, device=device)
        
        input_shape = torch.tensor(vol.shape[-3:], device=device)
        
        start = (c_t - target_shape // 2).floor().long()
        end = start + target_shape
        
        output_shape = (*vol.shape[:-3], *target_shape)
        cropped_vol = torch.zeros(output_shape, dtype=vol.dtype, device=device)
        
        src_start = torch.maximum(start, torch.tensor([0,0,0], device=device))
        src_end = torch.minimum(end, input_shape)
        
        dst_start = (src_start - start).long()
        dst_end = dst_start + (src_end - src_start).long()
        
        cropped_vol[
            ...,
            dst_start[0]:dst_end[0],
            dst_start[1]:dst_end[1],
            dst_start[2]:dst_end[2]
        ] = vol[
            ...,
            src_start[0]:src_end[0],
            src_start[1]:src_end[1],
            src_start[2]:src_end[2]
        ]
        
        
        cropped_vol = self._normalization(cropped_vol)

        cropped_vol = self.transforms(cropped_vol)
        # cropped_vol = cropped_vol.repeat(1, 3, 1, 1) 
        if cropped_vol.dim() == 5:
            cropped_vol = cropped_vol.squeeze(1)  
        return cropped_vol

    def _transforms(self):
        train_transformer = transforms.Compose([
            transforms.Resize((256, 256)),  # (B, C, H, W) -> (B, C, 256, *)
            # transforms.RandomRotation(5),           # 随机旋转 ±30°
            # transforms.RandomResizedCrop(224),       # 随机裁剪并缩放至 224x224
            # transforms.RandomHorizontalFlip(),       # 随机水平翻转
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return train_transformer


    def _resize_torch(self, data, scale):
        return torch.nn.functional.interpolate(data, size=scale, mode="trilinear")

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
        
    def evaluate(self, results, logger=None):
        series_right_all = 0.0
        series_count_all = 0.0
        for result in results:
            pred, gt = result  # torch.Size([7, 6]) torch.Size([1, 7])
            pred = pred.cpu().numpy()
            gt = gt.cpu().numpy()

            pred = np.argmax(pred, axis=1)
            gt = gt[0]
            print(gt, pred)
            pred_results = gt == pred
            series_right_all += np.sum(pred_results)
            series_count_all += pred_results.shape[0]
        return {"series_accurach": float(series_right_all / (series_count_all + 1e-5))}