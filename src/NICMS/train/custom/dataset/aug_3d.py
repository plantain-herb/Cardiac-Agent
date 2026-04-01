"""
Example:
    .. code-block:: python

        >>> import torch
        >>> import SimpleITK as sitk
        >>> import numpy as np
        >>> aug_parameters = {
        "rot_range_x": (-5.0, 5.0),
        "scale_range_y": (0.8, 1.2),
        "elastic_alpha": [3.0, 3.0, 3.0],  # x,y,z
        "smooth_num": 4,
        "field_size": [17, 17, 17],  # x,y,z
        "size_o": [200, 512, 512],
        'out_style': 'none',
        'itp_mode_dict': {'img': 'nearest'}
        }
        >>> pipelines = Augmentation3d(aug_parameters)
        >>> mask_itk = sitk.ReadImage('a.nii.gz')
        >>> mask = sitk.GetArrayFromImage(mask_itk)
        >>> inputs = torch.from_numpy(mask[np.newaxis, np.newaxis]).float().cuda()
        >>> outs = pipelines({'img': inputs})

"""

import math
import random

import torch
import torch.nn as nn

from .registry import PIPELINES


@PIPELINES.register_module()
class Augmentation3d(nn.Module):
    r"""Augmentation 3D on GPUs for 3d segmentation, including: rotation, flip, elastic transform, gray shift and scale

    Args:
        aug_parameters : the parameters for augmentation, a dictionary value, for details:

    Keyword Args:
        rot_range_x: rotation range of along x axes, default: (0.0, 0.0)
        rot_range_y: rotation range of along y axes, default: (0.0, 0.0)
        rot_range_z: rotation range of along z axes, default: (0.0, 0.0)
        scale_range_x: scale range of x axes, The larger scale_range_x is set, the image smaller, default: (1.0, 1.0)
        scale_range_y: scale range of y axes, The larger scale_range_y is set, the image smallers, default: (1.0, 1.0)
        scale_range_z: scale range of z axes, The larger scale_range_z is set, the image smaller, default: (1.0, 1.0)
        shift_range_x: translation range of x axes, default: (0.0, 0.0)
        shift_range_y: translation range of y axes, default: (0.0, 0.0)
        shift_range_z: translation range of z axes, default: (0.0, 0.0)
        flip_x: the bool value of flip or not of x axes, default: False
        flip_y: the bool value of flip or not of y axes, default: False
        flip_z: the bool value of flip or not of z axes, default: False
        elastic_alpha: value for elastic transform, default: 0.0, not use elastic transform, then smooth_num and field_size are useless
        smooth_num: value for elastic transform, the larger is set, the smoother the image is, default: 4
        field_size: value for elastic transform, the larger is set, the smoother the image is, default: [17, 17, 17]
        are values for elastic transform. The smaller elastic_alpha is set, the smoother image is outputted. If elastic_alpha is set as 0, only affine transform is performed
        itp_mode_dict: dict(), 插值方式dict, key为具体字段，如{'img': 'bilinear'}, optional: ``'bilinear'`` | ``'nearest'``. Default: ``'bilinear'``
        pad_mode_list: List[str, ...], padding方式dict, 如{'img': 'zeros'}, optional: ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
        out_style: str,　生成最后固定大小的图像，resize或者crop或者none, optional: ``'crop'`` | ``'resize'`` | ``'none'``. Default: ``'resize'``

        size_o: the output shape
    """

    def __init__(self, aug_parameters: dict):
        super(Augmentation3d, self).__init__()

        self.rot_range_x = aug_parameters.setdefault('rot_range_x', (0., 0.))
        self.rot_range_y = aug_parameters.setdefault('rot_range_y', (0., 0.))
        self.rot_range_z = aug_parameters.setdefault('rot_range_z', (0., 0.))
        self.scale_range_x = aug_parameters.setdefault('scale_range_x', (1., 1.))
        self.scale_range_y = aug_parameters.setdefault('scale_range_y', (1., 1.))
        self.scale_range_z = aug_parameters.setdefault('scale_range_z', (1., 1.))
        self.shift_range_x = aug_parameters.setdefault('shift_range_x', (0., 0.))
        self.shift_range_y = aug_parameters.setdefault('shift_range_y', (0., 0.))
        self.shift_range_z = aug_parameters.setdefault('shift_range_z', (0., 0.))
        self.flip_x = aug_parameters.setdefault('flip_x', 0.2)
        self.flip_y = aug_parameters.setdefault('flip_y', 0.2)
        self.flip_z = aug_parameters.setdefault('flip_z', 0.2)
        self.elastic_alpha = aug_parameters.setdefault('elastic_alpha', (0., 0., 0.))
        self.smooth_num = aug_parameters.setdefault('smooth_num', 4)
        self.field_size = aug_parameters.setdefault('field_size', [17, 17, 17])
        self.out_style = aug_parameters.setdefault('out_style', 'resize')
        self.itp_mode_dict = aug_parameters.setdefault('itp_mode_dict', dict())
        self.pad_mode_dict = aug_parameters.setdefault('pad_mode_dict', dict())
        self.size_o = aug_parameters['size_o']

    def forward(self, data):
        """
        Args:
            data: dict, keys can be followings:

        Keyword Args:
            'img': required, 原始图像
            'mask': optional, 图像标注
            'others': optional, 可选
        Returns:

        """
        img = data['img']
        vol_list = [img]
        itp_mode_list = [self.itp_mode_dict.get('img', 'bilinear')]
        pad_mode_list = [self.pad_mode_dict.get('img', 'zeros')]

        if data.get('mask') is not None:
            vol_list.append(data['mask'])
            itp_mode_list.append(self.itp_mode_dict.get('mask', 'nearest'))
            pad_mode_list.append(self.pad_mode_dict.get('mask', 'zeros'))

        if data.get('others') is not None:
            vol_list.extend(data['others'])
            itp_mode_list.extend(self.itp_mode_dict.get('others', ['bilinear'] * len(data['others'])))
            pad_mode_list.extend(self.pad_mode_dict.get('others', ['zeros'] * len(data['others'])))

        assert len(vol_list) == len(itp_mode_list
                                    ) == len(pad_mode_list), 'vol_list, itp_mode_list and pad_mode_list must match'

        vol_aug_list = self.__data_aug__(vol_list, itp_mode_list, pad_mode_list)
        if self.out_style == 'crop':
            vol_aug_list = self.__crop__(vol_aug_list)
        elif self.out_style == 'resize':
            for i in range(len(vol_aug_list)):
                if itp_mode_list[i] == 'nearest':
                    align_corners = None
                else:
                    align_corners = False
                vol_aug_list[i] = self.__resize__(
                    vol_aug_list[i], self.size_o, mode=itp_mode_list[i], align_corners=align_corners
                )
        elif self.out_style == 'none':
            pass
        else:
            raise ValueError('out_style only in one of crop, resize, none')

        data['img'] = vol_aug_list[0]
        temp_idx = 1
        if data.get('mask') is not None:
            data['mask'] = vol_aug_list[1]
            temp_idx += 1
        if data.get('others') is not None:
            data['others'] = vol_aug_list[temp_idx:]
        return data

    def __crop__(self, vol_list):
        center = [vol_list[0].size(2) // 2, vol_list[0].size(3) // 2, vol_list[0].size(4) // 2]
        z_s = center[0] - self.size_o[0] // 2
        z_e = z_s + self.size_o[0]
        y_s = center[1] - self.size_o[1] // 2
        y_e = y_s + self.size_o[1]
        x_s = center[2] - self.size_o[2] // 2
        x_e = x_s + self.size_o[2]

        vol_crop_list = []
        for vol in vol_list:
            vol_crop = vol[:, :, z_s:z_e, y_s:y_e, x_s:x_e]
            vol_crop_list.append(vol_crop)

        return vol_crop_list

    def __angle_axis_to_rotation_matrix__(self, angle_axis):

        def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
            # We want to be careful to only evaluate the square root if the
            # norm of the angle_axis vector is greater than zero. Otherwise
            # we get a division by zero.
            k_one = 1.0
            theta = torch.sqrt(theta2)
            wxyz = angle_axis / (theta + eps)
            wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            r00 = cos_theta + wx * wx * (k_one - cos_theta)
            r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
            r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
            r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
            r11 = cos_theta + wy * wy * (k_one - cos_theta)
            r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
            r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
            r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
            r22 = cos_theta + wz * wz * (k_one - cos_theta)
            rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
            return rotation_matrix.view(-1, 3, 3)

        def _compute_rotation_matrix_taylor(angle_axis):
            rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
            k_one = torch.ones_like(rx)
            rotation_matrix = torch.cat([k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
            return rotation_matrix.view(-1, 3, 3)

        # stolen from ceres/rotation.h

        _angle_axis = torch.unsqueeze(angle_axis, dim=1)
        theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
        theta2 = torch.squeeze(theta2, dim=1)

        # compute rotation matrices
        rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
        rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

        # create mask to handle both cases
        eps = 1e-6
        mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
        mask_pos = (mask).type_as(theta2)
        mask_neg = (mask == False).type_as(theta2)  # noqa

        # create output pose matrix
        batch_size = angle_axis.shape[0]
        rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
        rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
        # fill output matrix with masked values
        rotation_matrix[..., :3, :3] = \
            mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
        return rotation_matrix  # Nx4x4

    def __affine_elastic_transform_3d_gpu__(
        self,
        vol_list,
        rot,
        scale,
        shift,
        mode_list,
        pad_mode_list,
        alpha=[2.0, 2.0, 2.0],
        smooth_num=4,
        win=[5, 5, 5],
        field_size=[11, 11, 11]
    ):
        aff_matrix = self.__angle_axis_to_rotation_matrix__(rot)  # Nx4x4
        aff_matrix[:, 0, 3] = shift[:, 0]  # * data.size(4)
        aff_matrix[:, 1, 3] = shift[:, 1]  # * data.size(3)
        aff_matrix[:, 2, 3] = shift[:, 2]  # * data.size(2)
        # if scale:
        aff_matrix[:, 0, 0] *= scale[:, 0]
        aff_matrix[:, 1, 0] *= scale[:, 0]
        aff_matrix[:, 2, 0] *= scale[:, 0]
        aff_matrix[:, 0, 1] *= scale[:, 1]
        aff_matrix[:, 1, 1] *= scale[:, 1]
        aff_matrix[:, 2, 1] *= scale[:, 1]
        aff_matrix[:, 0, 2] *= scale[:, 2]
        aff_matrix[:, 1, 2] *= scale[:, 2]
        aff_matrix[:, 2, 2] *= scale[:, 2]

        aff_matrix = aff_matrix[:, 0:3, :]

        grid = torch.nn.functional.affine_grid(aff_matrix, vol_list[0].size()).cuda()

        pad = [win[i] // 2 for i in range(3)]
        fs = field_size
        dz = torch.rand(1, 1, fs[0] + pad[0] * 2, fs[1] + pad[1] * 2, fs[2] + pad[2] * 2).cuda()
        dy = torch.rand(1, 1, fs[0] + pad[0] * 2, fs[1] + pad[1] * 2, fs[2] + pad[2] * 2).cuda()
        dx = torch.rand(1, 1, fs[0] + pad[0] * 2, fs[1] + pad[1] * 2, fs[2] + pad[2] * 2).cuda()
        dz = (dz - 0.5) * 2.0 * alpha[0]
        dy = (dy - 0.5) * 2.0 * alpha[1]
        dx = (dx - 0.5) * 2.0 * alpha[2]

        for _ in range(smooth_num):
            dz = self.__smooth_3d__(dz, win)
            dy = self.__smooth_3d__(dy, win)
            dx = self.__smooth_3d__(dx, win)

        dz = dz[:, :, pad[0]:pad[0] + fs[0], pad[1]:pad[1] + fs[1], pad[2]:pad[2] + fs[2]]
        dy = dy[:, :, pad[0]:pad[0] + fs[0], pad[1]:pad[1] + fs[1], pad[2]:pad[2] + fs[2]]
        dx = dx[:, :, pad[0]:pad[0] + fs[0], pad[1]:pad[1] + fs[1], pad[2]:pad[2] + fs[2]]

        size_3d = [vol_list[0].size(2), vol_list[0].size(3), vol_list[0].size(4)]
        batch_size = vol_list[0].size(0)
        dz = self.__resize__(dz, size_3d).repeat(batch_size, 1, 1, 1, 1)
        dy = self.__resize__(dy, size_3d).repeat(batch_size, 1, 1, 1, 1)
        dx = self.__resize__(dx, size_3d).repeat(batch_size, 1, 1, 1, 1)

        grid[:, :, :, :, 0] += dz[:, 0, :, :, :]
        grid[:, :, :, :, 1] += dy[:, 0, :, :, :]
        grid[:, :, :, :, 2] += dx[:, 0, :, :, :]

        vol_o_list = []
        for i, vol in enumerate(vol_list):
            vol_o = torch.nn.functional.grid_sample(vol, grid, mode_list[i], pad_mode_list[i])
            vol_o_list.append(vol_o)
        return vol_o_list

    def __smooth_3d__(self, vol, win):
        kernel = torch.ones([1, vol.size(1), win[0], win[1], win[2]]).cuda()
        pad_size = [
            (int)((win[2] - 1) / 2), (int)((win[2] - 1) / 2), (int)((win[1] - 1) / 2), (int)((win[1] - 1) / 2),
            (int)((win[0] - 1) / 2), (int)((win[0] - 1) / 2)
        ]
        vol = torch.nn.functional.pad(vol, pad_size, 'replicate')
        vol_s = torch.nn.functional.conv3d(vol, kernel, stride=(1, 1, 1)) / torch.sum(kernel)
        return vol_s

    def __resize__(self, vol, size_tgt, mode='trilinear', align_corners=False):
        if mode == 'bilinear':
            mode = 'trilinear'
        vol_t = nn.functional.interpolate(vol, size=size_tgt, mode=mode, align_corners=align_corners)

        return vol_t

    def __data_aug__(self, vol_list, itp_mode_list, pad_mode_list):
        N = vol_list[0].size(0)
        rand_rot_x = (
            torch.rand(N, 1) * (self.rot_range_x[1] - self.rot_range_x[0]) + self.rot_range_x[0]
        ) / 180 * math.pi
        rand_rot_y = (
            torch.rand(N, 1) * (self.rot_range_y[1] - self.rot_range_y[0]) + self.rot_range_y[0]
        ) / 180 * math.pi
        rand_rot_z = (
            torch.rand(N, 1) * (self.rot_range_z[1] - self.rot_range_z[0]) + self.rot_range_z[0]
        ) / 180 * math.pi
        rand_rot = torch.cat([rand_rot_x, rand_rot_y, rand_rot_z], dim=1).cuda()

        rand_scale_x = torch.rand(N, 1) * (self.scale_range_x[1] - self.scale_range_x[0]) + self.scale_range_x[0]
        rand_scale_y = torch.rand(N, 1) * (self.scale_range_y[1] - self.scale_range_y[0]) + self.scale_range_y[0]
        rand_scale_z = torch.rand(N, 1) * (self.scale_range_z[1] - self.scale_range_z[0]) + self.scale_range_z[0]
        rand_scale = torch.cat([rand_scale_x, rand_scale_y, rand_scale_z], dim=1).cuda()

        rand_shift_x = torch.rand(N, 1) * (self.shift_range_x[1] - self.shift_range_x[0]) + self.shift_range_x[0]
        rand_shift_y = torch.rand(N, 1) * (self.shift_range_y[1] - self.shift_range_y[0]) + self.shift_range_y[0]
        rand_shift_z = torch.rand(N, 1) * (self.shift_range_z[1] - self.shift_range_z[0]) + self.shift_range_z[0]
        rand_shift = torch.cat([rand_shift_x, rand_shift_y, rand_shift_z], dim=1).cuda()

        # mode_list = ["nearest"] * len(vol_list)
        vol_aug_list = self.__affine_elastic_transform_3d_gpu__(
            vol_list,
            rand_rot,
            rand_scale,
            rand_shift,
            itp_mode_list,
            pad_mode_list,
            alpha=self.elastic_alpha,
            smooth_num=self.smooth_num,
            field_size=self.field_size
        )

        # flip

        if random.random() < self.flip_x:
            for i, vol_aug in enumerate(vol_aug_list):
                vol_aug_list[i] = torch.flip(vol_aug_list[i], dims=[4])

        if random.random() < self.flip_y:
            for i, vol_aug in enumerate(vol_aug_list):
                vol_aug_list[i] = torch.flip(vol_aug_list[i], dims=[3])

        if random.random() < self.flip_z:
            for i, vol_aug in enumerate(vol_aug_list):
                vol_aug_list[i] = torch.flip(vol_aug_list[i], dims=[2])

        return vol_aug_list
    

def set_default(param:dict, key:str, default_value, has_prob=True):
    value = default_value
    if key in param:
        value = param[key]
        if has_prob and (isinstance(value, tuple) or 
            isinstance(value, list)) and value[-1] == 0.0:
            value = default_value
    
    return value

@PIPELINES.register_module()
class Aug3dMini(nn.Module):
    r"""Augmentation 3D on GPUs for 3d segmentation, including: rotation, flip, elastic transform, gray shift and scale

    Args:
        aug_parameters : the parameters for augmentation, a dictionary value, for details:

    Keyword Args:
        rot_range_x: rotation range of along x axes, default: (0.0, 0.0, 0.0)
        rot_range_y: rotation range of along y axes, default: (0.0, 0.0, 0.0)
        rot_range_z: rotation range of along z axes, default: (0.0, 0.0, 0.0)
        scale_range_x: scale range of x axes, The larger scale_range_x is set, the image smaller, default: (0.0, 1.0, 1.0)
        scale_range_y: scale range of y axes, The larger scale_range_y is set, the image smallers, default: (0.0, 1.0, 1.0)
        scale_range_z: scale range of z axes, The larger scale_range_z is set, the image smaller, default: (0.0, 1.0, 1.0)
        shift_range_x: translation range of x axes, default: (0.0, 0.0, 0.0)
        shift_range_y: translation range of y axes, default: (0.0, 0.0, 0.0)
        shift_range_z: translation range of z axes, default: (0.0, 0.0, 0.0)
        flip_x: the bool value of flip or not of x axes, default: 0.5
        flip_y: the bool value of flip or not of y axes, default: 0.5
        flip_z: the bool value of flip or not of z axes, default: 0.5
        itp_mode_dict: dict(), 插值方式dict, key为具体字段，如{'img': 'bilinear'}, optional: ``'bilinear'`` | ``'nearest'``. Default: ``'bilinear'``
        pad_mode_list: List[str, ...], padding方式dict, 如{'img': 'zeros'}, optional: ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
    """

    def __init__(self, aug_parameters: dict):
        super(Aug3dMini, self).__init__()

        self.rot_range_x = set_default(aug_parameters, 'rot_range_x', (0., 0., 0.))
        self.rot_range_y = set_default(aug_parameters, 'rot_range_y', (0., 0., 0.))
        self.rot_range_z = set_default(aug_parameters, 'rot_range_z', (0., 0., 0.))
        self.scale_range_x = set_default(aug_parameters, 'scale_range_x', (1., 1., 0.))
        self.scale_range_y = set_default(aug_parameters, 'scale_range_y', (1., 1., 0.))
        self.scale_range_z = set_default(aug_parameters, 'scale_range_z', (1., 1., 0.))
        self.shift_range_x = set_default(aug_parameters, 'shift_range_x', (0., 0., 0.))
        self.shift_range_y = set_default(aug_parameters, 'shift_range_y', (0., 0., 0.))
        self.shift_range_z = set_default(aug_parameters, 'shift_range_z', (0., 0., 0.))
        self.flip_x = set_default(aug_parameters, 'flip_x', 0.)
        self.flip_y = set_default(aug_parameters, 'flip_y', 0.)
        self.flip_z = set_default(aug_parameters, 'flip_z', 0.)
        self.itp_mode_dict = set_default(aug_parameters, 'itp_mode_dict', dict(), False)
        self.pad_mode_dict = set_default(aug_parameters, 'pad_mode_dict', dict(), False)

    def forward(self, data):
        """
        Args:
            data: dict, keys can be followings:

        Keyword Args:
            'img': required, 原始图像
            'mask': optional, 图像标注
            'others': optional, 可选
        Returns:

        """
        img = data['img']
        vol_list = [img]
        itp_mode_list = [self.itp_mode_dict.get('img', 'bilinear')]
        pad_mode_list = [self.pad_mode_dict.get('img', 'zeros')]

        if data.get('mask') is not None:
            vol_list.append(data['mask'])
            itp_mode_list.append(self.itp_mode_dict.get('mask', 'nearest'))
            pad_mode_list.append(self.pad_mode_dict.get('mask', 'zeros'))

        if data.get('others') is not None:
            vol_list.extend(data['others'])
            itp_mode_list.extend(self.itp_mode_dict.get('others', ['bilinear'] * len(data['others'])))
            pad_mode_list.extend(self.pad_mode_dict.get('others', ['zeros'] * len(data['others'])))

        assert len(vol_list) == len(itp_mode_list
                                    ) == len(pad_mode_list), 'vol_list, itp_mode_list and pad_mode_list must match'

        vol_aug_list = self.__data_aug__(vol_list, itp_mode_list, pad_mode_list)
        
        data['img'] = vol_aug_list[0]
        temp_idx = 1
        if data.get('mask') is not None:
            data['mask'] = vol_aug_list[1]
            temp_idx += 1
        if data.get('others') is not None:
            data['others'] = vol_aug_list[temp_idx:]
        return data

    def __angle_axis_to_rotation_matrix__(self, angle_axis):

        def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
            # We want to be careful to only evaluate the square root if the
            # norm of the angle_axis vector is greater than zero. Otherwise
            # we get a division by zero.
            k_one = 1.0
            theta = torch.sqrt(theta2)
            wxyz = angle_axis / (theta + eps)
            wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            r00 = cos_theta + wx * wx * (k_one - cos_theta)
            r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
            r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
            r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
            r11 = cos_theta + wy * wy * (k_one - cos_theta)
            r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
            r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
            r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
            r22 = cos_theta + wz * wz * (k_one - cos_theta)
            rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
            return rotation_matrix.view(-1, 3, 3)

        def _compute_rotation_matrix_taylor(angle_axis):
            rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
            k_one = torch.ones_like(rx)
            rotation_matrix = torch.cat([k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
            return rotation_matrix.view(-1, 3, 3)

        # stolen from ceres/rotation.h

        _angle_axis = torch.unsqueeze(angle_axis, dim=1)
        theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
        theta2 = torch.squeeze(theta2, dim=1)

        # compute rotation matrices
        rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
        rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

        # create mask to handle both cases
        eps = 1e-6
        mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
        mask_pos = (mask).type_as(theta2)
        mask_neg = (mask == False).type_as(theta2)  # noqa

        # create output pose matrix
        batch_size = angle_axis.shape[0]
        rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
        rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
        # fill output matrix with masked values
        rotation_matrix[..., :3, :3] = \
            mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
        return rotation_matrix  # Nx4x4

    def __affine_elastic_transform_3d_gpu__(
        self,
        vol_list,
        rot,
        scale,
        shift,
        mode_list,
        pad_mode_list,
    ):
        aff_matrix = self.__angle_axis_to_rotation_matrix__(rot)  # Nx4x4
        aff_matrix[:, 0, 3] = shift[:, 0]  # * data.size(4)
        aff_matrix[:, 1, 3] = shift[:, 1]  # * data.size(3)
        aff_matrix[:, 2, 3] = shift[:, 2]  # * data.size(2)
        # if scale:
        aff_matrix[:, 0, 0] *= scale[:, 0]
        aff_matrix[:, 1, 0] *= scale[:, 0]
        aff_matrix[:, 2, 0] *= scale[:, 0]
        aff_matrix[:, 0, 1] *= scale[:, 1]
        aff_matrix[:, 1, 1] *= scale[:, 1]
        aff_matrix[:, 2, 1] *= scale[:, 1]
        aff_matrix[:, 0, 2] *= scale[:, 2]
        aff_matrix[:, 1, 2] *= scale[:, 2]
        aff_matrix[:, 2, 2] *= scale[:, 2]

        aff_matrix = aff_matrix[:, 0:3, :]
        aff_matrix = aff_matrix.to(vol_list[0].dtype)
        grid = torch.nn.functional.affine_grid(aff_matrix, vol_list[0].size())

        vol_o_list = []
        for i, vol in enumerate(vol_list):
            vol_o = torch.nn.functional.grid_sample(vol, grid, mode_list[i], pad_mode_list[i])
            vol_o_list.append(vol_o)
        return vol_o_list

    def __data_aug__(self, vol_list, itp_mode_list, pad_mode_list):
        N = vol_list[0].size(0)
        rand_rot_x = (
            torch.rand(N, 1) * (self.rot_range_x[1] - self.rot_range_x[0]) + self.rot_range_x[0]
        ) / 180 * math.pi
        apply_rand = torch.rand(N, 1) >= self.rot_range_x[2]
        rand_rot_x[apply_rand] = 0.0

        rand_rot_y = (
            torch.rand(N, 1) * (self.rot_range_y[1] - self.rot_range_y[0]) + self.rot_range_y[0]
        ) / 180 * math.pi
        apply_rand = torch.rand(N, 1) >= self.rot_range_y[2]
        rand_rot_y[apply_rand] = 0.0

        rand_rot_z = (
            torch.rand(N, 1) * (self.rot_range_z[1] - self.rot_range_z[0]) + self.rot_range_z[0]
        ) / 180 * math.pi
        apply_rand = torch.rand(N, 1) >= self.rot_range_z[2]
        rand_rot_z[apply_rand] = 0.0
        rand_rot = torch.cat([rand_rot_x, rand_rot_y, rand_rot_z], dim=1).cuda()

        rand_scale_x = torch.rand(N, 1) * (self.scale_range_x[1] - self.scale_range_x[0]) + self.scale_range_x[0]
        apply_rand = torch.rand(N, 1) >= self.scale_range_x[2]
        rand_scale_x[apply_rand] = 1.0

        rand_scale_y = torch.rand(N, 1) * (self.scale_range_y[1] - self.scale_range_y[0]) + self.scale_range_y[0]
        apply_rand = torch.rand(N, 1) >= self.scale_range_y[2]
        rand_scale_y[apply_rand] = 1.0

        rand_scale_z = torch.rand(N, 1) * (self.scale_range_z[1] - self.scale_range_z[0]) + self.scale_range_z[0]
        apply_rand = torch.rand(N, 1) >= self.scale_range_z[2]
        rand_scale_z[apply_rand] = 1.0
        rand_scale = torch.cat([rand_scale_x, rand_scale_y, rand_scale_z], dim=1).cuda()

        rand_shift_x = torch.rand(N, 1) * (self.shift_range_x[1] - self.shift_range_x[0]) + self.shift_range_x[0]
        apply_rand = torch.rand(N, 1) >= self.shift_range_x[2]
        rand_shift_x[apply_rand] = 0.0

        rand_shift_y = torch.rand(N, 1) * (self.shift_range_y[1] - self.shift_range_y[0]) + self.shift_range_y[0]
        apply_rand = torch.rand(N, 1) >= self.shift_range_y[2]
        rand_shift_y[apply_rand] = 0.0

        rand_shift_z = torch.rand(N, 1) * (self.shift_range_z[1] - self.shift_range_z[0]) + self.shift_range_z[0]
        apply_rand = torch.rand(N, 1) >= self.shift_range_z[2]
        rand_shift_z[apply_rand] = 0.0

        rand_shift = torch.cat([rand_shift_x, rand_shift_y, rand_shift_z], dim=1).cuda()

        vol_aug_list = self.__affine_elastic_transform_3d_gpu__(
            vol_list,
            rand_rot,
            rand_scale,
            rand_shift,
            itp_mode_list,
            pad_mode_list,
        )

        # flip
        if random.random() < self.flip_x:
            for i, vol_aug in enumerate(vol_aug_list):
                vol_aug_list[i] = torch.flip(vol_aug_list[i], dims=[4])

        if random.random() < self.flip_y:
            for i, vol_aug in enumerate(vol_aug_list):
                vol_aug_list[i] = torch.flip(vol_aug_list[i], dims=[3])

        if random.random() < self.flip_z:
            for i, vol_aug in enumerate(vol_aug_list):
                vol_aug_list[i] = torch.flip(vol_aug_list[i], dims=[2])

        return vol_aug_list
