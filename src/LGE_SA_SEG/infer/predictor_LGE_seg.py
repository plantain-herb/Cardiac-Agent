# type: ignore[no-any-return]
import os
import sys
from os.path import abspath, dirname
from typing import IO, Dict
import cc3d
import numpy as np
import SimpleITK as sitk
import torch
import yaml
from scipy.ndimage import binary_dilation, zoom, binary_erosion, binary_fill_holes
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
import tarfile
import ctypes
import logging
import os
import numpy as np
from typing import Tuple
from numpy.ctypeslib import ndpointer
from train.custom.model.utils import build_network


def boundbox_3d(seg, padding=None):
    points = np.argwhere(seg > 0)
    if len(points) == 0:
        return np.zeros([0, 0, 0, 0, 0, 0], dtype=np.int_)
    zmin, zmax = np.min(points[:, 0]), np.max(points[:, 0])
    ymin, ymax = np.min(points[:, 1]), np.max(points[:, 1])
    xmin, xmax = np.min(points[:, 2]), np.max(points[:, 2])
    if padding is not None:
        zmin = max(0, zmin - padding)
        zmax = min(seg.shape[0], zmax + padding)
        ymin = max(0, ymin - padding)
        ymax = min(seg.shape[1], ymax + padding)
        xmin = max(0, xmin - padding)
        xmax = min(seg.shape[2], xmax + padding)
    return np.array([zmin, ymin, xmin, zmax, ymax, xmax], dtype=np.int_)

class LGESegHeartConfig:
    def __init__(self, network_heart_f, network_heart_crop_f, config: Dict):

        self.network_heart_f = network_heart_f
        self.stage1_thres = config.get('stage1_thres')
        self.stage2_thres = config.get('stage2_thres')
        if self.network_heart_f:
            from mmengine import Config

            if isinstance(self.network_heart_f, str):
                self.network_heart_cfg = Config.fromfile(self.network_heart_f)
            else:
                import tempfile

                with tempfile.TemporaryDirectory() as temp_config_dir:
                    with tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=".py") as temp_config_file:
                        with open(temp_config_file.name, "wb") as f:
                            f.write(self.network_heart_f.read())

                        self.network_heart_cfg = Config.fromfile(temp_config_file.name)

        self.network_heart_crop_f = network_heart_crop_f
        if self.network_heart_crop_f:
            from mmengine import Config

            if isinstance(self.network_heart_crop_f, str):
                self.network_heart_crop_cfg = Config.fromfile(self.network_heart_crop_f)
            else:
                import tempfile

                with tempfile.TemporaryDirectory() as temp_config_dir:
                    with tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=".py") as temp_config_file:
                        with open(temp_config_file.name, "wb") as f:
                            f.write(self.network_heart_crop_f.read())

                        self.network_heart_crop_cfg = Config.fromfile(temp_config_file.name)

    def __repr__(self) -> str:
        return str(self.__dict__)


class LGESegHeartModel:
    def __init__(self, model_heart_f: IO, model_heart_crop_f: IO, network_heart_f, network_heart_crop_f,
                 config_f):
        # TODO: 模型文件定制
        self.model_heart_f = model_heart_f
        self.model_heart_crop_f = model_heart_crop_f
        self.network_heart_f = network_heart_f
        self.network_heart_crop_f = network_heart_crop_f
        self.config_f = config_f


class LGESegHeartPredictor:
    def __init__(self, gpu: int, model: LGESegHeartModel):
        self.gpu = gpu
        self.model = model
        if self.model.config_f is not None:
            if isinstance(self.model.config_f, str):
                with open(self.model.config_f, "r") as config_f:
                    self.config = LGESegHeartConfig(
                        self.model.network_heart_f, self.model.network_heart_crop_f, yaml.safe_load(config_f)
                    )
            else:
                self.config = LGESegHeartConfig(
                    self.model.network_heart_f, self.model.network_heart_crop_f,
                    yaml.safe_load(self.model.config_f)
                )
        else:
            self.config = None  # type:ignore
        # self.win_level = self.config.win_level
        # self.win_width = self.config.win_width
        self.stage1_thres = self.config.stage1_thres
        self.stage2_thres = self.config.stage2_thres
        self.load_model()

    @classmethod
    def build_predictor_from_tar(cls, tar: tarfile.TarFile, gpu: int):
        files = tar.getnames()
        return LGESegHeartPredictor(
            gpu=gpu,
            model=LGESegHeartModel(
                model_heart_f=tar.extractfile(tar.getmember("seg_heart_first.pt")),
                model_heart_crop_f=tar.extractfile(tar.getmember("seg_heart_second.pt")),
                network_heart_f=tar.extractfile(tar.getmember("seg_heart_first_config.py")),
                network_heart_crop_f=tar.extractfile(tar.getmember("seg_heart_second_config.py")),
                config_f=tar.extractfile(tar.getmember("seg_heart_config.yaml")),
            ),
        )

    def load_model(self) -> None:
        self.net_heart = self._load_model(self.model.model_heart_f, self.config.network_heart_cfg, half=True)
        self.net_heart_crop = self._load_model(self.model.model_heart_crop_f,
                                                  self.config.network_heart_crop_cfg, half=True)
    def _load_model(self, model_f, network_f, half=False) -> None:
        if isinstance(model_f, str):
            # 根据后缀判断类型
            if model_f.endswith(".pth"):
                net = self.load_model_pth(model_f, network_f, half)
            else:
                net = self.load_model_jit(model_f, half)
        else:
            try:
                net = self.load_model_jit(model_f, half)
            except Exception:
                net = self.load_model_pth(model_f, network_f, half)
        return net

    def load_model_jit(self, model_f, half) -> None:
        # 加载静态图
        from torch import jit

        if not isinstance(model_f, str):
            model_f.seek(0)
        net = jit.load(model_f, map_location=f"cuda:{self.gpu}")
        # net = jit.load(model_f, map_location=f"cpu")
        net = net.eval()
        if half:
            net.half()
        # net.cuda(self.gpu)
        return net

    def load_model_pth(self, model_f, network_cfg, half) -> None:
        # 加载动态图
        import importlib.util
        import os
        custom_path = os.path.join(dirname(dirname(abspath(__file__))), "train", "custom", "__init__.py")
        spec = importlib.util.spec_from_file_location("custom", custom_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # sys.modules["custom"] = module

        config = network_cfg

        net = build_network(config.model, test_cfg=config.test_cfg)

        if not isinstance(model_f, str):
            model_f.seek(0)
        checkpoint = torch.load(model_f, map_location=f"cuda:{self.gpu}")
        net.load_state_dict(checkpoint["state_dict"], strict=False)
        net.eval()
        if half:
            net.half()
        net.cuda(self.gpu)

        sys.path.pop()
        remove_names = []
        for k in sys.modules.keys():
            if "custom." in k or "custom" == k or "starship.umtf" in k:
                remove_names.append(k)
        for k in remove_names:
            del sys.modules[k]

        return net

    # def load_model_pth(self, model_f, network_cfg, half) -> None:
    #     # 加载动态图
    #     from starship.umtf.common import build_network
    #
    #     sys.path.append(dirname(dirname(abspath(__file__))))
    #     #from train import custom  # noqa: F401
    #
    #     config = network_cfg
    #
    #     net = build_network(config.model, test_cfg=config.test_cfg)
    #
    #     if not isinstance(model_f, str):
    #         model_f.seek(0)
    #     checkpoint = torch.load(model_f, map_location=f"cuda:{self.gpu}")
    #     net.load_state_dict(checkpoint["state_dict"], strict=False)
    #     net.eval()
    #     if half:
    #         net.half()
    #     net.cuda(self.gpu)
    #     return net

    def resize_cpu(self, image, dims, order=0):
        """rescale the dims, such as isotropic transform.
        :param image: 2D or 3D image, dim = depth/channel, height, width
        :param dims: the expected dim for output image
        :param order: order value [0,5], means use nearest or  b-spline method
        :return: resampled image
        """
        image_rs = zoom(
            image,
            np.array(dims) / np.array(image.shape, dtype=np.float32),
            order=order,
            mode="constant",
            cval=image.min(),
        )
        return image_rs

    # def _normalization(self, vol):
    #     global_mean = 282.864980
    #     global_std = 273.596458
    #     vol_normalized = (vol - global_mean) / (global_std + 1e-8)
    #     return vol_normalized

    def _normalization(self, vol):
        hu_max = torch.max(vol)
        hu_min = torch.min(vol)
        vol_normalized = (vol - hu_min) / (hu_max - hu_min + 1e-8)
        return vol_normalized

    def _get_max_con(self, mask):
        import cc3d

        cc_seg, max_labels_count = cc3d.connected_components((mask > 0).astype("uint8"), return_N=True)
        max_label = 1
        max_counts = 0
        for label in range(1, max_labels_count + 1):
            curr_sum = np.sum(cc_seg == label)
            if curr_sum > max_counts:
                max_counts = curr_sum
                max_label = label
        return (cc_seg == max_label).astype("uint8")

    def _get_max_con_2(self, mask):
        bin_mask_new, num = cc3d.connected_components(mask, return_N=True)
        if num == 1 or num == 0:
            bin_mask = bin_mask_new.astype("uint8")
        else:
            labels, counts = np.unique(bin_mask_new, return_counts=True)
            labels = labels.tolist()
            labels = sorted(labels, key=lambda x: counts[labels.index(x)], reverse=True)
            bin_mask = (bin_mask_new == labels[1]).astype("uint8")

        return bin_mask

    def _crop_data(self, vol, c_t, src_shape, src_spacing, patch_size, tgt_spacing, config):
        half_patch_size = [v // 2 for v in patch_size]
        grid, start_point = self._get_sample_grid(
            c_t, half_patch_size, src_spacing, src_shape, tgt_spacing
        )
        grid = grid.cuda(self.gpu)
        vol = torch.nn.functional.grid_sample(
            vol, grid, mode="bilinear", align_corners=True, padding_mode="border"
        )
        vol = self._normalization(vol, config)
        return vol, start_point

    def _get_sample_grid(self, center_point, half_patch_size, src_spacing, src_shape, tgt_spacing):
        grid = []
        start_point = []
        for cent_px, ts, ps_half in zip(center_point, tgt_spacing, half_patch_size):
            p_s = cent_px - ps_half * ts
            p_e = cent_px + ps_half * ts - (ts / 2)
            start_point.append(p_s)
            grid.append(torch.arange(p_s, p_e, ts, device=self.gpu))
        start_point = torch.tensor(start_point)
        grid = torch.meshgrid(*grid)
        grid = list(map(lambda x: x.unsqueeze(-1), grid))  # shape (h,d,w,(zyx))
        grid = torch.cat(grid, dim=-1)
        grid *= 2
        grid /= src_spacing[None, None, None, :]
        grid /= (src_shape - 1)[None, None, None, :]
        grid -= 1
        # change z,y,x to x,y,z
        grid = torch.flip(grid, dims=[3]).unsqueeze(0)
        return grid, start_point

    def _get_final_result(self, vol, roi_range, config):
        # vol = vol.cuda()
        # print(vol.shape)
        ori_shape = np.array(vol.shape[1:])
        if np.any((roi_range[3:] - roi_range[:3]) == 0):
            return np.zeros(np.array(vol.shape[1:])).astype("uint8")
        z_length = ori_shape[0]
        # vol = torch.from_numpy(vol)[None, None].float()
        heat_map1 = torch.zeros((z_length, 160, 160)).cuda(self.gpu)
        heat_map2 = torch.zeros((z_length, 160, 160)).cuda(self.gpu)
        heat_map3 = torch.zeros((z_length, 160, 160)).cuda(self.gpu)
        res = np.zeros((ori_shape[0], ori_shape[1], ori_shape[2]))
        # def _normalization(vol):
        #     hu_max = torch.max(vol)
        #     hu_min = torch.min(vol)
        #     vol_normalized = (vol - hu_min) / (hu_max - hu_min + 1e-8)
        #     return vol_normalized
        def _normalization(self, vol):
            global_mean = 282.864980
            global_std = 273.596458
            vol_normalized = (vol - global_mean) / (global_std + 1e-8)
            return vol_normalized

        # gpu_flag = False
        roi_range = roi_range.astype(np.int_)
        zmin, ymin, xmin, zmax, ymax, xmax = roi_range
        padding = 10
        zmin = max(0, zmin - padding)
        zmax = min(ori_shape[0], zmax + padding)
        ymin = max(0, ymin - padding)
        ymax = min(ori_shape[1], ymax + padding)
        xmin = max(0, xmin - padding)
        xmax = min(ori_shape[2], xmax + padding)

        vol_crop = vol[:, zmin: zmax, ymin: ymax, xmin: xmax]
        # 400*400*350=56000000

        hu_volume = self._normalization(vol_crop)
        config = self.config.network_heart_crop_cfg
        vol_shape = np.array(hu_volume.size()[1:])
        
        pad = (0, 0, 0, 0, 1, 1)  # 最后两个数字对应Z轴的前后填充
        hu_volume_padded = F.pad(hu_volume, pad, mode='constant', value=0)
        hu_volume_padded = hu_volume_padded.cuda(self.gpu)

        with torch.no_grad(), torch.cuda.device(self.gpu):
            net_cuda = self.net_heart_crop.cuda(self.gpu)
            for idx in range(z_length):
                data = hu_volume_padded[:, idx:idx+3, :, :]
                print(idx, data.shape,111111111111)
                data = F.interpolate(
                                    data,
                                    size=(160, 160),
                                    mode='bilinear',  
                                    align_corners=False  # 不强制对齐角落像素
                                )
                data = data.cuda().detach().half()
                pred_seg1, pred_seg2, pred_seg3 = self.net_heart_crop.forward_test(data)
                # -----------
                pred_seg1 = pred_seg1.squeeze()
                pred_seg2 = pred_seg2.squeeze()
                pred_seg3 = pred_seg3.squeeze()
                heat_map1[idx, :, :] += pred_seg1
                heat_map2[idx, :, :] += pred_seg2
                heat_map3[idx, :, :] += pred_seg3
            net_cuda.cpu()
            del net_cuda
            torch.cuda.empty_cache()

        # print(heat_map.shape,22)
        heat_map1 = F.interpolate(
                            heat_map1.unsqueeze(0),
                            size=(int(vol_shape[1]), int(vol_shape[2])),
                            mode='bilinear',  
                            align_corners=False  
                        )
        heat_map2 = F.interpolate(
                            heat_map2.unsqueeze(0),
                            size=(int(vol_shape[1]), int(vol_shape[2])),
                            mode='bilinear',  
                            align_corners=False  
                        )
        heat_map3 = F.interpolate(
                            heat_map3.unsqueeze(0),
                            size=(int(vol_shape[1]), int(vol_shape[2])),
                            mode='bilinear',  
                            align_corners=False  
                        )
        bin_seg1 = (heat_map1[0].detach().cpu().numpy() > self.stage2_thres).astype(np.uint8)
        bin_seg2 = (heat_map2[0].detach().cpu().numpy() > self.stage2_thres).astype(np.uint8)
        bin_seg3 = (heat_map3[0].detach().cpu().numpy() > self.stage2_thres).astype(np.uint8)
        # bin_seg = self.clean_mask_by_maxconnect(bin_seg)
        final_bin_seg_1 = bin_seg1 + bin_seg2 * 2
        final_bin_seg_1[final_bin_seg_1 > 2] = 2
        final_bin_seg = final_bin_seg_1 + bin_seg3 * 3
        final_bin_seg[final_bin_seg > 3] = 3
        res[zmin: zmax, ymin: ymax, xmin: xmax] = final_bin_seg

        return res

    def predict_coarse_heart(self, hu_volume: np.ndarray):
        hu_volume = self._normalization(hu_volume)
        config = self.config.network_heart_cfg
        vol_shape = np.array(hu_volume.size()[1:])
        z_length = vol_shape[0]
        heat_map = torch.zeros((z_length, 512, 512)).cuda(self.gpu)

        pad = (0, 0, 0, 0, 1, 1)  # 最后两个数字对应Z轴的前后填充
        hu_volume_padded = F.pad(hu_volume, pad, mode='constant', value=0)
        hu_volume_padded = hu_volume_padded.cuda(self.gpu)

        with torch.no_grad(), torch.cuda.device(self.gpu):
            net_cuda = self.net_heart.cuda(self.gpu)
            for idx in range(z_length):
                data = hu_volume_padded[:, idx:idx+3, :, :]
                data = F.interpolate(
                                    data,
                                    size=(512, 512),
                                    mode='bilinear',  
                                    align_corners=False  # 不强制对齐角落像素
                                )
                data = data.cuda().detach().half()
                pred_seg = self.net_heart.forward_test(data)
                pred_seg = pred_seg.squeeze()
                

                heat_map[idx, :, :] += pred_seg
            net_cuda.cpu()
            del net_cuda
            torch.cuda.empty_cache()

        # print(heat_map.shape,22)
        heat_map = F.interpolate(
                            heat_map.unsqueeze(0),
                            size=(int(vol_shape[1]), int(vol_shape[2])),
                            mode='bilinear',  # 适合3D医学图像的三线性插值
                            align_corners=False  # 不强制对齐角落像素
                        )
        bin_seg = (heat_map[0].detach().cpu().numpy() > self.stage1_thres).astype(np.uint8)
        bin_seg = self.clean_mask_by_maxconnect(bin_seg)
        if bin_seg.any():
            if np.sum(bin_seg) < 4:
                roi_range = np.concatenate([[0, 0, 0], [vol_shape[0], vol_shape[1], vol_shape[2]]], axis=0)
            else:
                zmin, ymin, xmin, zmax, ymax, xmax = boundbox_3d(bin_seg > 0)
                zmax, ymax, xmax = zmax - 1, ymax - 1, xmax - 1
                roi_range = np.array([zmin, ymin, xmin, zmax, ymax, xmax])

        else:
            roi_range = np.concatenate([[0, 0, 0], [vol_shape[0], vol_shape[1], vol_shape[2]]], axis=0)
        return bin_seg, roi_range

    def _get_corase_input(self, hu_volume, src_spacing, config):
        vol_shape = np.array(hu_volume.size()[2:])
        half_patch_size_inner = np.array(config.patch_size_inner, dtype=np.int_) // 2
        tgt_spacing = np.array([config.isotropy_spacing] * 3)
        ha_range = [(0, v) for v in vol_shape]
        crop_range = [
            np.arange(hr_min * ss + hpsi * ts, hr_max * ss + hpsi * ts, (hpsi * 2 - 2) * ts)
            for (hr_min, hr_max), hpsi, ss, ts in zip(ha_range, half_patch_size_inner, src_spacing, tgt_spacing)
        ]
        # print(vol_shape, vol_phy_shape, crop_range)
        for z_c in crop_range[0]:
            for y_c in crop_range[1]:
                for x_c in crop_range[2]:
                    c_t = np.array([z_c, y_c, x_c])
                    data, sp = self._crop_data(
                        hu_volume,
                        c_t,
                        torch.from_numpy(vol_shape).cuda(self.gpu),
                        torch.from_numpy(src_spacing).cuda(self.gpu),
                        config.patch_size,
                        np.array([config.isotropy_spacing] * 3),
                        config,
                    )
                    yield data, sp

    # def _dynamic_window_level(self, vol, indices, config, win_width=800):
    #     # print('-----------', vol.shape, indices.shape)

    #     points = vol[indices[:, 0], indices[:, 1], indices[:, 2]]
    #     min_hu = np.min(points)
    #     max_hu = np.max(points)
    #     win_width = max_hu - min_hu
    #     bin_count = np.bincount(points - min_hu)
    #     win_level = np.mean(np.argsort(bin_count)[-10:]) + min_hu
    #     config._win_level = win_level
    #     config._win_width = win_width

    def predict(self, img: np.ndarray):
        # 在转换之前复制数组，确保步长为正
        img_copy = img.copy()
        img = torch.from_numpy(img_copy.astype(np.float32)).float()[None]  # type: ignore
        heart_coarse, roi_range = self.predict_coarse_heart(img)
        # src_spacing = np.array(src_spacing)
        # value = np.max(heart_coarse)
        # if value == 0:
        #     heart = np.zeros_like(img[0][0])
        #     return heart.astype("uint8")

        heart_final = self._get_final_result(img, roi_range, self.config.network_heart_crop_cfg)
        # value_final = np.max(heart_final)
        # if value_final != 0:
        #     heart_itk = sitk.GetImageFromArray(heart_final)
        #     heart_itk = sitk.BinaryFillhole(heart_itk)
        #     heart_final = sitk.GetArrayFromImage(heart_itk)
        # return heart_final
        return heart_final

    def post_porcess(self, heart_mask):

        heart_mask = self.open_remove_weak_connect(heart_mask)

        heart_mask = binary_fill_holes(heart_mask)

        mask = np.zeros(heart_mask.shape, dtype="uint8")

        mask[heart_mask > 0] = 1

        return mask

    def open_remove_weak_connect(self, mask, radius=[3, 3, 3]):
        mask = binary_erosion(mask, np.ones(radius), iterations=2)
        mask = binary_dilation(mask, np.ones(radius), iterations=2)
        return mask

    # def open_remove_weak_connect_1(self, mask, radius=[3, 3, 3]):
    #     if np.sum(mask) == 0:
    #         return mask
    #     str_3D = np.ones(radius)
    #     imMask = sitk.GetImageFromArray(mask)
    #     z_min, z_max, y_min, y_max, x_min, x_max = boundbox_3d(mask)
    #     crop_bbox = (x_min, y_min, z_min, x_max - x_min, y_max - y_min, z_max - z_min)
    #     imMask = st.cropByBoundingBox(imMask, crop_bbox)
    #     imMask = sitk.GetArrayFromImage(imMask).astype('uint8')
    #     imMask = binary_erosion(imMask, str_3D, iterations=2)
    #     imMask = binary_dilation(imMask, str_3D, iterations=2)
    #     vol = np.zeros_like(mask)
    #     shape = vol.shape
    #     vol[max(0, crop_bbox[2]):min(shape[0], crop_bbox[2] + crop_bbox[5]),
    #     max(0, crop_bbox[1]):min(shape[1], crop_bbox[1] + crop_bbox[4]),
    #     max(0, crop_bbox[0]):min(shape[2], crop_bbox[0] + crop_bbox[3])] = imMask
    #     return vol

    def open_remove_weak_connect_2(self, mask):
        # mask = dilate_erode_cuda(mask, flag=2, connect_size=3, iterations=2, gpu=self.gpu)
        mask = dilate_erode_cuda(mask, flag=0, connect_size=3, iterations=2, gpu=self.gpu)
        mask = dilate_erode_cuda(mask, flag=1, connect_size=3, iterations=2, gpu=self.gpu)
        return mask

    # @staticmethod
    def clean_mask_by_maxconnect(self, mask):
        """
        提取3D掩码中最大的连通域，并返回仅包含该连通域的01掩码
        
        :param mask: 输入的3D掩码（可以是0和非0值组成的数组）
        :return: 仅包含最大连通域的01掩码（1表示最大连通域，0表示背景）
        """
        # 确保输入是二值掩码（非0值视为前景）
        binary_mask = (mask > 0).astype(np.int32)
        
        # 计算连通分量
        connected_components, num_components = cc3d.connected_components(
            binary_mask, 
            return_N=True
        )
        
        # 如果没有连通分量，返回全0掩码
        if num_components == 0:
            return np.zeros_like(binary_mask, dtype=np.int32)
        
        # 找到最大的连通域
        max_volume = -1
        max_component_id = 0
        for component_id in range(1, num_components + 1):
            # 计算当前连通域的体积
            component_volume = np.sum(connected_components == component_id)
            if component_volume > max_volume:
                max_volume = component_volume
                max_component_id = component_id
        
        # 生成仅包含最大连通域的01掩码（1表示最大连通域，0表示其他）
        max_connect_mask = (connected_components == max_component_id).astype(np.int32)
        
        return max_connect_mask

    def _cal_patch_size(self, size_3d, scale_xy=False):
        if scale_xy:
            scale = [1.0, 512 / size_3d[1], 512 / size_3d[2]]
        else:
            scale = [1.0, 1.0, 1.0]
        patch_size = self.config.patch_size
        patch_size = [int(round(patch_size[i] / scale[i])) for i in range(3)]
        return patch_size

    def _cal_core_size(self, patch_size):
        ratio = [self.config.core_size[i] / self.config.patch_size[i] for i in range(3)]
        core_size = [int(patch_size[i] * ratio[i]) for i in range(3)]
        return core_size

    def _add_padding(self, vol, patch_size, core_size):
        gap = [(patch_size[i] - core_size[i]) // 2 for i in range(3)]
        z_min = 0
        y_min = 0
        x_min = 0
        z_max = vol.shape[0]
        y_max = vol.shape[1]
        x_max = vol.shape[2]
        vol = vol[z_min:z_max, y_min:y_max, x_min:x_max]
        pad = (gap[0], patch_size[0], gap[1], patch_size[1], gap[2], patch_size[2])
        vol = torch.nn.functional.pad(vol, pad, "constant", value=0)
        start_point = gap
        return vol, start_point

    @staticmethod
    def _get_patch_position_list(size_3d, patch_size, core_size):
        patch_pos_list = []
        for z in range(0, size_3d[0] - patch_size[0], core_size[0]):
            for y in range(0, size_3d[1] - patch_size[1], core_size[1]):
                for x in range(0, size_3d[2] - patch_size[2], core_size[2]):
                    patch_pos_list.append([z, y, x])

        return patch_pos_list

    @staticmethod
    def resize_torch(data, scale):
        return torch.nn.functional.interpolate(data, size=scale, mode="trilinear", align_corners=True)

    def free(self):
        # TODO: add free logic
        if self.net is not None:
            del self.net
        torch.cuda.empty_cache()
