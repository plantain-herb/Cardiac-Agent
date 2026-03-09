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
import infer.sitktools as st
from scipy.ndimage import binary_dilation, zoom, binary_erosion, binary_fill_holes, generate_binary_structure
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
import tarfile
from scipy.ndimage import morphology
import os
import numpy as np
from typing import Tuple
from numpy.ctypeslib import ndpointer
from train.custom.model.utils import build_network
from skimage.morphology import skeletonize
# from starboost.dense.region_growth import region_growth_3d
# from starboost.dense.connected_components import connected_components_labeling
# from starboost.base import DenseTensor

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

class CtAbdomenSegDYConfig:
    def __init__(self, network_DY_f, network_DY_crop_f, network_DY_crop_r, config: Dict):

        self.network_DY_f = network_DY_f
        if self.network_DY_f:
            from mmengine.config import Config

            if isinstance(self.network_DY_f, str):
                self.network_DY_cfg = Config.fromfile(self.network_DY_f)
            else:
                import tempfile

                with tempfile.TemporaryDirectory() as temp_config_dir:
                    with tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=".py") as temp_config_file:
                        with open(temp_config_file.name, "wb") as f:
                            f.write(self.network_DY_f.read())

                        self.network_DY_cfg = Config.fromfile(temp_config_file.name)
        self.network_DY_crop_f = network_DY_crop_f
        if self.network_DY_crop_f:
            from mmengine.config import Config

            if isinstance(self.network_DY_crop_f, str):
                self.network_DY_crop_cfg = Config.fromfile(self.network_DY_crop_f)
            else:
                import tempfile

                with tempfile.TemporaryDirectory() as temp_config_dir:
                    with tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=".py") as temp_config_file:
                        with open(temp_config_file.name, "wb") as f:
                            f.write(self.network_DY_crop_f.read())

                        self.network_DY_crop_cfg = Config.fromfile(temp_config_file.name)

        self.network_DY_crop_r = network_DY_crop_r
        if self.network_DY_crop_r:
            from mmengine.config import Config

            if isinstance(self.network_DY_crop_r, str):
                self.network_DY_cropR_cfg = Config.fromfile(self.network_DY_crop_r)
            else:
                import tempfile

                with tempfile.TemporaryDirectory() as temp_config_dir:
                    with tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=".py") as temp_config_file:
                        with open(temp_config_file.name, "wb") as f:
                            f.write(self.network_DY_crop_r.read())

                        self.network_DY_cropR_cfg = Config.fromfile(temp_config_file.name)

        # self.win_level = self.network_DY_cfg.get("win_level")
        # self.win_width = self.network_DY_cfg.get("win_width")

    def __repr__(self) -> str:
        return str(self.__dict__)


class CtAbdomenSegDYModel:
    def __init__(self, model_DY_f: IO, model_DY_crop_f: IO, model_DY_crop_r: IO, network_DY_f, network_DY_crop_f, network_DY_crop_r, config_f):
        # TODO: 模型文件定制
        self.model_DY_f = model_DY_f
        self.network_DY_f = network_DY_f
        self.model_DY_crop_f = model_DY_crop_f
        self.network_DY_crop_f = network_DY_crop_f
        self.model_DY_crop_r = model_DY_crop_r
        self.network_DY_crop_r = network_DY_crop_r
        self.config_f = config_f


class CtAbdomenSegDYPredictor:
    def __init__(self, gpu: int, model: CtAbdomenSegDYModel):
        self.gpu = gpu
        self.model = model
        if self.model.config_f is not None:
            if isinstance(self.model.config_f, str):
                with open(self.model.config_f, "r") as config_f:
                    self.config = CtAbdomenSegDYConfig(
                        self.model.network_DY_f, self.model.network_DY_crop_f, self.model.network_DY_crop_r, yaml.safe_load(config_f)
                    )
            else:
                self.config = CtAbdomenSegDYConfig(
                    self.model.network_DY_f,
                    self.model.network_DY_crop_f,
                    self.model.network_DY_crop_r,
                    yaml.safe_load(self.model.config_f)
                )
        else:
            self.config = None  # type:ignore
        self.load_model()

    @classmethod
    def build_predictor_from_tar(cls, tar: tarfile.TarFile, gpu: int):
        files = tar.getnames()
        return CtAbdomenSegDYPredictor(
            gpu=gpu,
            model=CtAbdomenSegDYModel(
                model_DY_f=tar.extractfile(tar.getmember("seg_DY_first.pt")),
                network_DY_f=tar.extractfile(tar.getmember("seg_DY_first_config.py")),
                config_f=tar.extractfile(tar.getmember("seg_DY_config.yaml")),
            ),
        )

    def load_model(self) -> None:
        self.net_DY = self._load_model(self.model.model_DY_f, self.config.network_DY_cfg, half=False)
        self.net_DY_crop = self._load_model(self.model.model_DY_crop_f,
                                                  self.config.network_DY_crop_cfg, half=False)
        self.net_DY_cropR = self._load_model(self.model.model_DY_crop_r,
                                                  self.config.network_DY_cropR_cfg, half=False)

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
    
    def dilate_erode(self, vol, flag=1, connect_size=3, iterations=1):
        """
        实现膨胀、腐蚀、开运算和闭运算
        :param vol: 输入的 3D 张量
        :param flag: 操作标志，0 表示腐蚀，1 表示膨胀，2 表示开运算，3 表示闭运算
        :param connect_size: 连接大小，用于定义结构元素的大小
        :param iterations: 迭代次数
        :return: 处理后的 3D 张量
        """
        if isinstance(vol, torch.Tensor):
            vol = vol.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
        else:
            vol = torch.tensor(vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        padding = (connect_size - 1) // 2

        def erosion(vol):
            for _ in range(iterations):
                vol = F.max_pool3d(-vol, kernel_size=connect_size, stride=1, padding=padding)
                vol = -vol
            return vol

        def dilation(vol):
            for _ in range(iterations):
                vol = F.max_pool3d(vol, kernel_size=connect_size, stride=1, padding=padding)
            return vol

        if flag == 0:
            # 腐蚀
            result = erosion(vol)
        elif flag == 1:
            # 膨胀
            result = dilation(vol)
        elif flag == 2:
            # 开运算（先腐蚀后膨胀）
            result = erosion(vol)
            result = dilation(result)
        elif flag == 3:
            # 闭运算（先膨胀后腐蚀）
            result = dilation(vol)
            result = erosion(result)
        else:
            raise ValueError("Invalid flag value. Allowed values are 0, 1, 2, 3.")

        return result.squeeze(0).squeeze(0)

    def DY_predict(
        self, hu_volume: np.ndarray, src_spacing: np.ndarray
    ):
        img_copy = hu_volume.copy()
        img = torch.from_numpy(img_copy.astype(np.float32)).float()[None, None] 
        heart_mask, roi_range = self.predict_coarse(img, src_spacing)

        DY_mask1, DY_mask2, DY_mask6 = self._predict_L(
            hu_volume, heart_mask, src_spacing, pyramid=True
        )
        DY_mask1 = self._postprocess_DY(DY_mask1)
        DY_mask2 = self._postprocess_DY(DY_mask2)
        DY_mask6 = self._postprocess_DY(DY_mask6)
        # print(np.sum(DY_mask1), np.sum(DY_mask2), np.sum(DY_mask6))
        DY_mask_L = DY_mask1 + 2 * DY_mask2 + 6 * DY_mask6
        DY_mask_L[(DY_mask_L ==3)|(DY_mask_L ==8)|(DY_mask_L == 9)] = 2 
        DY_mask_L[DY_mask_L ==7] = 1 

        DY_mask3, DY_mask4, DY_mask5 = self._predict_R(
            hu_volume, heart_mask, src_spacing, pyramid=True
        )
        DY_mask3 = self._postprocess_DY(DY_mask3)
        DY_mask4 = self._postprocess_DY(DY_mask4)
        DY_mask5 = self._postprocess_DY(DY_mask5)
        # print(np.sum(DY_mask3), np.sum(DY_mask4), np.sum(DY_mask5))
        DY_mask_R = DY_mask3 * 3 + 4 * DY_mask4 + 5 * DY_mask5
        DY_mask_R[(DY_mask_R ==7)|(DY_mask_R ==9)|(DY_mask_R ==12)] = 4 
        DY_mask_R[DY_mask_R ==8] = 3 
        DY_mask_L[DY_mask_L == 6] = 10
        DY_mask_R[DY_mask_R == 5] = 20
        DY_mask = DY_mask_L + DY_mask_R
        DY_mask[DY_mask == 11] = 6
        DY_mask[DY_mask == 5] = 2
        DY_mask[DY_mask == 7] = 2
        DY_mask[DY_mask == 6] = 2
        DY_mask[DY_mask == 10] = 6
        DY_mask[DY_mask == 22] = 2
        DY_mask[DY_mask == 30] = 6
        DY_mask[DY_mask == 20] = 5
        

        print(np.unique(DY_mask), np.unique(DY_mask_R), np.unique(DY_mask_L))
        return DY_mask
    

    def _get_sample_grid_stage1(self, center_point, half_patch_size, src_spacing, src_shape, tgt_spacing):
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
    
    def _crop_data_stage1(self, vol, c_t, src_shape, src_spacing, patch_size, tgt_spacing, config):
        half_patch_size = [v // 2 for v in patch_size]
        grid, start_point = self._get_sample_grid_stage1(
            c_t, half_patch_size, src_spacing, src_shape, tgt_spacing
        )
        grid = grid.cuda(self.gpu)
        vol = torch.nn.functional.grid_sample(
            vol, grid, mode="bilinear", align_corners=True, padding_mode="border"
        )
        vol = self._normalization(vol)
        return vol, start_point
    
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
                    data, sp = self._crop_data_stage1(
                        hu_volume,
                        c_t,
                        torch.from_numpy(vol_shape).cuda(self.gpu),
                        torch.from_numpy(src_spacing).cuda(self.gpu),
                        config.patch_size,
                        np.array([config.isotropy_spacing] * 3),
                        config,
                    )
                    yield data, sp


    def predict_coarse(self, hu_volume: np.ndarray, src_spacing: np.ndarray):
        config = self.config.network_DY_cfg
        patch_size = torch.from_numpy(np.array(config.patch_size))
        patch_size_inner = torch.from_numpy(np.array(config.patch_size_inner))
        vol_shape = np.array(hu_volume.size()[2:])
        result_shape = vol_shape * src_spacing / config.isotropy_spacing + 1
        result_shape = result_shape.astype(np.int_)
        result_shape_t = torch.from_numpy(result_shape)
        heat_map = torch.zeros(tuple(result_shape)).cuda(self.gpu)
        # print("heatmap shape", heat_map.shape, spacing_zyx)
        heat_map_counter = torch.zeros(tuple(result_shape), dtype=torch.int16).cuda(self.gpu)
        hu_volume = hu_volume.cuda(self.gpu)
        with torch.no_grad(), torch.cuda.device(self.gpu):
            net_cuda = self.net_DY.cuda(self.gpu)
            for idx, (data, start_point) in enumerate(BackgroundGenerator(self._get_corase_input(hu_volume, src_spacing, config), max_prefetch=1)):
                data = data.cuda().detach() # .half()
                pred_seg = self.net_DY.forward_test(data)
                pred_seg = pred_seg.squeeze()
                p_s = start_point / config.isotropy_spacing
                p_s_origin = torch.round(p_s).int()

                p_s = p_s_origin + torch.div((patch_size - patch_size_inner), 2, rounding_mode='floor')
                p_e = p_s + patch_size_inner
                p_s1 = torch.div((patch_size - patch_size_inner), 2, rounding_mode='floor')
                p_e1 = p_s1 + patch_size_inner

                p_s[1:] = p_s_origin[1:]
                p_e[1:] = p_s[1:] + patch_size[1:]
                p_s1[1:] = 0
                p_e1[1:] = patch_size[1:]

                p_s_delta = (p_s < 0) * torch.abs(p_s)
                p_s1 = p_s1 + p_s_delta
                p_s = torch.tensor(list(map(lambda x: torch.clamp(x[0], 0, x[1]), zip(p_s, result_shape))))

                p_e_delta = (p_e > result_shape_t) * torch.abs(p_e - result_shape)
                p_e1 = p_e1 - p_e_delta
                p_e = torch.tensor(list(map(lambda x: torch.clamp(x[0], 0, x[1]), zip(p_e, result_shape))))

                heat_map[p_s[0]: p_e[0], p_s[1]: p_e[1], p_s[2]: p_e[2]] += pred_seg[
                                                                            p_s1[0]: p_e1[0], p_s1[1]: p_e1[1],
                                                                            p_s1[2]: p_e1[2]
                                                                            ]
                heat_map_counter[p_s[0]: p_e[0], p_s[1]: p_e[1], p_s[2]: p_e[2]] += 1
            net_cuda.cpu()
            del net_cuda
            torch.cuda.empty_cache()

        heat_map_counter = torch.clamp(heat_map_counter, min=1, max=256)
        heat_map /= heat_map_counter
        heat_map = heat_map.unsqueeze(0).unsqueeze(0)
        z_step = 1 / config.isotropy_spacing * src_spacing[0]
        zs = (np.arange(0, result_shape[0], z_step)[: vol_shape[0]] - result_shape[0] / 2.0) / (result_shape[0] / 2.0)
        y_step = 1 / config.isotropy_spacing * src_spacing[1]
        ys = (np.arange(0, result_shape[1], y_step)[: vol_shape[1]] - result_shape[1] / 2.0) / (result_shape[1] / 2.0)
        x_step = 1 / config.isotropy_spacing * src_spacing[2]
        xs = (np.arange(0, result_shape[2], x_step)[: vol_shape[2]] - result_shape[2] / 2.0) / (result_shape[2] / 2.0)
        zs, ys, xs = torch.from_numpy(zs), torch.from_numpy(ys), torch.from_numpy(xs)
        grid = torch.meshgrid([zs, ys, xs])
        grid = list(map(lambda x: x.unsqueeze(-1), grid))
        grid = torch.cat(grid, dim=-1)
        grid = torch.flip(grid, dims=[3]).unsqueeze(0).float()

        # 300 * 512 * 512
        if np.prod(vol_shape) < 300 * 512 * 512:
            grid = grid.cuda(self.gpu)
            heat_map = torch.nn.functional.grid_sample(
                heat_map, grid, align_corners=True
            )
        else:
            heat_map = torch.nn.functional.grid_sample(heat_map.cpu(), grid, align_corners=True)

        bin_seg = (heat_map[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        bin_seg_maxconn = self._get_max_con_2(bin_seg)
        if bin_seg_maxconn.any():
            if np.sum(bin_seg_maxconn) < 4:
                roi_range = np.concatenate([[0, 0, 0], [vol_shape[0], vol_shape[1], vol_shape[2]]], axis=0)
            else:
                zmin, ymin, xmin, zmax, ymax, xmax = boundbox_3d(bin_seg_maxconn > 0)
                zmax, ymax, xmax = zmax - 1, ymax - 1, xmax - 1
                roi_range = np.array([zmin, ymin, xmin, zmax, ymax, xmax])

        else:
            roi_range = np.concatenate([[0, 0, 0], [vol_shape[0], vol_shape[1], vol_shape[2]]], axis=0)
        return bin_seg_maxconn, roi_range
    
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


    def _predict_L(
        self,
        hu_volume: np.ndarray,
        heart_mask: np.ndarray,
        src_spacing: np.ndarray,
        pyramid=False,
    ):
        DY_mask1, DY_mask2, DY_mask3 = self._get_DY_mask_L(
            hu_volume, heart_mask, src_spacing, pyramid
        )
        return DY_mask1, DY_mask2, DY_mask3
    
    def _predict_R(
        self,
        hu_volume: np.ndarray,
        heart_mask: np.ndarray,
        src_spacing: np.ndarray,
        pyramid=False,
    ):
        DY_mask1, DY_mask2, DY_mask3 = self._get_DY_mask_R(
            hu_volume, heart_mask, src_spacing, pyramid
        )
        return DY_mask1, DY_mask2, DY_mask3

    def _postprocess_DY(self, DY_mask):
        config = self.config.network_DY_cfg
        from scipy.ndimage import morphology

        def get_max_conn(mask):
            bin_mask_new, num = cc3d.connected_components((mask > 0).astype("uint8"), return_N=True, connectivity=6)
            if num == 1 or num == 0:
                bin_mask = bin_mask_new.astype("uint8")
            else:
                labels, counts = np.unique(bin_mask_new, return_counts=True)
                labels = labels.tolist()
                labels = sorted(labels, key=lambda x: counts[labels.index(x)], reverse=True)
                bin_mask_1 = (bin_mask_new == labels[1]).astype("uint8")
                bin_mask = bin_mask_1
            return bin_mask

        result = get_max_conn(DY_mask)
        # result = get_max_conn_2d(DY_mask)

        return result

    def _get_phy_crop_box(self, vol_shape, spacing_zyx, heart_mask, config):

        roi_bbox = self._get_bbox(
            heart_mask, shift_upper=[40, 40, 40], shift_lower=[40, 40, 40]
        )

        ha_range = (
            (roi_bbox[0], roi_bbox[3]),
            (roi_bbox[1], roi_bbox[4]),
            (roi_bbox[2], roi_bbox[5]),
        )  # ha_range = ((zmin, zmax), (ymin, ymax), (xmin, xmax))

        half_patch_size = np.array(config.patch_size, dtype=np.int_) // 2  # type:ignore
        half_patch_size_inner = (
            np.array(config.patch_size_inner, dtype=np.int_) // 2
        )  # type:ignore

        tgt_spacing = np.array(config.isotropy_spacing)
        crop_range = [
            np.arange(
                hr_min * ss + hpsi * ts, hr_max * ss + hpsi * ts, (hpsi * 2 - 2) * ts
            )
            for (hr_min, hr_max), hpsi, ss, ts in zip(
                ha_range, half_patch_size_inner, spacing_zyx, tgt_spacing
            )
        ]
        crop_outer_box = [
            (np.min(cr) - hpsi * ts, np.max(cr) + hpsi * ts)
            for cr, hpsi, ts in zip(crop_range, half_patch_size, tgt_spacing)
        ]

        return crop_range, crop_outer_box


    def _get_bbox(self, mask, shift_upper=0, shift_lower=0):
        from typing import Sequence

        if isinstance(shift_upper, Sequence):
            shift_z_upper, shift_y_upper, shift_x_upper = shift_upper
        else:
            shift_z_upper, shift_y_upper, shift_x_upper = (
                shift_upper,
                shift_upper,
                shift_upper,
            )

        if isinstance(shift_lower, Sequence):
            shift_z_lower, shift_y_lower, shift_x_lower = shift_lower
        else:
            shift_z_lower, shift_y_lower, shift_x_lower = (
                shift_lower,
                shift_lower,
                shift_lower,
            )

        shape = mask.shape

        if np.sum(mask) < 500:
            z_min_ori, y_min_ori, x_min_ori, z_max_ori, y_max_ori, x_max_ori  = 0, 0, 0, mask.shape[0], mask.shape[1], mask.shape[2]
        else:
            z_min_ori, y_min_ori, x_min_ori, z_max_ori, y_max_ori, x_max_ori = boundbox_3d(mask > 0)
        coor = np.array([z_min_ori, y_min_ori, x_min_ori, z_max_ori, y_max_ori, x_max_ori])

        if np.prod(coor[3:] - coor[:3]) < 10:
            return np.array([0] * 6)

        coor[0] = max(0, coor[0] - shift_z_lower)
        coor[1] = max(0, coor[1] - shift_y_lower)
        coor[2] = max(0, coor[2] - shift_x_lower)
        coor[3] = min(shape[0], coor[3] + shift_z_upper)
        coor[4] = min(shape[1], coor[4] + shift_y_upper)
        coor[5] = min(shape[2], coor[5] + shift_x_upper)

        return coor

    # def _window_array(self, vol, config):
    #     win = [
    #         config.win_level - config.win_width / 2,
    #         config.win_level + config.win_width / 2,
    #     ]
    #     vol = torch.clamp(vol, win[0], win[1])
    #     vol -= win[0]
    #     vol /= config.win_width
    #     return vol
    def _normalization(self, vol):
        hu_max = torch.max(vol)
        hu_min = torch.min(vol)
        vol_normalized = (vol - hu_min) / (hu_max - hu_min + 1e-8)
        return vol_normalized
    
    def robust_scaler(self, X):
        median = torch.median(X)
        q1 = torch.quantile(X, 0.05) 
        q3 = torch.quantile(X, 0.95) 
        iqr = q3 - q1
        scaled_X = (X - median) / iqr
        scaled_X_nor = self._normalization(scaled_X)
        return scaled_X_nor

    def _crop_data(self, vol, c_t, src_shape, src_spacing, config, device, pyramid=False):
        patch_size, tgt_spacing = config.patch_size, np.array(config.isotropy_spacing)
        half_patch_size = [v // 2 for v in patch_size]

        if not pyramid:
            grid, start_point = self._get_sample_grid(
                c_t, half_patch_size, src_spacing, src_shape, tgt_spacing, device, rot_mat=None
            )
            # grid = grid.cuda(device=self.gpu)
            vol = torch.nn.functional.grid_sample(
                vol, grid, mode="bilinear", align_corners=True, padding_mode="border"
            )
            vol = self._normalization(vol, config)
        else:
            data_pyramid = []
            for level in range(config.data_pyramid_level):
                grid, sp = self._get_sample_grid(
                    c_t,
                    half_patch_size,
                    src_spacing,
                    src_shape,
                    tgt_spacing,
                    device, 
                    rot_mat=None,
                )
                if level == 0:
                    start_point = sp
                vol_patch = torch.nn.functional.grid_sample(
                    vol,
                    grid,
                    mode="bilinear",
                    align_corners=True,
                    padding_mode="border",
                )[0]
                vol_patch = self._normalization(vol_patch)
                data_pyramid.append(vol_patch)
                tgt_spacing = tgt_spacing * config.data_pyramid_step

            data_pyramid = data_pyramid[::-1]
            vol = torch.cat(data_pyramid, dim=0)[None]
        return vol, start_point

    def _get_sample_grid(
        self,
        center_point,
        half_patch_size,
        src_spacing,
        src_shape,
        tgt_spacing,
        device, 
        rot_mat=None,
    ):
        grid = []
        start_point = []
        for cent_px, ts, ps_half in zip(center_point, tgt_spacing, half_patch_size):
            p_s = cent_px - ps_half * ts
            p_e = cent_px + ps_half * ts - (ts / 2)
            start_point.append(p_s)
            grid.append(torch.arange(p_s, p_e, ts))

        start_point = np.array(start_point)
        grid = torch.meshgrid(*grid)
        grid = [g[:, :, :, None] for g in grid]
        grid = torch.cat(grid, dim=-1)  # shape (d,h,w,(zyx))

        grid = grid.to(device)

        if rot_mat is not None:
            grid -= center_point[None, None, None, :]
            grid = torch.matmul(grid, torch.linalg.inv(rot_mat))
            grid += center_point[None, None, None, :]
        grid *= 2
        grid /= src_spacing[None, None, None, :]
        grid /= (src_shape - 1)[None, None, None, :]
        grid -= 1

        grid = torch.flip(grid, dims=[3])[None]
        return grid, start_point

    def _pre_process(self, hu_vulume, src_shape, spacing_zyx, crop_range, config, device, pyramid):
        hu_vulume = hu_vulume.copy()
        compose_data = torch.from_numpy(hu_vulume.astype(np.float32)).float()[None, None].to(device)
        src_shape = torch.from_numpy(src_shape).to(device)
        spacing_zyx = torch.from_numpy(spacing_zyx.copy()).to(device)

        for z_t in crop_range[0]:
            for y_t in crop_range[1]:
                for x_t in crop_range[2]:
                    c_t = (z_t, y_t, x_t)
                    data, sp = self._crop_data(
                        compose_data, c_t, src_shape, spacing_zyx, config, device, pyramid
                    )
                    yield data, sp

    def _get_DY_mask_L(
        self,
        hu_volume: np.ndarray,
        heart_mask: np.ndarray,
        spacing_zyx: np.ndarray,
        pyramid: bool,
        constant_shift: int = 30,
    ):
        # 获取血管分割结果
        config = self.config.network_DY_crop_cfg
        patch_size = np.array(config.patch_size)
        patch_size_inner = np.array(config.patch_size_inner)
        ori_shape = np.array(hu_volume.shape)
        isotropy_spacing = np.array(config.isotropy_spacing)
        device = "cpu"
        if np.prod(ori_shape) < 600 * 512 * 512:
            device = f"cuda:{self.gpu}"

        crop_range, _ = self._get_phy_crop_box(
            ori_shape, spacing_zyx, heart_mask, config
        )
        heat_map1 = np.zeros(tuple(ori_shape))
        heat_map_counter1 = np.zeros(tuple(ori_shape))

        heat_map2 = np.zeros(tuple(ori_shape))
        heat_map_counter2 = np.zeros(tuple(ori_shape))

        heat_map3 = np.zeros(tuple(ori_shape))
        heat_map_counter3 = np.zeros(tuple(ori_shape))

        with torch.no_grad(), torch.cuda.device(self.gpu):  # type: ignore
            for idx, (data, start_point) in enumerate(
                BackgroundGenerator(
                    self._pre_process(
                        hu_volume, ori_shape, spacing_zyx, crop_range, config, device, pyramid
                    ),
                    max_prefetch=1,
                )
            ):
                data = data.detach().cuda()

                pred_seg1, pred_seg2, pred_seg3 = self.net_DY_crop.forward_test(data)
                pred_seg1 = torch.sigmoid(pred_seg1)
                pred_seg1 = pred_seg1.detach()[0, 0]

                pred_seg2 = torch.sigmoid(pred_seg2)
                pred_seg2 = pred_seg2.detach()[0, 0]

                pred_seg3 = torch.sigmoid(pred_seg3)
                pred_seg3 = pred_seg3.detach()[0, 0]

                p_s = (
                    start_point
                    + ((patch_size - patch_size_inner) // 2) * isotropy_spacing
                )
                p_e = p_s + patch_size_inner * isotropy_spacing
                p_s_pixel = np.round(p_s / spacing_zyx).astype(np.int_)  # type: ignore
                p_e_pixel = np.round(p_e / spacing_zyx).astype(np.int_)  # type: ignore

                p_s_pixel = np.clip(p_s_pixel, 0, ori_shape - 1)
                p_e_pixel = np.clip(p_e_pixel, 0, ori_shape)

                p_s = p_s_pixel * spacing_zyx

                tmp_crop_shape = p_e_pixel - p_s_pixel
                bin_crop_array1, bin_crop_array2, bin_crop_array3 = self._crop_back(
                    pred_seg1.to(device),
                    pred_seg2.to(device),
                    pred_seg3.to(device),   
                    p_s,
                    start_point,
                    torch.from_numpy(patch_size).to(device),  # type: ignore
                    tmp_crop_shape,
                    torch.from_numpy(isotropy_spacing).to(device),  # type: ignore
                    torch.from_numpy(spacing_zyx.copy()).to(device),  # type: ignore
                    device,
                )

                p_s = p_s_pixel
                p_e = p_e_pixel
                heat_map1[p_s[0] : p_e[0], p_s[1] : p_e[1], p_s[2] : p_e[2]] += (
                    bin_crop_array1.float().detach().cpu().numpy()
                )
                heat_map_counter1[p_s[0] : p_e[0], p_s[1] : p_e[1], p_s[2] : p_e[2]] += 1

                heat_map2[p_s[0] : p_e[0], p_s[1] : p_e[1], p_s[2] : p_e[2]] += (
                    bin_crop_array2.float().detach().cpu().numpy()
                )
                heat_map_counter2[p_s[0] : p_e[0], p_s[1] : p_e[1], p_s[2] : p_e[2]] += 1

                heat_map3[p_s[0] : p_e[0], p_s[1] : p_e[1], p_s[2] : p_e[2]] += (
                    bin_crop_array3.float().detach().cpu().numpy()
                )
                heat_map_counter3[p_s[0] : p_e[0], p_s[1] : p_e[1], p_s[2] : p_e[2]] += 1

        with torch.cuda.device(self.gpu):
            torch.cuda.empty_cache()

        heat_map_counter1 = np.clip(heat_map_counter1, a_min=1, a_max=256)  # type: ignore
        heat_map1 /= heat_map_counter1

        heat_map_counter2 = np.clip(heat_map_counter2, a_min=1, a_max=256)  # type: ignore
        heat_map2 /= heat_map_counter2

        heat_map_counter3 = np.clip(heat_map_counter3, a_min=1, a_max=256)  # type: ignore
        heat_map3 /= heat_map_counter3

        return (heat_map1 > 0.65).astype(np.uint8), (heat_map2 > 0.65).astype(np.uint8), (heat_map3 > 0.75).astype(np.uint8)
    
    def _get_DY_mask_R(
        self,
        hu_volume: np.ndarray,
        heart_mask: np.ndarray,
        spacing_zyx: np.ndarray,
        pyramid: bool,
        constant_shift: int = 30,
    ):
        # 获取血管分割结果
        config = self.config.network_DY_crop_cfg
        patch_size = np.array(config.patch_size)
        patch_size_inner = np.array(config.patch_size_inner)
        ori_shape = np.array(hu_volume.shape)
        isotropy_spacing = np.array(config.isotropy_spacing)
        device = "cpu"
        if np.prod(ori_shape) < 600 * 512 * 512:
            device = f"cuda:{self.gpu}"

        crop_range, _ = self._get_phy_crop_box(
            ori_shape, spacing_zyx, heart_mask, config
        )
        heat_map1 = np.zeros(tuple(ori_shape))
        heat_map_counter1 = np.zeros(tuple(ori_shape))

        heat_map2 = np.zeros(tuple(ori_shape))
        heat_map_counter2 = np.zeros(tuple(ori_shape))

        heat_map3 = np.zeros(tuple(ori_shape))
        heat_map_counter3 = np.zeros(tuple(ori_shape))

        with torch.no_grad(), torch.cuda.device(self.gpu):  # type: ignore
            for idx, (data, start_point) in enumerate(
                BackgroundGenerator(
                    self._pre_process(
                        hu_volume, ori_shape, spacing_zyx, crop_range, config, device, pyramid
                    ),
                    max_prefetch=1,
                )
            ):
                data = data.detach().cuda()

                pred_seg1, pred_seg2, pred_seg3 = self.net_DY_cropR.forward_test(data)
                pred_seg1 = torch.sigmoid(pred_seg1)
                pred_seg1 = pred_seg1.detach()[0, 0]

                pred_seg2 = torch.sigmoid(pred_seg2)
                pred_seg2 = pred_seg2.detach()[0, 0]

                pred_seg3 = torch.sigmoid(pred_seg3)
                pred_seg3 = pred_seg3.detach()[0, 0]

                p_s = (
                    start_point
                    + ((patch_size - patch_size_inner) // 2) * isotropy_spacing
                )
                p_e = p_s + patch_size_inner * isotropy_spacing
                p_s_pixel = np.round(p_s / spacing_zyx).astype(np.int_)  # type: ignore
                p_e_pixel = np.round(p_e / spacing_zyx).astype(np.int_)  # type: ignore

                p_s_pixel = np.clip(p_s_pixel, 0, ori_shape - 1)
                p_e_pixel = np.clip(p_e_pixel, 0, ori_shape)

                p_s = p_s_pixel * spacing_zyx

                tmp_crop_shape = p_e_pixel - p_s_pixel
                bin_crop_array1, bin_crop_array2, bin_crop_array3 = self._crop_back(
                    pred_seg1.to(device),
                    pred_seg2.to(device),
                    pred_seg3.to(device),   
                    p_s,
                    start_point,
                    torch.from_numpy(patch_size).to(device),  # type: ignore
                    tmp_crop_shape,
                    torch.from_numpy(isotropy_spacing).to(device),  # type: ignore
                    torch.from_numpy(spacing_zyx.copy()).to(device),  # type: ignore
                    device,
                )

                p_s = p_s_pixel
                p_e = p_e_pixel
                heat_map1[p_s[0] : p_e[0], p_s[1] : p_e[1], p_s[2] : p_e[2]] += (
                    bin_crop_array1.float().detach().cpu().numpy()
                )
                heat_map_counter1[p_s[0] : p_e[0], p_s[1] : p_e[1], p_s[2] : p_e[2]] += 1

                heat_map2[p_s[0] : p_e[0], p_s[1] : p_e[1], p_s[2] : p_e[2]] += (
                    bin_crop_array2.float().detach().cpu().numpy()
                )
                heat_map_counter2[p_s[0] : p_e[0], p_s[1] : p_e[1], p_s[2] : p_e[2]] += 1

                heat_map3[p_s[0] : p_e[0], p_s[1] : p_e[1], p_s[2] : p_e[2]] += (
                    bin_crop_array3.float().detach().cpu().numpy()
                )
                heat_map_counter3[p_s[0] : p_e[0], p_s[1] : p_e[1], p_s[2] : p_e[2]] += 1

        with torch.cuda.device(self.gpu):
            torch.cuda.empty_cache()

        heat_map_counter1 = np.clip(heat_map_counter1, a_min=1, a_max=256)  # type: ignore
        heat_map1 /= heat_map_counter1

        heat_map_counter2 = np.clip(heat_map_counter2, a_min=1, a_max=256)  # type: ignore
        heat_map2 /= heat_map_counter2

        heat_map_counter3 = np.clip(heat_map_counter3, a_min=1, a_max=256)  # type: ignore
        heat_map3 /= heat_map_counter3

        return (heat_map1 > 0.65).astype(np.uint8), (heat_map2 > 0.75).astype(np.uint8), (heat_map3 > 0.75).astype(np.uint8)

    def _crop_back(
        self,
        src_array1,
        src_array2,
        src_array3,
        start_point_tgt,
        start_point_src,
        src_shape,
        tgt_shape,
        src_spacing,
        tgt_spacing,
        device,
        mode="bilinear",
    ):
        grid = []
        for spt, sps, tsp, tsh in zip(
            start_point_tgt, start_point_src, tgt_spacing, tgt_shape
        ):
            p_s = spt - sps
            p_e = p_s + tsp * tsh - tsp / 2
            grid.append(torch.arange(p_s, p_e, tsp))
        grid = torch.meshgrid(*grid)
        grid = [g[:, :, :, None] for g in grid]
        grid = torch.cat(grid, dim=-1)  # shape (d,h,w,(zyx))
        
        grid = grid.to(device)

        grid *= 2
        grid /= src_spacing[None, None, None, :]
        grid /= (src_shape - 1)[None, None, None, :]
        grid -= 1
        # change z,y,x to x,y,z
        grid = torch.flip(grid, dims=[3])[None]
        ret1 = torch.nn.functional.grid_sample(
            src_array1[None, None],
            grid,
            mode=mode,
            align_corners=True,
            padding_mode="border",
        )[0, 0]
        ret2 = torch.nn.functional.grid_sample(
            src_array2[None, None],
            grid,
            mode=mode,
            align_corners=True,
            padding_mode="border",
        )[0, 0]
        ret3 = torch.nn.functional.grid_sample(
            src_array3[None, None],
            grid,
            mode=mode,
            align_corners=True,
            padding_mode="border",
        )[0, 0]
        return ret1, ret2, ret3


    def free(self):
        # TODO: add free logic
        if self.net is not None:
            del self.net
        torch.cuda.empty_cache()
