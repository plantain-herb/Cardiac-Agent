# type: ignore[no-any-return]
import sys
from os.path import abspath, dirname
from typing import IO, Dict
import os
import numpy as np
import random
import SimpleITK as sitk
import torch
import yaml
import tarfile
import functools

def save_nii(arr, output_path_file):
    im_sitk = sitk.GetImageFromArray(arr)
    # im_sitk.SetOrigin(affine.origin)
    # im_sitk.SetSpacing(affine.spacing)
    # im_sitk.SetDirection(affine.direction)
    sitk.WriteImage(im_sitk, output_path_file)
    return


class CineClassificationConfig:
    def __init__(self, network_f, config: Dict):
        # TODO: 模型配置文件

        self.network_f = network_f
        if self.network_f is not None:
            # from mmcv import Config
            from mmengine import Config

            if isinstance(self.network_f, str):
                self.network_cfg = Config.fromfile(self.network_f)
            else:
                import tempfile

                with tempfile.TemporaryDirectory() as temp_config_dir:
                    with tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=".py") as temp_config_file:
                        with open(temp_config_file.name, "wb") as f:
                            f.write(self.network_f.read())

                        self.network_cfg = Config.fromfile(temp_config_file.name)

    def __repr__(self) -> str:
        return str(self.__dict__)


class CineClassificationModel:
    def __init__(
        self, model_f: IO, network_f, config_f,
    ):
        # TODO: 模型文件定制
        self.model_f = model_f
        self.network_f = network_f
        self.config_f = config_f


class CineClassificationPredictor:
    def __init__(self, gpu: int, model: CineClassificationModel):
        self.gpu = gpu
        self.model = model
        if self.model.config_f is not None:
            if isinstance(self.model.config_f, str):
                with open(self.model.config_f, "r") as config_f:
                    self.config = CineClassificationConfig(self.model.network_f, yaml.safe_load(config_f),)
            else:
                self.config = CineClassificationConfig(
                    self.model.network_f, yaml.safe_load(self.model.config_f),
                )
        else:
            self.config = None
        self.load_model()

    @classmethod
    def build_predictor_from_tar(cls, tar: tarfile.TarFile, gpu: int):
        files = tar.getnames()

        model_segUrinary_vessel = CineClassificationModel(
            model_f=tar.extractfile(tar.getmember("Infar_phase.pt")) 
            if "Infar_phase.pt" in files
            else tar.extractfile(tar.getmember("Infar_phase.pth")),
            network_f=tar.extractfile(tar.getmember("phase_cls_config.py")),
            config_f=tar.extractfile(tar.getmember("cls_phase.yaml")),
        )

        return CineClassificationPredictor(gpu=gpu, model=model_segUrinary_vessel)

    def load_model(self) -> None:
        if isinstance(self.model.model_f, str):
            # 根据后缀判断类型
            if self.model.model_f.endswith(".pth"):
                self.load_model_pth()
            else:
                self.load_model_jit()
        else:
            try:
                self.load_model_jit()
            except Exception:
                self.load_model_pth()

    def load_model_jit(self) -> None:
        # 加载静态图
        from torch import jit

        if not isinstance(self.model.model_f, str):
            self.model.model_f.seek(0)
        self.net = jit.load(self.model.model_f, map_location=f"cuda:{self.gpu}")
        self.net.cuda(self.gpu)
    
    def load_model_pth(self) -> None:
        # 加载动态图
        from train.custom.model.utils import build_network

        import importlib.util
        import os
        custom_path = os.path.join(dirname(dirname(abspath(__file__))),"train","custom","__init__.py") 
        spec = importlib.util.spec_from_file_location("custom", custom_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        #sys.modules["custom"] = module
        
        config = self.config.network_cfg

        self.net = build_network(config.model, test_cfg=config.test_cfg)

        if not isinstance(self.model.model_f, str):
            self.model.model_f.seek(0)
        checkpoint = torch.load(self.model.model_f, map_location=f"cuda:{self.gpu}")
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()
        self.net.cuda(self.gpu)

        sys.path.pop()
        remove_names = []
        for k in sys.modules.keys():
            if "custom." in k or "custom" == k or "starship.umtf" in k:
                remove_names.append(k)
        for k in remove_names:
            del sys.modules[k]

    def _normalization(self, vol):
        hu_max = torch.max(vol)
        hu_min = torch.min(vol)
        vol_normalized = (vol - hu_min) / (hu_max - hu_min + 1e-8)
        return vol_normalized
    
    def _get_crop_center(self, vol_shape, seq):
        z_center = vol_shape[0] // 2
        y_center = vol_shape[1] // 2
        if seq == "sa" or seq == "lge": 
            x_center = vol_shape[2] // 2 - 10  # 所有sa的都向左偏移
        else:
            x_center = vol_shape[2] // 2

        z_center = max(0, min(z_center, vol_shape[0] - 1))
        y_center = max(0, min(y_center, vol_shape[1] - 1))
        x_center = max(0, min(x_center, vol_shape[2] - 1))

        center = np.array([z_center, y_center, x_center])

        return center

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

        if cropped_vol.dim() == 6:
            cropped_vol = cropped_vol.squeeze(2)  

        if cropped_vol.dim() != 5:
            raise ValueError(f"输入 Tensor 的维度必须是 5D 或 6D，但得到的是 {cropped_vol.dim()}D")

        return cropped_vol

    # 计算滑动窗口中心（z轴）
    def _get_z_centers(self, vol_shape, crop_z, num_crops):
        z = vol_shape[0]
        if z <= crop_z:
            return [z // 2] * num_crops
        start = crop_z // 2
        end = z - crop_z // 2 - 1
        centers = [int(round(start + (end - start) * i / (num_crops - 1))) for i in range(num_crops)]
        return centers
    
    # def predict(self, vols: list[np.ndarray]):
    def predict(self, vols: list[np.ndarray], num_crops=3): #num_crops：滑动几次

        config = self.config.network_cfg

        crop_shapes = [[80, 192, 192], [288, 144, 144], [9, 144, 144]]  # 4ch,sa,lge
        vols_torch = [torch.from_numpy(v.astype(np.float32))[None, None] for v in vols]
        # print(vols_torch[0].shape)
        
        # 生成每个序列的crop中心
        # z_centers_list: [[4ch_z1,4ch_z2,4ch_z3], [sa_z1,sa_z2,sa_z3], [lge_z1,lge_z2,lge_z3]]
        z_centers_list = [self._get_z_centers(v.shape[2:], crop_shapes[i][0], num_crops) for i, v in enumerate(vols_torch)]
        preds = []

        with torch.no_grad(), torch.cuda.device(self.gpu):
            for i in range(num_crops):
                    datas = []
                    for d in range(3): # 3个序列
                        vol = vols_torch[d] # torch.Tensor [1,1,D,H,W]
                        crop_shape = crop_shapes[d]
                        z_center = z_centers_list[d][i]
                        y_center = vol.shape[-2] // 2
                        if d != 0:  # sa和lge的x轴中心左偏移10
                            x_center = vol.shape[-1] // 2 - 10
                        else:
                            x_center = vol.shape[-1] // 2
                        center = np.array([z_center, y_center, x_center])
                        data = self._crop_data(vol, center, crop_shape)
                        datas.append(data.cuda().detach())

                    pred = self.net.forward_test(*datas)
                    pred = pred.cpu().detach().numpy()[0]
                    preds.append(pred)

        # 对所有预测概率求平均
        preds = np.stack(preds, axis=0)  # preds shape: [num_crops, num_classes]
        avg_pred = np.mean(preds, axis=0)

        return preds, avg_pred


        # phy_centers_4ch = self._get_crop_center(vols[0].shape, "4ch")
        # phy_centers_sa = self._get_crop_center(vols[1].shape, "sa")
        # phy_centers_lge = self._get_crop_center(vols[2].shape, "lge")
        # phy_centers = np.array([phy_centers_4ch, phy_centers_sa, phy_centers_lge])

        # vols_torch = [torch.from_numpy(v.astype(np.float32))[None, None] for v in vols]

        # with torch.no_grad(), torch.cuda.device(self.gpu):
        #     datas = []
        #     data_4ch = self._crop_data(vols_torch[0], phy_centers[0], [80, 192, 192])
        #     data_sa = self._crop_data(vols_torch[1], phy_centers[1], [288, 144, 144])
        #     data_lge = self._crop_data(vols_torch[2], phy_centers[2], [9, 144, 144])
        #     datas.append(data_4ch.cuda().detach())
        #     datas.append(data_sa.cuda().detach())
        #     datas.append(data_lge.cuda().detach())
        #     # flow = flow.cuda().detach()  
        #     pred = self.net.forward_test(*datas)
        #     pred = pred.cpu().detach().numpy()[0]
        
        # return pred

    # def _get_cls_result(self, hu_volume, Infar_phy_center: np.ndarray):
    #     config = self.config.network_cfg

    #     with torch.no_grad(), torch.cuda.device(self.gpu):
    #         data = self._get_cls_input(hu_volume, Infar_phy_center, config)
    #         data = data.cuda().detach()
    #         # flow = flow.cuda().detach()
    #         pred = self.net.forward_test(data)
    #         pred = pred.cpu().detach().numpy()[0]
    #     return pred

    def _get_cls_result(self, vols_torch, phy_centers: np.ndarray):
        """
        vols_torch: list of 3 torch.Tensor, each [1,1,D,H,W]
        phy_centers: list of 3 [z,y,x]
        """
        config = self.config.network_cfg

        with torch.no_grad(), torch.cuda.device(self.gpu):
            # data = self._get_cls_input(hu_volume, Infar_phy_center, config)
            # data = data.cuda().detach()
            datas = []
            for v, c in zip(vols_torch, phy_centers):
                data = self._get_cls_input(v, c, config)
                datas.append(data.cuda().detach())

            # flow = flow.cuda().detach()
            pred = self.net.forward_test(*datas)
            pred = pred.cpu().detach().numpy()[0]
        return pred

    def _get_cls_input(self, hu_volume, c_t, config):
        tgt_shape = config.patch_size
        # vol_shape = np.array(hu_volume.size()[2:])
        # print("vol_shape",vol_shape)
        data = self._crop_data(hu_volume, c_t, tgt_shape)
        # print("shape:",data.shape)
        return data

    def free(self):
        # TODO: add free logic
        if self.net is not None:
            del self.net
        torch.cuda.empty_cache()
