import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from custom.model.utils import build_backbone, build_head
from custom.dataset.utils import build_pipelines
from custom.model.registry import NETWORKS
from typing import Optional, Union, List
from prefetch_generator import BackgroundGenerator

@NETWORKS.register_module()
class Gl_Heart_Network(nn.Module):

    def __init__(
        self,
        backbone,
        head,
        apply_sync_batchnorm=False,
        pipeline=[],
        train_cfg=None,
        test_cfg=None
    ):
        super(Gl_Heart_Network, self).__init__()

        self.backbone = build_backbone(backbone)
        self.head = build_head(head)

        self._pipeline = build_pipelines(pipeline)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

    @torch.jit.ignore
    def forward(self, outer_img, outer_mask, inner_img, inner_mask, inner_grids):
        outs, _ = self.backbone(outer_img, inner_img, inner_grids, None)
        head_outs = self.head(outs)
        loss = self.head.loss(head_outs, [outer_mask, inner_mask])
        return loss

    @torch.jit.export
    def forward_test(self, outer_img, inner_img, inner_grids, outer_feature: Optional[List[torch.Tensor]]=None):
        outs, outer_feature = self.backbone(outer_img, inner_img, inner_grids, outer_feature)
        head_outs, head_outs_color = self.head.forward_test(outs)

        # pred_bin = (torch.sigmoid(head_outs) > 0.5).cpu().numpy().astype(np.uint8)[0]

        # pred_color = torch.argmax(head_outs_color, dim=1)
        # pred_color += 1
        # pred_color = pred_color.cpu().numpy().astype(np.uint8)[0]
        # pred_color = pred_color * pred_bin

        return head_outs, head_outs_color, outer_feature

    def cal_dice(self, mask1, mask2, smooth=1e-5):
        sum_mask1 = torch.sum(mask1)
        sum_mask2 = torch.sum(mask2)
        inter = torch.sum(mask2 * mask1)
        return (2 * inter + smooth) / (sum_mask1 + sum_mask2 + smooth)

    def single_test(self, img, seg, src_spacing):
        patch_size_outer = np.array([192, 192, 192])
        patch_size_block = np.array([96, 96, 96])
        patch_size_block_inner = np.array([72, 72, 72])
        tgt_spacing_block = 0.8
        tgt_spacing_block = np.array([tgt_spacing_block] * 3)
        win_level_list = [60, 90]
        win_width_list = [300, 150]
        loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)
        img = img.cpu()
        src_spacing = src_spacing[0].cpu()

        def _get_inner_crop_box(src_shape, src_spacing):
            zmin, ymin, xmin = 0, 0, 0
            zmax, ymax, xmax = src_shape
            vessel_range = ((zmin, zmax), (ymin, ymax), (xmin, xmax))

            half_patch_size = patch_size_block // 2
            half_patch_size = half_patch_size.astype(np.int_)

            half_patch_size_inner = patch_size_block_inner // 2
            half_patch_size_inner = half_patch_size_inner.astype(np.int_)

            crop_range = [
                np.arange(hr_min * ss + hpsi * ts, hr_max * ss + hpsi * ts, (hpsi * 2 - 2) * ts)
                for (hr_min, hr_max), hpsi, ss, ts in zip(vessel_range, half_patch_size_inner, src_spacing, tgt_spacing_block)
            ]
            return crop_range

        def _window_array(vol, win_level, win_width):
            win = [win_level - win_width / 2, win_level + win_width / 2]
            vol = torch.clamp(vol, win[0], win[1])
            vol -= win[0]
            vol /= win_width
            return vol
        
        def _get_outer_inputs(vol, src_shape, src_spacing):
            # 获取outer输入
            vessel_width = src_shape
            vessel_center = src_shape / 2

            outer_tgt_spacing = src_spacing * vessel_width / patch_size_outer
            outer_tgt_spacing = np.array([np.max(outer_tgt_spacing)] * 3)
            outer_ct = vessel_center * src_spacing

            outer_grid, start_points = _get_sample_grid(outer_ct, patch_size_outer // 2, torch.from_numpy(src_spacing), torch.from_numpy(src_shape), outer_tgt_spacing)
            vol = F.grid_sample(vol, grid=outer_grid, align_corners=True)

            if vol.shape[1] == 1:
                vol = [_window_array(vol, win_level, win_width) for win_level, win_width in zip(win_level_list, win_width_list)]
                vol = torch.cat(vol, dim=1)
            else:
                vol_volume, gauss_mask = torch.split(vol, dim=1, split_size_or_sections=1)
                vol_window = [_window_array(vol_volume, win_level, win_width) for win_level, win_width in zip(win_level_list, win_width_list)]
                vol = torch.cat(vol_window, dim=1)

            return vol, outer_tgt_spacing, outer_grid, start_points

        def _get_inner_inputs(vol, outer_shape, src_spacing, outer_spacing, inner_spacing, outer_sp, src_shape, crop_range):
            src_shape = np.array(vol.shape[2:])
            half_patch_size = patch_size_block // 2

            for z_t in crop_range[0]:
                for y_t in crop_range[1]:
                    for x_t in crop_range[2]:
                        c_t = (z_t, y_t, x_t)
                        inner_grid, inner_sp = _get_sample_grid(c_t, half_patch_size, torch.from_numpy(src_spacing), torch.from_numpy(src_shape), inner_spacing, None)
                        inner_vol = F.grid_sample(vol, grid=inner_grid, align_corners=True)
                        
                        if inner_vol.shape[1] == 1:
                            inner_vol = [_window_array(inner_vol, win_level, win_width) for win_level, win_width in zip(win_level_list, win_width_list)]
                            inner_vol = torch.cat(inner_vol, dim=1)
                            #vol = _window_array(vol, config._win_level, config._win_width)
                        else:
                            inner_vol_window = [_window_array(inner_vol[:, :1], win_level, win_width) for win_level, win_width in zip(win_level_list, win_width_list)]
                            inner_vol_window = torch.cat(inner_vol_window, dim=1)
                            inner_vol = torch.cat([inner_vol_window, inner_vol[:, 1:]], dim=1)

                        c_t_inner_outer = np.array(c_t - outer_sp)
                        inner_outer_grid, _ = _get_sample_grid(c_t_inner_outer, half_patch_size // 2, torch.from_numpy(outer_spacing), torch.from_numpy(outer_shape), inner_spacing * 2, None)
                        yield inner_vol, inner_sp, inner_outer_grid
            return vol

        def _get_sample_grid(center_point, half_patch_size, src_spacing, src_shape, tgt_spacing, rot_mat=None):
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

            if rot_mat is not None:
                grid -= center_point[None, None, None, :]
                grid = torch.matmul(grid, torch.linalg.inv(rot_mat))
                grid += center_point[None, None, None, :]
            grid *= 2
            grid /= src_spacing[None, None, None, :]
            grid /= (src_shape - 1)[None, None, None, :]
            grid -= 1
            # change z,y,x to x,y,z
            grid = torch.flip(grid, dims=[3])[None]
            return grid, start_point

        def _crop_back( src_array, start_point_tgt, start_point_src, src_shape, tgt_shape, src_spacing, tgt_spacing, mode='bilinear', ):
            grid = []
            for spt, sps, tsp, tsh in zip(start_point_tgt, start_point_src, tgt_spacing, tgt_shape):
                p_s = spt - sps
                p_e = p_s + tsp * tsh - tsp / 2
                grid.append(torch.arange(p_s, p_e, tsp))
            grid = torch.meshgrid(*grid)
            grid = [g[:, :, :, None] for g in grid]
            grid = torch.cat(grid, dim=-1)  # shape (d,h,w,(zyx))

            grid *= 2
            grid /= src_spacing[None, None, None, :]
            grid /= (src_shape - 1)[None, None, None, :]
            grid -= 1
            # change z,y,x to x,y,z
            grid = torch.flip(grid, dims=[3])[None]
            ret = torch.nn.functional.grid_sample(
                src_array[None, None], grid, mode=mode, align_corners=True, padding_mode='border'
            )[0, 0]
            return ret

        src_shape = np.array(img.shape[2:])
        src_spacing = np.array(src_spacing)
        outer_vol, outer_tgt_spacing, outer_grid, outer_sp = _get_outer_inputs(img, src_shape, src_spacing)
        outer_shape = np.array(outer_vol.shape[2:])
        crop_inner_range = _get_inner_crop_box(src_shape, src_spacing)
        heat_map = np.zeros(tuple(src_shape))
        heat_map_counter = np.zeros(tuple(src_shape))

        outer_feature = None
        with torch.no_grad():
            for idx, (inner_vol, inner_sp, inner_outer_grid) in enumerate(
                    BackgroundGenerator(
                        _get_inner_inputs(img, outer_shape, src_spacing, outer_tgt_spacing, tgt_spacing_block, outer_sp, src_shape, crop_inner_range), max_prefetch=1
                    )
                ):
                inner_vol = inner_vol.detach().cuda()
                inner_outer_grid = inner_outer_grid.detach().cuda()

                
                pred_seg, _ = self.forward_test(outer_vol.cuda(), inner_vol[:, None], inner_outer_grid[:, None], outer_feature)

                pred_seg = pred_seg.detach()[0, 0].cpu().float()
                p_s = inner_sp + ((patch_size_block - patch_size_block_inner) // 2) * tgt_spacing_block
                p_e = p_s + patch_size_block_inner * tgt_spacing_block
                p_s_pixel = np.round(p_s / src_spacing).astype(np.int_)
                p_e_pixel = np.round(p_e / src_spacing).astype(np.int_)

                p_s_pixel = np.clip(p_s_pixel, 0, src_shape - 1)
                p_e_pixel = np.clip(p_e_pixel, 0, src_shape)

                p_s = p_s_pixel * src_spacing

                tmp_crop_shape = p_e_pixel - p_s_pixel
                crop_array = _crop_back(
                    pred_seg,
                    p_s,
                    inner_sp,
                    torch.from_numpy(patch_size_block),
                    tmp_crop_shape,
                    torch.tensor(tgt_spacing_block),
                    torch.from_numpy(src_spacing),
                )

                p_s = p_s_pixel
                p_e = p_e_pixel
                heat_map[p_s[0]:p_e[0], p_s[1]:p_e[1], p_s[2]:p_e[2]] += (crop_array.float().detach().numpy())
                heat_map_counter[p_s[0]:p_e[0], p_s[1]:p_e[1], p_s[2]:p_e[2]] += 1
        heat_map_counter = np.clip(heat_map_counter, a_min=1, a_max=256)
        heat_map /= heat_map_counter
        seg = seg[0,0].cpu()
        heat_map = torch.from_numpy(heat_map).float()
        loss = loss_func(heat_map, seg)
        loss = loss.mean()
        dice = self.cal_dice(heat_map, seg)
        dice_loss = 1.0 - dice
        return [loss.numpy(), dice_loss.numpy()]

    def _apply_sync_batchnorm(self):
        print('apply sync batch norm')
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)
