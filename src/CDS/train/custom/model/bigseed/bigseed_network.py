import torch
import torch.nn as nn
from custom.model.utils import build_backbone, build_head
from custom.dataset.utils import build_pipelines
from custom.model.registry import NETWORKS

@NETWORKS.register_module()
class SegHeart_Network(nn.Module):
    def __init__(
        self, backbone, head, apply_sync_batchnorm=False, pipeline=[], train_cfg=None, test_cfg=None
    ):
        super(SegHeart_Network, self).__init__()

        self.backbone = build_backbone(backbone)
        self.head = build_head(head)

        self._pipeline = build_pipelines(pipeline)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

        self._show_count = 1

    @torch.jit.ignore
    def forward(self, img, mask):
        with torch.no_grad():
            # 数据pipeline(augmentation)处理
            # result = self._pipeline(dict(img=img, mask=mask, others=[liver]))
            # img, mask, liver = result['img'], result['mask'], result['others'][0]

            img = img.detach()
            mask = mask.detach()


        outs = self.backbone(img)
        head_outs = self.head(*outs)
        loss = self.head.loss(head_outs, mask)
        return loss

    @torch.jit.export
    def forward_test(self, img):
        img = img.detach()
        outs = self.backbone(img)
        head_outs = self.head(outs[0], outs[1], outs[2])
        return head_outs[2], head_outs[-1]

    def _apply_sync_batchnorm(self):
        print("apply sync batch norm")
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)



@NETWORKS.register_module()
class SegDY_Network(nn.Module):
    def __init__(
        self, backbone, head, apply_sync_batchnorm=False, pipeline=[], train_cfg=None, test_cfg=None
    ):
    # def __init__(
    #     self, backbone, head, other_win_range, apply_sync_batchnorm=False, pipeline=[], train_cfg=None, test_cfg=None
    # ):
        super(SegDY_Network, self).__init__()

        self.backbone = build_backbone(backbone)
        self.head = build_head(head)

        self._pipeline = build_pipelines(pipeline)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

        self._show_count = 1

    @torch.jit.ignore
    def forward(self, img, mask1, mask2):
        with torch.no_grad():
            # 数据pipeline(augmentation)处理
            # img_other = torch.clamp(img, min=self._other_win_range[0], max=self._other_win_range[1])
            # img_other -= self._other_win_range[0]
            # img_other /= self._other_win_range[1] - self._other_win_range[0]

            img = img.detach()
            mask1 = mask1.detach()
            mask2 = mask2.detach()

        # outs = self.backbone(img, img_other)
        outs = self.backbone(img)
        head_outs = self.head(*outs)
        loss = self.head.loss(head_outs, mask1, mask2)
        return loss

    @torch.jit.export
    def forward_test(self, img):
        # img_other = torch.clamp(img, min=self._other_win_range[0], max=self._other_win_range[1])
        # img_other -= self._other_win_range[0]
        # img_other /= self._other_win_range[1] - self._other_win_range[0]

        img = img.detach()
        # outs = self.backbone(img, img_other)
        outs = self.backbone(img)
        head_outs = self.head(outs[0], outs[1])
        # EJB
        # return head_outs[-1]
        # DY
        return head_outs[1], head_outs[-1]

    def _apply_sync_batchnorm(self):
        print("apply sync batch norm")
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)