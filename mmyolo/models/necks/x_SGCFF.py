# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.ops import CARAFEPack, DeformConv2dPack
from mmdet.models.backbones.csp_darknet import CSPLayer
from mmdet.models.necks.sgcff import ST
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from .base_yolo_neck import BaseYOLONeck
import torch.nn.functional as F


@MODELS.register_module()
class X_SGCFF(BaseYOLONeck):

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 use_depthwise: bool = False,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        self.num_csp_blocks = round(num_csp_blocks * deepen_factor)
        self.use_depthwise = use_depthwise

        super().__init__(
            in_channels=[
                int(channel * widen_factor) for channel in in_channels
            ],
            out_channels=int(out_channels * widen_factor),
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

        self.rebuild = Rebuild(channels=128, ffn=32)

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        if idx == 2:
            layer = ConvModule(
                self.in_channels[idx],
                self.in_channels[idx - 1],
                1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            layer = nn.Identity()

        return layer

    def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
        """build upsample layer."""
        return nn.Upsample(scale_factor=2, mode='nearest')

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        if idx == 1:
            return CSPLayer(
                self.in_channels[idx - 1] * 2,
                self.in_channels[idx - 1],
                num_blocks=self.num_csp_blocks,
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        elif idx == 2:
            return nn.Sequential(
                CSPLayer(
                    self.in_channels[idx - 1] * 2,
                    self.in_channels[idx - 1],
                    num_blocks=self.num_csp_blocks,
                    add_identity=False,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    self.in_channels[idx - 1],
                    self.in_channels[idx - 2],
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule
        return conv(
            self.in_channels[idx],
            self.in_channels[idx],
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return CSPLayer(
            self.in_channels[idx] * 2,
            self.in_channels[idx + 1],
            num_blocks=self.num_csp_blocks,
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        """build out layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The out layer.
        """
        return ConvModule(
            self.in_channels[idx],
            self.out_channels,
            1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)


    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                                                     feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        results[2], results[1], results[0] = self.rebuild(results[2], results[1], results[0])

        return tuple(results)


class Rebuild(nn.Module):
    def __init__(self,
                 channels: int = 96,
                 ffn: int = 24
                 ):
        super().__init__()
        self.channels = channels
        self.down = F.adaptive_max_pool2d
        self.up5_4 = CARAFEPack(channels, 2)
        self.up4_3 = CARAFEPack(channels, 2)
        self.st = ST(embed_dims=channels, feedforward_channels=ffn)

        self.fu5 = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.BatchNorm2d(channels),
            DeformConv2dPack(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        )
        self.fu4 = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.BatchNorm2d(channels),
            DeformConv2dPack(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        )
        self.fu3 = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.BatchNorm2d(channels),
            DeformConv2dPack(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        )

    def forward(self, L5, L4, L3):
        _, _, H4, W4 = L4.shape
        _, _, H5, W5 = L5.shape
        size4 = (H4, W4)
        size5 = (H5, W5)

        P5_4 = self.up5_4(L5)
        P3_4 = self.down(L3, size4)

        a = (P5_4 + L4 + P3_4) / 3.0
        a = self.st(a)

        A5 = torch.cat([self.down(a, size5), L5], dim=1)
        A5 = self.fu5(A5)

        A4 = torch.cat([a, L4], dim=1)
        A4 = self.fu4(A4)

        A3 = torch.cat([self.up4_3(a), L3], dim=1)
        A3 = self.fu3(A3)

        return A5, A4, A3
