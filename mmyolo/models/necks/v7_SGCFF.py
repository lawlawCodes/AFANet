# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import CARAFEPack, DeformConv2dPack
from mmdet.models.necks.sgcff import ST
from mmdet.utils import ConfigType, OptMultiConfig
import torch.nn.functional as F

from mmyolo.registry import MODELS
from ..layers import MaxPoolAndStrideConvBlock, RepVGGBlock, SPPFCSPBlock
from .base_yolo_neck import BaseYOLONeck


@MODELS.register_module()
class V7_SGCFF(BaseYOLONeck):

    def __init__(self,
                 in_channels: List[int],
                 out_channels: List[int],
                 block_cfg: dict = dict(
                     type='ELANBlock',
                     middle_ratio=0.5,
                     block_ratio=0.25,
                     num_blocks=4,
                     num_convs_in_block=1),
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 spp_expand_ratio: float = 0.5,
                 is_tiny_version: bool = False,
                 use_maxpool_in_downsample: bool = True,
                 use_in_channels_in_downsample: bool = False,
                 use_repconv_outs: bool = True,
                 upsample_feats_cat_first: bool = False,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):

        self.is_tiny_version = is_tiny_version
        self.use_maxpool_in_downsample = use_maxpool_in_downsample
        self.use_in_channels_in_downsample = use_in_channels_in_downsample
        self.spp_expand_ratio = spp_expand_ratio
        self.use_repconv_outs = use_repconv_outs
        self.block_cfg = block_cfg
        self.block_cfg.setdefault('norm_cfg', norm_cfg)
        self.block_cfg.setdefault('act_cfg', act_cfg)

        super().__init__(
            in_channels=[
                int(channel * widen_factor) for channel in in_channels
            ],
            out_channels=[
                int(channel * widen_factor) for channel in out_channels
            ],
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            upsample_feats_cat_first=upsample_feats_cat_first,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

        self.rebuild = Rebuild(act_cfg=dict(type='LeakyReLU', negative_slope=0.1))

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        if idx == len(self.in_channels) - 1:
            layer = SPPFCSPBlock(
                self.in_channels[idx],
                self.out_channels[idx],
                expand_ratio=self.spp_expand_ratio,
                is_tiny_version=self.is_tiny_version,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            layer = ConvModule(
                self.in_channels[idx],
                self.out_channels[idx],
                1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        return layer

    def build_upsample_layer(self, idx: int) -> nn.Module:
        """build upsample layer."""
        return nn.Sequential(
            ConvModule(
                self.out_channels[idx],
                self.out_channels[idx - 1],
                1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            nn.Upsample(scale_factor=2, mode='nearest'))

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        block_cfg = self.block_cfg.copy()
        block_cfg['in_channels'] = self.out_channels[idx - 1] * 2
        block_cfg['out_channels'] = self.out_channels[idx - 1]
        return MODELS.build(block_cfg)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        if self.use_maxpool_in_downsample and not self.is_tiny_version:
            return MaxPoolAndStrideConvBlock(
                self.out_channels[idx],
                self.out_channels[idx + 1],
                use_in_channels_of_middle=self.use_in_channels_in_downsample,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            return ConvModule(
                self.out_channels[idx],
                self.out_channels[idx + 1],
                3,
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
        block_cfg = self.block_cfg.copy()
        block_cfg['in_channels'] = self.out_channels[idx + 1] * 2
        block_cfg['out_channels'] = self.out_channels[idx + 1]
        return MODELS.build(block_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        """build out layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The out layer.
        """
        if len(self.in_channels) == 4:
            # P6
            return nn.Identity()

        out_channels = self.out_channels[idx] * 2

        if self.use_repconv_outs:
            return RepVGGBlock(
                self.out_channels[idx],
                out_channels,
                3,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            return ConvModule(
                self.out_channels[idx],
                out_channels,
                3,
                padding=1,
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
                 channel3: int = 128,
                 channel4: int = 256,
                 channel5: int = 512,
                 ffn: int = 64,
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001)
                 ):
        super().__init__()

        self.channel5_4 = ConvModule(channel5, channel4, 1,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.rechannel5 = ConvModule(channel4, channel5, 1,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.channel3_4 = ConvModule(channel3, channel4, 1,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.rechannel3 = ConvModule(channel4, channel3, 1,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.down = F.adaptive_max_pool2d
        self.up5_4 = CARAFEPack(channel4, 2)
        self.up4_3 = CARAFEPack(channel4, 2)

        self.st = ST(feedforward_channels=ffn, embed_dims=channel4)

        self.fu5 = nn.Sequential(
            nn.Conv2d(2 * channel5, channel5, 1),
            nn.BatchNorm2d(channel5),
            DeformConv2dPack(in_channels=channel5, out_channels=channel5, kernel_size=3, padding=1)
        )
        self.fu4 = nn.Sequential(
            nn.Conv2d(2 * channel4, channel4, 1),
            nn.BatchNorm2d(channel4),
            DeformConv2dPack(in_channels=channel4, out_channels=channel4, kernel_size=3, padding=1)
        )
        self.fu3 = nn.Sequential(
            nn.Conv2d(2 * channel3, channel3, 1),
            nn.BatchNorm2d(channel3),
            DeformConv2dPack(in_channels=channel3, out_channels=channel3, kernel_size=3, padding=1)
        )


    def forward(self, L5, L4, L3):
        _, _, H4, W4 = L4.shape
        _, _, H5, W5 = L5.shape
        size4 = (H4, W4)
        size5 = (H5, W5)

        B5 = self.channel5_4(L5)
        B3 = self.channel3_4(L3)

        P5_4 = self.up5_4(B5)
        P3_4 = self.down(B3, size4)
        a = (P5_4 + L4 + P3_4) / 3.0
        a = self.st(a)

        A5 = self.rechannel5(self.down(a, size5))
        A5 = torch.cat([A5, L5], dim=1)
        A5 = self.fu5(A5)

        A4 = torch.cat([a, L4], dim=1)
        A4 = self.fu4(A4)

        A3 = self.rechannel3(self.up4_3(a))
        A3 = torch.cat([A3, L3], dim=1)
        A3 = self.fu3(A3)

        return A5, A4, A3
