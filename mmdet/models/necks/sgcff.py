from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor
from mmdet.models.backbones.swin import SwinBlockSequence
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
from mmcv.ops.deform_conv import DeformConv2dPack
from mmcv.ops.carafe import CARAFEPack


@MODELS.register_module()
class SGCFF(BaseModule):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        init_cfg: MultiConfig = dict(type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:
            self.add_extra_convs = 'on_input'

        self.rebuild = Rebuild()
        self.ah = AHLayer(256)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

        laterals[3], laterals[2], laterals[1], laterals[0] = self.rebuild(laterals[3], laterals[2], laterals[1], laterals[0])

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        outs[-1] = self.ah(outs[-1])

        return tuple(outs)


class AHLayer(BaseModule):
    def __init__(self,
                 channel,
                 reduction=16,
                 spatial_kernel=7,
                 init_cfg: MultiConfig = dict(type='Xavier', layer='Conv2d', distribution='uniform')
                 ) -> None:
        super(AHLayer, self).__init__(init_cfg=init_cfg)

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class ST(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        feedforward_channels=64,
        depth=2,
        window_size=7
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.LN1 = nn.LayerNorm(embed_dims)
        self.swinhead = SwinBlockSequence(
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            depth=depth,
            window_size=window_size
        )
        self.LN2 = nn.LayerNorm(embed_dims)

    def forward(self, x):
        shortcut = x
        hw_shape = (x.shape[2], x.shape[3])
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.LN1(x)
        x, shape, _, _ = self.swinhead(x, hw_shape)
        x = self.LN2(x)
        x = x.view(-1, *shape, self.embed_dims).permute(0, 3, 1, 2).contiguous()
        out = x + shortcut

        return out


class Rebuild(nn.Module):
    def __init__(self,
                 channels: int = 256,
                 ffn: int = 64
                 ):
        super().__init__()
        self.channels = channels
        self.down = F.adaptive_max_pool2d
        self.up5_4 = CARAFEPack(channels, 2)
        self.up4_3 = CARAFEPack(channels, 2)
        self.up4_2 = CARAFEPack(channels, 4)

        self.st = ST(feedforward_channels=ffn)

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
        self.fu2 = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.BatchNorm2d(channels),
            DeformConv2dPack(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        )

    def forward(self, L5, L4, L3, L2):
        _, _, H4, W4 = L4.shape
        _, _, H5, W5 = L5.shape
        size4 = (H4, W4)
        size5 = (H5, W5)

        P5_4 = self.up5_4(L5)
        P3_4 = self.down(L3, size4)
        P2_4 = self.down(L2, size4)
        a = (P5_4 + L4 + P3_4 + P2_4) / 4.0

        a = self.st(a)

        B5 = torch.cat([self.down(a, size5), L5], dim=1)
        B5 = self.fu5(B5)

        B4 = torch.cat([a, L4], dim=1)
        B4 = self.fu4(B4)

        B3 = torch.cat([self.up4_3(a), L3], dim=1)
        B3 = self.fu3(B3)

        B2 = torch.cat([self.up4_2(a), L2], dim=1)
        B2 = self.fu2(B2)

        return B5, B4, B3, B2
