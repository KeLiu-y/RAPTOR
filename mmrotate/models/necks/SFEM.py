
import torch
import torch.nn as nn
from typing import List, Tuple

from mmcv.runner import BaseModule
from ..builder import ROTATED_NECKS
import torch.nn.functional as F
# --- 同样，在此处包含或导入 FEFM 和 Conv 的定义 ---
# 示例: from ..utils.arf_modules import FEFM, Conv

from mmdet.models.necks import FPN

class LightweightConv(nn.Module):
    """
    深度可分离卷积: 3x3 Depthwise Conv + 1x1 Pointwise Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


# --- 1. 重构 SimpleSAFM 模块 ---
class SimpleSAFM(nn.Module):
    def __init__(self, dim, ratio=3):
        super().__init__()
        self.dim = dim
        self.chunk_dim = (dim // ratio // 8) * 8 if (dim // ratio) > 8 else dim // ratio
        if self.chunk_dim == 0:
            self.chunk_dim = dim
        
        # 使用新的轻量化卷积替换标准3x3卷积
        self.proj = LightweightConv(dim, dim, kernel_size=3)
        
        self.dwconv = nn.Conv2d(self.chunk_dim, self.chunk_dim, 3, 1, 1, groups=self.chunk_dim, bias=False)
        self.out = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]
        x_proj = self.proj(x)
        x0, x1 = torch.split(x_proj, [self.chunk_dim, self.dim - self.chunk_dim], dim=1)
        x2 = F.adaptive_max_pool2d(x0, (h // 8, w // 8))
        x2 = self.dwconv(x2)
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=False)
        x2 = self.act(x2) * x0
        x_out = torch.cat([x1, x2], dim=1)
        x_out = self.out(self.act(x_out))
        return x_out


# --- 2. 重构 CCM 模块 (倒置残差结构) ---
class CCM(nn.Module):
    def __init__(self, dim, ffn_scale=1.5):
        super().__init__()
        hidden_dim = int(dim * ffn_scale)
        # 1x1 升维卷积
        self.conv1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=False)
        # 3x3 深度卷积
        self.conv_dw = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False)
        # 1x1 降维卷积
        self.conv2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv_dw(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


# --- AttBlock 和 SAFMNeck 保持不变，因为它们只是调用上面的模块 ---
class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=1.5, ratio=3):
        super().__init__()
        self.conv1 = SimpleSAFM(dim, ratio=ratio)
        self.conv2 = CCM(dim, ffn_scale)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x

@ROTATED_NECKS.register_module()
class SAFMNeck(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 n_blocks_per_level,
                 ffn_scale=1.5,
                 ratio=3,
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        assert len(in_channels) == len(n_blocks_per_level)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        assert self.num_outs >= self.num_ins

        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = nn.Conv2d(in_channels[i], out_channels, kernel_size=1)
            fpn_block = nn.Sequential(
                *[AttBlock(dim=out_channels, ffn_scale=ffn_scale, ratio=ratio) for _ in range(n_blocks_per_level[i])]
            )
            self.lateral_convs.append(l_conv)
            self.fpn_blocks.append(fpn_block)
        
        self.extra_convs = nn.ModuleList()
        for i in range(self.num_ins, self.num_outs):
            self.extra_convs.append(nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1))

    def forward(self, inputs):
        assert len(inputs) == self.num_ins
        
        laterals = [
            self.lateral_convs[i](inputs[i])
            for i in range(self.num_ins)
        ]
        
        for i in range(self.num_ins - 1, 0, -1):
            prev_shape = laterals[i-1].shape[2:]
            laterals[i-1] = laterals[i-1] + F.interpolate(laterals[i], size=prev_shape, mode='bilinear', align_corners=False)
        
        outs = [
            self.fpn_blocks[i](laterals[i]) for i in range(self.num_ins)
        ]
        
        if self.num_outs > self.num_ins:
            for i in range(self.num_ins, self.num_outs):
                outs.append(self.extra_convs[i - self.num_ins](outs[-1]))

        return tuple(outs[:self.num_outs])