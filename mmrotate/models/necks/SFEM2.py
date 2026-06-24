
import torch
import torch.nn as nn
from typing import List, Tuple

from mmcv.runner import BaseModule
from ..builder import ROTATED_NECKS
import torch.nn.functional as F
# --- 同样，在此处包含或导入 FEFM 和 Conv 的定义 ---
# 示例: from ..utils.arf_modules import FEFM, Conv

from mmdet.models.necks import FPN

class SimpleSAFM(nn.Module):
    """Part of the Spatial Adaptive Modulation module."""
    def __init__(self, dim, ratio=4):
        super().__init__()
        self.dim = dim
        self.chunk_dim = dim // ratio
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.dwconv = nn.Conv2d(self.chunk_dim, self.chunk_dim, 3, 1, 1, groups=self.chunk_dim, bias=False)
        self.out = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.act = nn.GELU()
    def forward(self, x):
        h, w = x.size()[-2:]
        x0, x1 = self.proj(x).split([self.chunk_dim, self.dim - self.chunk_dim], dim=1)
        x2 = F.adaptive_max_pool2d(x0, (h // 8, w // 8))
        x2 = self.dwconv(x2)
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=False)
        x2 = self.act(x2) * x0
        x = torch.cat([x1, x2], dim=1)
        x = self.out(self.act(x))
        return x

class CCM(nn.Module):
    """Convolutional Channel Mixer, part of the AttBlock."""
    def __init__(self, dim, ffn_scale, use_se=False):
        super().__init__()
        hidden_dim = int(dim * ffn_scale)
        self.conv1 = nn.Conv2d(dim, hidden_dim, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False)
        self.act = nn.GELU()
    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))

class AttBlock(nn.Module):
    """Core processing block for spatial modulation, used in SAFMNPP and TriAdNetV2."""
    def __init__(self, dim, ffn_scale=1.5, use_se=False, ratio=3):
        super().__init__()
        self.conv1 = SimpleSAFM(dim, ratio=ratio)
        self.conv2 = CCM(dim, ffn_scale, use_se)
    def forward(self, x):
        # The entire block is residual
        return x + self.conv2(self.conv1(x))

class SAFMNPP(nn.Module):
    """
    Spatially Adaptive Feature Modulation with PixelShuffle (Upsampling Module).
    Intended for use in the neck (e.g., SAFPN).
    """
    def __init__(self, input_dim, dim, n_blocks=6, ffn_scale=1.5, use_se=False, upscaling_factor=2):
        super().__init__()
        self.scale = upscaling_factor
        self.to_feat = nn.Conv2d(input_dim, dim, 3, 1, 1, bias=False)
        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale, use_se) for _ in range(n_blocks)])
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, input_dim * upscaling_factor ** 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(upscaling_factor)
        )
    def forward(self, x):
        res = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x = self.to_feat(x)
        x = self.feats(x)
        return self.to_img(x) + res

@ROTATED_NECKS.register_module()
class SAFPN(FPN):
    """
    Spatially Adaptive FPN (Neck).

    This FPN replaces the standard top-down upsampling (interpolation)
    with the powerful SAFMNPP module, enabling learned feature enhancement
    and upsampling.
    """
    def __init__(self, safmnpp_cfg: dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.safmnpp_cfg = safmnpp_cfg or {}
        if not self.safmnpp_cfg:
            print("WARNING: SAFPN initialized without a 'safmnpp_cfg'. It will be empty.")
            
        # Create a list of SAFMNPP upsampling modules
        self.upsample_blocks = nn.ModuleList()
        # The number of upsampling blocks is one less than the number of FPN levels
        for _ in range(len(self.in_channels) - 1):
            # In an FPN, the upsampling module's input channels are typically
            # equal to the FPN's unified `out_channels`.
            upsample_module = SAFMNPP(
                input_dim=self.out_channels,
                dim=self.out_channels, # Internal dim can be configured if needed
                upscaling_factor=2,
                **self.safmnpp_cfg
            )
            self.upsample_blocks.append(upsample_module)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        assert len(inputs) == len(self.in_channels)

        # 1. Build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 2. Build top-down path using SAFMNPP
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # === CORE MODIFICATION ===
            # Instead of F.interpolate, we use our learned upsampler
            prev_feat = self.upsample_blocks[i-1](laterals[i])
            laterals[i - 1] = laterals[i - 1] + prev_feat
            # =========================

        # 3. Build outputs
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # 4. Add extra levels (standard FPN logic)
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for _ in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)