import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
import warnings
import logging
import math
import einops

_logger = logging.getLogger(__name__)
torch.fx.wrap('len')

class to_channels_first(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.permute(0, 3, 1, 2)

class to_channels_last(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.permute(0, 2, 3, 1)

def _get_reference_points(
    spatial_shapes: List[int], device: Optional[torch.device], kernel_h: int, kernel_w: int,
    dilation_h: int, dilation_w: int, pad_h: int=0, pad_w: int=0, stride_h: int=1, stride_w: int=1
):
    _, H_, W_, _ = spatial_shapes
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            (dilation_h * (kernel_h - 1)) / 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out, dtype=torch.float32, device=device
        ),
        torch.linspace(
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out, dtype=torch.float32, device=device
        ),
        indexing='ij'
    )
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_
    ref = torch.stack((ref_x, ref_y), -1).reshape(1, H_out, W_out, 1, 2)
    return ref

def _generate_dilation_grids(
    spatial_shapes: List[int], kernel_h: int, kernel_w: int, dilation_h: int,
    dilation_w: int, group: int, device: Optional[torch.device],
):
    _, H_, W_, _ = spatial_shapes
    points_list = []
    x, y = torch.meshgrid(
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) + (kernel_w - 1) * dilation_w,
            kernel_w, dtype=torch.float32, device=device
        ),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) + (kernel_h - 1) * dilation_h,
            kernel_h, dtype=torch.float32, device=device
        ),
        indexing='ij'
    )
    points_list.extend([x / W_, y / H_])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2).\
        repeat(1, group, 1).permute(1, 0, 2)
    grid = grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)
    return grid

def dcnv4_core_pytorch(
    input, offset, mask, kernel_h: int, kernel_w: int, stride_h: int, stride_w: int,
    pad_h: int, pad_w: int, dilation_h: int, dilation_w: int, group: int,
    group_channels: int, offset_scale: float
):
    input = F.pad(input, [0, 0, pad_h, pad_h, pad_w, pad_w])
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape

    ref = _get_reference_points(
        input.shape, input.device, kernel_h, kernel_w, dilation_h,
        dilation_w, pad_h, pad_w, stride_h, stride_w
    )
    grid = _generate_dilation_grids(
        input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.device
    )
    spatial_norm = torch.tensor([W_in, H_in], device=input.device).reshape(1, 1, 1, 2).\
        repeat(1, 1, 1, group * kernel_h * kernel_w)

    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1).flatten(3, 4) + \
        offset * offset_scale / spatial_norm

    P_ = kernel_h * kernel_w
    sampling_grids = 2 * sampling_locations - 1
    input_ = input.view(N_, H_in * W_in, group * group_channels).transpose(1, 2).\
        reshape(N_ * group, group_channels, H_in, W_in)
    sampling_grid_ = sampling_grids.view(N_, H_out * W_out, group, P_, 2).transpose(1, 2).\
        flatten(0, 1)
    sampling_input_ = F.grid_sample(
        input_, sampling_grid_, mode='bilinear',
        padding_mode='zeros', align_corners=False
    )

    mask = mask.view(N_, H_out * W_out, group, P_).transpose(1, 2).\
        reshape(N_ * group, 1, H_out * W_out, P_)
    output = (sampling_input_ * mask).sum(-1).\
        view(N_, group * group_channels, H_out * W_out)

    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous()

def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.", stacklevel=2)
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)

class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class RountingFunction(nn.Module):
    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__()
        self.kernel_number = kernel_number
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels, kernel_number, bias=True)
        self.dropout2= nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels, kernel_number, bias=False)
        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi
        
        trunc_normal_(self.dwc.weight, std=.02)
        trunc_normal_(self.fc_alpha.weight, std=.02)
        trunc_normal_(self.fc_theta.weight, std=.02)

    def forward(self, x):
        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)
        
        alphas = self.dropout1(x)
        alphas = self.fc_alpha(alphas)
        alphas = torch.sigmoid(alphas)

        angles = self.dropout2(x)
        angles = self.fc_theta(angles)
        angles = self.act_func(angles)
        angles = angles * self.proportion
        return alphas, angles


# =================================================================================
# Main DCNv4 Module (FIXED AGAIN)
# =================================================================================

class DCNv4_pytorch(nn.Module):
    def __init__(
            self,
            channels=128,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            dw_kernel_size=3,
            remove_center=False,
            output_bias=True,
            without_pointwise=False,
            use_rotation=False,
            proportion=40.0,
            **kwargs
            ):
        super().__init__()
        if channels % group != 0:
            raise ValueError(f'channels must be divisible by group, but got {channels} and {group}')

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.dw_kernel_size = dw_kernel_size
        self.remove_center = int(remove_center)
        self.without_pointwise = without_pointwise

        self.P = int(kernel_size * kernel_size - self.remove_center)
        self.K = self.group * self.P

        if dw_kernel_size is not None:
            self.offset_mask_dw = nn.Conv2d(
                channels, channels, dw_kernel_size, stride=1,
                padding=(dw_kernel_size - 1) // 2, groups=channels
            )
        self.offset_mask = nn.Linear(channels, int(math.ceil((self.K * 3) / 8) * 8))

        if not without_pointwise:
            self.value_proj = nn.Linear(channels, channels)
            self.output_proj = nn.Linear(channels, channels, bias=output_bias)

        self.use_rotation = use_rotation
        if self.use_rotation:
            self.rounting_func = RountingFunction(
                in_channels=channels,
                kernel_number=self.group,
                proportion=proportion
            )

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.offset_mask.weight.data, 0.)
        constant_(self.offset_mask.bias.data, 0.)
        if not self.without_pointwise:
            xavier_uniform_(self.value_proj.weight.data)
            constant_(self.value_proj.bias.data, 0.)
            xavier_uniform_(self.output_proj.weight.data)
            if self.output_proj.bias is not None:
                constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        N, C, H, W = input.shape
        L = H * W

        x = input.permute(0, 2, 3, 1).view(N, L, C)
        if not self.without_pointwise:
            x = self.value_proj(x)
        x = x.reshape(N, H, W, -1)

        if self.dw_kernel_size is not None:
            offset_mask_input = self.offset_mask_dw(input)
            offset_mask_input = offset_mask_input.permute(0, 2, 3, 1).view(N, L, C)
        else:
            offset_mask_input = input.permute(0, 2, 3, 1).view(N, L, C)

        offset_mask_padded = self.offset_mask(offset_mask_input)
        offset_mask_padded = offset_mask_padded.reshape(N, H, W, -1)

        offset_mask = offset_mask_padded[..., :self.K * 3]
        offset_mask = offset_mask.reshape(N, H, W, self.K, 3)

        offset = offset_mask[..., :2]
        mask = offset_mask[..., 2].sigmoid()
        
        if self.use_rotation:
            alphas, angles = self.rounting_func(input)
            
            offset = offset.view(N, H, W, self.group, self.P, 2)
            mask = mask.view(N, H, W, self.group, self.P)

            # --- FIX STARTS HERE ---
            # 将 cos_a 和 sin_a 变形为6D张量以匹配 of_x/of_y 的维度
            cos_a = torch.cos(angles).view(N, 1, 1, self.group, 1, 1)
            sin_a = torch.sin(angles).view(N, 1, 1, self.group, 1, 1)
            # --- FIX ENDS HERE ---

            of_x, of_y = offset.chunk(2, dim=-1)
            of_x_rot = of_x * cos_a - of_y * sin_a
            of_y_rot = of_x * sin_a + of_y * cos_a
            offset = torch.cat([of_x_rot, of_y_rot], dim=-1)

            alphas = alphas.view(N, 1, 1, self.group, 1)
            mask = mask * alphas
        
        offset = offset.reshape(N, H, W, -1)
        mask = mask.reshape(N, H, W, -1)

        x = dcnv4_core_pytorch(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale
        )

        if not self.without_pointwise:
            x = self.output_proj(x)

        x = x.permute(0, 3, 1, 2)
        return x

