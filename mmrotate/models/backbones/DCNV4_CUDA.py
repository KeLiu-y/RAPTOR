import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
import math
import warnings
import einops
import time
import inspect



from DCNv4.functions import DCNv4Function



def get_dcnv4_args(dcn_func_cls):
    sig = inspect.signature(dcn_func_cls.forward)
    params = list(sig.parameters.keys())
    if 'ctx' in params: params.remove('ctx')
    print(f"🔍 [自动侦测] DCNv4 算子需要的参数列表: {params}")
    return params

DCN_EXPECTED_ARGS = get_dcnv4_args(DCNv4Function)



def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        def norm_cdf(x): return (1. + math.erf(x / math.sqrt(2.))) / 2.
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

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
        alphas = torch.sigmoid(self.fc_alpha(self.dropout1(x)))
        angles = self.act_func(self.fc_theta(self.dropout2(x))) * self.proportion
        return alphas, angles


class DCNv4_Rotated_CUDA(nn.Module):
    def __init__(self, channels=128, kernel_size=3, stride=1, pad=1, dilation=1, group=4,
                 offset_scale=1.0, dw_kernel_size=3, remove_center=False, output_bias=True,
                 without_pointwise=False, use_rotation=False, proportion=40.0, **kwargs):
        super().__init__()
        if channels % group != 0:
            raise ValueError(f'channels must be divisible by group')

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
        self.im2col_step = 64 
        self.P = int(kernel_size * kernel_size - self.remove_center)
        self.K = self.group * self.P

        if dw_kernel_size is not None:
            self.offset_mask_dw = nn.Conv2d(channels, channels, dw_kernel_size, stride=1, padding=(dw_kernel_size - 1) // 2, groups=channels)
        self.offset_mask = nn.Linear(channels, int(math.ceil((self.K * 3) / 8) * 8))

        if not without_pointwise:
            self.value_proj = nn.Linear(channels, channels)
            self.output_proj = nn.Linear(channels, channels, bias=output_bias)

        self.use_rotation = use_rotation
        if self.use_rotation:
            self.rounting_func = RountingFunction(channels, self.group, proportion=proportion)

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
        x = input.permute(0, 2, 3, 1)
        if not self.without_pointwise:
            x = self.value_proj(x)
        x = x.reshape(N, H, W, -1) 

        if self.dw_kernel_size is not None:
            offset_mask_input = self.offset_mask_dw(input).permute(0, 2, 3, 1)
        else:
            offset_mask_input = input.permute(0, 2, 3, 1)

        offset_mask = self.offset_mask(offset_mask_input).reshape(N, H, W, -1)
        offset_mask = offset_mask[..., :self.K * 3].reshape(N, H, W, self.K, 3)
        offset = offset_mask[..., :2]
        mask = offset_mask[..., 2].sigmoid()

        # --- 旋转逻辑 ---
        if self.use_rotation:
            alphas, angles = self.rounting_func(input)
            offset = offset.view(N, H, W, self.group, self.P, 2)
            mask = mask.view(N, H, W, self.group, self.P)
            cos_a = torch.cos(angles).view(N, 1, 1, self.group, 1, 1)
            sin_a = torch.sin(angles).view(N, 1, 1, self.group, 1, 1)
            of_x, of_y = offset.chunk(2, dim=-1)
            of_x_rot = of_x * cos_a - of_y * sin_a
            of_y_rot = of_x * sin_a + of_y * cos_a
            offset = torch.cat([of_x_rot, of_y_rot], dim=-1)
            alphas = alphas.view(N, 1, 1, self.group, 1)
            mask = mask * alphas

        # --- 🚀 修复核心：合并 + 补齐Padding ---
        if mask.dim() == 5:
            mask = mask.unsqueeze(-1)
        if offset.dim() == 4:
             offset = offset.view(N, H, W, self.group, self.P, 2)
        if mask.dim() == 4:
             mask = mask.view(N, H, W, self.group, self.P, 1)

        # 1. 拼接
        offset_mask = torch.cat([offset, mask], dim=-1) # [..., 3]
        
        # 2. 展平
        offset_mask = offset_mask.reshape(N, H, W, -1)
        
        # 3. 🔥【关键修复】补齐到 8 的倍数 🔥
        current_dim = offset_mask.shape[-1]
        if current_dim % 8 != 0:
            pad_dim = 8 - (current_dim % 8)
            # F.pad 在最后一维右侧补0
            offset_mask = F.pad(offset_mask, (0, pad_dim), "constant", 0)
            
        offset_mask = offset_mask.contiguous()
        x = x.contiguous()

        # 动态传参
        all_possible_args = {
            'input': x, 'offset_mask': offset_mask,
            'kernel_h': self.kernel_size, 'kernel_w': self.kernel_size,
            'stride_h': self.stride, 'stride_w': self.stride,
            'pad_h': self.pad, 'pad_w': self.pad,
            'dilation_h': self.dilation, 'dilation_w': self.dilation,
            'group': self.group, 'group_channels': self.group_channels,
            'offset_scale': self.offset_scale,
            'im2col_step': self.im2col_step,
            'remove_center': self.remove_center
        }

        call_args = []
        for param_name in DCN_EXPECTED_ARGS:
            if param_name in all_possible_args:
                call_args.append(all_possible_args[param_name])
            else:
                raise ValueError(f"DCNv4 需要参数 '{param_name}'，但我们没有提供！")

        x = DCNv4Function.apply(*call_args)

        if not self.without_pointwise:
            x = self.output_proj(x)
        x = x.permute(0, 3, 1, 2)
        return x





