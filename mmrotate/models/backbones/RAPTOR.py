
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math
import warnings
import copy


from .Adaptive_Scale_Branch import AdaptiveScaleBranch
from .Adaptive_Channel_Attention import SEBranch
from .DCNV4_CUDA import DCNv4_Rotated_CUDA
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from ..builder import ROTATED_BACKBONES
from timm.models.layers import DropPath, trunc_normal_


from mamba_ssm import Mamba
import selective_scan_cuda

class SelectiveScan(torch.autograd.Function):
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}" 
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows

        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False  
        )
        
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


class CrossScan(torch.autograd.Function):
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs
    def backward(ctx, ys: torch.Tensor):

        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)
class CrossMerge(torch.autograd.Function):
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    

    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs, None, None


def cross_selective_scan(
    x: torch.Tensor=None,
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    out_norm: torch.nn.Module=None,
    nrows = -1,
    delta_softplus = True,
    to_dtype=True,
    force_fp32=True,):
    B, D, H, W = x.shape
    D_A, N = A_logs.shape
    K, D_dt, R = dt_projs_weight.shape
    L = H * W

    if nrows < 1:
        nrows = K

    xs = CrossScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    

    xs = xs.view(B, -1, L) 
    dts = dts.contiguous().view(B, -1, L) 


    As = -torch.exp(A_logs.to(torch.float)).repeat(K, 1) 
    

    Ds = Ds.to(torch.float).repeat(K)

    delta_bias = dt_projs_bias.view(-1).to(torch.float)


    Bs = Bs.contiguous().view(B, K, N, L)
    Cs = Cs.contiguous().view(B, K, N, L)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, H, W)

    y: torch.Tensor = CrossMerge.apply(ys)
    y = y.transpose(dim0=1, dim1=2).contiguous() 
    y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)

class SS2D_v2(nn.Module):
    """
    Corrected SS2D module.
    This version adds the missing 'self.act' definition in the __init__ method.
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        ssm_ratio=2.0,
        dt_rank="auto",
        k_scan=4,
        **kwargs,):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.k_scan = k_scan

        self.in_proj = nn.Linear(d_model, self.expand, bias=False)
        self.conv2d = nn.Conv2d(self.expand, self.expand, kernel_size=d_conv, padding=d_conv // 2, groups=self.expand, bias=True)
        self.act = nn.SiLU() 
        self.x_proj_weight = nn.Parameter(torch.randn(self.k_scan, self.dt_rank + 2 * self.d_state, self.expand))
        self.x_proj_bias = nn.Parameter(torch.randn(self.k_scan, self.dt_rank + 2 * self.d_state))

        self.dt_projs_weight = nn.Parameter(torch.randn(self.k_scan, self.expand, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.randn(self.k_scan, self.expand))

        self.A_logs = nn.Parameter(torch.randn(self.expand, self.d_state))
        self.Ds = nn.Parameter(torch.randn(self.expand))


        self.out_norm = nn.LayerNorm(self.expand)
        self.out_proj = nn.Linear(self.expand, self.d_model, bias=False)

    def forward(self, x):

        x_proj = self.in_proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

     
        x_conv = self.act(self.conv2d(x_proj)) 


        y = cross_selective_scan(
            x=x_conv,
            x_proj_weight=self.x_proj_weight,
            x_proj_bias=self.x_proj_bias,
            dt_projs_weight=self.dt_projs_weight,
            dt_projs_bias=self.dt_projs_bias,
            A_logs=self.A_logs,
            Ds=self.Ds,
            out_norm=self.out_norm,
        ) 
    
        y_out = self.out_proj(y).permute(0, 3, 1, 2)

        return y_out
    
class TViMBlock(nn.Module):
    """ 使用新版 SS2D_v2 的 TViMBlock """
    def __init__(self, hidden_dim: int, drop_path: float = 0., **kwargs):
        super().__init__()
        # 实例化新版的 SS2D_v2
        self.op = SS2D_v2(d_model=hidden_dim, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input: torch.Tensor):
        return input + self.drop_path(self.op(input))


class LoGFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, sigma, norm_layer, act_layer):
        super().__init__()
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3, bias=False)
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel - kernel.mean()
        if kernel.sum() != 0:
            kernel = kernel / kernel.sum()
        self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), groups=out_c, bias=False)
        self.LoG.weight.data = kernel.unsqueeze(0).unsqueeze(0).repeat(out_c, 1, 1, 1)
        self.LoG.weight.requires_grad = False
        self.act = act_layer()
        self.norm1 = build_norm_layer(norm_layer, out_c)[1]
        self.norm2 = build_norm_layer(norm_layer, out_c)[1]

    def forward(self, x):
        x = self.conv_init(x)
        LoG_features = self.LoG(x)
        LoG_edge = self.act(self.norm1(LoG_features))
        x = self.norm2(x + LoG_edge)
        return x

class RotationalDCNv4Branch(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, group=4, **kwargs):
        super().__init__()
        self.dcn = DCNv4_Rotated_CUDA(channels=in_channels, use_rotation=True, group=group, pad=1, **kwargs)
        self.bn = build_norm_layer(norm_layer, out_channels)[1]

    def forward(self, x):
        return self.bn(self.dcn(x))

class LWEG_Stem(nn.Module):
    def __init__(self, in_chans, stem_dim, norm_layer, act_layer):
        super().__init__()
        self.feature_extractor = LoGFilter(in_chans, stem_dim // 2, kernel_size=7, sigma=0.5, norm_layer=norm_layer, act_layer=act_layer)
        self.downsampler = nn.Sequential(
            nn.Conv2d(stem_dim // 2, stem_dim, kernel_size=3, stride=2, padding=1, bias=False),
            build_norm_layer(norm_layer, stem_dim)[1],
            act_layer()
        )
        self.final_downsampler = nn.Sequential(
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3, stride=2, padding=1, bias=False),
            build_norm_layer(norm_layer, stem_dim)[1],
            act_layer()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.downsampler(x)
        x = self.final_downsampler(x)
        return x

class A_DRFD(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.output_channels = channels * 2
        self.path1 = nn.Sequential(
            nn.Conv2d(channels, self.output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            build_norm_layer(dict(type='BN'), self.output_channels)[1]
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
            build_norm_layer(dict(type='BN'), channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(channels, self.output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            build_norm_layer(dict(type='BN'), self.output_channels)[1]
        )
        self.path3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels, self.output_channels, kernel_size=1, bias=False),
            build_norm_layer(dict(type='BN'), self.output_channels)[1]
        )
        inter_channels = max(channels // r, 32)
        self.gate_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 3, kernel_size=1, bias=False)
        )

    def forward(self, x):
        gates = F.softmax(self.gate_generator(x), dim=1)
        out1, out2, out3 = self.path1(x), self.path2(x), self.path3(x)
        output = gates[:, 0:1, :, :] * out1 + gates[:, 1:2, :, :] * out2 + gates[:, 2:3, :, :] * out3
        return F.relu(output)


class SimpleDownsample(nn.Module):
    """
    普通的下采样模块：Conv 3x3, Stride 2
    """
    def __init__(self, in_channels, norm_layer):
        super().__init__()
        self.output_channels = in_channels * 2
        
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, self.output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            build_norm_layer(norm_layer, self.output_channels)[1]
        )

    def forward(self, x):
        return self.down(x)

def get_safe_dcn_group(channels):
    """
    针对 DCNv4 的自动分组计算函数。
    目标：确保 (channels / groups) 是 4 的倍数 (D % d_stride == 0)。
    优先顺序：每组 16 通道 -> 每组 8 通道 -> 每组 4 通道。
    """

    if channels % 16 == 0:
        return channels // 16

    if channels % 8 == 0:
        return channels // 8
    

    if channels % 4 == 0:
        return channels // 4
        
    return 1




class OptimizedLWEGBlock(nn.Module):
    def __init__(self, in_channels, stage_index, norm_layer, act_layer=nn.GELU, drop_path=0.):
        super().__init__()
        

        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            build_norm_layer(norm_layer, in_channels)[1],
            act_layer()
        )


        dcn_group = get_safe_dcn_group(in_channels)
        

        self.local_branch = RotationalDCNv4Branch(in_channels, in_channels, norm_layer, group=dcn_group)

        self.global_branch = TViMBlock(hidden_dim=in_channels, drop_path=drop_path)


        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            build_norm_layer(norm_layer, in_channels)[1],
            nn.Sigmoid()
        )
        

        self.se = SEBranch(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.act = act_layer()

    def forward(self, x):
        residual = x
        enhanced_x = self.feature_enhancer(x)

        local_features = self.local_branch(enhanced_x)
        global_features = self.global_branch(enhanced_x)
        
        combined = torch.cat([local_features, global_features], dim=1)
        gate_weights = self.gate(combined)
        fused_features = local_features * gate_weights + global_features * (1 - gate_weights)

        y = self.act(fused_features)
        y_calibrated = self.se(y)
        
        output = residual + self.drop_path(y_calibrated)
        return output

@ROTATED_BACKBONES.register_module()
class RAPTOR(BaseModule):
    def __init__(
        self,
        in_chans=3,
        base_channels=96,
        depths=[3, 3, 9, 3],
        norm_layer=dict(type='GN', num_groups=32, requires_grad=True),
        act_layer=nn.GELU,
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.2,
        init_cfg=None
    ):
        super().__init__(init_cfg)

        self.out_indices = out_indices
        self.stem = LWEG_Stem(in_chans, base_channels, norm_layer, act_layer)

        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_idx = 0
        current_channels = base_channels

        for i, depth in enumerate(depths):
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(OptimizedLWEGBlock(
                    in_channels=current_channels,
                    stage_index=i,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    drop_path=dpr[dpr_idx + j]
                ))
            self.stages.append(nn.Sequential(*stage_blocks))

            if i < len(depths) - 1:

                downsampler = A_DRFD(channels=current_channels)
                
                self.downsamplers.append(downsampler)
                current_channels = downsampler.output_channels

            
            dpr_idx += depth

        for i_stage in out_indices:
            channels_after_stage = base_channels * (2**i_stage)
            norm_layer_ = build_norm_layer(norm_layer, channels_after_stage)[1]
            self.add_module(f'norm{i_stage}', norm_layer_)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)
        return tuple(outs)

    def init_weights(self):

        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0)
        else:
            super().init_weights()