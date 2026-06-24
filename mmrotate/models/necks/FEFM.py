import torch
import torch.nn as nn
from typing import List, Tuple

from mmcv.runner import BaseModule
from ..builder import ROTATED_NECKS

# --- 同样，在此处包含或导入 FEFM 和 Conv 的定义 ---
# 示例: from ..utils.arf_modules import FEFM, Conv
import torch
import torch.nn as nn
import torch.fft



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class FEFM(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, reduction=8):
        """
        Frequency Exhaustive Fusion Mechanism (FEFM) - Final Stabilized Version
        """
        super().__init__()
        mid_channels = max(in_channels1, in_channels2)
        
        # Input convolutions to align channels
        self.conv_r = nn.Conv2d(in_channels1, mid_channels, kernel_size=1)
        self.conv_n = nn.Conv2d(in_channels2, mid_channels, kernel_size=1)

        # Convolutions for Q, K, V feature generation
        self.point_conv_Q = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        self.depth_conv_Q = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels)
        self.point_conv_K = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        self.depth_conv_K = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels)
        self.point_conv_V = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        self.depth_conv_V = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels)

        # Final output convolution
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

        # Learnable raw parameters for safe, constrained learning
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))
        self.raw_lambd = nn.Parameter(torch.tensor(0.0))
        self.raw_beta = nn.Parameter(torch.tensor(0.0))
        
        # Epsilon for numerical stability in divisions
        self.epsilon = 1e-8
        
        # Channel attention mechanism
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        d = max(int(mid_channels / reduction), 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(mid_channels, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, mid_channels * 2, 1, bias=False))
        self.softmax = nn.Softmax(dim=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, in_feats):
        F_R, F_N = in_feats[0], in_feats[1]
        F_R = self.conv_r(F_R)
        F_N = self.conv_n(F_N)

        # Channel attention fusion
        F_concat = torch.cat([F_R, F_N], dim=1)
        B, C_total, H, W = F_concat.shape
        mid_channels = C_total // 2
        F_concat_reshaped = F_concat.view(B, 2, mid_channels, H, W)
        feats_sum = torch.sum(F_concat_reshaped, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, 2, mid_channels, 1, 1))
        F_weighted = torch.sum(F_concat_reshaped * attn, dim=1)

        # Feature encoding
        Q = self.depth_conv_Q(self.point_conv_Q(F_weighted))
        K = self.depth_conv_K(self.point_conv_K(F_weighted))
        V = self.depth_conv_V(self.point_conv_V(F_weighted))

        # Get constrained parameters for stability
        beta = torch.sigmoid(self.raw_beta)
        lambd = torch.sigmoid(self.raw_lambd)
        alpha = torch.exp(self.raw_alpha) + self.epsilon
        original_dtype = Q.dtype
        
        # --- Frequency Domain Processing ---
        F_Q = torch.fft.fft2(Q.float(), dim=(-2, -1))
        F_K = torch.fft.fft2(K.float(), dim=(-2, -1))

        # Element-wise product of frequency components
        elem_product = F_Q * F_K
        
        # --- ‼️ FINAL FIX: STABILIZE THE PRODUCT ‼️ ---
        # Squash the values of the product to prevent explosion.
        # This is the most critical fix for the numerical instability.
        elem_product_real = torch.tanh(elem_product.real)
        elem_product_imag = torch.tanh(elem_product.imag)
        elem_product = torch.complex(elem_product_real, elem_product_imag)
        
        # Common Feature Reinforcement (CFR)
        B, C, H_fft, W_fft = F_Q.shape
        F_Q_flat = F.normalize(F_Q.view(B, C, -1), p=2, dim=-1) # Normalize for stable attention
        F_K_flat = F.normalize(F_K.view(B, C, -1), p=2, dim=-1)

        attn_matrix = torch.matmul(F_Q_flat, F_K_flat.transpose(1, 2))
        attn_weights = torch.softmax(attn_matrix.abs() / alpha, dim=-1)
        attn_weights_complex = torch.complex(attn_weights, torch.zeros_like(attn_weights))
        
        elem_product_flat = elem_product.view(B, C, -1)
        F_CFR_flat = torch.matmul(attn_weights_complex, elem_product_flat)
        F_CFR = F_CFR_flat.view(B, C, H_fft, W_fft)

        cfr_spatial = torch.fft.ifft2(F_CFR, dim=(-2, -1)).real.to(original_dtype)
        
        # Differential Feature Reinforcement (DFR)
        F_DFR = V - lambd * V * cfr_spatial
        
        # Final Fusion
        freq_output = Q * cfr_spatial + F_DFR
        output = beta * freq_output + (1 - beta) * F_weighted
        output = self.out_conv(output)

        return output
# 假设辅助模块已定义，现在是 Neck 的核心代码
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    default_act = nn.SiLU() 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    def forward_fuse(self, x):
        return self.act(self.conv(x))

@ROTATED_NECKS.register_module()
class ARF_FPN_Neck(BaseModule):
    """
    Adaptive Receptive Field - Frequency Exhaustive Fusion Mechanism FPN (ARF-FPN).
    
    该 Neck 接收来自任何 Backbone 的多尺度特征图，并使用 FEFM 模块进行
    自上而下的特征融合。
    """
    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 start_level: int = 0,
                 end_level: int = -1,
                 add_extra_convs: bool = False,
                 init_cfg=None):
        super(ARF_FPN_Neck, self).__init__(init_cfg)
        
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.end_level = self.num_ins if end_level == -1 else end_level

        # 1. 横向连接层 (Lateral Convolutions)
        # 将输入的多尺度特征图通道数统一到 out_channels
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            l_conv = Conv(in_channels[i], out_channels, k=1)
            self.lateral_convs.append(l_conv)
            
        # 2. FEFM 融合层 (FEFM Fusion Layers)
        # FEFM 模块用于自上而下的融合
        self.fefm_modules = nn.ModuleList()
        # 需要的 FEFM 模块数量比横向连接层少一个
        for i in range(len(self.lateral_convs) - 1):
            # FEFM 接收两个通道数相同的特征图
            self.fefm_modules.append(
                FEFM(in_channels1=out_channels, in_channels2=out_channels, out_channels=out_channels)
            )

        # 3. 如果需要，为额外的输出层添加卷积 (可选)
        self.extra_convs = None
        if add_extra_convs and num_outs > len(self.in_channels):
            self.extra_convs = nn.ModuleList()
            for i in range(num_outs - self.num_ins):
                # 在最后一个融合后的特征图上进行池化，以获得更大的感受野
                self.extra_convs.append(
                    nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1))

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    def init_weights(self):
        """
        Initiate the parameters.
        """
        # Call the parent's init_weights to initialize Conv2d, etc.
        super(ARF_FPN_Neck, self).init_weights()
            
    def forward(self, inputs: Tuple[torch.Tensor]):
        """
        inputs: 来自 backbone 的特征图元组
        """
        assert len(inputs) == len(self.in_channels)
        
        # 1. 执行横向连接
        laterals = [
            self.lateral_convs[i](inputs[i + self.start_level])
            for i in range(len(self.lateral_convs))
        ]
        
        # 2. 执行自上而下的 FEFM 融合
        # 从最高层特征开始 (索引 -1)
        for i in range(len(laterals) - 1, 0, -1):
            prev_feat = self.upsample(laterals[i])
            current_feat = laterals[i-1]
            
            # 使用 FEFM 进行融合，而不是简单的加法
            # FEFM 模块的索引与 prev_feat 的索引相对应
            fefm_idx = i - 1
            laterals[i-1] = self.fefm_modules[fefm_idx]([current_feat, prev_feat])
        
        # 3. 收集输出
        outs = [laterals[i] for i in range(len(laterals))]
        
        # 4. 添加额外的层级
        if self.extra_convs:
            for conv in self.extra_convs:
                outs.append(conv(outs[-1]))

        return tuple(outs)


# # file: test_neck.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Tuple
# import torch.fft
# import math

# # ===================================================================
# # Part 1: All necessary class definitions
# # We copy the FEFM class and the stable Conv helper class here.
# # ===================================================================

# class FEFM(nn.Module):
#     # --- Paste the complete, final FEFM class code from Part 1 above here ---
#     def __init__(self, in_channels1, in_channels2, out_channels, reduction=8):
#         """
#         Frequency Exhaustive Fusion Mechanism (FEFM) - Final Stabilized Version
#         """
#         super().__init__()
#         mid_channels = max(in_channels1, in_channels2)
#         self.conv_r = nn.Conv2d(in_channels1, mid_channels, kernel_size=1)
#         self.conv_n = nn.Conv2d(in_channels2, mid_channels, kernel_size=1)
#         self.point_conv_Q = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
#         self.depth_conv_Q = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels)
#         self.point_conv_K = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
#         self.depth_conv_K = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels)
#         self.point_conv_V = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
#         self.depth_conv_V = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels)
#         self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
#         self.raw_alpha = nn.Parameter(torch.tensor(0.0))
#         self.raw_lambd = nn.Parameter(torch.tensor(0.0))
#         self.raw_beta = nn.Parameter(torch.tensor(0.0))
#         self.epsilon = 1e-8
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         d = max(int(mid_channels / reduction), 4)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(mid_channels, d, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(d, mid_channels * 2, 1, bias=False))
#         self.softmax = nn.Softmax(dim=1)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, in_feats):
#         F_R, F_N = in_feats[0], in_feats[1]
#         F_R = self.conv_r(F_R)
#         F_N = self.conv_n(F_N)
#         F_concat = torch.cat([F_R, F_N], dim=1)
#         B, C_total, H, W = F_concat.shape
#         mid_channels = C_total // 2
#         F_concat_reshaped = F_concat.view(B, 2, mid_channels, H, W)
#         feats_sum = torch.sum(F_concat_reshaped, dim=1)
#         attn = self.mlp(self.avg_pool(feats_sum))
#         attn = self.softmax(attn.view(B, 2, mid_channels, 1, 1))
#         F_weighted = torch.sum(F_concat_reshaped * attn, dim=1)
#         Q = self.depth_conv_Q(self.point_conv_Q(F_weighted))
#         K = self.depth_conv_K(self.point_conv_K(F_weighted))
#         V = self.depth_conv_V(self.point_conv_V(F_weighted))
#         beta = torch.sigmoid(self.raw_beta)
#         lambd = torch.sigmoid(self.raw_lambd)
#         alpha = torch.exp(self.raw_alpha) + self.epsilon
#         original_dtype = Q.dtype
#         F_Q = torch.fft.fft2(Q.float(), dim=(-2, -1))
#         F_K = torch.fft.fft2(K.float(), dim=(-2, -1))
#         elem_product = F_Q * F_K
#         elem_product_real = torch.tanh(elem_product.real)
#         elem_product_imag = torch.tanh(elem_product.imag)
#         elem_product = torch.complex(elem_product_real, elem_product_imag)
#         B, C, H_fft, W_fft = F_Q.shape
#         F_Q_flat = F.normalize(F_Q.view(B, C, -1), p=2, dim=-1)
#         F_K_flat = F.normalize(F_K.view(B, C, -1), p=2, dim=-1)
#         attn_matrix = torch.matmul(F_Q_flat, F_K_flat.transpose(1, 2))
#         attn_weights = torch.softmax(attn_matrix.abs() / alpha, dim=-1)
#         attn_weights_complex = torch.complex(attn_weights, torch.zeros_like(attn_weights))
#         elem_product_flat = elem_product.view(B, C, -1)
#         F_CFR_flat = torch.matmul(attn_weights_complex, elem_product_flat)
#         F_CFR = F_CFR_flat.view(B, C, H_fft, W_fft)
#         cfr_spatial = torch.fft.ifft2(F_CFR, dim=(-2, -1)).real.to(original_dtype)
#         F_DFR = V - lambd * V * cfr_spatial
#         freq_output = Q * cfr_spatial + F_DFR
#         output = beta * freq_output + (1 - beta) * F_weighted
#         output = self.out_conv(output)
#         return output

# class Conv(nn.Module):
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         def autopad(k, p=None, d=1):
#             if d > 1: k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
#             if p is None: p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#             return p
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         num_groups = 32
#         if c2 > 0 and c2 % num_groups != 0:
#             for ng in range(min(num_groups, c2), 0, -1):
#                 if c2 % ng == 0:
#                     num_groups = ng
#                     break
#         self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=c2)
#         self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

#     def forward(self, x):
#         return self.act(self.gn(self.conv(x)))


# class ARF_FPN_Neck(nn.Module):
#     def __init__(self, in_channels: List[int], out_channels: int, num_outs: int, **kwargs):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_ins = len(in_channels)
#         self.num_outs = num_outs
        
#         self.lateral_convs = nn.ModuleList()
#         for i in range(self.num_ins):
#             l_conv = Conv(in_channels[i], out_channels, k=1)
#             self.lateral_convs.append(l_conv)
            
#         self.fefm_modules = nn.ModuleList()
#         for i in range(self.num_ins - 1):
#             self.fefm_modules.append(
#                 FEFM(in_channels1=out_channels, in_channels2=out_channels, out_channels=out_channels))

#         self.extra_convs = nn.ModuleList()
#         for i in range(num_outs - self.num_ins):
#             self.extra_convs.append(
#                 nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1))
        
#         # We no longer need the nn.Upsample layer defined here.
#         # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

#     def forward(self, inputs: Tuple[torch.Tensor]):
#         assert len(inputs) == len(self.in_channels)
        
#         laterals = [
#             self.lateral_convs[i](inputs[i]) for i in range(len(self.lateral_convs))
#         ]
        
#         # Top-down fusion pathway
#         for i in range(len(laterals) - 1, 0, -1):
#             current_feat = laterals[i-1]
#             prev_feat_raw = laterals[i]
            
#             # ⬇️ --- THIS IS THE FIX --- ⬇️
#             # Instead of upsampling by a fixed factor, we resize prev_feat to the exact
#             # spatial size of current_feat using F.interpolate.
#             prev_feat = F.interpolate(prev_feat_raw, size=current_feat.shape[2:], mode='nearest')
#             # ⬆️ --- END OF FIX --- ⬆️

#             fefm_idx = i - 1
#             laterals[i-1] = self.fefm_modules[fefm_idx]([current_feat, prev_feat])
        
#         outs = list(laterals)
        
#         if len(self.extra_convs) > 0:
#             last_feat = outs[-1]
#             for conv in self.extra_convs:
#                 last_feat = conv(last_feat)
#                 outs.append(last_feat)

#         return tuple(outs)
# # ===================================================================
# # Part 2: Standalone Test Execution
# # ===================================================================

# if __name__ == '__main__':
#     print("--- Standalone ARF_FPN_Neck Test ---")

#     # 1. Define neck configuration based on your settings
#     neck_config = dict(
#         in_channels=[64, 128, 256, 512], # Matches backbone output
#         out_channels=256,
#         num_outs=5
#     )
#     print(f"Attempting to build ARF_FPN_Neck with config:\n{neck_config}\n")

#     # 2. Instantiate the Neck
#     try:
#         neck_model = ARF_FPN_Neck(**neck_config)
#         neck_model.eval() # Set to evaluation mode
#         print("✅ Neck model built successfully!")
#     except Exception as e:
#         print(f"❌ Failed to build Neck model. Error: {e}")
#         exit()
        
#     # 3. Create mock input data (mimicking backbone output)
#     #    Batch size is set to 2 to test the case that previously failed.
#     batch_size = 2
#     mock_backbone_output = (
#         torch.randn(batch_size, 64, 100, 100), # Level 0
#         torch.randn(batch_size, 128, 50, 50),  # Level 1
#         torch.randn(batch_size, 256, 25, 25),  # Level 2
#         torch.randn(batch_size, 512, 13, 13),  # Level 3
#     )
#     print("\nCreated mock input data with 4 feature levels.")
#     for i, tensor in enumerate(mock_backbone_output):
#         print(f"  Input Level {i} shape: {tensor.shape}")
        
#     # 4. Perform a forward pass
#     print("\n--- Performing forward pass... ---")
#     try:
#         with torch.no_grad(): # No need to calculate gradients for this test
#             output_features = neck_model(mock_backbone_output)
        
#         print("\n✅ Forward pass completed successfully!")
#         print("--- Output Feature Map Shapes ---")
        
#         # 5. Check the output
#         if len(output_features) == neck_config['num_outs']:
#              print(f"✅ Correct number of output feature maps ({len(output_features)}).")
#         else:
#              print(f"❌ Incorrect number of outputs! Expected {neck_config['num_outs']}, got {len(output_features)}.")

#         for i, tensor in enumerate(output_features):
#             print(f"  Output Level {i} shape: {tensor.shape}")
#             if torch.isnan(tensor).any() or torch.isinf(tensor).any():
#                 print(f"  ❌ WARNING: Output tensor at level {i} contains NaN or Inf values!")
#             else:
#                 print(f"  ✅ Output tensor at level {i} is clean (no NaN/Inf).")

#     except Exception as e:
#         print(f"\n❌ An error occurred during the forward pass: {e}")