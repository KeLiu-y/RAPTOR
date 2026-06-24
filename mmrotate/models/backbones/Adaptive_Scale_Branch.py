# 自适应尺度分支
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveScaleBranch(nn.Module):
    """
    An adaptive scale branch using a dynamically weighted multi-dilation convolution approach.
    It captures multi-scale context and fuses it based on input features.
    """
    def __init__(self, channels, r=4):
        super(AdaptiveScaleBranch, self).__init__()
        inter_channels = max(channels // r, 32) # 防止中间层通道数过小
        
        # 三个并行的、不同膨胀率的深度可分离卷积，用于捕捉不同尺度的信息
        self.path1_d1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.path2_d3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.path3_d5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=5, dilation=5, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

        # 用于生成动态融合权重的注意力模块
        self.attention_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels * 3, 1, bias=False), # 输出3个分支的权重
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. 通过三个并行分支提取多尺度特征
        path1_out = self.path1_d1(x)
        path2_out = self.path2_d3(x)
        path3_out = self.path3_d5(x)
        
        # 2. 生成动态注意力权重
        # (B, C, H, W) -> (B, C*3, 1, 1)
        attention_weights = self.attention_generator(x)
        
        # 将权重分割成三份
        w1, w2, w3 = torch.chunk(attention_weights, 3, dim=1) # each: (B, C, 1, 1)
        
        # 3. 加权融合
        out = self.relu(w1 * path1_out + w2 * path2_out + w3 * path3_out)
        
        return out



