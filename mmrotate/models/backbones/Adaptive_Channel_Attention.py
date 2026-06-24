import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBranch(nn.Module):
    """
    An enhanced Squeeze-and-Excitation Branch that utilizes both average and max pooling
    to aggregate spatial information, providing richer channel-wise statistics.
    """
    # ✅ FIX: Changed the parameter name from 'r' to 'reduction' to match the calling code.
    def __init__(self, channels, reduction=16):
        super(SEBranch, self).__init__()
        
        # Calculate the intermediate channel size, ensuring it's not too small
        # ✅ FIX: Use the 'reduction' parameter here as well.
        inter_channels = max(channels // reduction, 32)

        # Shared Multi-Layer Perceptron (MLP)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, inter_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()

        # --- Squeeze Phase ---
        # 1. Average Pooling Path
        avg_pool_out = F.adaptive_avg_pool2d(x, 1).view(batch_size, num_channels)
        
        # 2. Max Pooling Path
        max_pool_out = F.adaptive_max_pool2d(x, 1).view(batch_size, num_channels)

        # --- Excitation Phase ---
        # Both paths go through the same shared MLP
        avg_mlp_out = self.shared_mlp(avg_pool_out)
        max_mlp_out = self.shared_mlp(max_pool_out)

        # Combine the features from both paths
        combined_features = avg_mlp_out + max_mlp_out
        
        # Apply sigmoid to get the channel weights (attention scores)
        channel_weights = self.sigmoid(combined_features)
        
        # Reshape weights to (B, C, 1, 1) for broadcasting
        channel_weights = channel_weights.view(batch_size, num_channels, 1, 1)

        # --- Scale Phase ---
        # Multiply the original feature map by the learned channel weights
        return x * channel_weights