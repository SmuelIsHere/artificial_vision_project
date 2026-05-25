import torch
import torch.nn as nn


class AttentionModule(nn.Module):

    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_attention  = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, padding=0),
        )

        self.sigmoid = nn.Sigmoid()

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_spatial = torch.cat([avg_out, max_out], dim=1)
        x_spatial = self.spatial_attention(x_spatial)

        avg_out = self.channel_attention(self.avg_pool(x))
        max_out = self.channel_attention(self.max_pool(x))
        x_channel = avg_out + max_out
        x_channel = self.sigmoid(x_channel)

        out = x * x_channel * x_spatial
        return out