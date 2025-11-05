import torch
import torch.nn as nn

class ChannelCircularPad2d(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # top down replicating padding
        top = x[:, :, -1:, :]  # take the lowest row → to add to the top
        bottom = x[:, :, :1, :]  # take the top row → to add to the lowest
        x = torch.cat([top, x, bottom], dim=2)  # (B, C, H+2, W)

        # left right zero padding
        zero_col = torch.zeros(B, C, H + 2, 1, device=x.device, dtype=x.dtype)
        x = torch.cat([zero_col, x, zero_col], dim=3)  # (B, C, H+2, W+2)

        return x
