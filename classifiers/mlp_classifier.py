from torch import nn
from utils.registry import registry


@registry.register_classifier("MLPBottleneckClassifier")
class MLPBottleneckClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        D = args.feature_dim         # 4096 for SSL, seq_len for raw data
        K = args.bottleneck_dim      # e.g., 16 or 32
        H = args.hidden_dim          # e.g., 128; 设置为 None 代表直接到 num_classes

        # 逐通道投影：共享权重，对所有 8 个通道用同一线性
        self.channel_proj = nn.Linear(D, K)

        # Head：可 1 层或 2 层
        if H == 0:
            self.head = nn.Linear(8 * K, args.num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(8 * K, H),
                nn.ReLU(),
                nn.Linear(H, args.num_classes)
            )

    def forward(self, x):            # x: (B, D, 8), ssl features: (B, 4096, 8), raw data: (B, seq_len, 8)
        x = x.permute(0, 2, 1)       # x: (B, 8, D)
        B, C, D = x.shape            # C=8
        x = self.channel_proj(x)     # (B, 8, K)
        x = x.reshape(B, -1)         # (B, 8*K)
        return self.head(x)
