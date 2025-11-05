from torch import nn
from utils.registry import registry


@registry.register_classifier("EMGPureCNN1d")
class EMGPureCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=args.conv1_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(args.conv1_channels, args.conv2_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(args.conv2_channels, args.conv3_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # (B, conv3_channels, 1)
        )
        self.classifier = nn.Linear(args.conv3_channels, args.num_classes)

    def forward(self, x):  # x: (B, seq_len, 8)
        x = x.permute(0, 2, 1)  # → (B, 8, seq_len)
        x = self.net(x)         # (B, conv3_channels, 1)
        x = x.view(x.size(0), -1)  # (B, conv3_channels)
        return self.classifier(x)  # (B, num_classes)
