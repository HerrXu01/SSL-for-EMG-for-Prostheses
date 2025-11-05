from torch import nn
from utils.registry import registry
from utils.padding import ChannelCircularPad2d


@registry.register_classifier("EMGPureCNN")
class EMGPureCNN(nn.Module):
    def __init__(self, args):
        super().__init__()

        if args.enable_channel_circular_padding:
            self.cnn_layers = nn.Sequential(
                ChannelCircularPad2d(),
                nn.Conv2d(in_channels=1, out_channels=args.conv1_channels, kernel_size=3, padding=0),
                nn.ReLU(),
                nn.Conv2d(args.conv1_channels, args.conv2_channels, kernel_size=3, padding=1),
                nn.ReLU()
            )
        else:
            self.cnn_layers = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=args.conv1_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(args.conv1_channels, args.conv2_channels, kernel_size=3, padding=1),
                nn.ReLU()
            )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(args.conv2_channels, args.num_classes)

    def forward(self, x):  # x: (B, seq_len, 8)
        x = x.permute(0, 2, 1).unsqueeze(1)  # → (B, 1, 8, seq_len)
        x = self.cnn_layers(x)  # (B, 64, 8, seq_len)
        x = self.pool(x)  # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 64)
        return self.classifier(x)  # (B, num_classes)

