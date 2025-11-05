from torch import nn
from utils.registry import registry
from utils.padding import ChannelCircularPad2d


@registry.register_classifier("EMGCompactFeatureCNN")
class EMGCompactFeatureCNN(nn.Module):
    def __init__(self, args):
        super().__init__()

        # MLP: maps 4096 → 256 for each channel
        self.projector = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

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

    def forward(self, x):  # x: (B, 8, 4096)
        B, C, D = x.shape  # C=8, D=4096

        # Apply projector MLP to each channel independently
        x = self.projector(x)          # → (B, 8, 256)

        x = x.unsqueeze(1)             # → (B, 1, 8, 256)
        x = self.cnn_layers(x)         # → (B, conv2_channels, 8, 256)
        x = self.pool(x)               # → (B, conv2_channels, 1, 1)
        x = x.view(B, -1)              # → (B, conv2_channels)
        return self.classifier(x)      # → (B, num_classes)
