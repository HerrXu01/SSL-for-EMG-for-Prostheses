from torch import nn
from utils.registry import registry
from utils.padding import ChannelCircularPad2d


@registry.register_classifier("EMGFeatureCNN")
class EMGFeatureCNN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.token_projector = nn.Sequential(
            nn.Linear(args.input_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.embed_dim),
            nn.ReLU()
        )

        if args.enable_channel_circular_padding:
            self.cnn_layers = nn.Sequential(
                ChannelCircularPad2d(),
                nn.Conv2d(args.embed_dim, 128, kernel_size=3, padding=0),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU()
            )
        else:
            self.cnn_layers = nn.Sequential(
                nn.Conv2d(args.embed_dim, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU()
            )
            
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, args.num_classes)

    def forward(self, x):  # x: (B, 8, num_tokens, 4096)
        B, C, T, D = x.shape
        x = x.view(B * C * T, D)
        x = self.token_projector(x)  # (B*C*T, embed_dim)
        x = x.view(B, C, T, -1).permute(0, 3, 1, 2)  # (B, embed_dim, 8, num_tokens)
        x = self.cnn_layers(x)  # (B, 64, 8, num_tokens)
        x = self.pool(x).view(B, -1)  # (B, 64)
        return self.classifier(x)     # (B, num_classes)
