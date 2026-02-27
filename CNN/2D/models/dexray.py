import torch
import torch.nn as nn

class DexRayNet(nn.Module):
    def __init__(self, input_size=128*128):
        super(DexRayNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=12),
            nn.Conv1d(64, 128, kernel_size=12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=12)
        )

        # compute flatten size AFTER defining feature extractor
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size)
            out = self.feature_extractor(dummy)
            flattened_size = out.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Binary output
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze(dim=-1)
