import torch
import torch.nn as nn
import torch.nn.functional as F

class CRGBMemCNN(nn.Module):
    def __init__(self):
        super(CRGBMemCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=25)

        # Calculate flatten size dynamically with dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)  # Adjust if your actual input size is different
            x = self.pool1(F.relu(self.conv1(dummy_input)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            self.flatten_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_size, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Avoids hardcoding flatten size
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
