import torch
import torch.nn as nn
import torch.nn.functional as F

class DiDroidNet(nn.Module):
    def __init__(self, num_classes=2):
        super(DiDroidNet, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # assuming input image is 224x224
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Output: [batch, 32, 112, 112]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Output: [batch, 64, 56, 56]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Output: [batch, 128, 28, 28]

        # Flatten
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Fully connected layers with dropout and ReLU
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
