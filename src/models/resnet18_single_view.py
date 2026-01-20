import torch
import torch.nn as nn
from torchvision import models


class ResNet18SingleView(nn.Module):
    """
    Single-view baseline model using ResNet-18 pretrained on ImageNet.
    Input: (B, 1, 512, 512)
    Output: (B, 1) logits
    """

    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)

        # Change first conv layer to accept 1-channel input
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace final FC layer for binary classification
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(1)
