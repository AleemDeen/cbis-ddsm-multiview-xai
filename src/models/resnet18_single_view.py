import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18SingleView(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet18SingleView, self).__init__()
        # Use modern weights parameter to avoid warnings
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Change the first layer to accept 1-channel (grayscale) mammograms
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Add Dropout before the final layer (0.3 is a good balance for medical imaging)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x, return_features=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        features = self.model.layer4(x)

        pooled = self.model.avgpool(features)
        pooled = torch.flatten(pooled, 1)
        logits = self.model.fc(pooled)

        if return_features:
            return logits, features

        return logits