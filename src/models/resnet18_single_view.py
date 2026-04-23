import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18SingleView(nn.Module):
    """
    Single-view mammogram classifier built on a pretrained ResNet18 backbone.

    The network takes a single grayscale mammogram (CC or MLO) and outputs a
    malignancy logit. Pretrained ImageNet weights are used as the starting
    point — even though mammograms are greyscale and look nothing like natural
    photos, the low-level edge detectors from ImageNet transfer well and speed
    up convergence considerably on the relatively small CBIS-DDSM dataset.
    """

    def __init__(self, num_classes=1):
        super(ResNet18SingleView, self).__init__()

        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Replace the original 3-channel input conv with a single-channel one.
        # Mammograms are greyscale, so feeding 3 identical channels would waste
        # computation and add an unnecessary inductive bias.
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Swap the default 1000-class head for a single malignancy logit.
        # Dropout is added before the linear layer to reduce overfitting —
        # CBIS-DDSM training splits contain only ~850 patients.
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x, return_features=False):
        # Run the standard ResNet stem: conv → BN → ReLU → maxpool
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        # Pass through the four residual stages
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        features = self.model.layer4(x)   # (B, 512, 16, 16) spatial feature map

        pooled = self.model.avgpool(features)
        pooled = torch.flatten(pooled, 1)
        logits = self.model.fc(pooled)

        # return_features=True is used during training to expose the layer4 map
        # for inline GradCAM localisation loss computation
        if return_features:
            return logits, features

        return logits
