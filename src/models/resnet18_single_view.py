import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18MultiView(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # Two separate ResNet branches
        self.cc_branch = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.mlo_branch = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Adapt first conv for grayscale
        self.cc_branch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.mlo_branch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove FC layers
        self.cc_branch.fc = nn.Identity()
        self.mlo_branch.fc = nn.Identity()

        # Final classifier (512 + 512 features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512 * 2, num_classes)
        )

    def forward(self, cc, mlo):
        cc_feat = self.cc_branch(cc)
        mlo_feat = self.mlo_branch(mlo)

        combined = torch.cat([cc_feat, mlo_feat], dim=1)
        logits = self.classifier(combined)

        return logits