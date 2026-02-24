import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18MultiView(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet18MultiView, self).__init__()

        # Shared backbone
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove original FC
        self.backbone.fc = nn.Identity()

        # Final classifier after concatenation
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512 * 2, num_classes)
        )

    def extract_features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        features = self.backbone.layer4(x)

        pooled = self.backbone.avgpool(features)
        pooled = torch.flatten(pooled, 1)

        return pooled, features

    def forward(self, cc, mlo, return_features=False):
        cc_pooled, cc_feat = self.extract_features(cc)
        mlo_pooled, mlo_feat = self.extract_features(mlo)

        fused = torch.cat([cc_pooled, mlo_pooled], dim=1)
        logits = self.classifier(fused)

        if return_features:
            return logits, cc_feat, mlo_feat

        return logits