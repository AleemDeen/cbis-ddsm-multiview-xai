import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class SegDecoder(nn.Module):
    """
    U-Net style decoder for ResNet18 features.

    Takes skip connections from all four ResNet stages:
      f1: B×64×128×128   (layer1)
      f2: B×128×64×64    (layer2)
      f3: B×256×32×32    (layer3)
      f4: B×512×16×16    (layer4)

    Produces a B×1×512×512 ROI probability mask.
    """

    def __init__(self):
        super().__init__()
        # Decode f4 → 32×32, fuse with f3
        self.up4   = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True))
        self.fuse4 = nn.Sequential(nn.Conv2d(256 + 256, 256, 3, padding=1), nn.ReLU(inplace=True))
        # Decode → 64×64, fuse with f2
        self.up3   = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.fuse3 = nn.Sequential(nn.Conv2d(128 + 128, 128, 3, padding=1), nn.ReLU(inplace=True))
        # Decode → 128×128, fuse with f1
        self.up2   = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.fuse2 = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, padding=1), nn.ReLU(inplace=True))
        # Final upsample 128×128 → 512×512
        self.out   = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, f1, f2, f3, f4):
        x = F.interpolate(self.up4(f4), scale_factor=2, mode="bilinear", align_corners=False)
        x = self.fuse4(torch.cat([x, f3], dim=1))   # 32×32

        x = F.interpolate(self.up3(x), scale_factor=2, mode="bilinear", align_corners=False)
        x = self.fuse3(torch.cat([x, f2], dim=1))   # 64×64

        x = F.interpolate(self.up2(x), scale_factor=2, mode="bilinear", align_corners=False)
        x = self.fuse2(torch.cat([x, f1], dim=1))   # 128×128

        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)  # 512×512
        return self.out(x)


class ResNet18MultiView(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # Two separate ResNet branches
        self.cc_branch  = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.mlo_branch = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Adapt first conv for grayscale
        self.cc_branch.conv1  = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.mlo_branch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove FC layers (pool output is 512-d)
        self.cc_branch.fc  = nn.Identity()
        self.mlo_branch.fc = nn.Identity()

        # Final classifier (512 + 512 features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512 * 2, num_classes),
        )

    def _branch_forward(self, branch, x):
        """Forward through a ResNet branch, returning both pooled features and layer4 maps."""
        x = branch.conv1(x)
        x = branch.bn1(x)
        x = branch.relu(x)
        x = branch.maxpool(x)
        x = branch.layer1(x)
        x = branch.layer2(x)
        x = branch.layer3(x)
        features = branch.layer4(x)          # (B, 512, H', W')
        pooled   = branch.avgpool(features)  # (B, 512, 1, 1)
        pooled   = torch.flatten(pooled, 1)  # (B, 512)
        return pooled, features

    def forward(self, cc, mlo, return_features=False):
        cc_pooled,  cc_features  = self._branch_forward(self.cc_branch,  cc)
        mlo_pooled, mlo_features = self._branch_forward(self.mlo_branch, mlo)

        combined = torch.cat([cc_pooled, mlo_pooled], dim=1)
        logits   = self.classifier(combined)

        if return_features:
            return logits, cc_features, mlo_features
        return logits


class ResNet18MultiViewSeg(nn.Module):
    """
    Multi-view ResNet18 with U-Net segmentation decoders.

    The backbone + classifier are identical to ResNet18MultiView and can be
    initialised from its saved weights (strict=False — seg head keys are new).

    The decoder uses skip connections from layer1-4 of each branch, giving
    the seg head access to both fine-grained spatial detail (layer1) and
    high-level semantic features (layer4).
    """

    def __init__(self, num_classes=1):
        super().__init__()

        self.cc_branch  = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.mlo_branch = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.cc_branch.conv1  = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.mlo_branch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.cc_branch.fc  = nn.Identity()
        self.mlo_branch.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512 * 2, num_classes),
        )

        # U-Net decoders with skip connections
        self.cc_seg_head  = SegDecoder()
        self.mlo_seg_head = SegDecoder()

    def _branch_forward(self, branch, x):
        """Returns pooled vector + all four intermediate feature maps for skip connections."""
        x  = branch.conv1(x)
        x  = branch.bn1(x)
        x  = branch.relu(x)
        x  = branch.maxpool(x)
        f1 = branch.layer1(x)                       # B×64×128×128
        f2 = branch.layer2(f1)                      # B×128×64×64
        f3 = branch.layer3(f2)                      # B×256×32×32
        f4 = branch.layer4(f3)                      # B×512×16×16
        pooled = torch.flatten(branch.avgpool(f4), 1)  # B×512
        return pooled, f1, f2, f3, f4

    def forward(self, cc, mlo, return_masks=False):
        cc_pooled,  *cc_feats  = self._branch_forward(self.cc_branch,  cc)
        mlo_pooled, *mlo_feats = self._branch_forward(self.mlo_branch, mlo)

        combined = torch.cat([cc_pooled, mlo_pooled], dim=1)
        logits   = self.classifier(combined)

        if return_masks:
            cc_mask  = self.cc_seg_head(*cc_feats)
            mlo_mask = self.mlo_seg_head(*mlo_feats)
            return logits, cc_mask, mlo_mask
        return logits
