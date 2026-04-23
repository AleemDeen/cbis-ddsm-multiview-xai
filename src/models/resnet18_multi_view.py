import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class SegDecoder(nn.Module):
    """
    U-Net style decoder that reconstructs a full-resolution ROI probability mask
    from the encoder's intermediate feature maps.

    Skip connections from all four ResNet stages are passed in so the decoder
    can combine fine spatial detail from early layers (f1: 128×128) with the
    high-level semantic information captured by deeper layers (f4: 16×16).
    Without skips, the upsampled coarse features alone produce blurry, poorly
    localised masks.

    Input feature map shapes:
      f1: B × 64  × 128 × 128  (layer1 output)
      f2: B × 128 × 64  × 64   (layer2 output)
      f3: B × 256 × 32  × 32   (layer3 output)
      f4: B × 512 × 16  × 16   (layer4 output)

    Output: B × 1 × 512 × 512  — sigmoid ROI probability mask
    """

    def __init__(self):
        super().__init__()

        # Each up-block halves the channel count before fusing with the skip connection
        self.up4   = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True))
        self.fuse4 = nn.Sequential(nn.Conv2d(256 + 256, 256, 3, padding=1), nn.ReLU(inplace=True))

        self.up3   = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.fuse3 = nn.Sequential(nn.Conv2d(128 + 128, 128, 3, padding=1), nn.ReLU(inplace=True))

        self.up2   = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.fuse2 = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, padding=1), nn.ReLU(inplace=True))

        # Final 4× bilinear upsample brings 128×128 → 512×512 to match input resolution
        self.out = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),  # output is a per-pixel probability, not a logit
        )

    def forward(self, f1, f2, f3, f4):
        # Upsample f4 (16→32), then concatenate with the f3 skip connection
        x = F.interpolate(self.up4(f4), scale_factor=2, mode="bilinear", align_corners=False)
        x = self.fuse4(torch.cat([x, f3], dim=1))   # 32×32

        # Upsample to 64×64, fuse f2
        x = F.interpolate(self.up3(x), scale_factor=2, mode="bilinear", align_corners=False)
        x = self.fuse3(torch.cat([x, f2], dim=1))   # 64×64

        # Upsample to 128×128, fuse f1
        x = F.interpolate(self.up2(x), scale_factor=2, mode="bilinear", align_corners=False)
        x = self.fuse2(torch.cat([x, f1], dim=1))   # 128×128

        # Final 4× upsample to match the 512×512 input mammogram
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return self.out(x)


class ResNet18MultiView(nn.Module):
    """
    Dual-branch ResNet18 classifier for paired CC + MLO mammogram views.

    Two independent ResNet18 encoders process the CC and MLO images separately.
    Their global average-pooled feature vectors are concatenated into a 1024-d
    representation, which is then passed through a shared linear classifier.

    Keeping the branches independent (rather than sharing weights) lets each
    branch specialise to the characteristics of its respective view, while the
    concatenation forces the classifier to reason jointly over both projections.
    """

    def __init__(self, num_classes=1):
        super().__init__()

        # Load ImageNet-pretrained weights for both branches
        self.cc_branch  = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.mlo_branch = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Adapt the first conv for single-channel (greyscale) mammograms
        self.cc_branch.conv1  = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.mlo_branch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the ResNet classification heads — pooled 512-d vectors feed
        # the shared classifier directly
        self.cc_branch.fc  = nn.Identity()
        self.mlo_branch.fc = nn.Identity()

        # Shared classifier: concatenated 512+512 features → single malignancy logit
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512 * 2, num_classes),
        )

    def _branch_forward(self, branch, x):
        """Run a full ResNet branch, returning the pooled feature vector and the
        layer4 spatial feature map (used for GradCAM during inference)."""
        x = branch.conv1(x)
        x = branch.bn1(x)
        x = branch.relu(x)
        x = branch.maxpool(x)
        x = branch.layer1(x)
        x = branch.layer2(x)
        x = branch.layer3(x)
        features = branch.layer4(x)          # (B, 512, H', W') — spatial map
        pooled   = branch.avgpool(features)  # (B, 512, 1, 1)
        pooled   = torch.flatten(pooled, 1)  # (B, 512)
        return pooled, features

    def forward(self, cc, mlo, return_features=False):
        cc_pooled,  cc_features  = self._branch_forward(self.cc_branch,  cc)
        mlo_pooled, mlo_features = self._branch_forward(self.mlo_branch, mlo)

        # Concatenate pooled features from both views before classification
        combined = torch.cat([cc_pooled, mlo_pooled], dim=1)
        logits   = self.classifier(combined)

        if return_features:
            return logits, cc_features, mlo_features
        return logits


class ResNet18MultiViewSeg(nn.Module):
    """
    Multi-view ResNet18 with a U-Net segmentation decoder attached to each branch.

    Extends ResNet18MultiView by routing skip connections (f1–f4) from each
    encoder branch into a dedicated SegDecoder. This produces a per-pixel ROI
    probability mask for each view in addition to the shared malignancy logit.

    The backbone weights are compatible with ResNet18MultiView, so this model
    can be initialised from a pre-trained mv_baseline.pt checkpoint and then
    fine-tuned with segmentation supervision — the decoder weights are new and
    will not be found in the base checkpoint (strict=False handles this).
    """

    def __init__(self, num_classes=1):
        super().__init__()

        self.cc_branch  = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.mlo_branch = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.cc_branch.conv1  = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.mlo_branch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.cc_branch.fc  = nn.Identity()
        self.mlo_branch.fc = nn.Identity()

        # Shared classifier — frozen during seg fine-tuning to preserve classification AUC
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512 * 2, num_classes),
        )

        # One decoder per view — they do not share weights because CC and MLO
        # have different spatial characteristics
        self.cc_seg_head  = SegDecoder()
        self.mlo_seg_head = SegDecoder()

    def _branch_forward(self, branch, x):
        """Returns the pooled classification vector plus all four intermediate
        feature maps needed for the U-Net skip connections."""
        x  = branch.conv1(x)
        x  = branch.bn1(x)
        x  = branch.relu(x)
        x  = branch.maxpool(x)
        f1 = branch.layer1(x)                          # B × 64  × 128 × 128
        f2 = branch.layer2(f1)                         # B × 128 × 64  × 64
        f3 = branch.layer3(f2)                         # B × 256 × 32  × 32
        f4 = branch.layer4(f3)                         # B × 512 × 16  × 16
        pooled = torch.flatten(branch.avgpool(f4), 1)  # B × 512
        return pooled, f1, f2, f3, f4

    def forward(self, cc, mlo, return_masks=False):
        cc_pooled,  *cc_feats  = self._branch_forward(self.cc_branch,  cc)
        mlo_pooled, *mlo_feats = self._branch_forward(self.mlo_branch, mlo)

        combined = torch.cat([cc_pooled, mlo_pooled], dim=1)
        logits   = self.classifier(combined)

        if return_masks:
            # Pass the four skip-connection maps into each decoder
            cc_mask  = self.cc_seg_head(*cc_feats)
            mlo_mask = self.mlo_seg_head(*mlo_feats)
            return logits, cc_mask, mlo_mask

        return logits
