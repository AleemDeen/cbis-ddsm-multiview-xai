import torch
import torch.nn.functional as F


class ScoreCAM:
    """
    Score-CAM: gradient-free class activation mapping.

    Unlike Grad-CAM, Score-CAM does not rely on gradients. Instead, it measures
    the contribution of each activation channel by masking the input image with
    an upsampled version of that channel and measuring the change in the model's
    output score. Channels whose mask produces a high score are weighted more
    heavily in the final heatmap.

    The gradient-free approach avoids saturation issues that can occur with
    Grad-CAM when activations are very large, at the cost of being significantly
    slower (one forward pass per activation channel).

    Reference: Wang et al., "Score-CAM: Score-Weighted Visual Explanations for
    Convolutional Neural Networks", CVPR Workshops 2020.
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None

        # A forward hook is all that is needed — no backward pass is required
        self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def generate(self, input_tensor):
        """
        Compute the Score-CAM heatmap for a single image.

        Args:
            input_tensor: (1, C, H, W) image tensor

        Returns:
            Normalised heatmap tensor of shape (H, W) in [0, 1]
        """
        self.model.eval()

        # Forward pass to populate self.activations via the hook
        with torch.no_grad():
            logits = self.model(input_tensor)
            baseline_score = torch.sigmoid(logits)  # baseline confidence (no masking)

        activations = self.activations  # (1, num_channels, H', W')
        b, c, h, w  = activations.shape

        cam = torch.zeros((h, w), device=input_tensor.device)

        for i in range(c):
            # Upsample each activation channel to input resolution
            activation_map = activations[:, i:i+1, :, :]
            upsampled = F.interpolate(
                activation_map,
                size=input_tensor.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

            # Normalise the upsampled map to [0, 1] so it acts as a soft mask
            norm_map = upsampled - upsampled.min()
            norm_map = norm_map / (norm_map.max() + 1e-8)

            # Mask the input with this channel's activation and score the result
            masked_input = input_tensor * norm_map
            with torch.no_grad():
                output = self.model(masked_input)
                weight = torch.sigmoid(output)  # how much does this region contribute?

            # Accumulate channel contributions weighted by their score
            cam += weight.squeeze() * activation_map.squeeze()

        # ReLU keeps only regions that increase the predicted score
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam
