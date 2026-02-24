import torch
import torch.nn.functional as F


class ScoreCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None

        self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def generate(self, input_tensor):
        self.model.eval()

        with torch.no_grad():
            logits = self.model(input_tensor)
            baseline_score = torch.sigmoid(logits)

        activations = self.activations  # (1, C, H, W)
        b, c, h, w = activations.shape

        cam = torch.zeros((h, w), device=input_tensor.device)

        for i in range(c):
            activation_map = activations[:, i:i+1, :, :]  # (1,1,H,W)

            upsampled = F.interpolate(
                activation_map,
                size=input_tensor.shape[2:],
                mode="bilinear",
                align_corners=False
            )

            norm_map = upsampled - upsampled.min()
            norm_map = norm_map / (norm_map.max() + 1e-8)

            masked_input = input_tensor * norm_map

            with torch.no_grad():
                output = self.model(masked_input)
                weight = torch.sigmoid(output)

            cam += weight.squeeze() * activation_map.squeeze()

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam