import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM implementation for ResNet-style CNNs.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        input_tensor: shape (1, C, H, W)
        returns: Grad-CAM heatmap (1, 1, h, w)
        """
        self.model.zero_grad()

        logits = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[:, target_class]
        score.backward()

        # Gradient-weighted activations
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)

        # Normalise
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam
