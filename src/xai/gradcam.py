import torch
import torch.nn.functional as F


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for ResNet-style CNNs.

    Grad-CAM produces a coarse spatial heatmap that highlights which regions of
    the input image most influenced the model's prediction. It does this by
    weighting the target layer's activation channels by the gradient of the
    predicted class score with respect to those activations, then summing and
    applying ReLU to retain only positively contributing regions.

    This class uses PyTorch forward and backward hooks to capture activations
    and gradients without modifying the model itself.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization", ICCV 2017.
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # Store the layer's output activations on the forward pass
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            # Store the gradient flowing back through this layer
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        Compute the Grad-CAM heatmap for a single image.

        Args:
            input_tensor: (1, C, H, W) image tensor
            target_class: class index to explain; defaults to the predicted class

        Returns:
            Normalised heatmap tensor of shape (1, 1, h, w) in [0, 1]
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)

        # If no target is specified, explain the top-scoring class
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        # Backpropagate the class score to compute gradients at the target layer
        score = logits[:, target_class]
        score.backward()

        # Global average-pool the gradients to get per-channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activation channels, then ReLU to keep only
        # regions that positively contribute to the predicted class
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))

        # Normalise to [0, 1] so the heatmap can be overlaid on the image
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam
