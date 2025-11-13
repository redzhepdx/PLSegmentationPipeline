from typing import List

import pytorch_toolbelt.inference.functional as F
from torch import Tensor, nn
from torch.nn.functional import interpolate


def flipud_image2mask(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image segmentation that averages predictions
    for input image and horizontally flipped one.

    For segmentation we need to reverse the transformation after making a prediction
    on augmented input.
    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    output = model(image) + F.torch_flipud(model(F.torch_flipud(image)))
    one_over_2 = float(1.0 / 2.0)
    return output * one_over_2


def fliphv_image2mask(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image segmentation that averages predictions
    for input image and horizontally and vertically flipped one.

    For segmentation we need to reverse the transformation after making a prediction
    on augmented input.
    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    flipped_image = F.torch_flipud(F.torch_fliplr(image))
    output = model(image) + F.torch_flipud(F.torch_fliplr(model(flipped_image)))
    one_over_2 = float(1.0 / 2.0)
    return output * one_over_2


class Flip4TTA(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        outputs = self.model(image)

        output_fliplr = F.torch_flipud(self.model(F.torch_flipud(image)))
        outputs += output_fliplr

        output_flipud = F.torch_fliplr(self.model(F.torch_fliplr(image)))
        outputs += output_flipud

        output_fliphv = self.model(F.torch_fliplr(F.torch_flipud(image)))
        output_fliphv = F.torch_flipud(F.torch_fliplr(output_fliphv))
        outputs += output_fliphv

        outputs /= 4

        return outputs


class MultiscaleWeightedTTAWrapper(nn.Module):
    """
    Weighted Multiscale TTA wrapper module
    """

    def __init__(self, model: nn.Module, scale_levels: List[float] = None, weights: List[float] = None):
        """
        Initialize multi-scale TTA wrapper

        :param model: Base model for inference
        :param scale_levels: List of additional scale levels,
            e.g: [0.5, 0.75, 1.25]
        :param weights: List of weights per scale [1.0, 0.5, 0.5, 1.25]
        """
        super().__init__()
        assert scale_levels, "scale_levels must be set"

        if weights is None:
            weights = [1.0] + [1.0 for _ in range(len(scale_levels))]

        self.model = model
        self.scale_levels = scale_levels
        self.weights = weights

    def forward(self, input: Tensor) -> Tensor:
        h = input.size(2)
        w = input.size(3)

        out_size = h, w
        output = self.model(input) * self.weights[0]

        if self.scale_levels:
            for scale, weight in zip(self.scale_levels, self.weights[1:]):
                dst_size = int(h * scale), int(w * scale)
                input_scaled = interpolate(input, dst_size, mode="bilinear", align_corners=False)
                output_scaled = self.model(input_scaled)
                output_scaled = interpolate(output_scaled, out_size, mode="bilinear", align_corners=False)
                output += (output_scaled * weight)
            output /= sum(self.weights)

        return output
