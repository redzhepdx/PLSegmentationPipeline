import math
import os
import pydoc
import random
import re
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any, Set

import cv2
import numpy as np
import pydensecrf.densecrf as densecrf
import pydensecrf.utils as crf_utils
import torch

random.seed(1337)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def dense_crf(inputs, predict_probs):
    h = predict_probs.shape[0]
    w = predict_probs.shape[1]

    predict_probs = np.expand_dims(predict_probs, 0)
    predict_probs = np.append(1 - predict_probs, predict_probs, axis=0)

    d = densecrf.DenseCRF2D(w, h, 2)
    U = crf_utils.unary_from_softmax(predict_probs)

    U = np.ascontiguousarray(U)
    inputs = np.ascontiguousarray(inputs)

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=10, compat=10)
    d.addPairwiseBilateral(sxy=100, srgb=100, rgbim=inputs, compat=1)

    Q = d.inference(5)
    Q = np.array(Q)
    Q = np.argmax(Q, axis=0)
    Q = Q.reshape((h, w)) * 255
    return Q


def haze_removal(image, w0=0.6, t0=0.1):
    darkImage = image.min(axis=2)
    maxDarkChannel = darkImage.max()
    darkImage = darkImage.astype(np.double)

    t = 1 - w0 * (darkImage / maxDarkChannel)
    T = t * 255
    T.dtype = 'uint8'

    t[t < t0] = t0

    J = image
    J[:, :, 0] = (image[:, :, 0] - (1 - t) * maxDarkChannel) / t
    J[:, :, 1] = (image[:, :, 1] - (1 - t) * maxDarkChannel) / t
    J[:, :, 2] = (image[:, :, 2] - (1 - t) * maxDarkChannel) / t

    return J


def fill_and_close(mask: np.ndarray) -> np.ndarray:
    # Thicken the edges
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=6)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.full_like(mask, 255)

    contours = list(filter(lambda c: cv2.contourArea(c) > 0, contours))
    contours = list(sorted(contours, key=lambda c: -1 * cv2.contourArea(c)))

    hull = cv2.convexHull(contours[0], False)

    if cv2.contourArea(contours[0]) < 500:
        return np.full_like(mask, 255)

    cv2.drawContours(mask, contours[1:], 0, color=0, thickness=-1)
    cv2.drawContours(mask, [hull], 0, color=255, thickness=-1)

    return mask


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)

    return pydoc.locate(object_type)(**kwargs)


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image)


def shuffle_list(*lists):
    if len(lists) > 1:
        lst = list(zip(*lists))

        random.shuffle(lst)
        return zip(*lst)
    else:
        lst = list(*lists)
        random.shuffle(lst)
        return lst


def split(*lists, split_percent=0.9):
    if len(lists) > 1:
        split_point = int(len(lists[0]) * split_percent)
        partitions = [(lst[:split_point], lst[split_point:]) for lst in lists]
        return partitions
    else:
        lst = list(*lists)
        split_point = int(len(lst) * split_percent)
        partitions = (lst[:split_point], lst[split_point:])
        return partitions


def split_with_txt(images: List[str], masks: List[str], bboxes: List[List], validation_image_names: Set[str]):
    train_images, validation_images = [], []
    train_masks, validation_masks = [], []
    train_bboxes, validation_bboxes = [], []

    for image, mask, bbox in zip(images, masks, bboxes):
        base_image_name = os.path.basename(image)

        if base_image_name in validation_image_names:
            validation_images.append(image)
            validation_masks.append(mask)
            validation_bboxes.append(bbox)
        else:
            train_images.append(image)
            train_masks.append(mask)
            train_bboxes.append(bbox)

    return (train_images, train_masks, train_bboxes), (validation_images, validation_masks, validation_bboxes)


def remove_small_regions(binary_mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(binary_mask, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE)

    # remove unusable contours with area 0
    contours = list(filter(lambda c: cv2.contourArea(c) > 0, contours))

    # sort contours descending by area size
    contours = list(sorted(contours, key=lambda c: -1 * cv2.contourArea(c)))

    # (x, y, w, h) = cv2.boundingRect(contours[0])
    cv2.fillPoly(binary_mask, pts=contours[1:], color=0)

    return binary_mask


def find_average(outputs: List, name: str) -> torch.Tensor:
    if len(outputs[0][name].shape) == 0:
        return torch.stack([x[name] for x in outputs]).mean()
    return torch.cat([x[name] for x in outputs]).mean()


def find_max(outputs: List, name: str) -> torch.Tensor:
    if len(outputs[0][name].shape) == 0:
        return torch.stack([x[name] for x in outputs]).max()
    return torch.cat([x[name] for x in outputs]).max()


def find_min(outputs: List, name: str) -> torch.Tensor:
    if len(outputs[0][name].shape) == 0:
        return torch.stack([x[name] for x in outputs]).min()
    return torch.cat([x[name] for x in outputs]).min()


def pad_to_size(
        target_size: Tuple[int, int],
        image: np.array,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
) -> Dict[str, Union[np.ndarray, Tuple[int, int, int, int]]]:
    """Pads the image on the sides to the target_size
    Args:
        target_size: (target_height, target_width)
        image:
        bboxes: np.array with shape (num_boxes, 4). Each row: [x_min, y_min, x_max, y_max]
        keypoints: np.array with shape (num_keypoints, 2), each row: [x, y]
    Returns:
        {
            "image": padded_image,
            "pads": (x_min_pad, y_min_pad, x_max_pad, y_max_pad),
            "bboxes": shifted_boxes,
            "keypoints": shifted_keypoints
        }
    """
    target_height, target_width = target_size

    image_height, image_width = image.shape[:2]

    if target_width < image_width:
        raise ValueError(f"Target width should bigger than image_width" f"We got {target_width} {image_width}")

    if target_height < image_height:
        raise ValueError(f"Target height should bigger than image_height" f"We got {target_height} {image_height}")

    if image_height == target_height:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = target_height - image_height
        y_min_pad = y_pad // 2
        y_max_pad = y_pad - y_min_pad

    if image_width == target_width:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = target_width - image_width
        x_min_pad = x_pad // 2
        x_max_pad = x_pad - x_min_pad

    result = {
        "pads": (x_min_pad, y_min_pad, x_max_pad, y_max_pad),
        "image": cv2.copyMakeBorder(image, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_CONSTANT),
    }

    if bboxes is not None:
        bboxes[:, 0] += x_min_pad
        bboxes[:, 1] += y_min_pad
        bboxes[:, 2] += x_min_pad
        bboxes[:, 3] += y_min_pad

        result["bboxes"] = bboxes

    if keypoints is not None:
        keypoints[:, 0] += x_min_pad
        keypoints[:, 1] += y_min_pad

        result["keypoints"] = keypoints

    return result


def pad(image: np.array, factor: int = 32, border: int = cv2.BORDER_CONSTANT) -> tuple:
    """Pads the image on the sides, so that it will be divisible by factor.
    Common use case: UNet type architectures.
    Args:
        image:
        factor:
        border: cv2 type border.
    Returns: padded_image
    """
    height, width = image.shape[:2]

    if height % factor == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = factor - height % factor
        y_min_pad = y_pad // 2
        y_max_pad = y_pad - y_min_pad

    if width % factor == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = factor - width % factor
        x_min_pad = x_pad // 2
        x_max_pad = x_pad - x_min_pad

    padded_image = cv2.copyMakeBorder(image, y_min_pad, y_max_pad, x_min_pad, x_max_pad, border)

    return padded_image, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


def unpad_from_size(
        pads: Tuple[int, int, int, int],
        image: Optional[np.array] = None,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Crops patch from the center so that sides are equal to pads.
    Args:
        image:
        pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
        bboxes: np.array with shape (num_boxes, 4). Each row: [x_min, y_min, x_max, y_max]
        keypoints: np.array with shape (num_keypoints, 2), each row: [x, y]
    Returns: cropped image
    {
            "image": cropped_image,
            "bboxes": shifted_boxes,
            "keypoints": shifted_keypoints
        }
    """
    x_min_pad, y_min_pad, x_max_pad, y_max_pad = pads

    result = {}

    if image is not None:
        height, width = image.shape[:2]
        result["image"] = image[y_min_pad: height - y_max_pad, x_min_pad: width - x_max_pad]

    if bboxes is not None:
        bboxes[:, 0] -= x_min_pad
        bboxes[:, 1] -= y_min_pad
        bboxes[:, 2] -= x_min_pad
        bboxes[:, 3] -= y_min_pad

        result["bboxes"] = bboxes

    if keypoints is not None:
        keypoints[:, 0] -= x_min_pad
        keypoints[:, 1] -= y_min_pad

        result["keypoints"] = keypoints

    return result


def rename_layers(state_dict: Dict[str, Any], rename_in_layers: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)

        result[key] = value

    return result


def state_dict_from_disk(
        file_path: Union[Path, str], rename_in_layers: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Loads PyTorch checkpoint from disk, optionally renaming layer names.
    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns:
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if rename_in_layers is not None:
        state_dict = rename_layers(state_dict, rename_in_layers)

    return state_dict
