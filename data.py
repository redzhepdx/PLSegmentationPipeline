import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import albumentations as albu
import cv2
import imageio
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset

from utils import tensor_from_rgb_image, pad, haze_removal


class SegmentationDataset(Dataset):
    def __init__(
            self,
            samples: List[Tuple[Path, Path, List]],
            transform: albu.Compose,
            length: int = None,
            remove_haze: bool = False
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.remove_haze = remove_haze

        if length is None:
            self.length = len(self.samples)
        else:
            self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % len(self.samples)

        image_path, mask_path, bbox = self.samples[idx]

        image = cv2.imread(image_path)

        if self.remove_haze:
            image = haze_removal(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = imageio.imread(mask_path)
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Crop
        image = image[bbox[0]: bbox[1], bbox[2]: bbox[3]]
        mask = mask[bbox[0]: bbox[1], bbox[2]: bbox[3]]
        mask[mask < 255] = 0

        # apply augmentations
        sample = self.transform(image=image, mask=mask)
        image, mask = sample["image"], sample["mask"]

        image, _ = pad(image, factor=64, border=cv2.BORDER_CONSTANT)
        mask, _ = pad(mask, factor=64, border=cv2.BORDER_CONSTANT)

        mask = (mask > 0).astype(np.uint8)

        mask = torch.from_numpy(mask)

        return {
            "image_id": os.path.basename(str(image_path)),
            "features": tensor_from_rgb_image(image),
            "masks": torch.unsqueeze(mask, 0).float(),
        }


class InferenceDataset(Dataset):
    def __init__(self, image_paths: List[Path], image_bboxes: List[List], transform: albu.Compose,
                 remove_haze: bool = False) -> None:
        self.image_paths = image_paths
        self.image_bboxes = image_bboxes
        self.transform = transform
        self.remove_haze = remove_haze

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        image_path = self.image_paths[idx]
        bbox = self.image_bboxes[idx]

        image = cv2.imread(image_path)
        if self.remove_haze:
            image = haze_removal(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]

        # Crop
        image = image[bbox[0]: bbox[1], bbox[2]: bbox[3]]

        non_transformed_image = image.copy()

        # Normalize
        image = self.transform(image=image)["image"]

        # Make the image size divisible by 64
        image, pads = pad(image, factor=64, border=cv2.BORDER_CONSTANT)
        non_transformed_image, _ = pad(non_transformed_image, factor=64, border=cv2.BORDER_CONSTANT)

        pad_dict = {
            "image": image,
            "pads": pads
        }

        return {
            "torch_image": tensor_from_rgb_image(pad_dict["image"]),
            "non_transformed_image": non_transformed_image,
            "image_path": str(image_path),
            "pads": pad_dict["pads"],
            "original_width": width,
            "original_height": height,
            "bbox": bbox
        }


def get_object_position(mask, margin=0):
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    min_x = 0
    min_y = 0
    max_y, max_x = mask.shape[:2]

    contours, _ = cv2.findContours(mask, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE)

    # remove unusable contours with area 0
    contours = list(filter(lambda c: cv2.contourArea(c) > 0, contours))

    # sort contours descending by area size
    contours = list(sorted(contours, key=lambda c: -1 * cv2.contourArea(c)))

    (x, y, w, h) = cv2.boundingRect(contours[0])

    return [max(min_y, y - margin), min(y + h + margin, max_y), max(min_x, x - margin), min(x + w + margin, max_x)]


def read_data(images_path: str, masks_path: str,
              crop_object: bool = True, mask_extension: str = ".gif") -> Tuple[List, List, List[List[int]]]:
    image_names = os.listdir(images_path)

    images, masks, bboxes = [], [], []

    for image_name in tqdm.tqdm(image_names):
        mask_name = f"{os.path.splitext(image_name)[0]}{mask_extension}"

        mask = imageio.imread(os.path.join(masks_path, mask_name))

        # Simple hack to remove possible small gray parts
        mask[mask < 255] = 0

        bbox = [0, mask.shape[0], 0, mask.shape[1]]

        if crop_object:
            bbox = get_object_position(mask, margin=100)

        images.append(os.path.join(images_path, image_name))
        masks.append(os.path.join(masks_path, mask_name))
        bboxes.append(bbox)

    return images, masks, bboxes
