import argparse
import os
import pickle
from typing import Dict

import albumentations.augmentations.functional as F
import numpy as np
import torch
import torch.nn as nn
import tqdm
import yaml
from albumentations.core.serialization import from_dict
from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import to_numpy
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader

from data import SegmentationDataset, read_data
from tools.tta import MultiscaleWeightedTTAWrapper, Flip4TTA
from utils import state_dict_from_disk, object_from_dict, tensor_from_rgb_image, remove_small_regions


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=str, help="Path to the config.", required=True)
    arg("-w", "--weight_path", type=str, help="Path to trained model", required=True)
    arg("-v", "--vis", action='store_true', default=False)
    return parser.parse_args()


@torch.no_grad()
def evaluate(dataloader: DataLoader, model: nn.Module, h_params: Dict, device=torch.device("cuda:0")) -> None:
    if h_params["inference_parameters"]["activation"]:
        print("Activation")
        model = nn.Sequential(model, nn.Sigmoid())

    # Create a test time augmentation wrapper [MultiScale + D4 TTA]
    if h_params["inference_parameters"]["test_time_augmentation"]:
        tta_model = MultiscaleWeightedTTAWrapper(
            Flip4TTA(model),
            scale_levels=[0.5],
            weights=[2.0, 1.0]
        )

    use_test_time_augmentation = bool(h_params["inference_parameters"]["test_time_augmentation"])

    scores = []

    for batch in tqdm.tqdm(dataloader):
        torch_images = batch["features"]
        image_paths = batch["image_id"]
        gt_mask = batch["masks"]
        non_transformed_image = torch_images.cpu().numpy()[0]

        threshold = h_params["inference_parameters"]["threshold"]

        batch_size = torch_images.shape[0]

        input_height, input_width = non_transformed_image.shape[:2]

        # If input image is too large, split into smaller patches, process and merge
        if input_height > h_params["inference_parameters"]["max_edge_h"] or \
                input_width > h_params["inference_parameters"]["max_edge_w"]:

            tile_size = (h_params["inference_parameters"]["split_edge"], h_params["inference_parameters"]["split_edge"])
            tile_step = (tile_size[0] // 2, tile_size[1] // 2)

            # Create a slicer
            tiler = ImageSlicer(non_transformed_image.shape, tile_size=tile_size, tile_step=tile_step,
                                weight="pyramid")

            # Allocate a CUDA buffer for holding entire mask
            merger = CudaTileMerger(tiler.target_shape, 1, tiler.weight)

            # Slice, Normalize and toTensor
            tiles = [tensor_from_rgb_image(F.normalize(tile,
                                                       mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225],
                                                       max_pixel_value=255.0))
                     for tile in tiler.split(non_transformed_image)]

            # Run predictions for tiles and accumulate them
            for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops)), batch_size=1, pin_memory=True):
                tiles_batch = tiles_batch.float().cuda()

                # Optional test time augmentation for each patch
                if use_test_time_augmentation:
                    pred_batch = tta_model(tiles_batch)
                else:
                    pred_batch = model(tiles_batch)

                # Gather the predicted patch mask
                merger.integrate_batch(pred_batch, coords_batch)

            # Normalize accumulated mask and convert back to numpy
            merged_mask = np.moveaxis(to_numpy(merger.merge()), 0, -1)
            predictions = torch.from_numpy(
                np.expand_dims(
                    tiler.crop_to_orignal_size(merged_mask), 0)
            ).permute(0, 3, 1, 2)

        else:
            # Optional test time augmentation for image
            if use_test_time_augmentation:
                predictions = tta_model(torch_images.to(device))
            else:
                # Direct prediction
                predictions = model(torch_images.to(device))

        # Visualize and Store Step
        for batch_idx in range(batch_size):
            # Get the mask prediction directly from logits
            mask = (predictions[batch_idx][0].cpu().numpy() > threshold).astype(np.uint8)

            output_mask = remove_small_regions(mask)
            gt_mask = gt_mask.cpu().numpy()[0][0].astype(np.uint8)

            iou_score = jaccard_score(y_pred=output_mask, y_true=gt_mask, average='micro')
            # print(f"Score : {iou_score}")

            scores.append(iou_score)

    scores = np.array(scores)
    avg_score = scores.mean()
    min_score = scores.min()
    max_score = scores.max()

    print(f"Average IOU : {avg_score}")
    print(f"Max IOU : {max_score}")
    print(f"Min IOU : {min_score}")


def main():
    args = parse_args()

    with open(args.config_path) as fp:
        h_params = yaml.load(fp, Loader=yaml.SafeLoader)

    # Create a model
    model = object_from_dict(h_params["model"])

    # Load the pre-trained weights
    corrections: Dict[str, str] = {"model.": ""}
    state_dict = state_dict_from_disk(file_path=args.weight_path, rename_in_layers=corrections)
    model.load_state_dict(state_dict)
    model.to(torch.device("cuda:0"))

    # Read Data
    root_path = h_params["test_data"]["root"]
    images_path = os.path.join(root_path, h_params["test_data"]["images_folder"])
    masks_path = os.path.join(root_path, h_params["test_data"]["masks_folder"])
    data_save_file_path = h_params["test_data"]["data_info_file"]
    crop_object = h_params["test_data"]["crop_object"]

    if not os.path.exists(data_save_file_path):
        images, masks, bboxes = read_data(images_path=images_path,
                                          masks_path=masks_path,
                                          crop_object=crop_object,
                                          mask_extension=h_params["test_data"]["mask_extension"])
        with open(data_save_file_path, "wb") as fp:
            pickle.dump([images, masks, bboxes], fp, protocol=-1)
    else:
        with open(data_save_file_path, "rb") as fp:
            [images, masks, bboxes] = pickle.load(fp)

    test_aug = from_dict(h_params["test_aug"])
    evaluation_set = list(zip(images, masks, bboxes))

    data_loader = DataLoader(
        SegmentationDataset(evaluation_set, test_aug, None),
        batch_size=1,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    # Predict
    evaluate(data_loader, model, h_params)


if __name__ == '__main__':
    main()
