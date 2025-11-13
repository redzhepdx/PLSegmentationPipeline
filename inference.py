import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Dict

import albumentations.augmentations.functional as F
import cv2
import numpy as np
import torch
import torch.nn as nn
import tqdm
import yaml
from albumentations.core.serialization import from_dict
from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import to_numpy
from torch.utils.data import DataLoader

from data import InferenceDataset, read_data
from tools.tta import MultiscaleWeightedTTAWrapper, Flip4TTA
from utils import state_dict_from_disk, object_from_dict, dense_crf, tensor_from_rgb_image, unpad_from_size, \
    remove_small_regions, sigmoid, fill_and_close


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=str, help="Path to the config.", required=True)
    arg("-w", "--weight_path", type=str, help="Path to trained model", required=True)
    arg("-v", "--vis", action='store_true', default=False)
    return parser.parse_args()


@torch.no_grad()
def predict(dataloader: DataLoader, model: nn.Module, h_params: Dict, device=torch.device("cuda:0"),
            vis=False) -> None:
    model.eval()
    os.makedirs(h_params["test_data"]["output_mask_path"], exist_ok=True)

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
    use_crf = bool(h_params["inference_parameters"]["use_crf"])

    for batch in tqdm.tqdm(dataloader):
        torch_images = batch["torch_image"]
        image_paths = batch["image_path"]
        non_transformed_image = batch["non_transformed_image"].cpu().numpy()[0]
        print(non_transformed_image.shape)

        pads = batch["pads"]
        bbox = batch["bbox"]

        heights = batch["original_height"]
        widths = batch["original_width"]

        threshold = h_params["inference_parameters"]["threshold"]

        batch_size = torch_images.shape[0]

        input_height, input_width = non_transformed_image.shape[:2]

        # If input image is too large, split into smaller patches, process and merge
        if input_height > h_params["inference_parameters"]["max_edge_h"] or \
                input_width > h_params["inference_parameters"]["max_edge_w"]:
            func_start = time.time()

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

            print(
                f"[Split and Merge]\n"
                f"Shape : {non_transformed_image.shape}\n"
                f"TTA : {use_test_time_augmentation}\n"
                f"TIME : {time.time() - func_start}\n")
        else:
            # Optional test time augmentation for image
            func_start = time.time()
            if use_test_time_augmentation:
                predictions = tta_model(torch_images.to(device))
            else:
                # Direct prediction
                predictions = model(torch_images.to(device))
            print(
                f"[Single Pass]\n"
                f"Shape : {non_transformed_image.shape}\n"
                f"TTA : {use_test_time_augmentation}\n"
                f"TIME : {time.time() - func_start}\n")

        # Visualize and Store Step
        for batch_idx in range(batch_size):
            image_name = Path(image_paths[batch_idx]).stem

            if threshold:
                # Get the mask prediction directly from logits
                mask = (predictions[batch_idx][0].cpu().numpy() > threshold).astype(np.uint8) * 255

                # Apply crf as post-processing algo
                if use_crf:
                    if isinstance(list(model.children())[-1], nn.Sigmoid):
                        probs = predictions
                        probs[probs < threshold] = 0.0
                    else:
                        # Probabilistic representation is necessary for CRF so Sigmoid(exp_logsigmoid trick)
                        probs = torch.nn.functional.logsigmoid(predictions).exp()
                        probs[probs < sigmoid(threshold)] = 0.0

                    # Apply crf to smoothen the output predictions
                    crf_mask = dense_crf(non_transformed_image,
                                         predict_probs=probs[batch_idx][0].cpu().numpy()).astype(np.uint8)

                # Optional Visualize
                if vis:
                    cv2.imshow("mask", cv2.resize(mask, (600, 600)))
                    cv2.imshow("image", cv2.resize(non_transformed_image, (600, 600)))

                    if use_crf:
                        cv2.imshow("crf", cv2.resize(crf_mask, (600, 600)))

                    cv2.waitKey(0)

                # Resize it to original size
                # mask = cv2.resize(
                #     mask, (widths[batch_idx].item(), heights[batch_idx].item()), interpolation=cv2.INTER_NEAREST
                # )

                # Create an empty original sized image
                output_mask = np.zeros(shape=(heights[batch_idx].item(), widths[batch_idx].item()))
                print("Output Mask : ", mask.shape)

                # Remove pads
                mask = unpad_from_size(pads, image=mask)["image"]
                print("Mask Unpad Shape : ", mask.shape)

                # Remove small disconnected regions
                start = time.time()
                mask = remove_small_regions(mask)
                print(f"[POST PROCESSING] Small disconnected region removal time : {time.time() - start}")

                # Place the prediction mask into empty image
                output_mask[bbox[0]: bbox[1], bbox[2]: bbox[3]] = mask

                if h_params["inference_parameters"]["use_crf"]:
                    print("USE CRF")
                    # Remove pads
                    crf_mask = unpad_from_size(pads, image=crf_mask)["image"]

                    # Remove small disconnected regions
                    crf_mask = remove_small_regions(crf_mask)

                    # Place the crf prediction mask into empty image
                    output_mask[bbox[0]: bbox[1], bbox[2]: bbox[3]] = crf_mask

                output_mask = fill_and_close(output_mask)

                cv2.imwrite(os.path.join(h_params["test_data"]["output_mask_path"], f"{image_name}.png"), output_mask)
            else:
                mask = predictions
                if not h_params["inference_parameters"]["activation"]:
                    mask = torch.sigmoid(predictions)
                mask = mask[batch_idx][0].cpu().numpy()
                np.save(os.path.join(h_params["test_data"]["output_mask_path"], f"{image_name}.npy"), mask)


def create_dataloader(h_params: Dict) -> torch.utils.data.DataLoader:
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

    dataset = InferenceDataset(images, bboxes, transform=test_aug, remove_haze=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )

    return data_loader


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

    # Create Data Loader
    data_loader = create_dataloader(h_params)

    # Predict
    predict(data_loader, model, h_params, vis=args.vis)


if __name__ == '__main__':
    main()
