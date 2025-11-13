import argparse
import os
import pickle
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as F
import segmentation_models_pytorch.utils.metrics as smp_metrics
import torch
import yaml
from albumentations.core.serialization import from_dict
from pytorch_lightning.loggers import WandbLogger
from pytorch_toolbelt.losses import BinaryFocalLoss, JaccardLoss
from tools.losses import BoundaryLoss
from torch.utils.data import DataLoader

from data import read_data, SegmentationDataset
from metrics import binary_mean_iou
from utils import shuffle_list, split, find_average, find_max, find_min, object_from_dict, state_dict_from_disk, \
    split_with_txt


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=str, help="Path to the config.", required=True)
    return parser.parse_args()


class SegmentationPipeline(pl.LightningModule):
    def __init__(self, h_params):
        super(SegmentationPipeline, self).__init__()
        self.h_params = h_params

        self.model = object_from_dict(self.h_params["model"])

        if "resume_from_checkpoint" in self.h_params:
            corrections: Dict[str, str] = {"model.": ""}

            state_dict = state_dict_from_disk(
                file_path=self.h_params["resume_from_checkpoint"],
                rename_in_layers=corrections,
            )
            self.model.load_state_dict(state_dict)

        self.losses = [("jaccard", 0.6, JaccardLoss(mode="binary", from_logits=True)),
                       ("boundary", 0.3, BoundaryLoss(mode="binary")),
                       ("focal", 0.9, BinaryFocalLoss())]

        self.metrics = {"FScore": smp_metrics.Fscore(), "Accuracy": smp_metrics.Accuracy(),
                        "Recal": smp_metrics.Recall(), "Precision": smp_metrics.Precision()}

        self.lr = h_params["optimizer"]["lr"]

        self.train_samples = None
        self.val_samples = None

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)

    def get_data(self, data_type: str) -> Tuple[List[str], List[str], List[List[int]]]:
        # Read Data
        root_path = self.h_params[data_type]["root"]
        images_path = os.path.join(root_path, self.h_params[data_type]["images_folder"])
        masks_path = os.path.join(root_path, self.h_params[data_type]["masks_folder"])
        data_save_file_path = self.h_params[data_type]["data_info_file"]
        crop_object = self.h_params[data_type]["crop_object"]

        if not os.path.exists(data_save_file_path):
            images, masks, bboxes = read_data(images_path=images_path,
                                              masks_path=masks_path,
                                              crop_object=crop_object,
                                              mask_extension=self.h_params["data"]["mask_extension"])
            with open(data_save_file_path, "wb") as fp:
                pickle.dump([images, masks, bboxes], fp, protocol=-1)
        else:
            with open(data_save_file_path, "rb") as fp:
                [images, masks, bboxes] = pickle.load(fp)

        # Shuffle Data
        images, masks, bboxes = shuffle_list(images, masks, bboxes)

        return images, masks, bboxes

    def setup(self, stage=0):

        images, masks, bboxes = self.get_data("data")

        if self.h_params["train_parameters"]["use_test_set"]:
            train_images, train_masks, train_bboxes = images, masks, bboxes
            val_images, val_masks, val_bboxes = self.get_data("test_data")

        else:
            if self.h_params["data"]["prev_validation_set"]:
                with open(self.h_params["data"]["prev_validation_set"], "r") as fp:
                    validation_image_names = fp.readlines()

                validation_image_names = set(validation_image_names)

                train_set, val_set = split_with_txt(images, masks, bboxes,
                                                    validation_image_names=validation_image_names)

                train_images, train_masks, train_bboxes = train_set
                val_images, val_masks, val_bboxes = val_set
            else:
                # Split Data
                train_images, val_images = split(images)
                train_masks, val_masks = split(masks)
                train_bboxes, val_bboxes = split(bboxes)

        self.train_samples = list(zip(train_images, train_masks, train_bboxes))
        self.val_samples = list(zip(val_images, val_masks, val_bboxes))

        print("Len train samples : ", len(self.train_samples))
        print("Len val samples : ", len(self.val_samples))

        os.makedirs(self.h_params["data"]["val_output_folder"], exist_ok=True)

    def train_dataloader(self) -> DataLoader:
        train_aug = from_dict(self.h_params["train_aug"])

        data_loader = DataLoader(
            SegmentationDataset(self.train_samples, train_aug, None),
            batch_size=self.h_params["train_parameters"]["batch_size"],
            num_workers=self.h_params["train_parameters"]["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        print("Train dataloader : ", len(data_loader))

        return data_loader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        val_aug = from_dict(self.h_params["val_aug"])

        data_loader = DataLoader(
            SegmentationDataset(self.val_samples, val_aug, None),
            batch_size=self.h_params["train_parameters"]["batch_size"],
            num_workers=self.h_params["train_parameters"]["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        print("Validation dataloader : ", len(data_loader))

        return data_loader

    def configure_optimizers(self):
        optimizer = object_from_dict(self.h_params["optimizer"],
                                     params=[x for x in self.model.parameters() if x.requires_grad])

        self.optimizers = optimizer

        if self.h_params.get("scheduler"):
            scheduler = object_from_dict(self.h_params["scheduler"], optimizer=optimizer)

            return {"optimizer": optimizer,
                    "lr_scheduler": scheduler,
                    "monitor": "val_mse"}

        return {"optimizer": optimizer,
                "monitor": "val_mse"}

    def training_step(self, batch: Dict, batch_idx: int):
        features = batch["features"]
        masks = batch["masks"]

        logits = self.forward(features)

        total_loss = 0
        logs = {}

        for loss_name, weight, loss in self.losses:
            loss_mask = loss(logits, masks)
            total_loss += weight * loss_mask
            logs[f"train_mask_{loss_name}"] = loss_mask

        self.log("train_loss", total_loss)

        logs[f"train_loss"] = total_loss
        logs["lr"] = self._get_current_lr()

        self.lr = logs["lr"]

        return {"loss": total_loss, "log": logs}

    def validation_step(self, batch: Dict, batch_idx: int):
        features = batch["features"]
        masks = batch["masks"]

        logits = self.forward(features)
        threshold = self.h_params["inference_parameters"]["threshold"]

        for idx in range(logits.shape[0]):
            # pred_mask = (preds[idx, :, :].detach().cpu().numpy() * 255).astype(np.uint8)
            pred_mask = (logits[idx, :, :].cpu().numpy() > threshold).astype(np.uint8) * 255
            cv2.imwrite(
                os.path.join(self.h_params["data"]["val_output_folder"], batch['image_id'][idx]), pred_mask[0, :, :]
            )

        result = {}
        for loss_name, _, loss in self.losses:
            result[f"val_mask_{loss_name}"] = loss(logits, masks)
            self.log(f"val_mask_{loss_name}", result[f"val_mask_{loss_name}"])

        result["val_iou"] = binary_mean_iou(logits, masks)
        self.log(f"val_iou", result["val_iou"])

        output = (logits > threshold).int()
        result["val_mse"] = F.mean_squared_error(output, masks)
        self.log("val_mse", result["val_mse"])

        for metric_name, metric in self.metrics.items():
            result[metric_name] = metric(output, masks)
            self.log(metric_name, result[metric_name])

        return result

    def validation_epoch_end(self, outputs: List[Any]) -> Dict:
        logs = {"epoch": self.trainer.current_epoch}

        avg_val_iou = find_average(outputs, "val_iou")
        avg_val_mse = find_average(outputs, "val_mse")

        max_val_mse = find_max(outputs, "val_mse")
        min_val_iou = find_min(outputs, "val_iou")

        logs["val_iou"] = avg_val_iou
        logs["val_mse"] = avg_val_mse

        logs["min_val_iou"] = min_val_iou
        logs["max_val_mse"] = max_val_mse

        self.log("val_iou", logs["val_iou"])
        self.log("val_mse", logs["val_mse"])

        self.log("min_val_iou", logs["min_val_iou"])
        self.log("max_val_mse", logs["max_val_mse"])

        for metric_name, metric in self.metrics.items():
            logs[metric_name] = find_average(outputs, metric_name)
            self.log(metric_name, logs[metric_name])

        return {"val_iou": avg_val_iou, "log": logs}

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers.param_groups][0]  # type: ignore
        return torch.Tensor([lr])[0].cuda()


def main():
    args = get_args()

    with open(args.config_path) as fp:
        h_params = yaml.load(fp, Loader=yaml.SafeLoader)

    pipeline = SegmentationPipeline(h_params=h_params)

    Path(h_params["checkpoint_callback"]["dirpath"]).mkdir(exist_ok=True, parents=True)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=30,
        progress_bar_refresh_rate=1,
        precision=16,
        num_sanity_val_steps=2,
        gradient_clip_val=5.0,
        sync_batchnorm=True,
        benchmark=True,
        logger=WandbLogger(h_params["experiment_name"]),
        checkpoint_callback=object_from_dict(h_params["checkpoint_callback"]),
        auto_lr_find=not h_params.get("scheduler"),
        stochastic_weight_avg=h_params.get("scheduler"),
        plugins="ddp",
        # profiler="pytorch"
    )

    if not h_params.get("scheduler"):
        trainer.tune(pipeline)

    trainer.fit(pipeline)


if __name__ == '__main__':
    main()
