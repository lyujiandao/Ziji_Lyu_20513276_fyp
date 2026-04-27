# -*- coding: utf-8 -*-
# CellViT Trainer Class
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

# import wandb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from skimage.color import rgba2rgb
from sklearn.metrics import accuracy_score
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_jaccard_index

from base_ml.base_early_stopping import EarlyStopping
from base_ml.base_trainer import BaseTrainer
from models.segmentation.cell_segmentation.cellvit import DataclassHVStorage
from cell_segmentation.utils.metrics import get_fast_pq, remap_label
from cell_segmentation.utils.tools import cropping_center
from models.segmentation.cell_segmentation.cellvit import CellViT
from utils.tools import AverageMeter

import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


class CellViTTrainer(BaseTrainer):
    """CellViT trainer class

    Args:
        model (CellViT): CellViT model that should be trained
        loss_fn_dict (dict): Dictionary with loss functions for each branch with a dictionary of loss functions.
            Name of branch as top-level key, followed by a dictionary with loss name, loss fn and weighting factor
            Example:
            {
                "nuclei_binary_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}},
                "hv_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}},
                "nuclei_type_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}}
            }
            Required Keys are:
                * nuclei_binary_map
                * hv_map
                * nuclei_type_map
        optimizer (Optimizer): Optimizer
        scheduler (_LRScheduler): Learning rate scheduler
        device (str): Cuda device to use, e.g., cuda:0.
        logger (logging.Logger): Logger module
        logdir (Union[Path, str]): Logging directory
        num_classes (int): Number of nuclei classes
        dataset_config (dict): Dataset configuration. Required Keys are:
            * "tissue_types": describing the present tissue types with corresponding integer
            * "nuclei_types": describing the present nuclei types with corresponding integer
        experiment_config (dict): Configuration of this experiment
        early_stopping (EarlyStopping, optional):  Early Stopping Class. Defaults to None.
        log_images (bool, optional): If images should be logged to WandB. Defaults to False.
        magnification (int, optional): Image magnification. Please select either 40 or 20. Defaults to 40.
        mixed_precision (bool, optional): If mixed-precision should be used. Defaults to False.
    """

    def __init__(
        self,
        model: CellViT,
        loss_fn_dict: dict,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        device: str,
        logger: logging.Logger,
        logdir: Union[Path, str],
        num_classes: int,
        dataset_config: dict,
        experiment_config: dict,
        early_stopping: EarlyStopping = None,
        log_images: bool = False,
        magnification: int = 40,
        mixed_precision: bool = False,
    ):
        super().__init__(
            model=model,
            loss_fn=None,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            logdir=logdir,
            experiment_config=experiment_config,
            early_stopping=early_stopping,
            accum_iter=1,
            log_images=log_images,
            mixed_precision=mixed_precision,
        )
        self.loss_fn_dict = loss_fn_dict
        self.num_classes = num_classes
        self.dataset_config = dataset_config
        self.tissue_types = dataset_config["tissue_types"]
        self.reverse_tissue_types = {v: k for k, v in self.tissue_types.items()}
        self.nuclei_types = dataset_config["nuclei_types"]
        self.magnification = magnification

        # setup logging objects
        self.loss_avg_tracker = {"Total_Loss": AverageMeter("Total_Loss", ":.4f")}
        
        # Add Gating Loss Tracker
        self.loss_avg_tracker["Gating_Loss"] = AverageMeter("Gating_Loss", ":.4f")

        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"] = AverageMeter(
                    f"{branch}_{loss_name}", ":.4f"
                )
        # REMOVED: self.batch_avg_tissue_acc = AverageMeter("Batch_avg_tissue_ACC", ":4.f")

        self.save_maps_every_n_epoches = 5  # Save predicted and GT maps every n epochs for qualitative analysis
        self.visualization_dir = Path(self.logdir) / "visualizations"
        self.visualization_dir.mkdir(exist_ok=True, parents=True)

    def train_epoch(
        self, epoch: int, train_dataloader: DataLoader, unfreeze_epoch: int = 50
    ) -> Tuple[dict, dict]:
        """Training logic for a training epoch

        Args:
            epoch (int): Current epoch number
            train_dataloader (DataLoader): Train dataloader
            unfreeze_epoch (int, optional): Epoch to unfreeze layers
        Returns:
            Tuple[dict, dict]: wandb logging dictionaries
                * Scalar metrics
                * Image metrics
        """
        # Handle Unfreezing (Need to handle DataParallel)
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
             self.model.module.train()
             if epoch >= unfreeze_epoch:
                # Assuming unfreeze_encoder is a method of CellViT
                if hasattr(self.model.module, "unfreeze_encoder"):
                    self.model.module.unfreeze_encoder()
        else:
            self.model.train()
            if epoch >= unfreeze_epoch:
                self.model.unfreeze_encoder()

        binary_dice_scores = []
        binary_jaccard_scores = []
        # REMOVED: tissue_pred = []
        # REMOVED: tissue_gt = [] (not needed for training loop metrics here)
        train_example_img = None

        # reset metrics
        self.loss_avg_tracker["Total_Loss"].reset()
        self.loss_avg_tracker["Gating_Loss"].reset() # Reset Gating Loss
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"].reset()
        # REMOVED: self.batch_avg_tissue_acc.reset()

        # randomly select a batch that should be displayed
        if self.log_images:
            select_example_image = int(torch.randint(0, len(train_dataloader), (1,)))
        else:
            select_example_image = None
        train_loop = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        for batch_idx, batch in train_loop:
            return_example_images = batch_idx == select_example_image
            batch_metrics, example_img = self.train_step(
                batch,
                batch_idx,
                len(train_dataloader),
                return_example_images=return_example_images,
            )
            if example_img is not None:
                train_example_img = example_img
            binary_dice_scores = (
                binary_dice_scores + batch_metrics["binary_dice_scores"]
            )
            binary_jaccard_scores = (
                binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
            )
            # REMOVED: tissue_pred.append(batch_metrics["tissue_pred"])
            # REMOVED: tissue_gt.append(batch_metrics["tissue_gt"])
            train_loop.set_postfix(
                {
                    "Loss": np.round(self.loss_avg_tracker["Total_Loss"].avg, 3),
                    "Gating": np.round(self.loss_avg_tracker["Gating_Loss"].avg, 4), # Show Gating Loss
                    "Dice": np.round(np.nanmean(binary_dice_scores), 3),
                    # REMOVED: "Pred-Acc": np.round(self.batch_avg_tissue_acc.avg, 3),
                }
            )

        # calculate global metrics
        binary_dice_scores = np.array(binary_dice_scores)
        binary_jaccard_scores = np.array(binary_jaccard_scores)
        # REMOVED: tissue_detection_accuracy = accuracy_score(...)

        scalar_metrics = {
            "Loss/Train": self.loss_avg_tracker["Total_Loss"].avg,
            "Gating_Loss/Train": self.loss_avg_tracker["Gating_Loss"].avg, # Log Gating Loss
            "Binary-Cell-Dice-Mean/Train": np.nanmean(binary_dice_scores),
            "Binary-Cell-Jacard-Mean/Train": np.nanmean(binary_jaccard_scores),
            # REMOVED: "Tissue-Multiclass-Accuracy/Train": tissue_detection_accuracy,
        }

        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                scalar_metrics[f"{branch}_{loss_name}/Train"] = self.loss_avg_tracker[
                    f"{branch}_{loss_name}"
                ].avg

        self.logger.info(
            f"{'Training epoch stats:' : <25} "
            f"Loss: {self.loss_avg_tracker['Total_Loss'].avg:.4f} - "
            f"Gating: {self.loss_avg_tracker['Gating_Loss'].avg:.4f} - "
            f"Binary-Cell-Dice: {np.nanmean(binary_dice_scores):.4f} - "
            f"Binary-Cell-Jacard: {np.nanmean(binary_jaccard_scores):.4f} "
            # REMOVED: f"Tissue-MC-Acc.: {tissue_detection_accuracy:.4f}"
        )

        image_metrics = {"Example-Predictions/Train": train_example_img}

        return scalar_metrics, image_metrics

    def train_step(
        self,
        batch: object,
        batch_idx: int,
        num_batches: int,
        return_example_images: bool,
    ) -> Tuple[dict, Union[plt.Figure, None]]:
        """Training step

        Args:
            batch (object): Training batch, consisting of images ([0]), masks ([1]), tissue_types ([2]) and figure filenames ([3])
            batch_idx (int): Batch index
            num_batches (int): Total number of batches in epoch
            return_example_images (bool): If an example preciction image should be returned

        Returns:
            Tuple[dict, Union[plt.Figure, None]]:
                * Batch-Metrics: dictionary with the following keys:
                * Example prediction image
        """
        # unpack batch
        imgs = batch[0].to(self.device)  # imgs shape: (batch_size, 3, H, W)
        masks = batch[
            1
        ]  # dict: keys: "instance_map", "nuclei_map", "nuclei_binary_map", "hv_map"
        tissue_types = batch[2]  # list[str]

        # Get Core Model for accessing custom methods like calculate_gating_loss
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model_core = self.model.module
        else:
            model_core = self.model

        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # make predictions
                predictions_ = self.model.forward(imgs)

                # reshaping and postprocessing
                predictions = self.unpack_predictions(predictions=predictions_)
                gt = self.unpack_masks(masks=masks, tissue_types=tissue_types)

                # calculate loss
                total_loss = self.calculate_loss(predictions, gt)

                # --- Add Gating Loss ---
                if hasattr(model_core, "calculate_gating_loss"):
                    gating_loss = model_core.calculate_gating_loss(predictions_)
                    total_loss += gating_loss
                    self.loss_avg_tracker["Gating_Loss"].update(gating_loss.item())

                # backward pass
                self.scaler.scale(total_loss).backward()

                if (
                    ((batch_idx + 1) % self.accum_iter == 0)
                    or ((batch_idx + 1) == num_batches)
                    or (self.accum_iter == 1)
                ):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.model.zero_grad()
        else:
            predictions_ = self.model.forward(imgs)
            predictions = self.unpack_predictions(predictions=predictions_)
            gt = self.unpack_masks(masks=masks, tissue_types=tissue_types)

            # calculate loss
            total_loss = self.calculate_loss(predictions, gt)

            # --- Add Gating Loss ---
            if hasattr(model_core, "calculate_gating_loss"):
                gating_loss = model_core.calculate_gating_loss(predictions_)
                total_loss += gating_loss
                self.loss_avg_tracker["Gating_Loss"].update(gating_loss.item())

            total_loss.backward()
            if (
                ((batch_idx + 1) % self.accum_iter == 0)
                or ((batch_idx + 1) == num_batches)
                or (self.accum_iter == 1)
            ):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.model.zero_grad()
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()

        batch_metrics = self.calculate_step_metric_train(predictions, gt)

        # if return_example_images:
        #     return_example_images = self.generate_example_image(
        #         imgs, predictions, gt, num_images=4, num_nuclei_classes=self.num_classes
        #     )
        # else:
        return_example_images = None

        return batch_metrics, return_example_images

    def validation_epoch(
        self, epoch: int, val_dataloader: DataLoader
    ) -> Tuple[dict, dict, float]:
        """Validation logic for a validation epoch

        Args:
            epoch (int): Current epoch number
            val_dataloader (DataLoader): Validation dataloader

        Returns:
            Tuple[dict, dict, float]: wandb logging dictionaries
                * Scalar metrics
                * Image metrics
                * Early stopping metric
        """
        self.model.eval()

        binary_dice_scores = []
        binary_jaccard_scores = []
        pq_scores = []
        cell_type_pq_scores = []
        # REMOVED: tissue_pred = []
        tissue_gt = []
        val_example_img = None

        # reset metrics
        self.loss_avg_tracker["Total_Loss"].reset()
        self.loss_avg_tracker["Gating_Loss"].reset() # Reset
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"].reset()
        # REMOVED: self.batch_avg_tissue_acc.reset()

        # randomly select a batch that should be displayed
        if self.log_images:
            select_example_image = int(torch.randint(0, len(val_dataloader), (1,)))
        else:
            select_example_image = None

        should_save_maps = (epoch % self.save_maps_every_n_epoches == 0) and self.log_images
        if should_save_maps:
            self.logger.info(f"Saving predicted and GT maps for epoch {epoch} for qualitative analysis.")

        val_loop = tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader))

        with torch.no_grad():
            for batch_idx, batch in val_loop:
                return_example_images = batch_idx == select_example_image
                batch_metrics, example_img = self.validation_step(
                    batch, batch_idx, return_example_images,epoch = epoch, should_save_maps=should_save_maps and batch_idx == 0,
                )
                if example_img is not None:
                    val_example_img = example_img
                binary_dice_scores = (
                    binary_dice_scores + batch_metrics["binary_dice_scores"]
                )
                binary_jaccard_scores = (
                    binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
                )
                pq_scores = pq_scores + batch_metrics["pq_scores"]
                cell_type_pq_scores = (
                    cell_type_pq_scores + batch_metrics["cell_type_pq_scores"]
                )
                # REMOVED: tissue_pred.append(batch_metrics["tissue_pred"])
                tissue_gt.append(batch_metrics["tissue_gt"])
                val_loop.set_postfix(
                    {
                        "Loss": np.round(self.loss_avg_tracker["Total_Loss"].avg, 3),
                        "Dice": np.round(np.nanmean(binary_dice_scores), 3),
                        # REMOVED: "Pred-Acc": np.round(self.batch_avg_tissue_acc.avg, 3),
                    }
                )
        tissue_types_val = [
            self.reverse_tissue_types[t].lower() for t in np.concatenate(tissue_gt)
        ]

        # calculate global metrics
        binary_dice_scores = np.array(binary_dice_scores)
        binary_jaccard_scores = np.array(binary_jaccard_scores)
        pq_scores = np.array(pq_scores)
        # REMOVED: tissue_detection_accuracy = accuracy_score(...)

        scalar_metrics = {
            "Loss/Validation": self.loss_avg_tracker["Total_Loss"].avg,
            "Gating_Loss/Validation": self.loss_avg_tracker["Gating_Loss"].avg, # Log Gating
            "Binary-Cell-Dice-Mean/Validation": np.nanmean(binary_dice_scores),
            "Binary-Cell-Jacard-Mean/Validation": np.nanmean(binary_jaccard_scores),
            # REMOVED: "Tissue-Multiclass-Accuracy/Validation": tissue_detection_accuracy,
            "bPQ/Validation": np.nanmean(pq_scores),
            "mPQ/Validation": np.nanmean(
                [np.nanmean(pq) for pq in cell_type_pq_scores]
            ),
        }

        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                scalar_metrics[
                    f"{branch}_{loss_name}/Validation"
                ] = self.loss_avg_tracker[f"{branch}_{loss_name}"].avg

        # calculate local metrics
        # per tissue class
        for tissue in self.tissue_types.keys():
            tissue = tissue.lower()
            tissue_ids = np.where(np.asarray(tissue_types_val) == tissue)
            scalar_metrics[f"{tissue}-Dice/Validation"] = np.nanmean(
                binary_dice_scores[tissue_ids]
            )
            scalar_metrics[f"{tissue}-Jaccard/Validation"] = np.nanmean(
                binary_jaccard_scores[tissue_ids]
            )
            scalar_metrics[f"{tissue}-bPQ/Validation"] = np.nanmean(
                pq_scores[tissue_ids]
            )
            scalar_metrics[f"{tissue}-mPQ/Validation"] = np.nanmean(
                [np.nanmean(pq) for pq in np.array(cell_type_pq_scores)[tissue_ids]]
            )

        # calculate nuclei metrics
        for nuc_name, nuc_type in self.nuclei_types.items():
            if nuc_name.lower() == "background":
                continue
            scalar_metrics[f"{nuc_name}-PQ/Validation"] = np.nanmean(
                [pq[nuc_type] for pq in cell_type_pq_scores]
            )

        self.logger.info(
            f"{'Validation epoch stats:' : <25} "
            f"Loss: {self.loss_avg_tracker['Total_Loss'].avg:.4f} - "
            f"Gating: {self.loss_avg_tracker['Gating_Loss'].avg:.4f} - "
            f"Binary-Cell-Dice: {np.nanmean(binary_dice_scores):.4f} - "
            f"Binary-Cell-Jacard: {np.nanmean(binary_jaccard_scores):.4f} - "
            f"bPQ-Score: {np.nanmean(pq_scores):.4f} - "
            f"mPQ-Score: {scalar_metrics['mPQ/Validation']:.4f} "
            # REMOVED: f"Tissue-MC-Acc.: {tissue_detection_accuracy:.4f}"
        )

        image_metrics = {"Example-Predictions/Validation": val_example_img}

        return scalar_metrics, image_metrics, np.nanmean(pq_scores)

    def validation_step(
        self,
        batch: object,
        batch_idx: int,
        return_example_images: bool,
        epoch: int,
        should_save_maps: bool,
    ):
        """Validation step

        Args:
            batch (object): Training batch, consisting of images ([0]), masks ([1]), tissue_types ([2]) and figure filenames ([3])
            batch_idx (int): Batch index
            return_example_images (bool): If an example preciction image should be returned

        Returns:
            Tuple[dict, Union[plt.Figure, None]]:
                * Batch-Metrics: dictionary, structure not fixed yet
                * Example prediction image
        """
        # unpack batch, for shape compare train_step method
        imgs = batch[0].to(self.device)
        masks = batch[1]
        tissue_types = batch[2]

        self.model.zero_grad()
        self.optimizer.zero_grad()
        
        # Get Core Model for accessing custom methods like calculate_gating_loss
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model_core = self.model.module
        else:
            model_core = self.model

        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # make predictions
                predictions_ = self.model.forward(imgs)
                # reshaping and postprocessing
                predictions = self.unpack_predictions(predictions=predictions_)
                gt = self.unpack_masks(masks=masks, tissue_types=tissue_types)
                # calculate loss
                _ = self.calculate_loss(predictions, gt)
                
                # Update tracker with Gating Loss (No Backprop)
                if hasattr(model_core, "calculate_gating_loss"):
                    gating_loss = model_core.calculate_gating_loss(predictions_)
                    self.loss_avg_tracker["Gating_Loss"].update(gating_loss.item())

        else:
            predictions_ = self.model.forward(imgs)
            # reshaping and postprocessing
            predictions = self.unpack_predictions(predictions=predictions_)
            gt = self.unpack_masks(masks=masks, tissue_types=tissue_types)
            # calculate loss
            _ = self.calculate_loss(predictions, gt)
            
            # Update tracker with Gating Loss (No Backprop)
            if hasattr(model_core, "calculate_gating_loss"):
                gating_loss = model_core.calculate_gating_loss(predictions_)
                self.loss_avg_tracker["Gating_Loss"].update(gating_loss.item())

        if should_save_maps and epoch is not None: # Set to False for now, as saving maps can cause issues and is not essential, can be re-enabled after checking the saving function
            try:
                self.save_all_prediction_maps(
                    imgs=imgs,
                    predictions=predictions,
                    gt=gt,
                    epoch=epoch,
                    batch_idx=batch_idx,
                )
                self.logger.info(f"Saved predicted and GT maps for epoch {epoch}.")
            except AssertionError:
                self.logger.error(
                    "AssertionError for saving maps. Please check. Continue without saving maps."
                )

        # get metrics for this batch
        batch_metrics = self.calculate_step_metric_validation(predictions, gt)

        if return_example_images:
            try:
                return_example_images = self.generate_example_image(
                    imgs,
                    predictions,
                    gt,
                    num_images=4,
                    num_nuclei_classes=self.num_classes,
                )
            except AssertionError:
                self.logger.error(
                    "AssertionError for Example Image. Please check. Continue without image."
                )
                return_example_images = None
        else:
            return_example_images = None

        return batch_metrics, return_example_images

    def unpack_predictions(self, predictions: dict) -> DataclassHVStorage:
        """Unpack the given predictions. Main focus lays on reshaping and postprocessing predictions, e.g. separating instances

        Args:
            predictions (dict): Dictionary with the following keys:
                * tissue_types: Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
                * nuclei_binary_map: Logit output for binary nuclei prediction branch. Shape: (batch_size, 2, H, W)
                * hv_map: Logit output for hv-prediction. Shape: (batch_size, 2, H, W)
                * nuclei_type_map: Logit output for nuclei instance-prediction. Shape: (batch_size, num_nuclei_classes, H, W)

        Returns:
            DataclassHVStorage: Processed network output
        """
        
        # REMOVED: predictions["tissue_types"] = predictions["tissue_types"].to(self.device)
        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=1
        )  # shape: (batch_size, 2, H, W)
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        )  # shape: (batch_size, num_nuclei_classes, H, W)
        
        # Handle DataParallel Wrapper
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model_core = self.model.module
        else:
            model_core = self.model

        (
            predictions["instance_map"],
            predictions["instance_types"],
        ) = model_core.calculate_instance_map(
            predictions, self.magnification
        )  # shape: (batch_size, H, W)
        predictions["instance_types_nuclei"] = model_core.generate_instance_nuclei_map(
            predictions["instance_map"], predictions["instance_types"]
        ).to(
            self.device
        )  # shape: (batch_size, num_nuclei_classes, H, W)

        if "regression_map" not in predictions.keys():
            predictions["regression_map"] = None

        predictions = DataclassHVStorage(
            nuclei_binary_map=predictions["nuclei_binary_map"],
            hv_map=predictions["hv_map"],
            nuclei_type_map=predictions["nuclei_type_map"],
            tissue_types=None, # REMOVED: predictions["tissue_types"],
            instance_map=predictions["instance_map"],
            instance_types=predictions["instance_types"],
            instance_types_nuclei=predictions["instance_types_nuclei"],
            batch_size=predictions["nuclei_binary_map"].shape[0], # CHANGED: was predictions["tissue_types"].shape[0],
            regression_map=predictions["regression_map"],
            num_nuclei_classes=self.num_classes,
        )

        return predictions

    def unpack_masks(self, masks: dict, tissue_types: list) -> DataclassHVStorage:
        """Unpack the given masks. Main focus lays on reshaping and postprocessing masks to generate one dict

        Args:
            masks (dict): Required keys are:
                * instance_map: Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W)
                * nuclei_binary_map: Binary nuclei segmentations. Shape: (batch_size, H, W)
                * hv_map: HV-Map. Shape: (batch_size, 2, H, W)
                * nuclei_type_map: Nuclei instance-prediction and segmentation (not binary, each instance has own integer).
                    Shape: (batch_size, num_nuclei_classes, H, W)

            tissue_types (list): List of string names of ground-truth tissue types

        Returns:
            DataclassHVStorage: GT-Results with matching shapes and output types
        """
        # get ground truth values, perform one hot encoding for segmentation maps
        gt_nuclei_binary_map_onehot = (
            F.one_hot(masks["nuclei_binary_map"], num_classes=2)
        ).type(
            torch.float32
        )  # background, nuclei
        nuclei_type_maps = torch.squeeze(masks["nuclei_type_map"]).type(torch.int64)
        gt_nuclei_type_maps_onehot = F.one_hot(
            nuclei_type_maps, num_classes=self.num_classes
        ).type(
            torch.float32
        )  # background + nuclei types

        # assemble ground truth dictionary
        gt = {
            "nuclei_type_map": gt_nuclei_type_maps_onehot.permute(0, 3, 1, 2).to(
                self.device
            ),  # shape: (batch_size, H, W, num_nuclei_classes)
            "nuclei_binary_map": gt_nuclei_binary_map_onehot.permute(0, 3, 1, 2).to(
                self.device
            ),  # shape: (batch_size, H, W, 2)
            "hv_map": masks["hv_map"].to(self.device),  # shape: (batch_size, H, W, 2)
            "instance_map": masks["instance_map"].to(
                self.device
            ),  # shape: (batch_size, H, W) -> each instance has one integer
            "instance_types_nuclei": (
                gt_nuclei_type_maps_onehot * masks["instance_map"][..., None]
            )
            .permute(0, 3, 1, 2)
            .to(
                self.device
            ),  # shape: (batch_size, num_nuclei_classes, H, W) -> instance has one integer, for each nuclei class
            "tissue_types": torch.Tensor([self.tissue_types[t] for t in tissue_types])
            .type(torch.LongTensor)
            .to(self.device),  # shape: batch_size
        }
        if "regression_map" in masks:
            gt["regression_map"] = masks["regression_map"].to(self.device)

        gt = DataclassHVStorage(
            **gt,
            batch_size=gt["tissue_types"].shape[0],
            num_nuclei_classes=self.num_classes,
        )
        return gt

    def calculate_loss(
        self, predictions: DataclassHVStorage, gt: DataclassHVStorage
    ) -> torch.Tensor:
        """Calculate the loss

        Args:
            predictions (DataclassHVStorage): Predictions
            gt (DataclassHVStorage): Ground-Truth values

        Returns:
            torch.Tensor: Loss
        """
        predictions = predictions.get_dict()
        gt = gt.get_dict()

        total_loss = 0

        for branch, pred in predictions.items():
            if branch in [
                "instance_map",
                "instance_types",
                "instance_types_nuclei",
                "tissue_types", # Added: explicitly skip tissue_types if it happens to be there (it is None now)
            ]:
                continue
            if pred is None: # Added check
                continue
            if branch not in self.loss_fn_dict:
                continue
            branch_loss_fns = self.loss_fn_dict[branch]
            for loss_name, loss_setting in branch_loss_fns.items():
                loss_fn = loss_setting["loss_fn"]
                weight = loss_setting["weight"]
                if loss_name == "msge":
                    loss_value = loss_fn(
                        input=pred,
                        target=gt[branch],
                        focus=gt["nuclei_binary_map"],
                        device=self.device,
                    )
                else:
                    # print(f"mypred:{pred.shape}")
                    # print(f"mytarget:{gt[branch].shape}")
                    loss_value = loss_fn(input=pred, target=gt[branch])
                total_loss = total_loss + weight * loss_value
                self.loss_avg_tracker[f"{branch}_{loss_name}"].update(
                    loss_value.detach().cpu().numpy()
                )
        self.loss_avg_tracker["Total_Loss"].update(total_loss.detach().cpu().numpy())

        return total_loss

    def calculate_step_metric_train(
        self, predictions: DataclassHVStorage, gt: DataclassHVStorage
    ) -> dict:
        """Calculate the metrics for the training step

        Args:
            predictions (DataclassHVStorage): Processed network output
            gt (DataclassHVStorage): Ground truth values
        Returns:
            dict: Dictionary with metrics. Keys:
                binary_dice_scores, binary_jaccard_scores, tissue_pred, tissue_gt
        """
        predictions = predictions.get_dict()
        gt = gt.get_dict()

        # REMOVED: Tissue Tpyes logits to probs and argmax to get class
        
        predictions["instance_map"] = predictions["instance_map"].detach().cpu()
        predictions["instance_types_nuclei"] = (
            predictions["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )
        gt["tissue_types"] = gt["tissue_types"].detach().cpu().numpy().astype(np.uint8)
        gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=1).type(
            torch.uint8
        )
        gt["instance_types_nuclei"] = (
            gt["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )

        # REMOVED: tissue_detection_accuracy = accuracy_score(...)
        # REMOVED: self.batch_avg_tissue_acc.update(tissue_detection_accuracy)

        binary_dice_scores = []
        binary_jaccard_scores = []

        batch_size = gt["tissue_types"].shape[0]

        for i in range(batch_size):
            # binary dice score: Score for cell detection per image, without background
            pred_binary_map = torch.argmax(predictions["nuclei_binary_map"][i], dim=0)
            target_binary_map = gt["nuclei_binary_map"][i]
            cell_dice = (
                dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
                .detach()
                .cpu()
            )
            binary_dice_scores.append(float(cell_dice))

            # binary aji
            cell_jaccard = (
                binary_jaccard_index(
                    preds=pred_binary_map,
                    target=target_binary_map,
                )
                .detach()
                .cpu()
            )
            binary_jaccard_scores.append(float(cell_jaccard))

        batch_metrics = {
            "binary_dice_scores": binary_dice_scores,
            "binary_jaccard_scores": binary_jaccard_scores,
            # REMOVED: "tissue_pred": pred_tissue,
            # REMOVED: "tissue_gt": gt["tissue_types"], # Not needed for train step
        }

        return batch_metrics

    def calculate_step_metric_validation(self, predictions: dict, gt: dict) -> dict:
        """Calculate the metrics for the training step

        Args:
            predictions (DataclassHVStorage): OrderedDict: Processed network output
            gt (DataclassHVStorage): Ground truth values
        Returns:
            dict: Dictionary with metrics. Keys:
                binary_dice_scores, binary_jaccard_scores, tissue_pred, tissue_gt
        """
        predictions = predictions.get_dict()
        gt = gt.get_dict()

        # REMOVED: Tissue Tpyes logits to probs and argmax to get class

        predictions["instance_map"] = predictions["instance_map"].detach().cpu()
        predictions["instance_types_nuclei"] = (
            predictions["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )
        instance_maps_gt = gt["instance_map"].detach().cpu()
        gt["tissue_types"] = gt["tissue_types"].detach().cpu().numpy().astype(np.uint8)
        gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=1).type(
            torch.uint8
        )
        gt["instance_types_nuclei"] = (
            gt["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )

        # REMOVED: tissue_detection_accuracy = accuracy_score(...)
        # REMOVED: self.batch_avg_tissue_acc.update(tissue_detection_accuracy)

        binary_dice_scores = []
        binary_jaccard_scores = []
        cell_type_pq_scores = []
        pq_scores = []

        batch_size = gt["tissue_types"].shape[0]

        for i in range(batch_size):
            # binary dice score: Score for cell detection per image, without background
            pred_binary_map = torch.argmax(predictions["nuclei_binary_map"][i], dim=0)
            target_binary_map = gt["nuclei_binary_map"][i]
            cell_dice = (
                dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
                .detach()
                .cpu()
            )
            binary_dice_scores.append(float(cell_dice))

            # binary aji
            cell_jaccard = (
                binary_jaccard_index(
                    preds=pred_binary_map,
                    target=target_binary_map,
                )
                .detach()
                .cpu()
            )
            binary_jaccard_scores.append(float(cell_jaccard))
            # pq values
            remapped_instance_pred = remap_label(predictions["instance_map"][i])
            remapped_gt = remap_label(instance_maps_gt[i])
            [_, _, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)
            pq_scores.append(pq)

            # pq values per class (skip background)
            nuclei_type_pq = []
            for j in range(0, self.num_classes):
                pred_nuclei_instance_class = remap_label(
                    predictions["instance_types_nuclei"][i][j, ...]
                )
                target_nuclei_instance_class = remap_label(
                    gt["instance_types_nuclei"][i][j, ...]
                )

                # if ground truth is empty, skip from calculation
                if len(np.unique(target_nuclei_instance_class)) == 1:
                    pq_tmp = np.nan
                else:
                    [_, _, pq_tmp], _ = get_fast_pq(
                        pred_nuclei_instance_class,
                        target_nuclei_instance_class,
                        match_iou=0.5,
                    )
                nuclei_type_pq.append(pq_tmp)

            cell_type_pq_scores.append(nuclei_type_pq)

        batch_metrics = {
            "binary_dice_scores": binary_dice_scores,
            "binary_jaccard_scores": binary_jaccard_scores,
            "pq_scores": pq_scores,
            "cell_type_pq_scores": cell_type_pq_scores,
            # REMOVED: "tissue_pred": pred_tissue,
            "tissue_gt": gt["tissue_types"],
        }

        return batch_metrics

    @staticmethod
    def generate_example_image(
        imgs: Union[torch.Tensor, np.ndarray],
        predictions: DataclassHVStorage,
        gt: DataclassHVStorage,
        num_nuclei_classes: int,
        num_images: int = 2,
    ) -> plt.Figure:
        """Generate example plot with image, binary_pred, hv-map and instance map from prediction and ground-truth

        Args:
            imgs (Union[torch.Tensor, np.ndarray]): Images to process, a random number (num_images) is selected from this stack
                Shape: (batch_size, 3, H', W')
            predictions (DataclassHVStorage): Predictions
            gt (DataclassHVStorage): gt
            num_nuclei_classes (int): Number of total nuclei classes including background
            num_images (int, optional): Number of example patches to display. Defaults to 2.

        Returns:
            plt.Figure: Figure with example patches
        """
        predictions = predictions.get_dict()
        gt = gt.get_dict()

        assert num_images <= imgs.shape[0]
        num_images = 4

        predictions["nuclei_binary_map"] = predictions["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions["hv_map"] = predictions["hv_map"].permute(0, 2, 3, 1)
        predictions["nuclei_type_map"] = predictions["nuclei_type_map"].permute(
            0, 2, 3, 1
        )
        predictions["instance_types_nuclei"] = predictions[
            "instance_types_nuclei"
        ].transpose(0, 2, 3, 1)

        gt["hv_map"] = gt["hv_map"].permute(0, 2, 3, 1)
        gt["nuclei_type_map"] = gt["nuclei_type_map"].permute(0, 2, 3, 1)
        predictions["instance_types_nuclei"] = predictions[
            "instance_types_nuclei"
        ].transpose(0, 2, 3, 1)

        h = gt["hv_map"].shape[1]
        w = gt["hv_map"].shape[2]

        sample_indices = torch.randint(0, imgs.shape[0], (num_images,))
        # convert to rgb and crop to selection
        sample_images = (
            imgs[sample_indices].permute(0, 2, 3, 1).contiguous().cpu().numpy()
        )  # convert to rgb
        sample_images = cropping_center(sample_images, (h, w), True)

        # get predictions
        pred_sample_binary_map = (
            predictions["nuclei_binary_map"][sample_indices, :, :, 1]
            .detach()
            .cpu()
            .numpy()
        )
        pred_sample_hv_map = (
            predictions["hv_map"][sample_indices].detach().cpu().numpy()
        )
        pred_sample_instance_maps = (
            predictions["instance_map"][sample_indices].detach().cpu().numpy()
        )
        pred_sample_type_maps = (
            torch.argmax(predictions["nuclei_type_map"][sample_indices], dim=-1)
            .detach()
            .cpu()
            .numpy()
        )

        # get ground truth labels
        gt_sample_binary_map = (
            gt["nuclei_binary_map"][sample_indices].detach().cpu().numpy()
        )
        gt_sample_hv_map = gt["hv_map"][sample_indices].detach().cpu().numpy()
        gt_sample_instance_map = (
            gt["instance_map"][sample_indices].detach().cpu().numpy()
        )
        gt_sample_type_map = (
            torch.argmax(gt["nuclei_type_map"][sample_indices], dim=-1)
            .detach()
            .cpu()
            .numpy()
        )

        # create colormaps
        hv_cmap = plt.get_cmap("jet")
        binary_cmap = plt.get_cmap("jet")
        instance_map = plt.get_cmap("viridis")

        # setup plot
        fig, axs = plt.subplots(num_images, figsize=(6, 2 * num_images), dpi=150)

        for i in range(num_images):
            placeholder = np.zeros((2 * h, 6 * w, 3))
            # orig image
            placeholder[:h, :w, :3] = sample_images[i]
            placeholder[h : 2 * h, :w, :3] = sample_images[i]
            # binary prediction
            placeholder[:h, w : 2 * w, :3] = rgba2rgb(
                binary_cmap(gt_sample_binary_map[i] * 255)
            )
            placeholder[h : 2 * h, w : 2 * w, :3] = rgba2rgb(
                binary_cmap(pred_sample_binary_map[i])
            )  # *255?
            # hv maps
            placeholder[:h, 2 * w : 3 * w, :3] = rgba2rgb(
                hv_cmap((gt_sample_hv_map[i, :, :, 0] + 1) / 2)
            )
            placeholder[h : 2 * h, 2 * w : 3 * w, :3] = rgba2rgb(
                hv_cmap((pred_sample_hv_map[i, :, :, 0] + 1) / 2)
            )
            placeholder[:h, 3 * w : 4 * w, :3] = rgba2rgb(
                hv_cmap((gt_sample_hv_map[i, :, :, 1] + 1) / 2)
            )
            placeholder[h : 2 * h, 3 * w : 4 * w, :3] = rgba2rgb(
                hv_cmap((pred_sample_hv_map[i, :, :, 1] + 1) / 2)
            )
            # instance_predictions
            placeholder[:h, 4 * w : 5 * w, :3] = rgba2rgb(
                instance_map(
                    (gt_sample_instance_map[i] - np.min(gt_sample_instance_map[i]))
                    / (
                        np.max(gt_sample_instance_map[i])
                        - np.min(gt_sample_instance_map[i] + 1e-10)
                    )
                )
            )
            placeholder[h : 2 * h, 4 * w : 5 * w, :3] = rgba2rgb(
                instance_map(
                    (
                        pred_sample_instance_maps[i]
                        - np.min(pred_sample_instance_maps[i])
                    )
                    / (
                        np.max(pred_sample_instance_maps[i])
                        - np.min(pred_sample_instance_maps[i] + 1e-10)
                    )
                )
            )
            # type_predictions
            placeholder[:h, 5 * w : 6 * w, :3] = rgba2rgb(
                binary_cmap(gt_sample_type_map[i] / num_nuclei_classes)
            )
            placeholder[h : 2 * h, 5 * w : 6 * w, :3] = rgba2rgb(
                binary_cmap(pred_sample_type_maps[i] / num_nuclei_classes)
            )

            # plotting
            axs[i].imshow(placeholder)
            axs[i].set_xticks([], [])

            # plot labels in first row
            if i == 0:
                axs[i].set_xticks(np.arange(w / 2, 6 * w, w))
                axs[i].set_xticklabels(
                    [
                        "Image",
                        "Binary-Cells",
                        "HV-Map-0",
                        "HV-Map-1",
                        "Cell Instances",
                        "Nuclei-Instances",
                    ],
                    fontsize=6,
                )
                axs[i].xaxis.tick_top()

            axs[i].set_yticks(np.arange(h / 2, 2 * h, h))
            axs[i].set_yticklabels(["GT", "Pred."], fontsize=6)
            axs[i].tick_params(axis="both", which="both", length=0)
            grid_x = np.arange(w, 6 * w, w)
            grid_y = np.arange(h, 2 * h, h)

            for x_seg in grid_x:
                axs[i].axvline(x_seg, color="black")
            for y_seg in grid_y:
                axs[i].axhline(y_seg, color="black")

        fig.suptitle(f"Patch Predictions for {num_images} Examples")

        fig.tight_layout()

        return fig
    
            # ===== 新增：所有Map保存函数 =====
    
    def save_all_prediction_maps(
        self,
        imgs: torch.Tensor,
        predictions,
        gt,
        epoch: int,
        batch_idx: int,
    ) -> None:
        """
        保存所有预测Map为PNG图像
        
        Args:
            imgs: 输入图像 (B, 3, H, W)
            predictions: DataclassHVStorage 预测对象
            gt: DataclassHVStorage 真值对象
            epoch: 当前epoch
            batch_idx: 批次索引
        """
        # 创建保存目录
        save_dir = self.visualization_dir / f"epoch_{epoch}" / f"batch_{batch_idx}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 转换为字典
        preds = predictions.get_dict()
        gt_dict = gt.get_dict()
        
        # 仅保存前2个样本，节省空间
        batch_size = min(2, imgs.shape[0])
        
        for sample_idx in range(batch_size):
            self._save_single_sample_maps(
                img=imgs[sample_idx],
                predictions=preds,
                gt=gt_dict,
                sample_idx=sample_idx,
                epoch=epoch,
                batch_idx=batch_idx,
                save_dir=save_dir,
            )

    def _save_single_sample_maps(
        self,
        img: torch.Tensor,
        predictions: dict,
        gt: dict,
        sample_idx: int,
        epoch: int,
        batch_idx: int,
        save_dir: Path,
    ) -> None:
        """保存单个样本的所有Map"""
        
        # 1. 保存原始图像
        self._save_original_image(img, save_dir, sample_idx)
        
        # 2. 保存HV Map
        if "hv_map" in predictions:
            self._save_hv_map(
                predictions["hv_map"][sample_idx],
                gt["hv_map"][sample_idx] if "hv_map" in gt else None,
                save_dir,
                sample_idx,
            )
        
        # 3. 保存Binary Map
        if "nuclei_binary_map" in predictions:
            self._save_binary_map(
                predictions["nuclei_binary_map"][sample_idx],
                gt["nuclei_binary_map"][sample_idx] if "nuclei_binary_map" in gt else None,
                save_dir,
                sample_idx,
            )
        
        # 4. 保存Type Map
        if "nuclei_type_map" in predictions:
            self._save_type_map(
                predictions["nuclei_type_map"][sample_idx],
                gt["nuclei_type_map"][sample_idx] if "nuclei_type_map" in gt else None,
                save_dir,
                sample_idx,
            )
        
        # 5. 保存Instance Map
        if "instance_map" in predictions:
            self._save_instance_map(
                predictions["instance_map"][sample_idx],
                gt["instance_map"][sample_idx] if "instance_map" in gt else None,
                save_dir,
                sample_idx,
            )
        
        # 6. 保存Cosine Similarity Map (DINOv3特有)
        if "cosine_sim_map" in predictions and predictions["cosine_sim_map"] is not None:
            self._save_cosine_similarity_map(
                predictions["cosine_sim_map"][sample_idx],
                save_dir,
                sample_idx,
            )
        
        # 7. 保存Token Weights (DINOv3特有)
        if "token_weights" in predictions and predictions["token_weights"] is not None:
            self._save_token_weights(
                predictions["token_weights"][sample_idx],
                save_dir,
                sample_idx,
            )
        
        # 8. 保存综合对比图
        self._save_comparison_figure(
            img=img,
            predictions=predictions,
            gt=gt,
            sample_idx=sample_idx,
            save_dir=save_dir,
        )

    def _save_original_image(
        self,
        img: torch.Tensor,
        save_dir: Path,
        sample_idx: int,
    ) -> None:
        """保存原始输入图像"""
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        
        # 反归一化
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 0.5 + 0.5).clip(0, 1)
        
        ax.imshow(img_np)
        ax.set_title(f"Original Image (Sample {sample_idx})", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f"01_original_image_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.debug(f"Saved original image for sample {sample_idx}")

    def _save_hv_map(
        self,
        hv_map: torch.Tensor,
        gt_hv_map: torch.Tensor = None,
        save_dir: Path = None,
        sample_idx: int = 0,
    ) -> None:
        """保存HV Map (水平和竖直梯度)"""
        hv_map = hv_map.detach().cpu().numpy()  # (2, H, W)
        
        if gt_hv_map is not None:
            gt_hv_map = gt_hv_map.detach().cpu().numpy()
            num_rows = 2
        else:
            num_rows = 1
        
        fig, axes = plt.subplots(num_rows, 2, figsize=(14, 6 * num_rows), dpi=150)
        
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        # 预测的H通道
        im0 = axes[0, 0].imshow(hv_map[0], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 0].set_title(f'Predicted HV - Horizontal (Sample {sample_idx})', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0], label='Value', fraction=0.046, pad=0.04)
        
        # 预测的V通道
        im1 = axes[0, 1].imshow(hv_map[1], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 1].set_title(f'Predicted HV - Vertical (Sample {sample_idx})', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], label='Value', fraction=0.046, pad=0.04)
        
        # 真值
        if gt_hv_map is not None:
            im2 = axes[1, 0].imshow(gt_hv_map[0], cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, 0].set_title(f'GT HV - Horizontal (Sample {sample_idx})', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            plt.colorbar(im2, ax=axes[1, 0], label='Value', fraction=0.046, pad=0.04)
            
            im3 = axes[1, 1].imshow(gt_hv_map[1], cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, 1].set_title(f'GT HV - Vertical (Sample {sample_idx})', fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
            plt.colorbar(im3, ax=axes[1, 1], label='Value', fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(save_dir / f"02_hv_map_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.debug(f"Saved HV map for sample {sample_idx}")

    def _save_binary_map(
        self,
        binary_map: torch.Tensor,
        gt_binary_map: torch.Tensor = None,
        save_dir: Path = None,
        sample_idx: int = 0,
    ) -> None:
        """保存Binary Map (细胞核vs背景)"""
        # 获取概率和预测
        binary_probs = binary_map.detach().cpu().numpy()  # (2, H, W)
        binary_pred = np.argmax(binary_probs, axis=0)  # (H, W)
        
        if gt_binary_map is not None:
            gt_binary_map = gt_binary_map.detach().cpu().numpy()
            gt_pred = np.argmax(gt_binary_map, axis=0)
            num_rows = 2
        else:
            num_rows = 1
        
        fig, axes = plt.subplots(num_rows, 2, figsize=(14, 6 * num_rows), dpi=150)
        
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        # 预测概率
        im0 = axes[0, 0].imshow(binary_probs[1], cmap='viridis', vmin=0, vmax=1)
        axes[0, 0].set_title(f'Predicted Nuclei Probability (Sample {sample_idx})', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0], label='Probability', fraction=0.046, pad=0.04)
        
        # 预测分类
        im1 = axes[0, 1].imshow(binary_pred, cmap='gray')
        axes[0, 1].set_title(f'Predicted Binary (0=bg, 1=nuclei) (Sample {sample_idx})', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], label='Class', fraction=0.046, pad=0.04)
        
        # 真值
        if gt_binary_map is not None:
            im2 = axes[1, 0].imshow(gt_binary_map[1], cmap='viridis', vmin=0, vmax=1)
            axes[1, 0].set_title(f'GT Nuclei Probability (Sample {sample_idx})', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            plt.colorbar(im2, ax=axes[1, 0], label='Probability', fraction=0.046, pad=0.04)
            
            im3 = axes[1, 1].imshow(gt_pred, cmap='gray')
            axes[1, 1].set_title(f'GT Binary (Sample {sample_idx})', fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
            plt.colorbar(im3, ax=axes[1, 1], label='Class', fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(save_dir / f"03_binary_map_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.debug(f"Saved binary map for sample {sample_idx}")

    def _save_type_map(
        self,
        type_map: torch.Tensor,
        gt_type_map: torch.Tensor = None,
        save_dir: Path = None,
        sample_idx: int = 0,
    ) -> None:
        """保存Type Map (细胞核分类)"""
        type_probs = type_map.detach().cpu().numpy()  # (C, H, W)
        type_pred = np.argmax(type_probs, axis=0)  # (H, W)
        
        if gt_type_map is not None:
            gt_type_map = gt_type_map.detach().cpu().numpy()
            gt_pred = np.argmax(gt_type_map, axis=0)
            num_rows = 2
        else:
            num_rows = 1
        
        fig, axes = plt.subplots(num_rows, 1, figsize=(10, 8 * num_rows), dpi=150)
        
        if num_rows == 1:
            axes = [axes]
        
        # 预测
        im0 = axes[0].imshow(type_pred, cmap='tab20', vmin=0, vmax=self.num_classes-1)
        axes[0].set_title(f'Predicted Nuclei Types (Sample {sample_idx})', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        cbar0 = plt.colorbar(im0, ax=axes[0], label='Nuclei Type', fraction=0.046, pad=0.04)
        cbar0.set_ticks(range(self.num_classes))
        
        # 真值
        if gt_type_map is not None:
            im1 = axes[1].imshow(gt_pred, cmap='tab20', vmin=0, vmax=self.num_classes-1)
            axes[1].set_title(f'GT Nuclei Types (Sample {sample_idx})', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            cbar1 = plt.colorbar(im1, ax=axes[1], label='Nuclei Type', fraction=0.046, pad=0.04)
            cbar1.set_ticks(range(self.num_classes))
        
        plt.tight_layout()
        plt.savefig(save_dir / f"04_type_map_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.debug(f"Saved type map for sample {sample_idx}")

    def _save_instance_map(
        self,
        instance_map: torch.Tensor,
        gt_instance_map: torch.Tensor = None,
        save_dir: Path = None,
        sample_idx: int = 0,
    ) -> None:
        """保存Instance Map (实例分割结果)"""
        instance_map = instance_map.detach().cpu().numpy()  # (H, W)
        
        if gt_instance_map is not None:
            gt_instance_map = gt_instance_map.detach().cpu().numpy()
            num_rows = 2
        else:
            num_rows = 1
        
        fig, axes = plt.subplots(num_rows, 1, figsize=(10, 8 * num_rows), dpi=150)
        
        if num_rows == 1:
            axes = [axes]
        
        # 预测
        pred_normalized = (instance_map - np.min(instance_map)) / (
            np.max(instance_map) - np.min(instance_map) + 1e-10
        )
        im0 = axes[0].imshow(pred_normalized, cmap='nipy_spectral')
        axes[0].set_title(
            f'Predicted Instance Map ({int(np.max(instance_map))} instances) (Sample {sample_idx})',
            fontsize=12, fontweight='bold'
        )
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], label='Instance ID', fraction=0.046, pad=0.04)
        
        # 真值
        if gt_instance_map is not None:
            gt_normalized = (gt_instance_map - np.min(gt_instance_map)) / (
                np.max(gt_instance_map) - np.min(gt_instance_map) + 1e-10
            )
            im1 = axes[1].imshow(gt_normalized, cmap='nipy_spectral')
            axes[1].set_title(
                f'GT Instance Map ({int(np.max(gt_instance_map))} instances) (Sample {sample_idx})',
                fontsize=12, fontweight='bold'
            )
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], label='Instance ID', fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(save_dir / f"05_instance_map_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.debug(f"Saved instance map for sample {sample_idx}")

    def _save_cosine_similarity_map(
        self,
        sim_map: torch.Tensor,
        save_dir: Path = None,
        sample_idx: int = 0,
    ) -> None:
        """保存Cosine Similarity Map (Token Gating的注意力热力图)"""
        sim_map = sim_map.detach().cpu().numpy()
        
        # 处理形状
        if sim_map.ndim == 3:
            sim_map = sim_map[0]  # (H, W)
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        
        im = ax.imshow(sim_map, cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'Cosine Similarity Map (Token Gating Attention) (Sample {sample_idx})', 
                     fontsize=12, fontweight='bold')
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, label='Similarity Score', fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(save_dir / f"06_cosine_similarity_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.debug(f"Saved cosine similarity map for sample {sample_idx}")

    def _save_token_weights(
        self,
        token_weights: torch.Tensor,
        save_dir: Path = None,
        sample_idx: int = 0,
    ) -> None:
        """保存Token Weights (Token Gating的权重分布)"""
        token_weights = token_weights.detach().cpu().numpy()
        
        # 处理形状
        if token_weights.ndim == 2:
            token_weights = token_weights.squeeze()  # (N,)
        
        # 尝试reshape为空间维度
        N = token_weights.shape[0]
        H_patch = W_patch = int(np.sqrt(N))
        
        if H_patch * W_patch == N:
            weight_map = token_weights.reshape(H_patch, W_patch)
            
            fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
            im = ax.imshow(weight_map, cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f'Token Gating Weights (Spatial View) (Sample {sample_idx})', 
                         fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, label='Gate Weight', fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(save_dir / f"07_token_weights_spatial_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # 直方图视图
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.hist(token_weights, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Gate Weight', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Token Gating Weights Distribution (Sample {sample_idx})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f"07_token_weights_histogram_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.debug(f"Saved token weights for sample {sample_idx}")

    def _save_comparison_figure(
        self,
        img: torch.Tensor,
        predictions: dict,
        gt: dict,
        sample_idx: int,
        save_dir: Path,
    ) -> None:
        """保存综合对比图 (所有map在一张图上)"""
        # 反归一化图像
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 0.5 + 0.5).clip(0, 1)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=150)
        
        # 第一行：原始数据和基础map
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title(f"Original Image (Sample {sample_idx})", fontweight='bold')
        axes[0, 0].axis('off')
        
        # HV水平
        if "hv_map" in predictions:
            hv_h = predictions["hv_map"][sample_idx][0].detach().cpu().numpy()
            im = axes[0, 1].imshow(hv_h, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[0, 1].set_title(f"HV - Horizontal (Sample {sample_idx})", fontweight='bold')
            axes[0, 1].axis('off')
            plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # HV竖直
        if "hv_map" in predictions:
            hv_v = predictions["hv_map"][sample_idx][1].detach().cpu().numpy()
            im = axes[0, 2].imshow(hv_v, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[0, 2].set_title(f"HV - Vertical (Sample {sample_idx})", fontweight='bold')
            axes[0, 2].axis('off')
            plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # Binary预测
        if "nuclei_binary_map" in predictions:
            binary_pred = torch.argmax(predictions["nuclei_binary_map"][sample_idx], dim=0).cpu().numpy()
            im = axes[0, 3].imshow(binary_pred, cmap='gray')
            axes[0, 3].set_title(f"Binary Prediction (Sample {sample_idx})", fontweight='bold')
            axes[0, 3].axis('off')
            plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)
        
        # 第二行：真值和其他预测
        if "nuclei_binary_map" in gt:
            gt_binary = torch.argmax(gt["nuclei_binary_map"][sample_idx], dim=0).cpu().numpy()
            im = axes[1, 0].imshow(gt_binary, cmap='gray')
            axes[1, 0].set_title(f"GT Binary (Sample {sample_idx})", fontweight='bold')
            axes[1, 0].axis('off')
            plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Type预测
        if "nuclei_type_map" in predictions:
            type_pred = torch.argmax(predictions["nuclei_type_map"][sample_idx], dim=0).cpu().numpy()
            im = axes[1, 1].imshow(type_pred, cmap='tab20', vmin=0, vmax=self.num_classes-1)
            axes[1, 1].set_title(f"Type Prediction (Sample {sample_idx})", fontweight='bold')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Instance预测
        if "instance_map" in predictions:
            inst_pred = predictions["instance_map"][sample_idx].cpu().numpy()
            inst_norm = (inst_pred - np.min(inst_pred)) / (np.max(inst_pred) - np.min(inst_pred) + 1e-10)
            im = axes[1, 2].imshow(inst_norm, cmap='nipy_spectral')
            axes[1, 2].set_title(f"Instance Prediction (Sample {sample_idx})", fontweight='bold')
            axes[1, 2].axis('off')
            plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        # Cosine相似度
        if "cosine_sim_map" in predictions and predictions["cosine_sim_map"] is not None:
            sim = predictions["cosine_sim_map"][sample_idx][0].cpu().numpy()
            im = axes[1, 3].imshow(sim, cmap='hot', vmin=0, vmax=1)
            axes[1, 3].set_title(f"Cosine Similarity (Sample {sample_idx})", fontweight='bold')
            axes[1, 3].axis('off')
            plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(save_dir / f"09_comparison_overview_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.debug(f"Saved comparison figure for sample {sample_idx}")