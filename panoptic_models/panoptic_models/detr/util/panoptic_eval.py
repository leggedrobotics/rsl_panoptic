# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import os

import numpy as np

from panoptic_models.deeplab2 import config_pb2
from panoptic_models.deeplab2.data import dataset
from panoptic_models.deeplab2.model import deeplab
from panoptic_models.deeplab2.utils import vis_coco_reduced
import cv2
import panoptic_models.detr.util.misc as utils
from panoptic_models import config
from panopticapi.utils import id2rgb
from typing import Dict

try:
    from panopticapi.evaluation import pq_compute
except ImportError:
    pass
import wandb

# class InstanceEvaluator(object):
#     """
#     This class evalutes the instance classification results.
#     The output of the model are bounding boxes and class labels in format cxcywh.
#     """
#     def __init__(self, num_classes, ignore_label, wandb_log=True):


class PanopticEvaluator(object):
    def __init__(self, ann_file, ann_folder, output_dir="panoptic_eval"):
        self.gt_json = ann_file
        self.gt_folder = ann_folder
        if utils.is_main_process():
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
        self.output_dir = output_dir
        self.predictions = []

    def update(self, predictions):
        for p in predictions:
            with open(os.path.join(self.output_dir, p["file_name"]), "wb") as f:
                f.write(p.pop("png_string"))

        self.predictions += predictions

    def synchronize_between_processes(self):
        all_predictions = utils.all_gather(self.predictions)
        merged_predictions = []
        for p in all_predictions:
            merged_predictions += p
        self.predictions = merged_predictions

    def summarize(self):
        if utils.is_main_process():
            json_data = {"annotations": self.predictions}
            predictions_json = os.path.join(self.output_dir, "predictions.json")
            with open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))
            return pq_compute(
                self.gt_json,
                predictions_json,
                gt_folder=self.gt_folder,
                pred_folder=self.output_dir,
            )
        return None


class SimplePanopticEvaluator(object):
    """
    This class should be used in conjuction with the SimplePanopticPostProcessor class
    """

    def __init__(
        self,
        ann_file,
        ann_folder,
        write_to_file=True,
        output_dir="panoptic_eval",
        wandb_log=True,
    ):
        self.gt_json = ann_file
        self.gt_folder = ann_folder
        if utils.is_main_process():
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
        self.output_dir = output_dir
        self.predictions = []
        self.write_to_file = write_to_file
        self.wandb_log = wandb_log

    def update(self, predictions: Dict, samples=None, visualize_mask=False):
        for i, p in enumerate(predictions):
            # resize the sample to the original size, which is the size of p["panoptic_id"]
            panoptic_seg = p.pop("src")
            panoptic_id = p.pop("panoptic_id")
            panoptic_id = panoptic_id.cpu().numpy()
            # seg_im = id2rgb(panoptic_id)
            # we override the id with out panoptic label
            panoptic_pred, segments_info = config.create_panoptic_label(
                panoptic_id, p["segments_info"]
            )
            p["segments_info"] = segments_info
            # we need to save the image as pgn because panoptic api will load directly the png images
            # flip color channel form rgb to bgr
            panoptic_pred_rgb = id2rgb(panoptic_pred)
            # the id2rgb assumes opposite channel order [instance, cat, 0]
            # instead we generate with create panoptic label [0, cat, instance], which has been used for the ground thruth data
            panoptic_pred_rgb = panoptic_pred_rgb[:, :, ::-1]
            # this is used by panoptic api
            if self.write_to_file:
                cv2.imwrite(
                    os.path.join(self.output_dir, p["file_name"]), panoptic_pred_rgb
                )

            # this is used to visualize it
            if visualize_mask:
                assert samples is not None
                sample = samples.tensors[i].cpu().numpy()
                self.visualize_mask(panoptic_pred, sample, p["file_name"])

        # we popped src and panoptic id, only segment_info, image_id and image_name remains which are serializable
        self.predictions += predictions

    def visualize_mask(
        self, panoptic_pred: np.ndarray, sample: np.ndarray, sample_name: str
    ):
        """
        Args:
            panoptic_pred: [H, W] array of values = cat * 256 + instance id
            samples: numpy array of shape [H, W, 3]
        """
        # this is not useful for the evaluation but it is useful for the visualization as the id2rgb generates mostly blackish images
        # resize the samples to their original size
        sample = sample.transpose((1, 2, 0))
        sample = cv2.resize(sample, (panoptic_pred.shape[1], panoptic_pred.shape[0]))

        image_output, panoptic_map = vis_coco_reduced.vis_panoptic_seg(
            sample, panoptic_pred
        )
        # save image_output on wandb
        if self.wandb_log:
            wandb.log({"image_output": wandb.Image(image_output)})
        # save the image
        # this is for visualization
        cv2.imwrite(os.path.join(self.output_dir, "vis_" + sample_name), image_output)

    def synchronize_between_processes(self):
        all_predictions = utils.all_gather(self.predictions)
        merged_predictions = []
        for p in all_predictions:
            merged_predictions += p
        self.predictions = merged_predictions

    def summarize(self):
        # filter the self.gt_json so that it contains only the images that were predicted
        if utils.is_main_process():
            json_data = {"annotations": self.predictions}
            predictions_json = os.path.join(self.output_dir, "predictions.json")
            with open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))
            print("Computing panoptic metrics...")
            return pq_compute(
                self.gt_json,
                predictions_json,
                gt_folder=self.gt_folder,
                pred_folder=self.output_dir,
            )
        return None
