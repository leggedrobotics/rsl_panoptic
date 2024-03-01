# Simin Fei:
# Add PQ Evaluator, such that we are using the same evaluation script for Panoptic-DeepLab model and DETR model.
# The PQ Evaluator is using the panopticapi for computing pq.

import os
import io
import json
from PIL import Image
import cv2
import numpy as np

try:
    from panopticapi.evaluation import pq_compute
except ImportError:
    pass


class PQEvaluator(object):
    def __init__(self, ann_file, ann_folder, output_dir="panoptic_train"):
        self.gt_json = ann_file
        self.gt_folder = ann_folder
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir
        self.predictions = []

    def get_segments_info(
        self,
        panoptic_pred,
        thing_list=list(np.arange(1, 8)) + [26, 29, 30, 31],
        label_divisor=256,
    ):
        segments_info = []
        ids = np.unique(panoptic_pred)
        for idx in ids:
            category_id = idx // label_divisor
            isthing = category_id in thing_list
            selected_pixels = panoptic_pred == idx
            area = np.sum(selected_pixels)
            segment = {
                "id": int(idx),
                "isthing": isthing,
                "category_id": int(category_id),
                "area": int(area),
            }
            segments_info.append(segment)
        return segments_info

    def update(self, panoptic_pred, file_name, image_id):
        """
        Get panoptic prediction (RGB) and save it.
        Get panoptic result dict and update self.predictions.
        args:
            result: panoptic postprocessor result.
            file_name: image file name in annotation file.
            image_id: image id in annotation file.
        """
        segments_info = self.get_segments_info(panoptic_pred)
        # get panoptic pred RGB
        panoptic_pred_color = np.zeros(
            (panoptic_pred.shape[0], panoptic_pred.shape[1], 3), dtype=np.uint8
        )
        panoptic_pred_color[..., 0] = panoptic_pred % 256
        panoptic_pred_color[..., 1] = panoptic_pred // 256 % 256
        panoptic_pred_color[..., 2] = panoptic_pred // 256 // 256

        cv2.imwrite(
            os.path.join(self.output_dir, file_name.replace(".jpg", ".png")),
            panoptic_pred_color[..., ::-1],
        )

        res_pan = {}
        res_pan["image_id"] = image_id
        res_pan["file_name"] = file_name.replace(".jpg", ".png")
        res_pan["segments_info"] = segments_info
        self.predictions.append(res_pan)

    def summarize(self):
        json_data = {"annotations": self.predictions}
        predictions_json = os.path.join(self.output_dir, "predictions.json")
        with open(predictions_json, "w") as f:
            json.dump(json_data, f)
        return pq_compute(
            self.gt_json,
            predictions_json,
            gt_folder=self.gt_folder,
            pred_folder=self.output_dir,
        )
