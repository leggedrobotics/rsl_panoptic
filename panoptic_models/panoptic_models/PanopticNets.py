import torch
import torchvision.transforms as T

import time
import numpy as np
import os

# Imports for DETR
import panoptic_models.detr as detr
import panoptic_models.detr.configs.config
import panoptic_models.detr.models

# imports for Panoptic DeepLab
import tensorflow as tf
from panoptic_models.deeplab2 import config_pb2
from panoptic_models.deeplab2.model import utils
from panoptic_models.deeplab2.data import dataset
from panoptic_models.deeplab2.model import deeplab
from panoptic_models.deeplab2.utils import vis_coco_reduced
from google.protobuf import text_format


class PanopticNet(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x, vis=False):
        raise NotImplementedError


class Mask2Former(PanopticNet):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        # from detectron2.engine.defaults import DefaultPredictor
        # self.predictor = DefaultPredictor(self.config)
        from panoptic_models.mask2former.third_party.Mask2Former.demo.predictor import (
            VisualizationDemo,
        )

        self.predictor = VisualizationDemo(self.config)

        print("Mask2Former is initialized.")

    def __call__(self, image: np.ndarray, vis_seg=False, vis_label=False):
        """
        args:
            image: BGR image.
        returns:
            image_output: panoptic segmentation visualization, BGR
            panoptic_map: panoptic mask, BGR
        """
        start = time.time()
        # image = cv2.imread("/home/nubertj/Downloads/Construction-resized-300x180.jpg")
        if vis_seg:
            predictions, visualized_output = self.predictor.run_on_image(image)
        else:
            predictions = self.predictor.predictor(image)  # get raw predictions
        panoptic_seg, seg_infos = predictions["panoptic_seg"]
        segments = torch.zeros(panoptic_seg.shape).cuda()
        for sinfo in seg_infos:
            segments[panoptic_seg == sinfo["id"]] = sinfo["category_id"] + 1
        panoptic_mask = (
            255 * torch.ones((panoptic_seg.shape[0], panoptic_seg.shape[1], 3)).cuda()
        )
        panoptic_mask[..., 2] = segments
        if vis_seg:
            print("Pred. + vis. time: {:.3f}s".format(time.time() - start))
            return visualized_output, np.rot90(panoptic_mask.cpu().numpy(), k=3).astype(
                np.uint8
            )
        else:
            print("Pred. + vis. time: {:.3f}s".format(time.time() - start))
            return np.rot90(panoptic_mask.cpu().numpy(), k=3).astype(np.uint8)


class DETR(PanopticNet):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        if torch.cuda.is_available():
            self.device = "cuda"
            self.config.device = "cuda"
        else:
            self.device = "cpu"
            self.config.device = "cpu"
        self.transform = self._build_norm_transform()
        self.model, self.postprocessor = self._load_model_and_postprocessor()

    @staticmethod
    def _build_norm_transform():
        # standard PyTorch mean-std input image normalization
        transform = T.Compose(
            [T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        return transform

    def _load_model_and_postprocessor(self):
        """
        args:
            args: args for build_model
        returns:
            model: model in eval mode.
            postprocessor: postprocessor for panoptic segmentation.
        """
        model, _, postprocessors = detr.models.build_model(self.config)
        if self.config.frozen_weights is not None:
            checkpoint = torch.load(self.config.frozen_weights, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
        else:
            raise ValueError("frozen weights must be given in test mode!")
        model.eval()

        postprocessor = postprocessors["panoptic"]
        postprocessor.threshold = self.config.threshold
        model.to(self.device)
        return model, postprocessor

    def __call__(self, image: np.ndarray, vis_seg=False, vis_label=False):
        """
        args:
            image: BGR image.
        returns:
            image_output: panoptic segmentation visualization, BGR
            panoptic_map: panoptic mask, BGR
        """
        # preprocess
        start = time.time()
        image = image[..., ::-1].copy()
        image_orig = torch.from_numpy(image).to(self.device)
        image_orig = image_orig.permute(2, 0, 1)
        orig_shape = image_orig.shape[1:]
        image_orig = image_orig.unsqueeze(0)
        image_transformed = T.Resize(800)(image_orig) / 255.0
        image_tensor = self.transform(image_transformed)
        # print("preprocess time: {:.3f}s".format(time.time()-start))

        # infer and postprocessing
        out = self.model(image_tensor)

        post_procecessed = self.postprocessor(
            out,
            torch.as_tensor(image_tensor.shape[-2:]).unsqueeze(0),
            torch.as_tensor(orig_shape).unsqueeze(0),
        )[0]

        panoptic_seg = post_procecessed["src"]
        if vis_seg:
            panoptic_id = post_procecessed["panoptic_id"].cpu().numpy()
            segments_info = post_procecessed["segments_info"]
            panoptic_pred, segments_info = panoptic_models.config.create_panoptic_label(
                panoptic_id, segments_info
            )
            if vis_label:
                image_output, panoptic_map = vis_coco_reduced.vis_panoptic_seg(
                    image, panoptic_pred
                )
            else:
                image_output, panoptic_map = vis_coco_reduced.vis_seg(
                    image, panoptic_pred
                )
            print("total time: {:.3f}s".format(time.time() - start))
            return image_output[..., ::-1].astype(np.uint8), np.rot90(
                panoptic_map[..., ::-1], k=3
            ).astype(np.uint8)
        else:
            panoptic_mask = 255 * torch.ones(
                (panoptic_seg.shape[0], panoptic_seg.shape[1], 3)
            )
            panoptic_mask[..., 2] = panoptic_seg
            print("total time: {:.3f}s".format(time.time() - start))
            return np.rot90(panoptic_mask.cpu().numpy(), k=3).astype(np.uint8)
