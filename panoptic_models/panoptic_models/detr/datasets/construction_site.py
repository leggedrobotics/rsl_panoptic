# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from typing import List
from panopticapi.utils import rgb2id
from panoptic_models.detr.util.box_ops import masks_to_boxes
import torchvision
from tqdm import tqdm
import os
from multiprocessing import Pool
from functools import partial
import panoptic_models.detr.datasets.transforms as T


class PanopticRam:
    def __init__(
        self,
        img_folder,
        ann_folder,
        ann_file,
        transforms=None,
        return_masks=True,
        num_samples=0,
    ):
        with open(ann_file, "r") as f:
            self.coco = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.coco["images"] = sorted(self.coco["images"], key=lambda x: x["image_id"])
        # sanity check
        if "annotations" in self.coco:
            for img, ann in zip(self.coco["images"], self.coco["annotations"]):
                assert img["file_name"][:-4] == ann["file_name"][:-4]

        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.ann_file = ann_file
        self.transforms = transforms
        self.return_masks = return_masks
        self.imgs = []
        self.targets = []

        if num_samples > 0:
            # reduce size of coco['annotations'] to speed up loading
            print("Reducing size of coco['annotations'] to speed up loading")
            self.coco["annotations"] = self.coco["annotations"][:num_samples]

        # iterate over len(self.coco["annotations"]) with tqdm to show progress
        for ann in tqdm(self.coco["annotations"]):
            ann_info = ann
            img_path = Path(self.img_folder) / ann_info["file_name"].replace(
                ".png", ".jpg"
            )
            ann_path = Path(self.ann_folder) / ann_info["file_name"]

            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            if "segments_info" in ann_info:
                masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
                masks = rgb2id(masks)

                ids = np.array([ann["id"] for ann in ann_info["segments_info"]])
                masks = masks == ids[:, None, None]

                masks = torch.as_tensor(masks, dtype=torch.uint8)
                labels = torch.tensor(
                    [ann["category_id"] for ann in ann_info["segments_info"]],
                    dtype=torch.int64,
                )

            target = {}
            target["image_id"] = torch.tensor(
                [ann_info["image_id"] if "image_id" in ann_info else ann_info["id"]]
            )
            if self.return_masks:
                target["masks"] = masks
            target["labels"] = labels

            target["boxes"] = masks_to_boxes(masks)
            # convert from xyxy to xywh format
            boxes = target["boxes"]
            x, y, w_b, h_b = boxes.unbind(1)
            boxes = torch.stack((x, y, x + w_b, y + h_b), dim=1)
            target["boxes"] = boxes

            target["size"] = torch.as_tensor([int(h), int(w)])
            target["orig_size"] = torch.as_tensor([int(h), int(w)])
            if "segments_info" in ann_info:
                for name in ["iscrowd", "area"]:
                    target[name] = torch.tensor(
                        [ann[name] for ann in ann_info["segments_info"]]
                    )

            # if self.transforms is not None:
            #     img, target = self.transforms(img, target)
            self.imgs.append(img)
            self.targets.append(target)

    def __getitem__(self, idx):
        img, target = self.transforms(self.imgs[idx], self.targets[idx])
        return img, target

    def __len__(self):
        return len(self.imgs)


class Panoptic:
    def __init__(
        self,
        img_folder,
        ann_folder,
        ann_file,
        transforms=None,
        return_masks=True,
        num_samples=0,
    ):
        with open(ann_file, "r") as f:
            self.coco = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        try:
            self.coco["images"] = sorted(
                self.coco["images"], key=lambda x: x["image_id"]
            )
        except KeyError:
            self.coco["images"] = sorted(self.coco["images"], key=lambda x: x["id"])
        # sanity check
        if "annotations" in self.coco:
            for img, ann in zip(self.coco["images"], self.coco["annotations"]):
                assert img["file_name"][:-4] == ann["file_name"][:-4]

        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.ann_file = ann_file
        self.transforms = transforms
        self.return_masks = return_masks

        if num_samples > 0:
            # reduce size of coco['annotations'] to speed up loading
            print("Reducing size of coco['annotations'] to speed up loading")
            self.coco["annotations"] = self.coco["annotations"][:num_samples]

    def __getitem__(self, idx):
        ann_info = self.coco["annotations"][idx]
        img_path = Path(self.img_folder) / ann_info["file_name"].replace(".png", ".jpg")
        ann_path = Path(self.ann_folder) / ann_info["file_name"]

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if "segments_info" in ann_info:
            masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
            masks = rgb2id(masks)

            ids = np.array([ann["id"] for ann in ann_info["segments_info"]])
            masks = masks == ids[:, None, None]

            masks = torch.as_tensor(masks, dtype=torch.uint8)
            labels = torch.tensor(
                [ann["category_id"] for ann in ann_info["segments_info"]],
                dtype=torch.int64,
            )

        target = {}
        target["image_id"] = torch.tensor(
            [ann_info["image_id"] if "image_id" in ann_info else ann_info["image_id"]]
        )
        if self.return_masks:
            target["masks"] = masks
        target["labels"] = labels

        boxes = [ann["bbox"] for ann in ann_info["segments_info"]]
        # xywh to cxcywh
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0] += boxes[:, 2] / 2
        boxes[:, 1] += boxes[:, 3] / 2
        target["boxes"] = boxes

        target["size"] = torch.as_tensor([int(h), int(w)])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        if "segments_info" in ann_info:
            for name in ["iscrowd", "area"]:
                target[name] = torch.tensor(
                    [ann[name] for ann in ann_info["segments_info"]]
                )

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.coco["annotations"])


def build(image_set, args):
    img_folder_root = Path(args.dataset_panoptic_path)
    ann_folder_root = Path(args.dataset_panoptic_path)
    assert (
        img_folder_root.exists()
    ), f"provided COCO path {img_folder_root} does not exist"
    assert (
        ann_folder_root.exists()
    ), f"provided COCO path {ann_folder_root} does not exist"
    mode = "panoptic"
    PATHS = {
        "train": ("train", Path("annotations") / f"{mode}_train.json"),
        "val": ("val", Path("annotations") / f"{mode}_val.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder_path = img_folder_root / img_folder
    ann_folder = ann_folder_root / f"{mode}_{img_folder}"
    ann_file = ann_folder_root / ann_file

    dataset = Panoptic(
        img_folder_path,
        ann_folder,
        ann_file,
        transforms=T.make_construction_transforms(image_set),
        return_masks=args.masks,
        num_samples=args.num_samples,
    )

    return dataset
