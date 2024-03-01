# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.datasets
from PIL import Image
from typing import List
from panopticapi.utils import rgb2id
from panoptic_models.detr.util.box_ops import masks_to_boxes
import panoptic_models
import panoptic_models.detr.datasets.transforms as T

# from .coco import make_coco_transforms
from tqdm import tqdm
import os
from multiprocessing import Pool
from functools import partial

# from panoptic_models.detr.datasets.construction_site import make_transforms
import panoptic_models.detr.datasets.transforms as T
import panoptic_models.detr.datasets.construction_site as construction_site


class MultiPanopticRAM(torch.utils.data.Dataset):
    def __init__(
        self,
        img_folders: List[str],
        ann_folders: List[str],
        ann_files: List[str],
        datasets_name: [List[str]],
        image_set: [str],
        return_masks=True,
        num_samples_list: List[int] = None,
    ):
        self.datasets_name = datasets_name
        ann_files_loaded = []
        # extract dataset name from img_folder path, it's the second to last folder in the path
        # make transforms
        self.transforms = {}
        for idx, name in enumerate(datasets_name):
            self.transforms[name] = T.make_transforms(name, image_set)
        # parent directory of img_folders and ann_folders
        # datasets_dir = [Path(img_folder).parent for img_folder in img_folders]
        for ann_file in ann_files:
            with open(ann_file, "r") as f:
                ann_files_loaded.append(json.load(f))

        self.img_folders = img_folders
        self.ann_folders = ann_folders
        self.ann_files = ann_files_loaded
        self.return_masks = return_masks
        self.imgs = []
        self.targets = []
        # if pickles files already exist inside the dataset directory, load them
        # otherwise, create pickles files and save them
        # if num samples is not specified, use all samples
        if num_samples_list is None:
            self.num_samples_list = [0 for ann_file in ann_files_loaded]
        else:
            print("Loading only {} samples".format(num_samples_list))
            self.num_samples_list = num_samples_list

        # iterate over len(self.coco["annotations"]) with tqdm to show progress
        for ann_file, ann_folder, img_folder, num_samples, dataset_name in zip(
            self.ann_files,
            self.ann_folders,
            self.img_folders,
            self.num_samples_list,
            self.datasets_name,
        ):
            print("Loading {}".format(dataset_name))
            try:
                ann_file["images"] = sorted(ann_file["images"], key=lambda x: x["id"])
            except KeyError:
                ann_file["images"] = sorted(
                    ann_file["images"], key=lambda x: x["image_id"]
                )
            # sanity check
            if "annotations" in ann_file:
                for img, ann in zip(ann_file["images"], ann_file["annotations"]):
                    assert img["file_name"][:-4] == ann["file_name"][:-4]
            # use multiprocessing to speed up loading
            # example using a single core
            if num_samples == 0:
                num_samples = len(ann_file["images"])
            for i in tqdm(range(num_samples)):
                self.preprocess(
                    ann_file["annotations"][i], ann_folder, img_folder, dataset_name
                )

    def preprocess(self, ann, ann_folder, img_folder, dataset_name):
        ann_info = ann
        print(img_folder)
        img_path = Path(img_folder) / ann_info["file_name"].replace(".png", ".jpg")
        ann_path = Path(ann_folder) / ann_info["file_name"]

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

        # if self.transforms is not None:
        #     img, target = self.transforms[dataset_name](img, target)
        self.imgs.append(img)
        self.targets.append(target)

    def __getitem__(self, idx):
        # get dataset name based on the idx and the list of num_samples for each dataset
        dataset_name = self.datasets_name[0]
        local_idx = idx
        for i, num_samples in enumerate(self.num_samples_list):
            if local_idx < num_samples:
                dataset_name = self.datasets_name[i]
                break
            else:
                local_idx -= num_samples
        img, target = self.transforms[dataset_name](self.imgs[idx], self.targets[idx])
        return img, target

    def __len__(self):
        return len(self.imgs)


import panoptic_models.detr.datasets.coco as coco


class MultiPanoptic(torch.utils.data.Dataset):
    def __init__(
        self,
        img_folders: List[str],
        ann_folders: List[str],
        ann_files: List[str],
        datasets_name: [List[str]],
        image_set: [str],
        return_masks=True,
        num_samples_list: List[int] = None,
    ):
        self.datasets_name = datasets_name
        ann_files_loaded = []
        # extract dataset name from img_folder path, it's the second to last folder in the path
        # make transforms
        self.transforms = {}
        for idx, name in enumerate(datasets_name):
            self.transforms[name] = T.make_transforms(name, image_set)
        # parent directory of img_folders and ann_folders
        # datasets_dir = [Path(img_folder).parent for img_folder in img_folders]
        for ann_file in ann_files:
            with open(ann_file, "r") as f:
                ann_files_loaded.append(json.load(f))

        self.img_folders = img_folders
        self.ann_folders = ann_folders
        self.ann_files = ann_files_loaded
        self.return_masks = return_masks
        # if pickles files already exist inside the dataset directory, load them
        # otherwise, create pickles files and save them
        # if num samples contains 0, use all samples
        for i in range(len(num_samples_list)):
            if num_samples_list[i] == 0:
                num_samples_list[i] = len(ann_files_loaded[i]["images"])
        else:
            print(
                "For Pan Segmentation Loading only {} samples".format(num_samples_list)
            )
            self.num_samples_list = num_samples_list

    def preprocess(self, ann, ann_folder, img_folder, dataset_name):
        ann_info = ann
        img_path = Path(img_folder) / ann_info["file_name"].replace(".png", ".jpg")
        ann_path = Path(ann_folder) / ann_info["file_name"]

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

    def __getitem__(self, idx):
        # there are n different datasets each with a different number of samples
        # depending on the idx find the correct dataset and sample
        dataset_idx = 0
        idx_data = idx
        for ann_file, num_samples in zip(self.ann_files, self.num_samples_list):
            if idx_data < num_samples:
                break
            dataset_idx += 1
            idx_data -= num_samples
        return self.preprocess(
            self.ann_files[dataset_idx]["annotations"][idx_data],
            self.ann_folders[dataset_idx],
            self.img_folders[dataset_idx],
            self.datasets_name[dataset_idx],
        )

    def __len__(self):
        total_len = 0
        for ann_file in self.ann_files:
            total_len += len(ann_file["images"])
        return total_len


def build(image_set, datasets: List[str], masks=False, num_samples_list=None):
    """
    Build a panoptic dataset by merging multiple datasets together.
    Args:
        image_set: train, val or test
        datasets: dataset names found inside the datasets folder

    Returns:
    """
    # image set can be train, val or test
    print("Loading {}".format(datasets))
    assert image_set in [
        "train",
        "val",
        "test",
    ], f"image_set should be train, val or test, got {image_set}"
    (
        image_folders,
        image_annotated_folders,
        annotations_folders,
        instances_annotations_folders,
    ) = ([], [], [], [])
    # the datasets are located inside the datasets folder in the root of the project
    for dataset in datasets:
        # get root of the package
        dataset = f"datasets/{dataset}"
        # get root package path
        package_path = os.path.dirname(panoptic_models.__file__)
        # parent folder
        parent_folder = os.path.dirname(package_path)
        # get the path to the dataset folder
        dataset_path = parent_folder + "/" + dataset
        # check that the path exist else raise an error
        assert Path(dataset_path).exists(), f"{dataset_path} does not exist"
        # find folder names that contains above substrings
        train_folders = list(
            filter(lambda x: image_set in x, list(os.walk(dataset_path))[0][1])
        )
        images_annotated_folder = next(
            filter(lambda x: "panoptic_" + image_set in x, train_folders)
        )
        images_folder = next(filter(lambda x: "panoptic" not in x, train_folders))
        annotations_json = next(
            filter(
                lambda x: "panoptic_" + image_set in x,
                list(os.walk(dataset_path + "/annotations/"))[0][2],
            )
        )
        instance_annotation_json = next(
            filter(
                lambda x: "instances_" + image_set in x,
                list(os.walk(dataset_path + "/annotations/"))[0][2],
            )
        )
        # add the root of the dataset to the list
        image_folders.append(dataset_path + "/" + images_folder)
        image_annotated_folders.append(dataset_path + "/" + images_annotated_folder)
        annotations_folders.append(dataset_path + "/annotations/" + annotations_json)
        instances_annotations_folders.append(
            dataset_path + "/annotations/" + instance_annotation_json
        )
    # print the list of folders
    print(f"{image_set} folders: {image_folders}")
    print(f"{image_set} annotations folders: {annotations_folders}")
    print(f"{image_set} images annotated folders: {image_annotated_folders}")

    if masks:
        if len(image_folders) > 1:
            print("Merging datasets")
            dataset_object = MultiPanopticRAM(
                image_folders,
                image_annotated_folders,
                annotations_folders,
                datasets,
                image_set,
                return_masks=masks,
                num_samples_list=num_samples_list,
            )
        else:
            print("Single dataset")
            dataset_object = construction_site.Panoptic(
                image_folders[0],
                image_annotated_folders[0],
                annotations_folders[0],
                transforms=T.make_transforms(datasets[0], image_set),
                return_masks=masks,
                num_samples=num_samples_list[0],
            )
    else:
        if len(image_folders) > 1:
            print("Merging datasets")
            # dataset_object = MultiDetection(image_folders, image_annotated_folders, instances_annotations_folders, datasets, image_set,
            #                                return_masks=masks, num_samples_list=num_samples_list)
            dataset_object = MultiPanopticRAM(
                image_folders,
                image_annotated_folders,
                annotations_folders,
                datasets,
                image_set,
                return_masks=masks,
                num_samples_list=num_samples_list,
            )
        else:
            print("Single dataset")
            # TODO: fix coco.CocoDetection that is not working for the construction dataset even though is in the same format as coco
            # dataset_object = coco.CocoDetection(image_folders[0], instances_annotations_folders[0], transforms=T.make_transforms(datasets[0], image_set), return_masks=masks)
            dataset_object = construction_site.Panoptic(
                image_folders[0],
                image_annotated_folders[0],
                annotations_folders[0],
                transforms=T.make_transforms(datasets[0], image_set),
                return_masks=masks,
                num_samples=num_samples_list[0],
            )
    return dataset_object
