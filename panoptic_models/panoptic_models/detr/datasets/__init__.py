# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision
import panoptic_models.detr.datasets.build_dataset as build_dataset
from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build(image_set, args):
    datasets = [dataset for dataset in args.dataset_file.split(",")]
    num_samples_list = [int(s) for s in args.num_samples.split(",")]
    return build_dataset.build(
        image_set, datasets, masks=args.masks, num_samples_list=num_samples_list
    )
