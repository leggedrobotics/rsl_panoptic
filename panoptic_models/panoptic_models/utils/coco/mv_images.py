# import packages
import shutil
import json
import os

# import parameters
from panoptic_models.panoptic_models.config.params_data import (
    COCO_SRC,
    COCO_DST,
    DATASET_SPLIT,
    OUTPUT_PATH_META,
)

# local paramters
_SRC_COCO_IMAGES = COCO_SRC + DATASET_SPLIT + "2017"
_DST_COCO_IMAGES = COCO_DST + DATASET_SPLIT + "2017/"

_SRC_ANNOTATIONS = COCO_SRC + f"panoptic_{DATASET_SPLIT}2017/"
_DST_ANNOTATIONS = COCO_DST + f"panoptic_{DATASET_SPLIT}2017/"


if __name__ == "__main__":
    """
    Script to copy images and annotations to the dataset_labeled directory where it can be combined with the segment_ai
    dataset
    """
    with open(OUTPUT_PATH_META, "r") as f:
        meta = json.load(f)

    # check whether destination directories exists, if not create them
    os.makedirs(_DST_COCO_IMAGES, exist_ok=True)
    os.makedirs(_DST_ANNOTATIONS, exist_ok=True)

    # move images and annotations
    images = meta["images"]
    anns = meta["annotations"]

    for image in images:
        file_name = image["file_name"]
        shutil.copyfile(
            os.path.join(_SRC_COCO_IMAGES, file_name),
            os.path.join(_DST_COCO_IMAGES, file_name),
        )

    for ann in anns:
        file_name = ann["file_name"]
        shutil.copyfile(
            os.path.join(_SRC_ANNOTATIONS, file_name),
            os.path.join(_DST_ANNOTATIONS, file_name),
        )
