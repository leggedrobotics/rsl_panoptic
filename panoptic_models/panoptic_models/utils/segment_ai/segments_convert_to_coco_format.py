"""
1. Convert annotation images downloaded from segments.ai into coco format (according to the category ids and thing
    list customely defined).
2. Convert the segments info downloaded from segments.ai into coco format.
3. Split into train, val and test set
"""

# import packages
import json
import cv2
import os
import numpy as np
import collections
from sklearn.model_selection import train_test_split
from copy import deepcopy
from tqdm import tqdm
from panopticapi.utils import id2rgb

# import scripts
from panoptic_models.mask2former.utils.logger import _logger
from panoptic_models.mask2former.utils.logger import parse_arguments, logger_level

# import parameters
from panoptic_models.config.params_data import (
    SEGMENTS_INFO_PATH,
    ANNOTATION_DIR,
    SEGMENTS_VAL_SIZE,
    SEGMENTS_TEST_SIZE,
    SEGMENTS_DST,
    RANDOM_SEED,
    IMAGE_PATH,
    SEGMENTS_MAX_SAMPLES,
)
from panoptic_models.config.labels_info import _CATEGORIES, unknown_id

# fixed parameters
_HEIGHT = 1376
_WIDTH = 1152
_IGNORE_LABEL = 0
_PANOPTIC_LABEL_DIVISOR = 256

# DST paths
_DST_TRAIN = SEGMENTS_DST + "/train/"
_DST_VAL = SEGMENTS_DST + "/val/"
_DST_TEST = SEGMENTS_DST + "/test/"
_DST_PAN_TRAIN = SEGMENTS_DST + "/panoptic_train/"
_DST_PAN_VAL = SEGMENTS_DST + "/panoptic_val/"
_DST_PAN_TEST = SEGMENTS_DST + "/panoptic_test/"
_DST_ANNO = SEGMENTS_DST + "/annotations/"


def _mask_to_bbox(mask):
    """
    Convert binary mask to bbox [x_min, y_min, h, w]
    args:
        mask: binary mask
    returns:
        bbox: xyhw
    """
    xs, ys = np.where(mask == True)
    x_min = int(np.min(xs))
    y_min = int(np.min(ys))
    x_max = int(np.max(xs))
    y_max = int(np.max(ys))
    return [x_min, y_min, (x_max - x_min), (y_max - y_min)]


def create_panoptic_label(annotation, segments):
    """
    Convert raw annotation image downloaded from segments.ai to coco format.
    Convert raw segments info downloaded from segments.ai to coco format.
    args:
        annotation: raw annotation image downloaded from segments.ai
        segments: raw segments info downloaded from segments.ai
    returns:
        panoptic_label_color: annotation image in coco format.
        meta_segments: segments info in coco format.
    """
    meta_segments = []
    semantic_label = np.ones(annotation.shape) * _IGNORE_LABEL
    instance_label = np.zeros(annotation.shape)
    instance_count = collections.defaultdict(int)
    for i, segment in enumerate(segments):
        seg_id = segment["id"]
        category_id = segment["category_id"]
        if category_id == 0:
            category_id = unknown_id
        selected_pixels = annotation == seg_id
        semantic_label[selected_pixels] = category_id

        instance_count[category_id] += 1
        if instance_count[category_id] >= _PANOPTIC_LABEL_DIVISOR:
            raise ValueError(
                "Too many instances for category %d in this image." % category_id
            )
        instance_label[selected_pixels] = instance_count[category_id]
        meta_segments_id = (
            category_id * _PANOPTIC_LABEL_DIVISOR + instance_count[category_id]
        )

        meta_segment = {
            "id": int(meta_segments_id),
            "category_id": int(category_id),
            "iscrowd": 0,
            "bbox": _mask_to_bbox(selected_pixels),
            "area": int(np.sum(selected_pixels)),
        }
        meta_segments.append(meta_segment)
    panoptic_label = semantic_label * _PANOPTIC_LABEL_DIVISOR + instance_label
    panoptic_label_color = id2rgb(panoptic_label)
    return panoptic_label_color, meta_segments


def remove_color(id_dict: dict) -> dict:
    del id_dict["color"]
    return id_dict


def save_meta(
    segments_info: dict,
    meta_images: list,
    meta_annotations: list,
    idx_list: list,
    dataset_name: str,
) -> None:
    meta = {
        "info": {"name": segments_info["name"], "version": segments_info["version"]},
        "images": [
            meta_image for meta_image in meta_images if meta_image["id"] in idx_list
        ],
        "annotations": [
            meta_annotation
            for meta_annotation in meta_annotations
            if meta_annotation["image_id"] in idx_list
        ],
        "categories": [remove_color(dict_id) for dict_id in deepcopy(_CATEGORIES)],
    }
    with open(os.path.join(_DST_ANNO, f"panoptic_{dataset_name}.json"), "w") as f:
        json.dump(meta, f)


def main():
    # open segmentation information
    assert os.path.isfile(
        SEGMENTS_INFO_PATH
    ), f"No file found under {SEGMENTS_INFO_PATH}"
    with open(SEGMENTS_INFO_PATH, "r") as f:
        segments_info = json.load(f)
    annotations_info = segments_info["annotations"]
    nb_images = len(annotations_info)

    # create output folders and determine which elements are splitted
    os.makedirs(_DST_TRAIN, exist_ok=True)
    os.makedirs(_DST_PAN_TRAIN, exist_ok=True)
    os.makedirs(_DST_ANNO, exist_ok=True)
    train_idx = np.arange(1, nb_images + 1, 1)
    if SEGMENTS_MAX_SAMPLES:
        train_idx = np.random.choice(
            train_idx, size=SEGMENTS_MAX_SAMPLES, replace=False
        )
    if SEGMENTS_VAL_SIZE > 0:
        os.makedirs(_DST_VAL, exist_ok=True)
        os.makedirs(_DST_PAN_VAL, exist_ok=True)
        train_idx, val_idx = train_test_split(
            train_idx, test_size=SEGMENTS_VAL_SIZE, random_state=RANDOM_SEED
        )
        _logger.info(f"Total of {len(val_idx)} samples split for validation!")
    if SEGMENTS_TEST_SIZE > 0:
        os.makedirs(_DST_TEST, exist_ok=True)
        os.makedirs(_DST_PAN_TEST, exist_ok=True)
        train_idx, test_idx = train_test_split(
            train_idx, test_size=SEGMENTS_TEST_SIZE, random_state=RANDOM_SEED
        )
        _logger.info(f"Total of {len(test_idx)} samples split for testing!")

    # determine image information
    meta_images = []
    meta_annotations = []

    for idx, ann_info in enumerate(tqdm(annotations_info, desc="Images Converted")):
        # process image meta information
        image_file = ann_info["image_file"]
        image_info = {
            "file_name": image_file,
            "height": int(_HEIGHT),
            "width": int(_WIDTH),
            "id": int(image_file[:-4]),
        }
        meta_images.append(image_info)

        # process image annotations
        segments = ann_info["segments"]
        annotation = cv2.imread(os.path.join(ANNOTATION_DIR, image_file))[..., -1]
        panoptic_label_color, meta_segments = create_panoptic_label(
            annotation, segments
        )

        annotation_info = {
            "segments_info": meta_segments,
            "file_name": image_file,
            "image_id": int(image_file[:-4]),
        }
        meta_annotations.append(annotation_info)

        # save annotation and move image
        if image_info["id"] in train_idx:
            path_image = _DST_TRAIN
            path_anno = _DST_PAN_TRAIN
            move_image = True
        elif image_info["id"] in val_idx:
            path_image = _DST_VAL
            path_anno = _DST_PAN_VAL
            move_image = True
        elif image_info["id"] in test_idx:
            path_image = _DST_TEST
            path_anno = _DST_PAN_TEST
            move_image = True
        else:
            move_image = False

        if move_image:
            os.rename(
                os.path.join(IMAGE_PATH, image_file[:-4] + ".jpg"),
                os.path.join(path_image, image_file[:-4] + ".jpg"),
            )
            cv2.imwrite(
                os.path.join(path_anno, image_file), panoptic_label_color[..., ::-1]
            )

    save_meta(
        segments_info, meta_images, meta_annotations, train_idx, dataset_name="train"
    )
    if SEGMENTS_VAL_SIZE > 0:
        save_meta(
            segments_info, meta_images, meta_annotations, val_idx, dataset_name="val"
        )
    if SEGMENTS_TEST_SIZE > 0:
        save_meta(
            segments_info, meta_images, meta_annotations, test_idx, dataset_name="test"
        )


if __name__ == "__main__":
    args = parse_arguments().parse_args()
    logger_level(argument=args)
    main()
