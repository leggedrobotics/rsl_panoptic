# Author: Pascal Roth, Simin Fei and Lorenzo Terenzi
# Modified from https://github.com/google-research/deeplab2/blob/main/data/coco_constants.py
# File containing the meta info of reduced COCO dataset.

# import numpy as np
from typing import Sequence, Mapping, Any
import copy
from panopticapi.utils import IdGenerator
import json
import numpy as np
import collections
import matplotlib.pyplot as plt
from panoptic_models.detr.util.plot_utils import plot_bounding_boxes
import torch
from panopticapi.utils import id2rgb

# ==
# Metadata for the reduced COCO and the labeled segments.ai datasets
# ==
# fixed parameters
_HEIGHT = 1376
_WIDTH = 1152
_IGNORE_LABEL = 0
_PANOPTIC_LABEL_DIVISOR = 256
_MIN_AREA = 15


def get_coco_meta() -> Sequence[Any]:
    """Get original coco meta."""
    return copy.deepcopy(_COCO_META)


def get_reduced_class_list():
    """Get reduced class list."""
    reduced_class_list = []
    for key in _CLASS_MAPPING.keys():
        reduced_class_list.append(key)
    return reduced_class_list


def get_id_mapping() -> Mapping[int, int]:
    """Creates a dictionary mapping the original category_id into continuous ones (reduced).
    Unneeded category_id are mapped to 0.

    Returns:
        A dictionary mapping original category id to contiguous category ids (reduced).
    """
    reduced_class_list = get_reduced_class_list()
    id_mapping = {}
    for meta in _COCO_META:
        if meta["name"] in reduced_class_list:
            id_mapping[meta["id"]] = _CLASS_MAPPING[meta["name"]]["id"]
        else:
            id_mapping[meta["id"]] = unknown_id
    return id_mapping


def get_reduced_class_ids():
    """Get the original ids from _COCO_META, which are in the reduced class list."""
    reduced_class_list = get_reduced_class_list()
    reduced_class_ids = []
    for meta in _COCO_META:
        if meta["name"] in reduced_class_list:
            reduced_class_ids.append(meta["id"])
    return reduced_class_ids


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
        # check area skip if too small
        if np.sum(selected_pixels) < _MIN_AREA:
            continue
        semantic_label[selected_pixels] = category_id

        instance_count[category_id] += 1
        if instance_count[category_id] >= _PANOPTIC_LABEL_DIVISOR:
            raise ValueError(
                "Too many instances for category %d in this image." % category_id
            )
        instance_label[selected_pixels] = instance_count[category_id]
        # so you can recover semantic id by // and instance id by % _PANOPTIC_LABEL_DIVISOR
        meta_segments_id = (
            category_id * _PANOPTIC_LABEL_DIVISOR + instance_count[category_id]
        )

        meta_segment = {
            "id": int(meta_segments_id),
            "category_id": int(category_id),
            "iscrowd": 0,
            "bbox": box_cxcywh_to_xywh(_mask_to_cxcywh(selected_pixels)),
            "area": int(np.sum(selected_pixels)),
        }
        meta_segments.append(meta_segment)
    # apply the panoptic transformation to the whole image
    panoptic_label = semantic_label * _PANOPTIC_LABEL_DIVISOR + instance_label
    return panoptic_label, meta_segments


def _mask_to_cxcywh(mask):
    """
    Convert binary mask to bbox [xc, yc, h, w].
    DETR uses bbox format [xc, yc, w, h] but normalized wrt the image size.
    args:
        mask: binary mask
    returns:
        bbox: xyhw
    """
    try:
        ys, xs = np.where(mask == True)
        x_min = int(np.min(xs))
        y_min = int(np.min(ys))
        x_max = int(np.max(xs))
        y_max = int(np.max(ys))
        return [
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            (x_max - x_min),
            (y_max - y_min),
        ]
    except ValueError:
        # empty mask case
        return [0, 0, 0, 0]


def mask_to_coco_bbox(mask):
    xs, ys = np.where(mask == True)
    x_min = int(np.min(xs))
    y_min = int(np.min(ys))
    x_max = int(np.max(xs))
    y_max = int(np.max(ys))
    return [x_min, y_min, x_max, y_max]


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return b


def box_cxcywh_to_xywh(x):
    x_c, y_c, w, h = x
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), w, h]
    return b


def box_cxcywh_to_coco(x):
    x_c, y_c, w, h = x
    b = [int((x_c - 0.5 * w)), int((y_c - 0.5 * h)), w, h]
    return b


def box_cxcywh_to_coco_torch(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [int((x_c - 0.5 * w)), int((y_c - 0.5 * h)), w, h]
    return torch.stack(b, dim=1)


_CATEGORIES = [
    {
        "id": 1,
        "name": "person",
        "supercategory": "person",
        "isthing": 1,
        "color": [220, 20, 60],
    },
    {
        "id": 2,
        "name": "bicycle",
        "supercategory": "vehicle",
        "isthing": 1,
        "color": [119, 11, 32],
    },
    {
        "id": 3,
        "name": "car",
        "supercategory": "vehicle",
        "isthing": 1,
        "color": [0, 0, 142],
    },
    {
        "id": 4,
        "name": "motorcycle",
        "supercategory": "vehicle",
        "isthing": 1,
        "color": [0, 0, 230],
    },
    {
        "id": 5,
        "name": "train",
        "supercategory": "vehicle",
        "isthing": 1,
        "color": [0, 80, 100],
    },
    {
        "id": 6,
        "name": "truck",
        "supercategory": "vehicle",
        "isthing": 1,
        "color": [0, 0, 70],
    },
    {
        "id": 7,
        "name": "boat",
        "supercategory": "vehicle",
        "isthing": 1,
        "color": [0, 0, 192],
    },
    {
        "id": 8,
        "name": "bridge",
        "supercategory": "building",
        "isthing": 0,
        "color": [150, 150, 100],
    },
    {
        "id": 9,
        "name": "building-other-merged",
        "supercategory": "building",
        "isthing": 0,
        "color": [116, 112, 0],
    },
    {
        "id": 10,
        "name": "gravel",
        "supercategory": "streets",
        "isthing": 0,
        "color": [124, 74, 181],
    },
    {
        "id": 11,
        "name": "railroad",
        "supercategory": "streets",
        "isthing": 0,
        "color": [230, 150, 140],
    },
    {
        "id": 12,
        "name": "road",
        "supercategory": "streets",
        "isthing": 0,
        "color": [128, 64, 128],
    },
    {
        "id": 13,
        "name": "pavement-merged",
        "supercategory": "ground",
        "isthing": 0,
        "color": [96, 96, 96],
    },
    {
        "id": 14,
        "name": "dirt-merged",
        "supercategory": "ground",
        "isthing": 0,
        "color": [208, 229, 228],
    },
    {
        "id": 15,
        "name": "sand",
        "supercategory": "ground",
        "isthing": 0,
        "color": [254, 212, 124],
    },
    {
        "id": 16,
        "name": "snow",
        "supercategory": "ground",
        "isthing": 0,
        "color": [255, 255, 255],
    },
    {
        "id": 17,
        "name": "floor-other-merged",
        "supercategory": "ground",
        "isthing": 0,
        "color": [96, 36, 108],
    },
    {
        "id": 18,
        "name": "grass-merged",
        "supercategory": "ground",
        "isthing": 0,
        "color": [152, 251, 152],
    },
    {
        "id": 19,
        "name": "wall-other-merged",
        "supercategory": "environment",
        "isthing": 0,
        "color": [102, 102, 156],
    },
    {
        "id": 20,
        "name": "rock-merged",
        "supercategory": "environment",
        "isthing": 0,
        "color": [0, 114, 143],
    },
    {
        "id": 21,
        "name": "water-other",
        "supercategory": "environment",
        "isthing": 0,
        "color": [58, 41, 149],
    },
    {
        "id": 22,
        "name": "tree-merged",
        "supercategory": "environment",
        "isthing": 0,
        "color": [107, 142, 35],
    },
    {
        "id": 23,
        "name": "fence-merged",
        "supercategory": "environment",
        "isthing": 0,
        "color": [190, 153, 153],
    },
    {
        "id": 24,
        "name": "sky-other-merged",
        "supercategory": "environment",
        "isthing": 0,
        "color": [70, 130, 180],
    },
    {
        "id": 25,
        "name": "floor-concrete",
        "supercategory": "ground",
        "isthing": 0,
        "color": [96, 96, 108],
    },
    {
        "id": 26,
        "name": "container",
        "supercategory": "building",
        "isthing": 1,
        "color": [248, 231, 28],
    },
    {
        "id": 27,
        "name": "self-arm",
        "supercategory": "self",
        "isthing": 0,
        "color": [218, 149, 38],
    },
    {
        "id": 28,
        "name": "self-leg",
        "supercategory": "self",
        "isthing": 0,
        "color": [218, 149, 98],
    },
    {
        "id": 29,
        "name": "stone",
        "supercategory": "material",
        "isthing": 1,
        "color": [0, 114, 113],
    },
    {
        "id": 30,
        "name": "gravel-pile",
        "supercategory": "material",
        "isthing": 1,
        "color": [124, 74, 141],
    },
    {
        "id": 31,
        "name": "sand-pile",
        "supercategory": "material",
        "isthing": 1,
        "color": [254, 212, 84],
    },
    {
        "id": 32,
        "name": "construction_machine",
        "supercategory": "unknown",
        "isthing": 1,
        "color": [255, 0, 127],
    },
    {
        "id": 33,
        "name": "bucket",
        "supercategory": "vehicle",
        "isthing": 1,
        "color": [255, 84, 127],
    },
    {
        "id": 34,
        "name": "gripper",
        "supercategory": "vehicle",
        "isthing": 1,
        "color": [255, 170, 127],
    },
]

# # unknown id goes to the last category
unknown_id = 35
unknown_dict = {
    "id": 35,
    "name": "unknown",
    "supercategory": "unknown",
    "isthing": 1,
    "color": [1, 1, 1],
}
# put the unknown class at the beginning of the list
_CATEGORIES.append(unknown_dict)

# wandb format for class labels
wandb_categories = []
for cat in _CATEGORIES:
    wandb_categories.append({"id": cat["id"], "name": cat["name"]})

# construct a dictionary with the format category["id"] = category
# this is used for
categories_dict = {}
for category in _CATEGORIES:
    categories_dict[category["id"]] = category
# create an IdGenerator object
id_generator = IdGenerator(categories_dict)

# original coco meta
_COCO_META = [
    {"color": [1, 1, 1], "isthing": 1, "id": 0, "name": "unknown"},
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
    {"color": [255, 255, 128], "isthing": 0, "id": 92, "name": "banner"},
    {"color": [147, 211, 203], "isthing": 0, "id": 93, "name": "blanket"},
    {"color": [150, 100, 100], "isthing": 0, "id": 95, "name": "bridge"},
    {"color": [168, 171, 172], "isthing": 0, "id": 100, "name": "cardboard"},
    {"color": [146, 112, 198], "isthing": 0, "id": 107, "name": "counter"},
    {"color": [210, 170, 100], "isthing": 0, "id": 109, "name": "curtain"},
    {"color": [92, 136, 89], "isthing": 0, "id": 112, "name": "door-stuff"},
    {"color": [218, 88, 184], "isthing": 0, "id": 118, "name": "floor-wood"},
    {"color": [241, 129, 0], "isthing": 0, "id": 119, "name": "flower"},
    {"color": [217, 17, 255], "isthing": 0, "id": 122, "name": "fruit"},
    {"color": [124, 74, 181], "isthing": 0, "id": 125, "name": "gravel"},
    {"color": [70, 70, 70], "isthing": 0, "id": 128, "name": "house"},
    {"color": [255, 228, 255], "isthing": 0, "id": 130, "name": "light"},
    {"color": [154, 208, 0], "isthing": 0, "id": 133, "name": "mirror-stuff"},
    {"color": [193, 0, 92], "isthing": 0, "id": 138, "name": "net"},
    {"color": [76, 91, 113], "isthing": 0, "id": 141, "name": "pillow"},
    {"color": [255, 180, 195], "isthing": 0, "id": 144, "name": "platform"},
    {"color": [106, 154, 176], "isthing": 0, "id": 145, "name": "playingfield"},
    {"color": [230, 150, 140], "isthing": 0, "id": 147, "name": "railroad"},
    {"color": [60, 143, 255], "isthing": 0, "id": 148, "name": "river"},
    {"color": [128, 64, 128], "isthing": 0, "id": 149, "name": "road"},
    {"color": [92, 82, 55], "isthing": 0, "id": 151, "name": "roof"},
    {"color": [254, 212, 124], "isthing": 0, "id": 154, "name": "sand"},
    {"color": [73, 77, 174], "isthing": 0, "id": 155, "name": "sea"},
    {"color": [255, 160, 98], "isthing": 0, "id": 156, "name": "shelf"},
    {"color": [255, 255, 255], "isthing": 0, "id": 159, "name": "snow"},
    {"color": [104, 84, 109], "isthing": 0, "id": 161, "name": "stairs"},
    {"color": [169, 164, 131], "isthing": 0, "id": 166, "name": "tent"},
    {"color": [225, 199, 255], "isthing": 0, "id": 168, "name": "towel"},
    {"color": [137, 54, 74], "isthing": 0, "id": 171, "name": "wall-brick"},
    {"color": [135, 158, 223], "isthing": 0, "id": 175, "name": "wall-stone"},
    {"color": [7, 246, 231], "isthing": 0, "id": 176, "name": "wall-tile"},
    {"color": [107, 255, 200], "isthing": 0, "id": 177, "name": "wall-wood"},
    {"color": [58, 41, 149], "isthing": 0, "id": 178, "name": "water-other"},
    {"color": [183, 121, 142], "isthing": 0, "id": 180, "name": "window-blind"},
    {"color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window-other"},
    {"color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree-merged"},
    {"color": [190, 153, 153], "isthing": 0, "id": 185, "name": "fence-merged"},
    {"color": [146, 139, 141], "isthing": 0, "id": 186, "name": "ceiling-merged"},
    {"color": [70, 130, 180], "isthing": 0, "id": 187, "name": "sky-other-merged"},
    {"color": [134, 199, 156], "isthing": 0, "id": 188, "name": "cabinet-merged"},
    {"color": [209, 226, 140], "isthing": 0, "id": 189, "name": "table-merged"},
    {"color": [96, 36, 108], "isthing": 0, "id": 190, "name": "floor-other-merged"},
    {"color": [96, 96, 96], "isthing": 0, "id": 191, "name": "pavement-merged"},
    {"color": [64, 170, 64], "isthing": 0, "id": 192, "name": "mountain-merged"},
    {"color": [152, 251, 152], "isthing": 0, "id": 193, "name": "grass-merged"},
    {"color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt-merged"},
    {"color": [206, 186, 171], "isthing": 0, "id": 195, "name": "paper-merged"},
    {"color": [152, 161, 64], "isthing": 0, "id": 196, "name": "food-other-merged"},
    {"color": [116, 112, 0], "isthing": 0, "id": 197, "name": "building-other-merged"},
    {"color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock-merged"},
    {"color": [102, 102, 156], "isthing": 0, "id": 199, "name": "wall-other-merged"},
    {"color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug-merged"},
]

# class mapping: mapping original classes to reduced classes
# indoor classes and unneeded classes are removed
_CLASS_MAPPING = {
    "unknown": {"name": "unknown", "id": 0, "color": [1, 1, 1]},
    "person": {"name": "person", "id": 1, "color": [220, 20, 60]},
    "bicycle": {"name": "bicycle", "id": 2, "color": [119, 11, 32]},
    "car": {"name": "car", "id": 3, "color": [0, 0, 142]},
    "bus": {"name": "car", "id": 3, "color": [0, 0, 142]},
    "motorcycle": {"name": "motorcycle", "id": 4, "color": [0, 0, 230]},
    "train": {"name": "train", "id": 5, "color": [0, 80, 100]},
    "truck": {"name": "truck", "id": 6, "color": [0, 0, 70]},
    "boat": {"name": "boat", "id": 7, "color": [0, 0, 192]},
    "bridge": {"name": "bridge", "id": 8, "color": [150, 100, 100]},
    "door-stuff": {"name": "building-other-merged", "id": 9, "color": [116, 112, 0]},
    "house": {"name": "building-other-merged", "id": 9, "color": [116, 112, 0]},
    "roof": {"name": "building-other-merged", "id": 9, "color": [116, 112, 0]},
    "stairs": {"name": "building-other-merged", "id": 9, "color": [116, 112, 0]},
    "window-other": {"name": "building-other-merged", "id": 9, "color": [116, 112, 0]},
    "building-other-merged": {
        "name": "building-other-merged",
        "id": 9,
        "color": [116, 112, 0],
    },
    "wall-other-merged": {
        "name": "building-other-merged",
        "id": 9,
        "color": [116, 112, 0],
    },
    "ceiling-merged": {
        "name": "building-other-merged",
        "id": 9,
        "color": [116, 112, 0],
    },
    "gravel": {"name": "gravel", "id": 10, "color": [124, 74, 181]},
    "railroad": {"name": "railroad", "id": 11, "color": [230, 150, 140]},
    "road": {"name": "road", "id": 12, "color": [128, 64, 128]},
    "pavement-merged": {"name": "pavement-merged", "id": 13, "color": [96, 96, 96]},
    "dirt-merged": {"name": "dirt-merged", "id": 14, "color": [208, 229, 228]},
    "sand": {"name": "sand", "id": 15, "color": [254, 212, 124]},
    "snow": {"name": "snow", "id": 16, "color": [255, 255, 255]},
    "floor-other-merged": {
        "name": "floor-other-merged",
        "id": 17,
        "color": [96, 36, 108],
    },
    "grass-merged": {"name": "grass-merged", "id": 18, "color": [152, 251, 152]},
    "wall-brick": {"name": "wall-other-merged", "id": 19, "color": [102, 102, 156]},
    "wall-tile": {"name": "wall-other-merged", "id": 19, "color": [102, 102, 156]},
    "wall-wood": {"name": "wall-other-merged", "id": 19, "color": [102, 102, 156]},
    "rock-merged": {"name": "rock-merged", "id": 20, "color": [0, 114, 143]},
    "wall-stone": {"name": "rock-merged", "id": 20, "color": [0, 114, 143]},
    "water-other": {"name": "water-other", "id": 21, "color": [58, 41, 149]},
    "river": {"name": "water-other", "id": 21, "color": [58, 41, 149]},
    "sea": {"name": "water-other", "id": 21, "color": [58, 41, 149]},
    "tree-merged": {"name": "tree-merged", "id": 22, "color": [107, 142, 35]},
    "fence-merged": {"name": "fence-merged", "id": 23, "color": [190, 153, 153]},
    "sky-other-merged": {"name": "sky-other-merged", "id": 24, "color": [70, 130, 180]},
}

CLASS_HAS_INSTANCE_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
