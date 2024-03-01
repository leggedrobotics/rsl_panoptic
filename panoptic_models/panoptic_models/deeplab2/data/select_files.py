import collections
import json
import math
import os
import numpy as np
import tensorflow as tf

from typing import Sequence, Tuple, Any

from absl import app
from absl import flags
from absl import logging

from data import coco_constants2
from deeplab2.data import data_utils
from deeplab2.data import dataset


FLAGS = flags.FLAGS

flags.DEFINE_string("coco_root", None, "coco dataset root folder.")

flags.DEFINE_string("dataset_split", "train", "dataset split: train or val")

flags.DEFINE_string(
    "output_path",
    None,
    "Path to save selected filenames. Example: /path/to/selected_filenames_train.json",
)

flags.DEFINE_boolean(
    "treat_crowd_as_ignore",
    True,
    "Whether to apply ignore labels to crowd pixels in " "panoptic label.",
)

flags.DEFINE_integer(
    "num_categories_at_least",
    3,
    "Each selected image contains at least $(num_categories_at_least) categories we need.",
)


_IGNORE_LABEL = dataset.COCO_PANOPTIC_INFORMATION.ignore_label
_CLASS_MAPPING = coco_constants2.get_id_mapping()
_CLASS_HAS_INSTANCE_LIST = coco_constants2.CLASS_HAS_INSTANCE_LIST
_PANOPTIC_LABEL_DIVISOR = dataset.COCO_PANOPTIC_INFORMATION.panoptic_label_divisor
_TARGET_CATEGORY_IDS = coco_constants2.get_reduced_class_ids()  # list


# A map from data type to folder name that saves the data.
_FOLDERS_MAP = {
    "train": {
        "image": "train2017",
        "label": "annotations",
    },
    "val": {
        "image": "val2017",
        "label": "annotations",
    },
}

# A map from data type to data format.
_DATA_FORMAT_MAP = {
    "image": "jpg",
    "label": "png",
}
_PANOPTIC_LABEL_FORMAT = "raw"


def _get_images(coco_root: str, dataset_split: str) -> Sequence[str]:
    """Gets files for the specified data type and dataset split.

    Args:
    coco_root: String, path to coco dataset root folder.
    dataset_split: String, dataset split ('train', 'val', 'test').

    Returns:
    A list of sorted file names.
    """
    pattern = "*.%s" % _DATA_FORMAT_MAP["image"]
    search_files = os.path.join(
        coco_root, _FOLDERS_MAP[dataset_split]["image"], pattern
    )
    filenames = tf.io.gfile.glob(search_files)
    return sorted(filenames)


def _get_panoptic_annotation(
    coco_root: str, dataset_split: str, annotation_file_name: str
) -> str:
    panoptic_folder = "panoptic_%s2017" % dataset_split
    return os.path.join(
        coco_root,
        _FOLDERS_MAP[dataset_split]["label"],
        panoptic_folder,
        annotation_file_name,
    )


def _read_segments(coco_root: str, dataset_split: str):
    """Reads segments information from json file.

    Args:
        coco_root: String, path to coco dataset root folder.
        dataset_split: String, dataset split.

    Returns:
        segments_dict: A dictionary that maps file prefix of annotation_file_name to
          a tuple of (panoptic annotation file name, segments). Please refer to
          _generate_panoptic_label() method on the detail structure of `segments`.

    Raises:
        ValueError: If found duplicated image id in annotations.
    """
    json_filename = os.path.join(
        coco_root,
        _FOLDERS_MAP[dataset_split]["label"],
        "panoptic_%s2017.json" % dataset_split,
    )
    with tf.io.gfile.GFile(json_filename) as f:
        panoptic_dataset = json.load(f)

    segments_dict = {}
    for annotation in panoptic_dataset["annotations"]:
        image_id = annotation["image_id"]
        if image_id in segments_dict:
            raise ValueError("Image ID %s already exists" % image_id)
        annotation_file_name = annotation["file_name"]
        segments = annotation["segments_info"]

        segments_dict[os.path.splitext(annotation_file_name)[-2]] = (
            annotation_file_name,
            segments,
        )

    return segments_dict


def _is_in_target_category_ids(segments):
    """
    Check whether there is at least one category id of a COCO image in the target category ids.
    Args:
        segments: A list of dictionaries containing information of every segment.
            Read from panoptic_${DATASET_SPLIT}2017.json. This method consumes
            the following fields in each dictionary:
              - id: panoptic id
              - category_id: semantic class id
              - area: pixel area of this segment
              - iscrowd: if this segment is crowd region

    Returns:
        bool, if there is three ids in the target category ids, return True, else return False.
    """
    count = 0
    for segment in segments:
        if segment["category_id"] in _TARGET_CATEGORY_IDS:
            count += 1
            if count >= FLAGS.num_categories_at_least:
                return True
    return False


def _get_annotation_filename(image_path: str, segments_dict: Any):
    """Creates labels for panoptic segmentation.

    Args:
      image_path: String, path to the image file.
      segments_dict:
        Read from panoptic_${DATASET_SPLIT}2017.json. This method consumes
        the following fields in each dictionary:
          - id: panoptic id
          - category_id: semantic class id
          - area: pixel area of this segment
          - iscrowd: if this segment is crowd region

    Returns:
      return annotation_filename if this image/annotation is selected, otherwise return None.
    """

    image_path = os.path.normpath(image_path)
    path_list = image_path.split(os.sep)
    file_name = path_list[-1]

    annotation_filename, segments = segments_dict[os.path.splitext(file_name)[-2]]

    if _is_in_target_category_ids(segments):
        return annotation_filename
    else:
        return None


def _select_files(coco_root: str, dataset_split: str) -> None:
    """Select files.

    Args:
      coco_root: String, path to coco dataset root folder.
      dataset_split: String, the dataset split (one of `train`, `val` and `test`).
      output_dir: String, directory to write output TFRecords to.
    """
    image_files = _get_images(coco_root, dataset_split)

    num_images = len(image_files)

    segments_dict = _read_segments(coco_root, dataset_split)

    file_names = []

    for i in range(num_images):
        annotation_filename = _get_annotation_filename(image_files[i], segments_dict)
        if annotation_filename is not None:
            file_names.append(annotation_filename)
    return file_names


def main(unused_argv: Sequence[str]) -> None:
    dataset_split = FLAGS.dataset_split
    logging.info("Starts processing dataset split %s.", dataset_split)
    file_names_selected = _select_files(FLAGS.coco_root, dataset_split)
    print("Number of images selected: ", len(file_names_selected))
    data = {
        "num_categories_at_least": FLAGS.num_categories_at_least,
        "annotation_file_names": file_names_selected,
    }
    if not FLAGS.output_path.endswith(".json"):
        output_path = FLAGS.output_path + ".json"
    else:
        output_path = FLAGS.output_path
    with open(output_path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    flags.mark_flags_as_required(["coco_root", "output_path"])
    app.run(main)
