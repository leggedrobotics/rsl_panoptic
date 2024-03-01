import json
import os
import numpy as np
import random
from typing import Sequence, Tuple, Any
from data import coco_constants2
from absl import app
from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "coco_meta_path", None, "Example: /coco_root/annotations/panoptic_train2017.json"
)

flags.DEFINE_string(
    "selected_filenames_path", None, "Example: /path/to/selected_filenames_train.json"
)

flags.DEFINE_string(
    "output_path", None, "Example: /output/path/panoptic_train2017.json."
)

flags.DEFINE_integer(
    "num_images_selected", None, "Number of images to select from COCO for training."
)


_REDUCED_CATEGORY_IDS = coco_constants2.get_reduced_class_ids()
_ID_MAPPING = coco_constants2.get_id_mapping()


def main(unused_argv: Sequence[str]) -> None:
    with open(FLAGS.selected_filenames_path, "r") as f:
        file_names_ = json.load(f)

    file_names = file_names_["annotation_file_names"]

    random_indexes = random.sample(
        range(400, len(file_names)), FLAGS.num_images_selected
    )
    file_names_selected = []
    for idx in random_indexes:
        file_names_selected.append(file_names[idx])
    print("Number of images selected: ", len(file_names_selected))

    with open(FLAGS.coco_meta_path, "rb") as f:
        train_meta = json.load(f)

    images = []
    annotations = []
    for i, image in enumerate(train_meta["images"]):
        if image["file_name"].replace(".jpg", ".png") in file_names_selected:
            images.append(image)

    for i, ann in enumerate(train_meta["annotations"]):
        if ann["file_name"] in file_names_selected:
            segments_info = []
            for segment in ann["segments_info"]:
                if segment["category_id"] in _REDUCED_CATEGORY_IDS:
                    segment["category_id"] = _ID_MAPPING[segment["category_id"]]
                    segments_info.append(segment)
            ann["segments_info"] = segments_info
            annotations.append(ann)

    train_meta["images"] = images
    train_meta["annotations"] = annotations
    assert len(images) == len(annotations)
    with open(FLAGS.output_path, "w") as f:
        json.dump(train_meta, f)


if __name__ == "__main__":
    flags.mark_flags_as_required(
        [
            "coco_meta_path",
            "selected_filenames_path",
            "output_path",
            "num_images_selected",
        ]
    )
    app.run(main)
