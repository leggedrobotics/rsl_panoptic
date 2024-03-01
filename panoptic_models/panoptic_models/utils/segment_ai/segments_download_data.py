"""
1. Download images and annotations from segments.ai.
2. Save raw segments info.
"""

# import packages
import json
import wget
import os

# import parameters
from panoptic_models.config.params_data import (
    SEGMENTS_INFO_PATH,
    ANNOTATION_DIR,
    META_FILENAME,
    VERSION,
    IMAGE_PATH,
    CATERPILLAR,
)


if __name__ == "__main__":
    with open(META_FILENAME, "r") as f:
        meta = json.load(f)
    samples = meta["dataset"]["samples"]

    # make sure image path exits
    os.makedirs(IMAGE_PATH, exist_ok=True)
    os.makedirs(ANNOTATION_DIR, exist_ok=True)
    dir, _ = os.path.split(SEGMENTS_INFO_PATH)
    os.makedirs(dir, exist_ok=True)

    i = 0
    annotation_meta = []
    # use tqdm to show progress
    for sample in samples:
        # filter data by setting 'label_status' to 'REVIEWED'. Options are ['REVIEWED', 'LABELED']
        if (
            sample["labels"]["ground-truth"] is not None
            and sample["labels"]["ground-truth"]["label_status"] == "REVIEWED"
        ):

            if CATERPILLAR and not sample["name"].startswith("frame"):
                continue
            elif not CATERPILLAR and sample["name"].startswith("frame"):
                continue

            i += 1
            segments = sample["labels"]["ground-truth"]["attributes"]["annotations"]
            segments_dict = {"image_file": "{:05d}.png".format(i), "segments": segments}
            annotation_meta.append(segments_dict)
            image = wget.download(
                sample["attributes"]["image"]["url"],
                os.path.join(IMAGE_PATH, "{:05d}.jpg".format(i)),
            )
            annotation = wget.download(
                sample["labels"]["ground-truth"]["attributes"]["segmentation_bitmap"][
                    "url"
                ],
                os.path.join(ANNOTATION_DIR, "{:05d}.png".format(i)),
            )

    data = {
        "name": "construction_site",
        "version": VERSION,
        "annotations": annotation_meta,
    }
    with open(SEGMENTS_INFO_PATH, "w") as f:
        json.dump(data, f)
