import shutil
import json
import os

if __name__ == "__main__":
    # change accordingly
    # -----------------------------------------------
    processed_meta_path = (
        "my_dataset_root/coco_panoptic_path/annotations/panoptic_train2017.json"
    )

    src_root_images = "coco_root/train2017"
    dst_root_images = "my_dataset_root/coco_path/train2017/"

    src_root_anns = "coco_root/annotations/panoptic_train2017/"
    dst_root_anns = "my_dataset_root/coco_panoptic_path/panoptic_train2017/"
    # -----------------------------------------------

    with open(processed_meta_path, "r") as f:
        meta = json.load(f)

    images = meta["images"]
    anns = meta["annotations"]

    for image in images:
        file_name = image["file_name"]
        shutil.copyfile(
            os.path.join(src_root_images, file_name),
            os.path.join(dst_root_images, file_name),
        )

    for ann in anns:
        file_name = ann["file_name"]
        shutil.copyfile(
            os.path.join(src_root_anns, file_name),
            os.path.join(dst_root_anns, file_name),
        )
