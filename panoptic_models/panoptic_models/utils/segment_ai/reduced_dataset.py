"""
This file removes a fraction of the images from the training dataset.
It removes the images from the annotations and the images from the images folders.
It also creates a new annotation file with the reduced dataset.
The folder structure is the same as the original dataset and it's:
- annotations
- train
- val
- panoptic_train
- panoptic_val
but we only have a fraction of the images in the train2017 and panoptic_train2017 folders.
"""

import os
import json
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm


def remove_annotation(json_path, fraction_to_remove, seed):
    """Remove a fraction of the images from the annotations"""
    with open(json_path, "r") as f:
        metadata = json.load(f)
        info = metadata["info"]
        images = metadata["images"]
        annotations = metadata["annotations"]

    # remove a fraction of the images from the annotations
    random.seed(seed)
    num_images = len(images)
    num_images_to_remove = int(num_images * fraction_to_remove)
    list_image_ids = [image["id"] for image in images]
    print("Removing %d images from the annotations" % num_images_to_remove)
    images_to_remove = random.sample(list_image_ids, num_images_to_remove)
    images_to_remove.sort()

    # remove images from the annotations
    new_images = []
    new_annotations = []
    id_map = {}
    new_id = 0
    for i, image in enumerate(images):
        if image["id"] not in images_to_remove:
            id_map[image["id"]] = new_id
            image["id"] = new_id
            new_images.append(image)
            new_id += 1
    for ann in annotations:
        if ann["image_id"] not in images_to_remove:
            ann["image_id"] = id_map[ann["image_id"]]
            new_annotations.append(ann)

    # update the metadata with the new information
    metadata["images"] = new_images
    metadata["annotations"] = new_annotations

    # delete the old annotation file and create a new one
    os.remove(json_path)
    new_json_path = json_path
    with open(new_json_path, "w") as f:
        json.dump(metadata, f)


def remove_images(image_folder, new_json_path):
    """Remove the images from the image folder"""
    with open(new_json_path, "r") as f:
        metadata = json.load(f)
        images = metadata["images"]
    print("Keeping %d images from the images folder %s" % (len(images), image_folder))
    # image_ids = [image['file_name'] for image in images] but without the extension
    image_ids = [image["file_name"].split(".")[0] for image in images]
    for image in tqdm(os.listdir(image_folder)):
        # without the extension
        image_wo_ext = image.split(".")[0]
        if image_wo_ext not in image_ids:
            os.remove(os.path.join(image_folder, image))


def main():
    parser = argparse.ArgumentParser()
    # dataset folder
    parser.add_argument(
        "--dataset_folder", type=str, default="data/segment_ai", help="dataset folder"
    )
    # list of fraction of images to remove
    parser.add_argument(
        "--list_fraction_to_remove",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8],
        help="list of fraction of images to remove",
    )
    # seed
    parser.add_argument("--seed", type=int, default=42, help="seed")
    args = parser.parse_args()

    for fraction_to_remove in args.list_fraction_to_remove:
        print("Removing a fraction of {} of the images".format(fraction_to_remove))
        # copy dataset folder but change it's name to dataset_folder_<fraction_to_remove>
        # convert the  fraction to remove to a string and keep only the first decimal
        fraction_to_remove_round = np.round(1 - fraction_to_remove, decimals=1)
        shutil.copytree(
            args.dataset_folder,
            args.dataset_folder + "_" + str(fraction_to_remove_round),
        )
        new_dataset_path = args.dataset_folder + "_" + str(fraction_to_remove_round)

        # remove a fraction of the images from the annotations
        json_path = os.path.join(
            new_dataset_path, "segments/annotations", "panoptic_train.json"
        )
        remove_annotation(json_path, fraction_to_remove, args.seed)

        # remove the images from the image folder
        image_folder = os.path.join(new_dataset_path, "segments/train")
        remove_images(image_folder, json_path)

        # remove the images from the panoptic folder
        image_folder = os.path.join(new_dataset_path, "segments/panoptic_train")
        remove_images(image_folder, json_path)


if __name__ == "__main__":
    main()
