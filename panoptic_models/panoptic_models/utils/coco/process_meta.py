# import packages
from copy import deepcopy
import os
import json
import random
from panoptic_models.data.utils.coco import coco_constants2

# import parameters
from panoptic_models.panoptic_models.config.params_data import (
    COCO_META_PATH,
    SELECTED_FILENAME_PATH,
    NUM_IMAGES_SELECTED,
    OUTPUT_PATH_META,
    RANDOM_SEED,
    COCO_M2F_ORIGINAL_PERFORMANCE_EVAL,
)
from panoptic_models.panoptic_models.config.params_meta import (
    _CATEGORIES,
    _SUPERCATEGORIES,
)

# import scripts
from panoptic_models.mask2former.utils.logger import _logger

# set local parameters
_REDUCED_CATEGORY_IDS = coco_constants2.get_reduced_class_ids()
_ID_MAPPING = coco_constants2.get_id_mapping()
random.seed(RANDOM_SEED)


def main() -> None:
    with open(SELECTED_FILENAME_PATH, "r") as f:
        file_names_ = json.load(f)

    # from the selected images, pick NUM_IMAGES_SELECTED at random
    file_names = file_names_["annotation_file_names"]
    random_indexes = random.sample(range(len(file_names)), NUM_IMAGES_SELECTED)
    file_names_selected = []
    for idx in random_indexes:
        file_names_selected.append(file_names[idx])
    _logger.info(f"Number of images selected: {len(file_names_selected)}")

    # open meta data
    with open(COCO_META_PATH, "rb") as f:
        train_meta = json.load(f)

    # init lists
    images = []
    image_id = []
    annotations = []
    if COCO_M2F_ORIGINAL_PERFORMANCE_EVAL:
        annotations_original = []

    # get selected images and their IDs
    for image in train_meta["images"]:
        if image["file_name"].replace(".jpg", ".png") in file_names_selected:
            images.append(image)
            image_id.append(image["id"])

    # get the annotations of the selected IDs
    for ann in train_meta["annotations"]:
        if ann["file_name"] not in file_names_selected:
            continue

        # annotations mapped to the reduced category set and if True saved in the original format
        # - original format: compare performance of model to performance of m2f on same dataset
        # - reduced category set: needed to train the new projection heads

        # reduced format
        segments_info = []
        for segment in ann["segments_info"]:
            if segment["category_id"] in _REDUCED_CATEGORY_IDS:
                segment["category_id"] = int(_ID_MAPPING[segment["category_id"]])
                segments_info.append(segment)
        ann["segments_info"] = segments_info
        annotations.append(ann)

        # original format
        if COCO_M2F_ORIGINAL_PERFORMANCE_EVAL:
            annotations_original.append(ann)

    # reduced category set
    cat_meta = []
    categories = deepcopy(_CATEGORIES)
    for cat in categories:
        # if cat['id'] == 0:
        #    continue
        del cat["color"]
        cat["supercategory"] = next(
            item["supercategory"]
            for item in _SUPERCATEGORIES
            if item["id"] == cat["id"]
        )
        cat_meta.append(cat)

    # save reduced meta data
    meta_red = deepcopy(train_meta)
    meta_red["images"] = images
    meta_red["annotations"] = annotations
    meta_red["categories"] = cat_meta

    assert len(images) == len(annotations)

    # make sure that output path exists, else create directory
    file_path, _ = os.path.split(OUTPUT_PATH_META)
    os.makedirs(file_path, exist_ok=True)

    with open(OUTPUT_PATH_META, "w") as f:
        json.dump(meta_red, f)

    # save original meta data for the selected images
    if COCO_M2F_ORIGINAL_PERFORMANCE_EVAL:
        train_meta["images"] = images
        train_meta["annotations"] = annotations_original

        filepath, filetype = os.path.splitext(OUTPUT_PATH_META)
        with open(filepath + "_original" + filetype, "w") as f:
            json.dump(train_meta, f)

    # adapt instance path (still errors when evaluating with segmentation)


#    if COCO_INSTANCE_PATH is not None and COCO_META_PATH is not None:
#        with open(COCO_INSTANCE_PATH, 'rb') as f:
#            train_instance = json.load(f)
#
#        instances = []
#        for ann in train_instance['annotations']:
#            if ann['image_id'] in image_id and ann['category_id'] in _REDUCED_CATEGORY_IDS:
#                ann['category_id'] = _ID_MAPPING[ann['category_id']]
#                instances.append(ann)
#
#        cat_instance = []
#        categories = deepcopy(_CATEGORIES)
#        for cat in categories:
#            if cat['id'] == 0:
#                continue
#            del cat['color']
#            # del cat['isthing']
#            cat['supercategory'] = _SUPERCATEGORIES[cat['id']]['supercategory']
#            cat_instance.append(cat)
#
#        train_instance['images'] = images
#        train_instance['annotations'] = instances
#        train_instance['categories'] = cat_instance
#
#        with open(OUTPUT_PATH_INSTANCE, 'w') as f:
#            json.dump(train_instance, f)


if __name__ == "__main__":
    main()
