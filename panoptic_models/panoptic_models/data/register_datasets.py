# ==
# Script to register reduced COCO dataset and Segments.ai dataset to be used by detectron2 and thus mask2former
# ==

# import packages
import numpy as np

# import local scripts
from detectron2.data.datasets import register_coco_panoptic
from detectron2.data import MetadataCatalog, DatasetCatalog
import panoptic_models.config.params_data as params_data
# import parameters
from panoptic_models.config.params_data import (
    COCO_DST,
    SEGMENTS_DST,
    COCO_M2F_ORIGINAL_PERFORMANCE_EVAL,
    CATERPILLAR,
)
from panoptic_models.config.labels_info import _CATEGORIES


class DatasetRegister:
    """
    Register Dataset in the Detectron2 format
    """

    def __init__(
        self, path_to_coco: str, path_to_segments: str, path_to_caterpillar: str = None
    ) -> None:
        self.coco_path = path_to_coco
        self.segments_path = path_to_segments
        self.caterpillar_path = path_to_caterpillar

        # get meta data
        self.meta = self.construct_meta_detectron2_format()

        # register coco train set
        json_panoptic_train = self.coco_path + "annotations/panoptic_train2017.json"
        image_root_train = self.coco_path + "train2017/"
        panoptic_root_train = self.coco_path + "panoptic_train2017"
        self.register_dataset(
            "coco_2017_train_red",
            image_root_train,
            panoptic_root_train,
            json_panoptic_train,
        )

        # register coco val set
        json_panoptic_val = self.coco_path + "annotations/panoptic_val2017.json"
        image_root_val = self.coco_path + "val2017"
        panoptic_root_val = self.coco_path + "panoptic_val2017"
        json_instance_val = self.coco_path + "annotations/instances_val2017.json"
        self.register_dataset(
            "coco_2017_val_red",
            image_root_val,
            panoptic_root_val,
            json_panoptic_val,
            json_instance_val,
        )

        # register segments train dataset
        json_panoptic_segments_train = (
            self.segments_path + "annotations/panoptic_train.json"
        )
        image_root_segments_train = self.segments_path + "train/"
        panoptic_root_segments_train = self.segments_path + "panoptic_train"
        self.register_dataset(
            "construction_site_train",
            image_root_segments_train,
            panoptic_root_segments_train,
            json_panoptic_segments_train,
        )

        # register segments val dataset
        json_panoptic_segments_val = (
            self.segments_path + "annotations/panoptic_val.json"
        )
        image_root_segments_val = self.segments_path + "val/"
        panoptic_root_segments_val = self.segments_path + "panoptic_val"
        self.register_dataset(
            "construction_site_val",
            image_root_segments_val,
            panoptic_root_segments_val,
            json_panoptic_segments_val,
        )

        # register segments test dataset
        json_panoptic_segments_test = (
            self.segments_path + "annotations/panoptic_test.json"
        )
        image_root_segments_test = self.segments_path + "test/"
        panoptic_root_segments_test = self.segments_path + "panoptic_test"
        self.register_dataset(
            "construction_site_test",
            image_root_segments_test,
            panoptic_root_segments_test,
            json_panoptic_segments_test,
        )

        # register caterpillar dataset
        if self.caterpillar_path:
            json_panoptic_caterpillar = (
                self.caterpillar_path + "annotations/panoptic_train.json"
            )
            image_root_caterpillar = self.caterpillar_path + "train/"
            panoptic_root_caterpillar = self.caterpillar_path + "panoptic_train"
            self.register_dataset(
                "caterpillar",
                image_root_caterpillar,
                panoptic_root_caterpillar,
                json_panoptic_caterpillar,
            )

        if COCO_M2F_ORIGINAL_PERFORMANCE_EVAL:
            coco_val_2017_meta = MetadataCatalog.get("coco_2017_val_panoptic")
            self.meta = {
                "thing_classes": coco_val_2017_meta.get("thing_classes"),
                "thing_colors": coco_val_2017_meta.get("thing_colors"),
                "thing_dataset_id_to_contiguous_id": coco_val_2017_meta.get(
                    "thing_dataset_id_to_contiguous_id"
                ),
                "stuff_classes": coco_val_2017_meta.get("stuff_classes"),
                "stuff_colors": coco_val_2017_meta.get("stuff_colors"),
                "stuff_dataset_id_to_contiguous_id": coco_val_2017_meta.get(
                    "stuff_dataset_id_to_contiguous_id"
                ),
            }
            json_panoptic_original_val = (
                self.coco_path + "annotations/panoptic_val2017_original.json"
            )
            image_root_original_val = self.coco_path + "val2017/"
            panoptic_root_original_val = self.coco_path + "panoptic_val2017"
            self.register_dataset(
                "coco_2017_val_original",
                image_root_original_val,
                panoptic_root_original_val,
                json_panoptic_original_val,
            )

            json_panoptic_original_train = (
                self.coco_path + "annotations/panoptic_train2017_original.json"
            )
            image_root_original_train = self.coco_path + "train2017/"
            panoptic_root_original_train = self.coco_path + "panoptic_train2017"
            self.register_dataset(
                "coco_2017_train_original",
                image_root_original_train,
                panoptic_root_original_train,
                json_panoptic_original_train,
            )

    def construct_meta_detectron2_format(self) -> dict:
        """
        Create Metadata for the datasets in the detectron2 standard format. This means that the classes are
        separated in 'stuff' and 'things', for each part an local and global id as well as a name and color is defined.
        """
        thing_dataset_id_to_contiguous_id = {}
        stuff_dataset_id_to_contiguous_id = {}
        colors = []
        classes = []
        for idx, item in enumerate(_CATEGORIES):
            colors.append(item["color"])
            classes.append(item["name"])
            if item["isthing"] == 1:
                thing_dataset_id_to_contiguous_id[item["id"]] = int(idx)
            else:
                stuff_dataset_id_to_contiguous_id[item["id"]] = int(idx)

        return {
            "thing_classes": classes,
            "thing_colors": colors,
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
            "stuff_classes": classes,
            "stuff_colors": colors,
            "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        }

    def register_dataset(
        self, name, image_root, panoptic_root, panoptic_json, instance_json=None
    ) -> None:
        register_coco_panoptic(
            name=name,
            metadata=self.meta,
            image_root=image_root,
            panoptic_root=panoptic_root,
            panoptic_json=panoptic_json,
            instances_json=instance_json,
        )

    def test_registering(self) -> None:
        # registration succeeded if not error here
        dataset_train_red = MetadataCatalog.get("coco_2017_train_red")
        dataset_val_red = MetadataCatalog.get("coco_2017_val_red")
        dataset_segment_train = MetadataCatalog.get("construction_site_train")
        dataset_segment_val = MetadataCatalog.get("construction_site_val")
        # this part is just for debugging and evaluating if dataset registration succeeded
        # get dataset information
        dataset_train_original = MetadataCatalog.get("coco_2017_train_panoptic")
        dataset_val_original = MetadataCatalog.get("coco_2017_val_panoptic")

    def get_all_datasets(self) -> None:
        print(DatasetCatalog.list())


def main():
    registering = DatasetRegister(COCO_DST, SEGMENTS_DST, CATERPILLAR)
    registering.test_registering()
    # print available datasets
    registering.get_all_datasets()


if __name__ == "__main__":
    main()
    print("All datasets have been registered!")
