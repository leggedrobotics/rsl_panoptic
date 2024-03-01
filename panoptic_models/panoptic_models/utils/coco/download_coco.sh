#!/bin/bash

#==
# Download and Prepare COCO Dataset
#==

# Download
printf "Download COCO 2017 Training Images\n"
wget http://images.cocodataset.org/zips/train2017.zip -P ./coco_dataset

printf "Download COCO 2017 Validation Images\n"
wget http://images.cocodataset.org/zips/val2017.zip -P ./coco_dataset

printf "Download COCO 2017 Panoptic Annotations\n"
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip -P ./coco_dataset

# Unzip
printf "Unzip downloaded datafiles\n"
unzip -q ./coco_dataset/train2017.zip -d ./coco_dataset
unzip -q ./coco_dataset/val2017.zip -d ./coco_dataset
unzip -q ./coco_dataset/panoptic_annotations_trainval2017.zip -d ./coco_dataset

# remove zip files
printf "Remove zip files\n"
rm ./coco_dataset/train2017.zip
rm ./coco_dataset/val2017.zip
rm ./coco_dataset/panoptic_annotations_trainval2017.zip

