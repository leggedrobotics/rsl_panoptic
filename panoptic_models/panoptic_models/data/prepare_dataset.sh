#==
# Automaticall prepare labeled dataset for the usage with mask2former
#==

echo "RSL Self-Supervised Learning for Panoptic Image Segmentation - Labeled Dataset Preparation \n";

echo "Construction Site Dataset\n"\
     "-------------------------\n"\
     "start downloading data\n"

# get package path
PACKAGE_PATH=$(python3 -c "import panoptic_models.mask2former; print(mask2former.__path__[0])")
echo "Package Path: $PACKAGE_PATH \n"
python3 $PACKAGE_PATH/data/utils/segment_ai/segments_download_data.py
echo "DONE! Start converting data to COCO and splitting it into train, val and test set \n"
python3 $PACKAGE_PATH/data/utils/segment_ai/segments_convert_to_coco_format.py
echo "DONE! Move construction dataset to dataset_labeled \n"
# TODO: include moving function
echo "DONE! Construction Dataset Prepared \n"
echo ""

echo "COCO Dataset\n"\
     "------------\n"\
     "start downloading data\n"

