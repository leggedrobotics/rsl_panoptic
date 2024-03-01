# COCO DATASET

Go to the [COCO dataset website](https://cocodataset.org/#download), and download [2017 train images](http://images.cocodataset.org/zips/train2017.zip), [2017 val images](http://images.cocodataset.org/zips/val2017.zip), and [2017 Panoptic Train/Val annotations](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip). Extract the images from the zip folders and proceed.
As an alternative use the script ```download_coco.sh```. 

## Prepare Dataset

Select parameters in `mask2former/config/params_data.py`. Then execute the following scripts in the given order:

```
python3 select_files.py
python3 process_meta.py
python3 mv_images.py
```
