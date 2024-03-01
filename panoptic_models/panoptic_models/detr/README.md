# DETR

The code is partially taken and modified from https://github.com/facebookresearch/detr. Several new files are also added. The code here is mainly for running the __panoptic_ros__ node. For DETR training, inference and evaluation, follow the instructions below.

## Installation\
You can use the docker container that we provide or follow the instruction below. 

```bash
cd panoptic_models/panoptic_models/detr
git clone git@github.com:facebookresearch/detr.git
```

Install PyTorch 1.5+ and torchvision 0.6+:

```
conda install -c pytorch pytorch torchvision
```

Install pycocotools cython and scipy:

```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Install panopticapi:

```
pip install git+https://github.com/cocodataset/panopticapi.git
```

Install Detectron2:

```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

For details, see [here](https://github.com/facebookresearch/detr).

To avoid import errors, make sure to set the PYTHONPATH

```bash
export PYTHONPATH="/path/to/m545_panoptic_segmentation/panoptic_models/panoptic_models/detr:$PYTHONPATH"
```

## Dataset

Dataset should be converted to COCO format for training and evaluation, and downloaded in the following path arrangement.  

```
coco_path:  
   - train2017 (training images)  

coco_panoptic_path:  
   - panoptic_train2017 (panoptic annotations)  
   - annotations:  
      - panoptic_train2017.json
```

The dataset used in this project includes part of the COCO panoptic dataset, and customly labeled dataset. For preparing the dataset, see [panoptic_models/utils/README.md](../../../../utils/README.md).

## Training
For finetuning the model on our custum dataset:
```bash
python3 panoptic_models/detr/main.py --dataset_file construction_site --val_dataset_file construction_site --dataset_filename construction --batch_size 1 --lr 0.00001 --no_aux_loss --output_dir <path_to_package>/panoptic_models/outputs/detr --resume <path_to_package>/panoptic_models/detr/weights/detr-r50-panoptic-00ce5173.pth --device cuda --epochs 1001 --save_epochs 100 --validation_epochs 25 --run_name instance_v8 --num_samples 0 --wandb_log --tag detection
```
If you don't have wandb remove it from the flags. 

Pretrained model url: https://dl.fbaipublicfiles.com/detr/detr-r50-panoptic-00ce5173.pth 

For training segmentation model (frozen backbone) run:
```bash
python3 panoptic_models/detr/main.py --dataset_file construction_site --val_dataset_file construction_site --dataset_filename construction --batch_size 1 --lr 0.00001 --no_aux_loss --output_dir <path_to_package>/panoptic_models/outputs/detr --frozen_weights <path_to_package>/panoptic_models/detr/weights/detr_instance_r50_construction_epoch_500.pth --device cuda --epochs 501 --save_epochs 100 --validation_epochs 25 --masks --run_name seg_v8_frozen --num_samples 0 --wandb_log --tag segmentation
```
For further finetuning of the whole network:
```bash
python3 panoptic_models/detr/main.py --dataset_file construction_site --val_dataset_file construction_site --dataset_filename construction --batch_size 1 --lr 0.00001 --no_aux_loss --output_dir <path_to_package>/panoptic_models/outputs/detr --resume <path_to_package>/panoptic_models/detr/weights/detr_50_panoptic_construction_frozen_epoch_500.pth --device cuda --epochs 501 --save_epochs 100 --validation_epochs 25 --masks --run_name seg_v8 --num_samples 0 --wandb_log --tag segmentation
```

## Inference and evaluation
To run the model:
```bash
python infer.py --frozen_weights=/path/to/weights/checkpoint.pth --masks --image_root=/path/to/test_images/ --output_dir=/path/to/output_dir/
```

For enabling pq evaluation during inference, add flag __--eval__, and add info for:  
- _image_root_: image root of evaluation data.  
- _ann_file_: gt annotation meta file.  
- _ann_folder_: gt annotations.  
- _ann_output_folder_: predicted annotation output folder. 

