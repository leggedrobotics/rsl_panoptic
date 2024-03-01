# Panoptic DeepLab

The code is partially taken and modified from https://github.com/google-research/deeplab2. Several new files are also added. The code here is mainly for running the __panoptic_ros__ node. For Panoptic-DeepLab training, inference and evaluation, follow the instructions below.


## Installation

```bash
cd panoptic_models/panoptic_models/deeplab2/
git clone git@github.com:google-research/deeplab2.git
```

For installation details of deeplab2, see https://github.com/google-research/deeplab2/blob/main/g3doc/setup/installation.md.

To avoid import errors, make sure to set the PYTHONPATH

```bash
export PYTHONPATH="/path/to/m545_panoptic_segmentation/deploy/panoptic_models/panoptic_models/deeplab2:$PYTHONPATH"
```


## Dataset

The dataset used in this project includes part of the COCO panoptic dataset, and customly labeled dataset. 

For preparing the dataset, see __m545_panoptic_segmentation/utils/README.md__.


## Before training

For training with the new dataset in this project, you need to make a few changes in deeplab2 first.

1. Change the dataset info in __deeplab2/data/dataset.py__ (line 219-232) to:

```python
COCO_PANOPTIC_INFORMATION = DatasetDescriptor(
    dataset_name=_COCO_PANOPTIC,
    splits_to_sizes={'train': 118287,
                     'val': 5000,
                     'test': 40670},
    num_classes=32,
    ignore_label=0,
    panoptic_label_divisor=256,
    class_has_instances_list=tuple(range(1, 8)) + (26, 29, 30, 31),
    is_video_dataset=False,
    colormap=_COCO_COLORMAP,
    is_depth_dataset=False,
    ignore_depth=None,
)
```


2. Change the training config in __deeplab2/configs/coco/panoptic_deeplab/resnet50_os32.textproto__:

  - __experiment_name__ (line 28): set experiment name.  
  - __model_options/initial_checkpoint__ (line 32): set the initial checkpoint path to pretrained model. Download link: https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/resnet50_os32_panoptic_deeplab_coco_train_2.tar.gz.  
  - __model_options/semantic_head/output_channels__ (line 80): 32.  
  - __trainer_options__ (line 85): set trainer_options as you like.  
  - __train_dataset_options/file_pattern__ (line 113): /data/path/tfrecords/train*.tfrecord.  
  - __train_dataset_options/batch_size__ (line 116): you may need to change batch_size.  
  - __eval_dataset_options/file_pattern__ (line 133): /data/path/tfrecords/val*.tfrecord.    
  - __evaluator_options/stuff_area_limit__ (line 145): you may need to change the stuff area limit smaller.


3. Change in __deeplab2/trainer/train_lib.py__:  

Do not initialize the semantic last layer with pretrained model, since the number of semantic output channels doesn't match. Comment line 152-159, add:

```python
del init_dict[common.CKPT_SEMANTIC_LAST_LAYER]
```


4. Change in __deeplab2/trainer/trainer.py__:

If you want to freeze the backbone and only train the semantic head and instance head, change line 245 to :

```python
training_vars = self._model._decoder._semantic_head.trainable_variables + \
                self._model._decoder._instance_center_head.trainable_variables + \
                self._model._decoder._instance_regression_head.trainable_variables
```


## Training

Now everything is set up, run the following script for training: 

```bash 
python deeplab2/trainer/train.py --config_file=deeplab2/configs/coco/panoptic_deeplab/resnet50_os32.textproto --mode=train --model_dir=${MODEL_DIR} --num_gpus=${NUM_GPUS}
```  


## Inference and evaluation

For inference only, run:

```bash
cd panoptic_deeplab
python infer.py \
    --config_file=${CONFIG_FILE_PATH} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --image_dir=${IMAGE_DIR} \
    --save_dir=${SAVE_DIR}
```

To enable evaluation, run:

```bash
cd panoptic_deeplab
python infer.py \
    --config_file=${CONFIG_FILE_PATH} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --image_dir=${IMAGE_DIR} \
    --save_dir=${SAVE_DIR} \
    --eval=True \
    --ann_file=${GT_ANNOTATION_META_FILE_PATH} \
    --ann_folder=${GT_ANNOTATION_FOLDER} \
    --ann_output_folder=${PREDICTED_ANNOTATION_OUTPUT_FOLDER}
```
 
