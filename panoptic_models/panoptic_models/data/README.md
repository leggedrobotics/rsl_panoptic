# Datasets 

## Labeled Dataset
Labeled Dataset used for the supervised training of the baseline and the supervised fine-tuning 
after self-supervised pre-training. Data sources are labeled images of test construction sites, a 
reduced version of the Coco Dataset and the "Caterpillar" test dataset

### Structure 

```
.dataset_path
    - coco
        - {train|val}2017 (samples)
        - panoptic_{train|val}2017 (panoptic segmentation maps)
        - annotations:  
            - panoptic_{train|val}2017.json  (annotation meta)
            - panoptic_{train|val}2017_original.json
    - segments
        - {train|test|val} (samples)
        - panoptic_{train|val|test} (panoptic segmentation maps)
        - annotations
    - caterpillar (if needed)
        - train (samples)
        - panoptic_train (panoptic segmentation maps)
        - annotations
```

The default for the *dataset_path* is given by *COCO_DST* and *SEG_DST* defined in the parameter file, however, an argument with a different path can be passed to ``` m2f_train.py --dataset DATASET-PATH ```. 

The two different annotations are used for later evaluation of the new model compared to the original implementation trained for all COCO classes. Thus, *panoptic_{train|val}2017_original* contains the original COCO annotations for the selected images. 

### Generation

If the standard labeled dataset should be generated, please use ```prepare_datasets.sh ``` described below. 
In the case that modifications should be performed or more control is intended, follow the description for the individual datasets:

- [COCO](./utils/coco/README.md)
- [Labelled Test Construction Sites and Caterpillar Dataset](./utils/segment_ai/README.md)

## Unlabeled Dataset
Unlabeled Dataset used for the self-supervised pre-training. Data acquired from the RSL 
Construction site and by scraping YouTube, in particular the channel [LetsDig18](https://www.youtube.com/c/letsdig18) and [Hedblom](https://www.youtube.com/c/HedblomSwe).

### Structure

```
.dataset_path
    - train
    - val
```

*dataset_path* is passed as argument to the pre-training function, i.e. ``` main_pretrain.py --data_path DATASET-PATH```. 
*train* and *val* have to include different folders with the samples, the names of the folders can represent certain classes/ origins/ ... but are not used during training.

### Generation

If the standard labeled dataset should be generated, please use ```prepare_datasets.sh ``` described below. 
In the case that modifications should be performed or more control is intended, follow the description for the individual datasets:

- [COCO](./utils/coco/README.md)
- [YouTube Scrape](./utils/yt_data/README.md)
- [Unlabeled Test Construction Sites](./utils/rosbag/README.md)


## Data Generation

:warning: Still under development, use single steps :warning:

All parameters required can be set under 
```
config/params_data.py
```

After setting the parameters, both the dataset used for the supervised as well as self-supervised training can be generated as follows:
```
sh prepare_dataset.sh
```
Should only one of the two be required, please use the generation scripts within the directories for the labeled or unlabeled dataset (more precise information can be found in the according README files within the directories). In the case that a particular dataset should be acquired, the corresponding function can be found under 'utils/DATASET_NAME'. 

## Own Datasets

The usage of additional datasets is easily possible. 
We have to differeniate between the SSL and supervised training:

- **SSL Training**: If data should be used in addition to the presented samples, just include it in generated *dataset_unlabeled* directory under either train or val. On the contrary, if only the new dataset should be used, any folder can be selected as long as it contains a train and val directory with corresponding subdirectories. 
- **Supervised Training**: Any dataset with COCO structure can be added, i.e. a directory with the following structure is mandatory:
  ```
  ./dataset_dir
    - {train|val|test}
    - panoptic_{train|val|test}
    - annotations 
  ```
  Then the dataset has to be added in `data/register_datasets.py`. Please refer to the examples given there how to add the dataset. 
  Afterwards, the dataset name can be included in the model configs and training with it can be started. 
