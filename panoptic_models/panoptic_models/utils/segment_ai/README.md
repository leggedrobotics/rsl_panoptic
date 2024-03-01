# Labeled Test Construction Site

The labels for the samples have been generated using [Segments.ai](https://segments.ai/) and can be found [here](https://segments.ai/leggedrobotics/construction_site/). 

**IMPORTANT**: before the data generation scripts can be executed, the segments.ai information of the dataset have to be downloaded from the website. This should be a json file with a certain version. The Caterpillar Dataset is included started with version 6.0, i.e. please download `construction_site-v6.0.json` or higher versions. Place the json file in the folder specified in config/params_data/SEGMENTS_SRC. 

## Data Generation
Select parameters in `mask2former/config/params_data.py`. Then execute the following scripts in the given order:

```
python3 segments_download_data.py
python3 segments_convert_to_coco_format.py
```

Within the conversion function to COCO format, the samples are automatically moved to the destination defined in `params_data.py`. 
In the case that not all samples are needed, the leftovers will remain in the download folder and can be manually deleted.