# M545 Panoptic Segmentation

This package is for deploying the panoptic segmentation developed by the Robotics Systems Lab.

## Panoptic Models 

This section encompasses our efforts in developing state-of-the-art panoptic segmentation models using a supervised learning approache. The models are designed to understand and segment images into meaningful parts, combining both object detection and semantic segmentation tasks to provide a comprehensive view of an image's content.

### Panoptic Segmentation

Our approach utilizes DETR and Mask2Former framework, aiming to generate hierarchical feature embeddings for panoptic segmentation. The models are trained on selected subsets of the COCO Dataset and a specially curated Test Construction Site Dataset, demonstrating their effectiveness in diverse environments. For detailed instructions on training and inference, refer to the [Mask2Former README](panoptic_models/panoptic_models/mask2former/README.md) and the [DETR README](panoptic_model/panoptic_models/detr/README.md) 

Key Highlights:
- Utilization of COCO Dataset and Construction site data for robust training.
- Employment of Mask2Former framework for accurate segmentation results.
- Sample visualizations of segmentation predictions available for reference.

For more details on datasets and model architecture, please see the specific README.md in the utils subfolder and the self_sup_seg/m2f_deploy directory.

## Panoptic Ros 

Integrating advanced panoptic segmentation capabilities with ROS (Robot Operating System), the Panoptic Ros project provides real-time image segmentation through a dedicated ROS node. It supports various models like DETR and Mask2Former, allowing seamless integration into robotics applications for enhanced environmental understanding.

### ROS Panoptic Segmentation Node

Key Features:
- Easy integration with ROS for image topic subscription and segmented image publishing.
- Support for multiple segmentation models with configurable ROS parameters.
- Real-time visualization options for segmentation results and labels within the ROS environment.

Installation and Launch:
- Docker container recommended for ease of setup; alternatively, dependencies are listed in the official DETR and Mask2Former repos.
- Launch the node using roslaunch command: `roslaunch panoptic_ros image_segmentation.launch`.

Parameters and Customization:
- Various parameters allow for customization of input topics, model selection, and visualization preferences.
- Extendability by implementing additional models and adjusting image processing steps to meet specific needs.

## Setup with Docker

pip install pandas
pip install tensorflow --upgrade
pip install wandb 
pip install timm
pip install scikit-learn
./rsl_panoptic/panoptic_models/panoptic_models/mask2former/third_party/Mask2Former/mask2former/modeling/pixel_decoder/ops/make.sh

### Build docker Image

For building the docker image execute:
```bash
docker build -t rsl_panoptic_seg -f Dockerfile .
```
To run the container and mount it on your home:
```bash
docker run --gpus all -it -e HOST_USERNAME=$(whoami) -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) -v /home/$(whoami):/home/$(whoami) rsl_panoptic_seg
```

### Running the Docker Image

For running the docker image we recommend the usage of the run.sh script from this [repository](https://github.com/leggedrobotics/m545_docker).

After building the image (or pulling it from docker hub), this can be done by typing
```bash
./bin/run.sh --type=gpu --tag=panoptic
```
## Deploying

### Download the pretrained model 

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">epochs</th>
<th valign="bottom">PQ</th>
<th valign="bottom">AP@0.5</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: detr_R50_bs2_200ep -->
 <tr><td align="left"><a href="deploy/panoptic_models/panoptic_models/detr/configs/config.py">Panoptic DETR</a></td>
<td align="center">R50</td>
<td align="center">200</td>
<td align="center">55.5</td>
<td align="center">76.5</td>
<td align="center">60.3</td>
<td align="center">47430278_4</td>
<td align="center"><a href="https://drive.google.com/file/d/1FHsLkN9JlOb2pmBVra96rAybEesiQCXS/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: panoptic_deeplab_R50_bs4_120000steps-->
 <tr><td align="left"><a href="deploy/panoptic_models/panoptic_models/PanopticNets.py">Panoptic DeepLab</a></td>
<td align="center">R50</td>
<td align="center">200</td>
<td align="center">62.0</td>
<td align="center">78.0</td>
<td align="center">68.9</td>
<td align="center">47430278_5</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1zaGx07VkxDUwZHnV2Eexwyn_JSyCNSHS?usp=sharing">model</a></td>
</tr>
</tbody></table>

The evaluation statistics are results from the validation set, which contains 40 customely labeled images.

### Building the Package
Inside the docker container, the image can be built using
```bash
catkin build panoptic_ros
```

### Launching
It can then be launched using
```bash
roslaunch panoptic_ros image_segmentation.launch
```

----------------------------------------------------

## Dependencies (if no docker image is used)

This repository uses python 3.9 and assumes you have installed on your machine CUDA 11.1 drivers. 
Plase use conda to install the dependencies: 
```bash
conda env create -f ros/src/conda/environment.yml
```
Then install the detectron model: 
```bash 
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
the panoptic segmentation api:
```bash
pip install git+https://github.com/cocodataset/panopticapi.git
```
the transformers models from huggingface:
```bash
pip install -q git+https://github.com/huggingface/transformers.git timm
```
