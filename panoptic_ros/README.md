# ROS Panoptic Segmentation Node

## Overview
This project integrates advanced panoptic segmentation models into a ROS (Robot Operating System) node, enabling real-time segmentation of images from a ROS-enabled camera. The core functionality is wrapped in a Python class, `PanopticRos`, which subscribes to camera topics, processes the images through the segmentation model, and publishes the segmented images and masks.

## Features
- **Integration with ROS**: Seamlessly subscribe to image topics and publish segmented images and masks.
- **Support for Multiple Models**: Configurable to use different segmentation models such as DETR, Mask2Former, and potentially others by setting ROS parameters.
- **Real-Time Visualization**: Optional visualization of segmentation results and labels directly from the ROS environment.

## Installation
We reccomend using the docker container provided, else check the dependencies necessary in the official repos of DETR and Mask2Former


## Launch
Start the container and:
```
roslaunch panoptic_ros image_segmentation.launch
```

## Parameters
- `image_topic` (String, default: `/camera/rgb/image_raw`): The topic from which to subscribe to input images.
- `compressed_image` (Bool, default: `True`): Whether to subscribe to a compressed image topic.
- `show_segmentation` (Bool, default: `False`): Enable/disable visualization of segmentation.
- `show_labels` (Bool, default: `False`): Enable/disable visualization of labels.
- `net` (String, required): The model to use (`DETR`, `Mask2Former`, or other supported models).

## Topics Published
- `/src/image` (sensor_msgs/Image): The segmented image.
- `/src/mask` (sensor_msgs/Image): The segmentation mask.

## Customization
- Implement additional segmentation models by extending the `panoptic_models.PanopticNets.PanopticNet` class and integrating them into the `PanopticRos` class.
- Adjust the image preprocessing and postprocessing steps within the callback functions to fit your specific requirements.

## License
This project is open-source and available under the MIT license.
