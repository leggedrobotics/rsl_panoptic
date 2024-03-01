# Unlabeled Test Construction Sites 

To use samples from test construction sites, the corresponding rosbags have to be downloaded. For access please contact the authors of this work. 

## Data Generation
When the rosbags have been downloaded, execute the following steps:

1. *Image Extraction*: run the following script to extract images from rosbags:
    ```
    python3 rosbag_frame_extractor.py -bf BAG-FILE -o OUTPUT-DIR -t IMAGE-TOPIC -n NUMBER-IMAGES -c COMPRESSED 
    ```
    If the intended number of images is not defined, all images in the rosbag will be extracted. The selection of images is random. The default for compressed images is False. 

2. *Image Rotation*: Images have to be rotated s.t. the excavator is upright. If that is not the case, images can be rotated by calling
   ```
   python3 rosbag_rotate_frame.py -p PATH-TO-IMAGE(-DIR) -a ROTATION-ANGLE
   ```

3. *Move Images': Manually move the image directories to the unlabeled dataset folder