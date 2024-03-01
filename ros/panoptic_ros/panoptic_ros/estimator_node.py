# this files contain the ros components of the panoptic segmentation
import numpy as np
import cv2

import rospy
from rospy.exceptions import ROSException
import sensor_msgs
import cv_bridge
import panoptic_models.PanopticNets
import panoptic_models
import gdown
import os
import rospkg


class PanopticRos(object):
    def __init__(
        self,
        panoptic_net: panoptic_models.PanopticNets.PanopticNet,
        node_name: str = "ros",
    ):
        # initilize ros node
        rospy.init_node(node_name)
        # initialize ros listener with param image topic
        image_sub_topic = rospy.get_param("image_topic", "/camera/rgb/image_raw")
        self.compressed_image = rospy.get_param("compressed_image", True)
        if self.compressed_image:
            rospy.Subscriber(
                image_sub_topic,
                sensor_msgs.msg.CompressedImage,
                self.compressed_image_callback,
                queue_size=1,
                buff_size=2**24,
            )
        else:
            rospy.Subscriber(
                image_sub_topic,
                sensor_msgs.msg.Image,
                self.image_callback,
                queue_size=1,
                buff_size=2**24,
            )
        # publisher of segmented image and mask
        image_topic = rospy.get_param("topics/publish/seg_topic_image", "/src/image")
        mask_topic = rospy.get_param("topics/publish/seg_topic_mask", "/src/mask")

        # visualization flag
        self.vis_panoptic_seg = rospy.get_param("show_segmentation", False)
        self.vis_label = rospy.get_param("show_labels", False)

        self.image_pub = rospy.Publisher(
            image_topic, sensor_msgs.msg.Image, queue_size=1
        )
        self.mask_pub = rospy.Publisher(mask_topic, sensor_msgs.msg.Image, queue_size=1)
        # else:
        # self.mask_pub = rospy.Publisher(mask_topic, PanopticMask, queue_size=1)
        test_image_topic = "test_image"
        self.test_pub = rospy.Publisher(
            test_image_topic, sensor_msgs.msg.Image, queue_size=1
        )
        # initialize cv bridge
        # ros bridge to convert ros image to opencv images
        self.bridge = cv_bridge.CvBridge()
        # panoptic segmentation network
        self.panoptic_net = panoptic_net

        print("-----------------------------------------------")
        print("Model is fully initialized, ready to go...")

    def compressed_image_callback(self, ros_image: sensor_msgs.msg.CompressedImage):
        """Publishes a segmented version of the image

        Receives the ros image, converts it to opencv image,
        passes it to the panoptic segmentation network,
        converts the output to an image, and publishes it again as a ros image.

        Args:
            ros_image (sensor_msgs.msg.Image)
        """
        try:
            np_arr = np.fromstring(ros_image.data, np.uint8)
            cv2_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # rotate image 90 degrees coutner clockwise
            cv2_image = cv2.rotate(cv2_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except cv_bridge.CvBridgeError as e:
            print(e)
        self.image_cv2_callback(cv2_image, header=ros_image.header)

    def image_callback(self, ros_image: sensor_msgs.msg.Image):
        try:
            cv2_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
            # rotate image 90 degrees coutner clockwise
            cv2_image = cv2.rotate(cv2_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except cv_bridge.CvBridgeError as e:
            print(e)

        self.image_cv2_callback(cv2_image, header=ros_image.header)

    def image_cv2_callback(self, cv2_image, header=None):
        # segment the image
        if self.vis_panoptic_seg:
            panoptic_image, panoptic_mask = self.panoptic_net(
                cv2_image, vis_seg=self.vis_panoptic_seg, vis_label=self.vis_label
            )
            print("type of panoptic image: ", type(panoptic_image))
            # covert src to ROS image
            panoptic_image_np = np.array(panoptic_image.get_image())

            ros_seg_image = self.bridge.cv2_to_imgmsg(panoptic_image_np, "rgb8")
            ros_seg_mask = self.bridge.cv2_to_imgmsg(panoptic_mask, "bgr8")
            #     # add the header to the ros image
            ros_seg_image.header = header
            ros_seg_mask.header = header
            self.image_pub.publish(ros_seg_image)
            self.mask_pub.publish(ros_seg_mask)
        else:
            panoptic_mask = self.panoptic_net(cv2_image, vis_seg=self.vis_panoptic_seg)
            # covert to rgb image from bgr
            panoptic_mask = cv2.cvtColor(panoptic_mask, cv2.COLOR_BGR2RGB)
            ros_seg_mask = self.bridge.cv2_to_imgmsg(panoptic_mask, "rgb8")
            ros_seg_mask.header = header
            self.mask_pub.publish(ros_seg_mask)


def check_weights_and_download(path_to_weights: str, weights_name: str, url: str):
    if not os.path.exists(path_to_weights):
        os.makedirs(path_to_weights)
    # if weights not in path_to_weights, download them
    if not os.path.exists(os.path.join(path_to_weights, weights_name)):
        print("Downloading weights...")
        # download file from gdrive
        gdown.download(url, os.path.join(path_to_weights, weights_name))


if __name__ == "__main__":
    try:
        net_name = rospy.get_param("net", "PanopticDeepLab")

    except ROSException:
        print("Please set the parameters in the launch file")
        exit(1)

    rospack = rospkg.RosPack()
    path_to_package = rospack.get_path("panoptic_models")

    if net_name == "DETR":
        # check if the weights are in the folder panoptic_models/detr/weights of the ros package panoptic_models
        # if not, download them from https://drive.google.com/file/d/1FHsLkN9JlOb2pmBVra96rAybEesiQCXS/view?usp=sharing
        # find path of package panoptic_models with rospack
        path_to_weights = os.path.join(path_to_package, "panoptic_models/detr/weights")
        weights_name = "detr_50_panoptic_construction_finetuned_200_epoch_200"
        url = "https://drive.google.com/file/d/1LPnB3d1yzKzbnO900iDSsAkRvr-3tFkU"
        check_weights_and_download(
            path_to_weights=path_to_weights, weights_name=weights_name, url=url
        )
        panoptic_net = panoptic_models.PanopticNets.DETR(
            config=panoptic_models.detr.configs.config
        )

    elif net_name == "Mask2Former":
        path_to_weights = os.path.join(
            path_to_package,
            "panoptic_models/mask2former/weights/mask2former_proj_heads/swin_t",
        )
        weights_name = "mask2former_finetuned.pth"
        # weights_name = 'model_0054999.pth'
        # weights_name = 'model_final_9fd0ae.pkl'
        url = "https://drive.google.com/file/d/1L8rWE5RPTC0mqB_JaYEM1GO1tcD0eqK0"
        check_weights_and_download(
            path_to_weights=path_to_weights, weights_name=weights_name, url=url
        )
        # Import
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from panoptic_models.mask2former.third_party.Mask2Former.mask2former import (
            add_maskformer2_config,
        )

        # Build up config
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        config_file = os.path.join(
            path_to_package,
            "panoptic_models/mask2former/config/mask2former_proj_heads/swin_t"
            "/maskformer2_swin_tiny_bs16_50ep.yaml",
        )
        print(config_file)
        cfg.merge_from_file(config_file)
        weights_path = os.path.join(path_to_weights, weights_name)
        import torch

        print("Model weights: ", weights_path)
        torch.load(weights_path, map_location="cpu")
        opts = ["MODEL.WEIGHTS", weights_path]
        # print model weights
        print("Model weights: ", opts[1])
        cfg.merge_from_list(opts)
        # cfg.MODEL.DEVICE = "cpu"
        # Metadata
        from panoptic_models.data.register_datasets import DatasetRegister
        from panoptic_models.mask2former.config.params_data import (
            COCO_DST,
            SEGMENTS_DST,
        )

        # register datasets
        registerer = DatasetRegister(COCO_DST, SEGMENTS_DST)
        registerer.get_all_datasets()
        print("New datasets added!")
        print("------------------------------------")
        print("Loaded configs for Mask2Former.")
        panoptic_net = panoptic_models.PanopticNets.Mask2Former(config=cfg)

    else:
        print(
            'Please set the net parameter to "PanopticDeepLab", "DETR" or "Mask2Former"'
        )
        exit(1)

    panoptic_ros = PanopticRos(panoptic_net)
    rospy.spin()
