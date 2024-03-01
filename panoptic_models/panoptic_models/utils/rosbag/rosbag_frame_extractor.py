"""
Extract images from a rosbag.
"""

# import packages
import random
import os
import cv2
import numpy as np

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# import scripts
from panoptic_models.mask2former.utils.logger import logger_level, parse_arguments
from panoptic_models.mask2former.utils.logger import _logger

# import parameters
from panoptic_models.panoptic_models.config.params_data import RANDOM_SEED

random.seed(RANDOM_SEED)


def main(args):
    """
    Extract a folder of images from a rosbag.
    """
    _logger.info(
        "Extract images from %s on topic %s into %s"
        % (args.bag_file, args.image_topic, args.output_dir)
    )

    bag = rosbag.Bag(args.bag_file, "r")
    # bridge = CvBridge()  # bridge is not used

    # if total number of images defined, select them randomly from rosbag
    message_count = bag.get_message_count(args.image_topic)
    if args.nb_images:
        assert (
            message_count > args.nb_images
        ), f"Should select {args.nb_images} but only total of {message_count} included in bag!"
        image_idx = random.sample(range(message_count), args.nb_images)
    else:
        image_idx = range(message_count)

    count = 0
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
        if t.to_sec() < args.time:
            count += 1
            continue
        if count not in image_idx:
            count += 1
            continue
        if args.compressed:
            im = np.fromstring(msg.data, np.uint8)
            cv_img = cv2.imdecode(im, cv2.IMREAD_COLOR)
        else:
            im = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1
            )
            cv_img = cv2.cvtColor(im, cv2.IMREAD_COLOR)
            # cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr16")

        _logger.debug(f"Wrote image {count}")
        cv2.imwrite(os.path.join(args.output_dir, "frame%06i.png" % count), cv_img)
        count += 1

    bag.close()

    return


if __name__ == "__main__":
    # parse args
    parser = parse_arguments()
    parser.add_argument(
        "-bf",
        "--bag_file",
        default="mask2former/data/utils/rosbag/rosbag_data/mapping.bag",  # required=True,
        help="Input ROS bag.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="mask2former/dataset_unlabeled/train/rosbag_heap1",  # required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "-t",
        "--image_topic",
        default="/camMainView/image",  # required=True,
        help="Image topic.",
    )
    parser.add_argument(
        "-n",
        "--nb_images",
        type=int,
        default=10000,
        help="Total number of image extracted from the ROS bag",
    )
    parser.add_argument(
        "-c",
        "--compressed",
        action="store_true",
        help="Compressed images within rosbag",
    )
    parser.add_argument(
        "-ti", "--time", default=0.0, type=float, help="Start time of rosbag"
    )
    args = parser.parse_args()
    # change logger level
    logger_level(args)

    main(args)
