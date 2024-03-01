"""
Loads an image and streams it at 10 hz
"""
import numpy as np
import cv2
import rospy
import sensor_msgs
import os
import cv_bridge

if __name__ == "__main__":
    # init node
    rospy.init_node("test_stream_node")
    # load image
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # get test image path
    # try to load image
    image = cv2.imread(cur_dir + "/test_image.jpg")
    # check if image is loaded
    if image is None:
        print("Error: Image not found")
        exit(1)
    # convert image to ROS message using cv_bridge
    bridge = cv_bridge.CvBridge()
    image_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")

    # publish image
    pub = rospy.Publisher("/camMainView/image_raw", sensor_msgs.msg.Image, queue_size=1)
    rate = rospy.Rate(40)
    while not rospy.is_shutdown():
        pub.publish(image_msg)
        rate.sleep()

    rospy.spin()
    cv2.destroyAllWindows()
    exit()
