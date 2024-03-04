import rosbag
import cv2
import os
import numpy as np

topic_name = "/camMainView/image_raw/compressed"
start_time_relative = (
    0.0  # Relative start time in seconds from the beginning of the bag
)
image_frequency = 1
rotation_angle = 90  # Default rotation angle (90 degrees counterclockwise)
bag_file = "heap_camera_2023-02-21-17-06-47.bag"
bag_path = "/media/lorenzo/ssd11/bags" + "/" + bag_file

# Ensure the output directory exists
output_dir = "extracted_" + bag_file.split(".")[0]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def compressed_imgmsg_to_cv2(compressed_img_msg):
    """
    Convert a compressed image message to a CV2 image.

    Args:
        compressed_img_msg: The compressed image message to convert.

    Returns:
        A numpy array representing the CV2 image.
    """
    np_arr = np.fromstring(compressed_img_msg.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode as a color image
    return cv_image


def rotate_image(image, angle):
    """
    Rotate an image by a given angle counterclockwise.

    Args:
        image: The image to rotate.
        angle: The angle by which to rotate the image (90, 180, 270 degrees).

    Returns:
        The rotated image.
    """
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        return image  # If the angle is not 90, 180, or 270, return the image as is


def extract_images(
    bag_path,
    topic_name,
    start_time_relative,
    image_frequency,
    rotation_angle,
    output_dir,
):
    """
    Extract and optionally rotate images from a ROS bag file at a specific frequency.
    """
    bag = rosbag.Bag(bag_path, "r")
    count = 0
    saved_count = 0
    time_skip = 1.0 / image_frequency
    start_time_absolute = None  # Absolute start time in the bag's timeline

    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        if start_time_absolute is None:
            start_time_absolute = (
                t.to_sec() + start_time_relative
            )  # Update start time based on the first message

        if t.to_sec() < start_time_absolute:
            continue

        if t.to_sec() >= start_time_absolute + (saved_count * time_skip):
            try:
                cv_image = compressed_imgmsg_to_cv2(msg)
                if rotation_angle in [90, 180, 270]:
                    cv_image = rotate_image(cv_image, rotation_angle)
            except Exception as e:
                print(f"Error converting or rotating image: {e}")
                continue

            cv2.imwrite(f"{output_dir}/image_{saved_count:04d}.png", cv_image)
            print(f"Saved image_{saved_count:04d}.png")
            saved_count += 1

        count += 1

    print(f"Finished extracting images. {saved_count} images were saved.")
    bag.close()


if __name__ == "__main__":
    extract_images(
        bag_path,
        topic_name,
        start_time_relative,
        image_frequency,
        rotation_angle,
        output_dir,
    )
