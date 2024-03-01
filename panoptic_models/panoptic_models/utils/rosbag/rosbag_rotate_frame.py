# ==
# Rotate the images from the rosbags
# ==

# import packages
import cv2
import os

from cv2 import rotate

# import scripts
from panoptic_models.mask2former.utils.logger import parse_arguments, logger_level
from panoptic_models.mask2former.utils.logger import _logger


def rotate_img(angle, img_path: str) -> None:
    """
    Rotate given image by either 90, 180, 270 degrees (clockwise)
    """
    # load image
    img = cv2.imread(img_path)

    # select angle and rotate
    if int(angle) == 90:
        img_rot = cv2.ROTATE_90_CLOCKWISE
    elif int(angle) == 180:
        img_rot = cv2.ROTATE_180
    elif int(angle) == 270:
        img_rot = cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        raise KeyError(
            f"Given angle {angle} not possible, either choose 90, 180 or 270 degrees (clockwise)"
        )
    img_rotated = cv2.rotate(img, img_rot)

    # save image
    cv2.imwrite(img_path, img_rotated)


def main(args):
    assert os.path.exists(args.path)

    if os.path.isfile(args.path):
        rotate_img(args.angle, args.path)
    elif os.path.isdir(args.path):
        [
            rotate_img(args.angle, os.path.join(args.path, f))
            for f in os.listdir(args.path)
            if (
                os.path.isfile(os.path.join(args.path, f))
                and f.endswith(("jpg", "png"))
            )
        ]

    _logger.info("DONE! Images are rotated!")


if __name__ == "__main__":
    parser = parse_arguments()
    parser.add_argument(
        "-p",
        "--path",
        default="mask2former/data/dataset_unlabeled/train/rosbag_heap1/frame002736.png",  # required=True,
        help="Path to image or directory with images",
    )
    parser.add_argument(
        "-a",
        "--angle",
        default=90,
        help="Rotation angle (can be either 90, 180, 270 degrees in clockwise direction)",
    )
    args = parser.parse_args()
    logger_level(args)

    main(args)
