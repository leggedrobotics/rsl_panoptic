# import packages
import numpy as np
import cv2
import random
import os

# import scripts
from panoptic_models.mask2former.utils.logger import _logger

# import parameters
from panoptic_models.panoptic_models.config.params_data import (
    FRAMES_PER_VIDEO,
    MAX_FRAME_NB,
)

FRAME_RATE = 100


class FrameExtractor:
    """
    Class to store the frames of all videos together with the dict of their additional information
    """

    def __init__(self, save_path: str):
        self.path = save_path
        self.total_frame_nb: int = 0

        # create path where images are saved if it does not exists
        if not os.path.exists(os.path.join(os.getcwd(), self.path)):
            os.makedirs(os.path.join(os.getcwd(), self.path))

    def extract_frames(self, video_info, video_path: str):
        """
        Method used for extracting the frames from the videos
        """
        # open video
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        frame_count = 0

        if not success:
            _logger.warning(f"Video {video_path} could not be opened!")
            raise KeyError("Open video failed!")

        # get frames which should be saved
        video_frame_nbr = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_list = random.sample(range(0, video_frame_nbr), FRAMES_PER_VIDEO)

        # make new subdirectory for video frames
        os.makedirs(f"/{self.path}/{video_info.video_id}/", exist_ok=True)

        # save frames as jpg images
        while success:
            if frame_count in frame_list:
                cv2.imwrite(
                    f"/{self.path}/{video_info.video_id}/{video_info.video_id}_frame{frame_count}.jpg",
                    image,
                )
            success, image = vidcap.read()
            frame_count += 1

        # release video
        vidcap.release()
        cv2.destroyAllWindows()

        # update total frame number
        self.total_frame_nb += FRAMES_PER_VIDEO

        _logger.info(
            f"Video: {video_info.video_id}: {FRAMES_PER_VIDEO} frames extracted"
        )

    def max_reached(self) -> bool:
        if MAX_FRAME_NB and self.total_frame_nb > MAX_FRAME_NB:
            _logger.info("Max number of frames reached!")
            return True
        else:
            return False
