########
# Scrape data from a youtube channel to get further unsupervised
########

# import packages
from pytube import YouTube
import os
import pickle
import http

# import scripts
from panoptic_models.mask2former.utils.logger import _logger
from panoptic_models.mask2former.utils.logger import parse_arguments, logger_level
from panoptic_models.data.utils.yt_data.yt_stats import YTstats
from panoptic_models.data.utils.yt_data.yt_frame_extractor import FrameExtractor

# import parameters
from panoptic_models.panoptic_models.config.params_data import (
    API_KEY,
    CHANNEL_ID,
    PATH_YT_DATA,
    REPO_PATH,
)


def main() -> None:
    # get the video IDs for all videos of the channel
    try:
        # load entire video data
        with open(f"{REPO_PATH}/data/utils/yt_data/video_data.pkl", "rb") as file:
            video_list = pickle.load(file)
            _logger.info("Channel video data loaded from file")
    except FileNotFoundError:
        yt = YTstats(API_KEY, channel_id=CHANNEL_ID)
        video_list = yt.get_channel_video_data()
        _logger.info("Channel video data acquired")
        # save video data to be loaded later
        with open(f"{REPO_PATH}/data/utils/yt_data/video_data.pkl", "wb") as file:
            pickle.dump(video_list, file)
            _logger.debug("Information for all videos on the channel saved")

    # video filter according to tracks
    tag_list = ["house", "track", "street", "truck"]
    video_list_red: list = []
    tag_filter_counter = 0
    for key, items in video_list.items():
        if "tags" in items.keys() and any(tag in items["tags"] for tag in tag_list):
            video_list_red.append(key)
            tag_filter_counter += 1
    _logger.debug(
        f"{tag_filter_counter} videos filtered that do not contain any of the following tags {tag_list}"
    )

    # initialize class that summarizes all frames
    frames = FrameExtractor(save_path=PATH_YT_DATA)

    # for each video perform
    # video_list_red.insert(0, '-VDVzMFdz7s')  # append test video

    # other great videos
    # video_list_red.insert(0, 'bufZh2CKT0Q')

    for single_video in video_list_red:

        if frames.max_reached():
            break

        try:
            video = YouTube(f"https://www.youtube.com/watch?v={single_video}")
            video_path = video.streams.get_highest_resolution().download()
            frames.extract_frames(video, video_path)
            os.remove(video_path)
        except http.client.IncompleteRead:
            _logger.info(
                f"Video {single_video} not included, leads to IncompleteRead error"
            )


if __name__ == "__main__":
    args = parse_arguments().parse_args()
    # define logger level
    logger_level(args)
    # start youtube scrape
    main()
