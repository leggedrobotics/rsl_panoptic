from typing import Optional
import os

REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(REPO_PATH)
RANDOM_SEED = 0

###
# File to store all parameters related to the data extraction
###

# Parameters of the YouTube scrape
API_KEY: str = "AIzaSyD9rL8RWCIljnqtgaNfLZf2vP8xAayVJTo"  # personal API Key, used to access the information of the channel
# CHANNEL_ID: str = 'UCXuY8udERTCYNHWQfw-CbJA' # channel ID of Hedblom
CHANNEL_ID: str = "UC-NlyhOdTK4C30NhptvD7UA"  # channel ID of LetsDig18
FRAMES_PER_VIDEO: int = 500
MAX_FRAME_NB: Optional[
    int
] = 30000  # if None, frames will be saved until all videos are handeled
PATH_YT_DATA: str = REPO_PATH + "/data/utils/yt_data/images_letsdig"

# Parameters to Download the segments.ai labeled dataset
SEGMENTS_SRC = REPO_PATH + "/data/"
SEGMENTS_DST = REPO_PATH + "/data/dataset_v8/segments"
META_FILENAME = SEGMENTS_SRC + "construction_site-v8.0.json"
VERSION = "v8.0"
IMAGE_PATH = SEGMENTS_SRC + VERSION + "/train/"
SEGMENTS_INFO_PATH = SEGMENTS_SRC + VERSION + "/annotations/segments_info_train.json"
ANNOTATION_DIR = SEGMENTS_SRC + VERSION + "/panoptic_train/"
SEGMENTS_VAL_SIZE = 0.15
SEGMENTS_TEST_SIZE = 0
SEGMENTS_MAX_SAMPLES = None
CATERPILLAR = False

# Parameters to adjust coco dataset
DATASET_SPLIT = "train"  # train or val

COCO_SRC = REPO_PATH + "/data/utils/coco/coco_dataset/"
COCO_DST = REPO_PATH + "/data/dataset_labeled/coco/"

COCO_META_PATH = COCO_SRC + f"annotations/panoptic_{DATASET_SPLIT}2017.json"
COCO_INSTANCE_PATH = (
    COCO_SRC + f"annotations/instances_{DATASET_SPLIT}2017.json"
    if DATASET_SPLIT == "val"
    else None
)

SELECTED_FILENAME_PATH = COCO_SRC + f"selected_filenames_{DATASET_SPLIT}.json"

OUTPUT_PATH_META = (
    COCO_DST + f"annotations/panoptic_{DATASET_SPLIT}2017.json"
)  # /path/to/selected_filenames_train.json
OUTPUT_PATH_INSTANCE = (
    COCO_DST + f"annotations/instances_{DATASET_SPLIT}2017.json"
    if DATASET_SPLIT == "val"
    else None
)

# Whether to apply ignore labels to crowd pixels in panoptic label
TREAT_CROWD_AS_IGNORE = True
# Each selected image contains at least $(num_categories_at_least) categories we need
NUM_CATEGORIES_AT_LEAST = 5 if DATASET_SPLIT == "train" else 15
# Number of images to select from COCO for training
NUM_IMAGES_SELECTED = 50000 if DATASET_SPLIT == "train" else 500
# If META data should be mapped to the reduced category set
# (default False, only True to compare model performance against original m2f implementation)
COCO_M2F_ORIGINAL_PERFORMANCE_EVAL = True
