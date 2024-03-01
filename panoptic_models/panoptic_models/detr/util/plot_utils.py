"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path, PurePath

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision.transforms.functional import to_tensor
import numpy as np
from panoptic_models.detr.util.box_ops import box_cxcywh_to_xywh_np
import os
import cv2


def plot_logs(
    logs,
    fields=("class_error", "loss_bbox_unscaled", "mAP"),
    ewm_col=0,
    log_name="log.txt",
):
    """
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    """
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(
                f"{func_name} info: logs param expects a list argument, converted to list[Path]."
            )
        else:
            raise ValueError(
                f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}"
            )

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(
                f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}"
            )
        if not dir.exists():
            raise ValueError(
                f"{func_name} - invalid directory in logs argument:\n{dir}"
            )
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == "mAP":
                coco_eval = (
                    pd.DataFrame(np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1])
                    .ewm(com=ewm_col)
                    .mean()
                )
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f"train_{field}", f"test_{field}"],
                    ax=axs[j],
                    color=[color] * 2,
                    style=["-", "--"],
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


def plot_precision_recall(files, naming_scheme="iter"):
    if naming_scheme == "exp_id":
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == "iter":
        names = [f.stem for f in files]
    else:
        raise ValueError(f"not supported {naming_scheme}")
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(
        files, sns.color_palette("Blues", n_colors=len(files)), names
    ):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data["precision"]
        recall = data["params"].recThrs
        scores = data["scores"]
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data["recall"][0, :, 0, -1].mean()
        print(
            f"{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, "
            + f"score={scores.mean():0.3f}, "
            + f"f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}"
        )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title("Precision / Recall")
    axs[0].legend(names)
    axs[1].set_title("Scores / Recall")
    axs[1].legend(names)
    return fig, axs


def plot_bounding_box(image, bbox, box_label, image_label):
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # as title use the image name
    ax.set_title(image_label)
    # Display the image
    ax.imshow(image)

    # Create a Rectangle patch
    # transform the box from (cx, cy, x, y) to (x, y, width, height)
    # box_xyxy = box_cxcywh_to_xyxy(bbox)
    # rect = patches.Rectangle((box_xyxy[0], box_xyxy[1]), box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1], linewidth=1, edgecolor='r', facecolor='none')

    box_xyxy = bbox
    rect = patches.Rectangle(
        (box_xyxy[0], box_xyxy[1]),
        box_xyxy[2],
        box_xyxy[3],
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )

    # Add the patch to the Axes
    ax.add_patch(rect)
    ax.text(bbox[0], bbox[1], box_label, color="red")

    plt.show()


def plot_bounding_boxes(
    image, orig_size, boxes, boxes_labels, image_label, save_path=None, normalized=True
):
    """
    Args:
        image: np.array (width, heigth, 3)
        boxes (list of size) [num_boxes, 4]: style cxcywh
        boxes_labels (list of size [num_boxes]): label of the object
    """
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # reverse normalization with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225] across the channels
    # using np
    if normalized:
        image = image.transpose((2, 0, 1))
        image = (
            image * np.array([0.229, 0.224, 0.225])[:, None, None]
            + np.array([0.485, 0.456, 0.406])[:, None, None]
        )
        image = image.transpose((1, 2, 0))
    # resizes the image to the original size
    size = (int(800 * orig_size[0] / orig_size[1]), 800)
    # Display the image
    ax.imshow(image)
    # flip original size to (width, height)
    # Create a Rectangle patch
    for box, box_label in zip(boxes, boxes_labels):
        # transform the box from (cx, cy, w, h) to (x, y, width, height)
        # resize to image size
        if normalized:
            # box_xywh = box_cxcywh_to_xywh_np(box)
            box_xywh = box
            box_xywh = box_xywh * np.array(
                [orig_size[1], orig_size[0], orig_size[1], orig_size[0]]
            )
        else:
            box_xywh = box
        rect = patches.Rectangle(
            (box_xywh[0], box_xywh[1]),
            box_xywh[2],
            box_xywh[3],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.text(box_xywh[0], box_xywh[1], box_label, color="red")

    if save_path:
        # get module path
        module_path = os.path.dirname(os.path.abspath(__file__)) + "/../../../"
        path = os.path.join(module_path, save_path)
        plt.savefig(path + image_label + ".png")
    else:
        plt.show()


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return b
