# ==
# Script to evaluate model performance over training iterations on the test set
# ==

# import packages
import argparse
import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# necessary to run it on the cluster
matplotlib.use("Agg")

import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import time

import numpy as np
import tqdm

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from detectron2.evaluation import verify_results


from panoptic_models.mask2former.third_party.Mask2Former.mask2former import (
    add_maskformer2_config,
)
from panoptic_models.mask2former.third_party.Mask2Former.demo.predictor import (
    VisualizationDemo,
)

# register datasets
from panoptic_models.data.register_datasets import DatasetRegister
from panoptic_models.panoptic_models.config.params_data import COCO_DST, SEGMENTS_DST
from panoptic_models.mask2former.m2f_deploy.m2f_train import Trainer


def get_parser():
    parser = argparse.ArgumentParser(description="m2f evaluation parser")
    parser.add_argument(
        "--config-file",
        default="mask2former/models/mask2former_backbone/swin_t/maskformer2_swin_tiny_bs1_small.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--model-dir",
        default="output/m2f_swin_tiny_backbone_mae-cat-conti_seg_255_coco_50000_bs1_fine_seg_255_coco_1250",
        help="the directory containing the different model weights",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Bool whether visualization is performed (As deault 5 images, to change use --nbr-images)",
    )
    parser.add_argument(
        "--nbr-images",
        type=int,
        default=5,
        help="Number of images to sample from test dataset and run visualization on",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mask2former/data/dataset_labeled_seg_050_coco_050",
        help="Path to dataset (expected folders: path/coco and path/segment), "
        "if None COCO_DST and SEGMENT_DST of config/params.py used",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--caterpillar",
        type=str,
        default="mask2former/data/caterpillar",
        help="Path to caterpillar dataset (for evaluation)",  # Adjustment for the given project
    )
    return parser


def setup_cfg(args, model_weights):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    print("------------------------------------------")
    print(model_weights)
    cfg.merge_from_list(
        ["MODEL.WEIGHTS", f"{os.path.join(args.model_dir, model_weights)}"]
    )
    cfg.DATASETS.TEST = cfg.DATASETS.TEST + ("construction_site_test",)
    cfg.freeze()
    return cfg


def get_scores(model_weights, args):
    # setup cfg and add model_weights
    cfg = setup_cfg(args, model_weights)

    # build model and get scores on test set
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    res = Trainer.test(cfg, model)
    if cfg.TEST.AUG.ENABLED:
        res.update(Trainer.test_with_TTA(cfg, model))
    if comm.is_main_process():
        verify_results(cfg, res)

    return {model_weights: res}, cfg


def process_scores(scores: dict, cfg, path) -> dict:
    """
    Save scores as dict of DataFrames
    """
    idx_names = ["PQ", "SQ", "RQ", "PQ_th", "SQ_th", "RQ_th", "PQ_st", "SQ_st", "RQ_st"]
    df_metrics = {
        dataset_single: pd.DataFrame(index=idx_names)
        for dataset_single in cfg.DATASETS.TEST
    }

    for metrices in scores:
        for model, dataset in metrices[0].items():
            name = (
                int((model[len("model_") : -len(".pth")]))
                if not model == "model_final.pth"
                else cfg.SOLVER.MAX_ITER
            )
            for dataset_name, values in dataset.items():
                df_metrics[dataset_name][name] = values["panoptic_seg"].values()

    [
        df_metrics[dataset_name].to_csv(
            os.path.join(path, f"eval_metric_{dataset_name}.csv")
        )
        for dataset_name in df_metrics.keys()
    ]
    return df_metrics


def get_dataset_name(dataset: str) -> str:
    """
    Adaptation to current project, assign names to certain datasets
    """
    if dataset == "eval_metric_caterpillar.csv":
        dataset_name = "CATERPILLAR Dataset"
    elif dataset == "eval_metric_coco_2017_val_red.csv":
        dataset_name = "COCO Validation Dataset"
    elif dataset == "eval_metric_construction_site_test.csv":
        dataset_name = "CONSTRUCTION Test Dataset"
    elif dataset == "eval_metric_construction_site_val.csv":
        dataset_name = "CONSTRUCTION Validation Dataset"
    else:
        dataset_name = dataset
    return dataset_name


def plot_scores(scores, cfg, path) -> None:
    """
    Plot metrics
    """
    # create subplot
    fig, axs = plt.subplots(3, 3, sharey="row", figsize=(12, 12), dpi=160)
    fig.suptitle(f"{cfg.MODEL.WEIGHTS.split('/')[1]}", fontsize=16, y=1.04)
    color_ar = ["r", "g", "b", "m", "c"]

    for i, dataset in enumerate(scores.keys()):
        dataset_name = get_dataset_name(dataset)
        x_values = scores[dataset].columns
        y_names = scores[dataset].index
        scores_np = scores[dataset].to_numpy()

        # sort array in case that models are not read in the right order
        x_values = np.array([int(x) for x in x_values])
        sort_idx = sorted(range(len(x_values)), key=lambda k: x_values[k])
        x_values = x_values[sort_idx]
        scores_np = scores_np[:, sort_idx]

        axs[0, 0].set_title(y_names[0])
        axs[0, 0].plot(x_values, scores_np[0, :], color=color_ar[i], label=dataset_name)
        axs[0, 1].set_title(y_names[1])
        axs[0, 1].plot(x_values, scores_np[1, :], color=color_ar[i], label=dataset_name)
        axs[0, 2].set_title(y_names[2])
        axs[0, 2].plot(x_values, scores_np[2, :], color=color_ar[i], label=dataset_name)
        axs[1, 0].set_title(y_names[3])
        axs[1, 0].plot(x_values, scores_np[3, :], color=color_ar[i], label=dataset_name)
        axs[1, 1].set_title(y_names[4])
        axs[1, 1].plot(x_values, scores_np[4, :], color=color_ar[i], label=dataset_name)
        axs[1, 2].set_title(y_names[5])
        axs[1, 2].plot(x_values, scores_np[5, :], color=color_ar[i], label=dataset_name)
        axs[2, 0].set_title(y_names[6])
        axs[2, 0].ticklabel_format(axis="x", style="sci")
        axs[2, 0].plot(x_values, scores_np[6, :], color=color_ar[i], label=dataset_name)
        axs[2, 1].set_title(y_names[7])
        axs[2, 1].ticklabel_format(axis="x", style="sci")
        axs[2, 1].plot(x_values, scores_np[7, :], color=color_ar[i], label=dataset_name)
        axs[2, 2].set_title(y_names[8])
        axs[2, 2].ticklabel_format(axis="x", style="sci")
        axs[2, 2].plot(x_values, scores_np[8, :], color=color_ar[i], label=dataset_name)

    fig.subplots_adjust(bottom=0.3, wspace=0.33)
    axs[2, 1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fancybox=False,
        shadow=False,
        ncol=1,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(path, "scores_dataset_comp.png"), bbox_inches="tight")
    plt.close()
    logger.info(
        f"Scores plot saved under {os.path.join(path, 'scores_dataset_comp.png')}"
    )


def vis_images(cfg, nbr_images, path, random_seed: int = 1):
    """
    Visualize images
    """

    # get information of the test set
    from detectron2.data import DatasetCatalog

    dataset = DatasetCatalog.get("construction_site_test")
    random.seed(random_seed)
    dataset = random.choices(dataset, k=nbr_images)
    # inti demonstration class
    demo = VisualizationDemo(cfg)
    # make dir for output visualization images
    save_dir = os.path.join(path, f"{(cfg.MODEL.WEIGHTS.split('/')[2])[:-len('.pth')]}")
    os.makedirs(save_dir, exist_ok=True)

    for data_single in tqdm.tqdm(dataset):
        # use PIL, to be consistent with evaluation
        img = read_image(data_single["file_name"], format="BGR")
        start_time = time.time()
        print("Image Shape:")
        print(img.shape)
        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: {} in {:.2f}s".format(
                data_single["file_name"],
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )
        out_filename = os.path.join(
            save_dir, os.path.basename(data_single["file_name"])
        )

        filename, file_extension = os.path.splitext(out_filename)
        if file_extension == ".jpg":
            out_filename = filename + ".png"

        visualized_output.save(out_filename)


def main(args):
    # register datasets
    if args.dataset:
        ds_coco = os.path.join(args.dataset, "coco/")
        ds_segment = os.path.join(args.dataset, "segments/")
        assert os.path.isdir(ds_coco), f"Expected Dataset {ds_coco} does not exist!"
        assert os.path.isdir(
            ds_segment
        ), f"Expected Dataset {ds_segment} does not exist!"
        registerer = DatasetRegister(ds_coco, ds_segment, args.caterpillar)
    else:
        registerer = DatasetRegister(COCO_DST, SEGMENTS_DST, args.caterpillar)
    registerer.get_all_datasets()
    print("New datasets added!")

    # get all models included in args.model_dir
    files = os.listdir(args.model_dir)
    model_weights = [file for file in files if file.endswith("pth")]

    # check if evaluation already has been performed
    csv_files = [file for file in os.listdir(args.model_dir) if file.endswith(".csv")]
    if csv_files:
        df_metrics = {
            csv_path_single[len("eval_metric_") : -len(".csv")]: pd.read_csv(
                os.path.join(args.model_dir, csv_path_single), index_col=0
            )
            for csv_path_single in csv_files
        }
        cfg_single = setup_cfg(args, model_weights[0])
        logger.info(
            "Evaluation already performed, metric results loaded from csv files"
        )
    else:
        # evaluate models to get scores
        scores = [get_scores(weights, args) for weights in model_weights]
        # process scores
        df_metrics = process_scores(scores, scores[0][1], args.model_dir)
        # get sample cfg data
        cfg_single = scores[0][1]
        logger.info("Evaluation performed, scores saved!")

    # plot scores
    plot_scores(df_metrics, cfg_single, args.model_dir)

    # if output dir included, select some frames of the test set and visualize their output
    if args.vis:
        [
            vis_images(cfg_single, args.nbr_images, args.model_dir)
            for _, cfg_single in scores
        ]


if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    main(args)
