# ==
# Plot evaluation metrics comparison between multiple runs
# ==

# import packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
from typing import List

# necessary to run it on the cluster
matplotlib.use("Agg")

# import scripts
from panoptic_models.mask2former.utils.logger import _logger


def get_parser():
    parser = argparse.ArgumentParser(description="m2f evaluation parser")
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        default=None,
        help="directories containing the different evaluation metric results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/metric_comp",
        help="directory where output files should be saved",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of Plot (otherwise dataset + all model names)",
    )
    return parser


def get_scores(model_dirs: List[str]) -> dict:
    """
    Load scores from the single evaluated models
    """
    metrics = {}
    for model_dir in model_dirs:
        # list all files in folder and filter for csv files, raise error if non are found
        metric_path = [file for file in os.listdir(model_dir) if file.endswith(".csv")]
        assert (
            len(metric_path) != 0
        ), f"No csv files found in {model_dir}. Run m2f_eval.py for the models in this directory first!"
        basename = os.path.basename(model_dir)
        metrics[basename] = {
            metric_path_single: pd.read_csv(
                os.path.join(model_dir, metric_path_single), index_col=0
            )
            for metric_path_single in metric_path
        }
        _logger.info(f"metrices from {model_dir} loaded")
    return metrics


def get_dataset_name(dataset: str) -> str:
    """
    Adaptation to current project, assign certain names for datasets
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


def get_model_name(model: str) -> str:
    """
    Adaptation for project, assign certain model names
    """
    if (
        model
        == "m2f_swin_tiny_backbone_m2f_seg_255_coco_50000_bs1_fine_seg_255_coco_1250"
    ):
        model_name = "Supervised Baseline"
    elif (
        model
        == "m2f_swin_tiny_backbone_mae_seg_255_coco_50000_bs1_fine_seg_255_coco_1250"
    ):
        model_name = "Self-Supervised Pre-Training"
    elif (
        model
        == "m2f_swin_tiny_backbone_mae-cat_seg_255_coco_50000_bs1_fine_seg_255_coco_1250"
    ):
        model_name = "Self-Supervised Pre-Training with CAT Data"
    elif (
        model
        == "m2f_swin_tiny_backbone_mae-cat-conti_seg_255_coco_50000_bs1_fine_seg_255_coco_1250"
    ):
        model_name = "M2F Backbone continued CAT (conv-finetuned)"
    elif (
        model
        == "m2f_swin_tiny_backbone_m2f_seg_255_coco_50000_bs1_fine_seg_050_coco_050"
    ):
        model_name = "Supervised Baseline"
    elif (
        model
        == "m2f_swin_tiny_backbone_mae_seg_255_coco_50000_bs1_fine_seg_050_coco_050"
    ):
        model_name = "Self-Supervised Pre-Training"
    elif (
        model
        == "m2f_swin_tiny_backbone_mae-cat_seg_255_coco_50000_bs1_fine_seg_050_coco_050"
    ):
        model_name = "Self-Supervised Pre-Training with CAT Data"
    elif (
        model
        == "m2f_swin_tiny_backbone_mae-cat-conti_seg_255_coco_50000_bs1_fine_seg_050_coco_050"
    ):
        model_name = "M2F Backbone continued CAT (finetuned)"
    elif (
        model
        == "m2f_swin_tiny_backbone_m2f_seg_255_coco_50000_bs1_fine_seg_255_coco_1250_small"
    ):
        model_name = "Supervised Baseline"
    elif (
        model
        == "m2f_swin_tiny_backbone_mae_seg_255_coco_50000_bs1_fine_seg_255_coco_1250_small"
    ):
        model_name = "Self-Supervised Pre-Training"
    elif (
        model
        == "m2f_swin_tiny_backbone_mae-cat_seg_255_coco_50000_bs1_fine_seg_255_coco_1250_small"
    ):
        model_name = "Self-Supervised Pre-Training with CAT Data"
    elif (
        model
        == "m2f_swin_tiny_backbone_mae-cat-conti_seg_255_coco_50000_bs1_fine_seg_255_coco_1250_small"
    ):
        model_name = "M2F Backbone continued CAT (conv-finetuned, sdecoder)"
    elif (
        model
        == "m2f_swin_tiny_backbone_m2f_seg_255_coco_50000_bs1_fine_seg_050_coco_050_small"
    ):
        model_name = "Supervised Baseline"
    elif (
        model
        == "m2f_swin_tiny_backbone_mae_seg_255_coco_50000_bs1_fine_seg_050_coco_050_small"
    ):
        model_name = "Self-Supervised Pre-Training"
    elif (
        model
        == "m2f_swin_tiny_backbone_mae-cat_seg_255_coco_50000_bs1_fine_seg_050_coco_050_small"
    ):
        model_name = "Self-Supervised Pre-Training with CAT Data"
    elif (
        model
        == "m2f_swin_tiny_backbone_mae-cat-conti_seg_255_coco_50000_bs1_fine_seg_050_coco_050_small"
    ):
        model_name = "M2F Backbone continued CAT (finetuned, sdecoder)"
    else:
        model_name = model
    return model_name


def plot_scores(metric_scores: dict, output_dir: str, name: str) -> None:
    # get model and dataset names
    models = list(metric_scores.keys())
    datasets = list(metric_scores[models[0]].keys())

    # create output directory if necessary
    os.makedirs(output_dir, exist_ok=True)

    # create plot for the different datasets and models
    for dataset in datasets:
        dataset_name = get_dataset_name(dataset)
        fig, axs = plt.subplots(
            3, 3, sharex="col", sharey="row", figsize=(12, 12), dpi=160
        )
        fig.suptitle(dataset_name, y=1.04, fontsize=16)
        color_ar = ["r", "g", "b", "m", "c"]
        model_str = dataset

        for i, model in enumerate(models):
            model_name = get_model_name(model)
            x_values = metric_scores[model][dataset].columns
            y_names = metric_scores[model][dataset].index
            scores_np = metric_scores[model][dataset].to_numpy()

            # sort array in case that models are not read in the right order
            x_values = np.array([int(x) for x in x_values])
            sort_idx = sorted(range(len(x_values)), key=lambda k: x_values[k])
            x_values = x_values[sort_idx]
            scores_np = scores_np[:, sort_idx]

            axs[0, 0].set_title(y_names[0])
            axs[0, 0].set_ylabel("Score Value [%]")
            axs[0, 0].plot(
                x_values, scores_np[0, :], color=color_ar[i], label=model_name
            )
            axs[0, 1].set_title(y_names[1])
            axs[0, 1].plot(
                x_values, scores_np[1, :], color=color_ar[i], label=model_name
            )
            axs[0, 2].set_title(y_names[2])
            axs[0, 2].plot(
                x_values, scores_np[2, :], color=color_ar[i], label=model_name
            )
            axs[1, 0].set_title(y_names[3])
            axs[1, 0].set_ylabel("Score Value")
            axs[1, 0].plot(
                x_values, scores_np[3, :], color=color_ar[i], label=model_name
            )
            axs[1, 1].set_title(y_names[4])
            axs[1, 1].plot(
                x_values, scores_np[4, :], color=color_ar[i], label=model_name
            )
            axs[1, 2].set_title(y_names[5])
            axs[1, 2].plot(
                x_values, scores_np[5, :], color=color_ar[i], label=model_name
            )
            axs[2, 0].set_title(y_names[6])
            axs[2, 0].ticklabel_format(axis="x", style="sci")
            axs[2, 0].set_xlim(0, max(x_values))
            axs[2, 0].set_xlabel("Iterations")
            axs[2, 0].set_ylabel("Score Value")
            axs[2, 0].plot(
                x_values, scores_np[6, :], color=color_ar[i], label=model_name
            )
            axs[2, 1].set_title(y_names[7])
            axs[2, 1].ticklabel_format(axis="x", style="sci")
            axs[2, 1].set_xlim(0, max(x_values))
            axs[2, 1].set_xlabel("Iterations")
            axs[2, 1].plot(
                x_values, scores_np[7, :], color=color_ar[i], label=model_name
            )
            axs[2, 2].set_title(y_names[8])
            axs[2, 2].ticklabel_format(axis="x", style="sci")
            axs[2, 2].set_xlim(0, max(x_values))
            axs[2, 2].set_xlabel("Iterations")
            axs[2, 2].plot(
                x_values, scores_np[8, :], color=color_ar[i], label=model_name
            )
            model_str = model_str + "_" + model

        fig.subplots_adjust(bottom=0.3, wspace=0.33)
        handles, labels = axs[2, 2].get_legend_handles_labels()
        axs[2, 1].legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            fancybox=False,
            shadow=False,
            ncol=1,
        )

        plt.tight_layout()
        if name:
            plt.savefig(
                os.path.join(output_dir, f"{name}_{dataset_name}_complete.png"),
                bbox_inches="tight",
            )
            _logger.info(
                f"Scores plot saved under {os.path.join(output_dir, f'{name}_{dataset_name}_complete.png')}"
            )
        else:
            plt.savefig(
                os.path.join(output_dir, f"{model_str}_complete.png"),
                bbox_inches="tight",
            )
            _logger.info(
                f"Scores plot saved under {os.path.join(output_dir, f'{model_str}_complete.png')}"
            )
        plt.close()

    # create plot for the different datasets and models ONLY MAIN METRICS
    for dataset in datasets:
        dataset_name = get_dataset_name(dataset)
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey="row", dpi=160)
        fig.suptitle(dataset_name, fontsize=16, y=1.08)
        color_ar = ["r", "g", "b", "m", "c"]
        model_str = dataset

        for i, model in enumerate(models):
            model_name = get_model_name(model)
            x_values = metric_scores[model][dataset].columns
            y_names = metric_scores[model][dataset].index
            scores_np = metric_scores[model][dataset].to_numpy()

            # sort array in case that models are not read in the right order
            x_values = np.array([int(x) for x in x_values])
            sort_idx = sorted(range(len(x_values)), key=lambda k: x_values[k])
            x_values = x_values[sort_idx]
            scores_np = scores_np[:, sort_idx]

            axs[0].set_title(y_names[0])
            axs[0].set_xlim(0, max(x_values))
            axs[0].set_xlabel("Iterations")
            axs[0].set_ylabel("Score Value [%]")
            axs[0].plot(x_values, scores_np[0, :], color=color_ar[i], label=model_name)
            axs[1].set_title(y_names[1])
            axs[1].set_xlim(0, max(x_values))
            axs[1].set_xlabel("Iterations")
            axs[1].plot(x_values, scores_np[1, :], color=color_ar[i], label=model_name)
            axs[2].set_title(y_names[2])
            axs[2].set_xlim(0, max(x_values))
            axs[2].set_xlabel("Iterations")
            axs[2].plot(x_values, scores_np[2, :], color=color_ar[i], label=model_name)

            model_str = model_str + "_" + model

        fig.subplots_adjust(bottom=0.3, wspace=0.33)
        axs[1].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            fancybox=False,
            shadow=False,
            ncol=1,
        )
        plt.tight_layout()
        if name:
            plt.savefig(
                os.path.join(output_dir, f"{name}_{dataset_name}.png"),
                bbox_inches="tight",
            )
            _logger.info(
                f"Scores plot saved under {os.path.join(output_dir, f'{name}_{dataset_name}.png')}"
            )
        else:
            plt.savefig(os.path.join(output_dir, f"{model_str}.png"))
            _logger.info(
                f"Scores plot saved under {os.path.join(output_dir, f'{model_str}.png')}"
            )
        plt.close()


def main(args):
    # get scores
    metric_scores = get_scores(args.model_dirs)
    # plot scores
    plot_scores(metric_scores, args.output_dir, args.name)


if __name__ == "__main__":
    args = get_parser().parse_args()
    print(args)
    main(args)
