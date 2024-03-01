import argparse
import random
import numpy as np
import torch
import pathlib
import panoptic_models.detr.util.misc as utils
from panoptic_models.detr.datasets import build_dataset
from panoptic_models.detr.engine import evaluate
from panoptic_models.detr.models import build_model
import panoptic_models.config as config
from panoptic_models.detr.datasets.transforms import make_transforms
from torch.utils.data import DataLoader
import cv2
from panoptic_models.deeplab2.utils import vis_coco_reduced
import torchvision.transforms as T
import os
import panoptic_models.PanopticNets
import panoptic_models.detr.configs.config


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )
    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries", default=100, type=int, help="Number of query slots"
    )
    parser.add_argument("--pre_norm", action="store_true")

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )
    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )

    # Dataset parameters
    parser.add_argument("--dataset_file", default="coco_panoptic")
    parser.add_argument("--image_folder", default="data/images", type=str)
    parser.add_argument(
        "--image_annotated_folder", default="data/images_annotated", type=str
    )
    parser.add_argument("--annotations_file", default="data/annotations", type=str)
    # parser.add_argument('--coco_path', type=str)
    # parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument(
        "--output_dir",
        default="panoptic_models/outputs/detr",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    # Distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    # training parameters
    # summary writer logdir
    parser.add_argument(
        "--logdir", default="logs", type=str, help="logdir for summary writer."
    )
    parser.add_argument(
        "--load_to_ram", action="store_true", help="load dataset to RAM."
    )
    parser.add_argument("--wandb_log", action="store_true", help="log to wandb.")

    return parser


def main(args):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    # load model from checkpoint
    checkpoint = torch.load(args.frozen_weights, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    postprocessor = postprocessors["panoptic"]
    postprocessor.threshold = 0.85
    model.eval()
    model.to("cuda")

    if args.eval:
        dataset_eval = panoptic_models.detr.datasets.construction_site.PanopticRam(
            args.image_folder,
            args.image_annotated_folder,
            args.annotations_file,
            transforms=make_transforms(),
            return_masks=True,
            num_samples=10,
        )

        data_loader_eval = DataLoader(
            dataset_eval,
            args.batch_size,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )

        test_stats = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_eval,
            device,
            args.output_dir,
            args.wandb_log,
        )
        print("Test stats:", test_stats)
    else:

        # if not eval then just save the output of the model for the images in args.image_folder
        # load images from args.image_folder as cv images
        # save the output of the model for the images in args.image_folder
        # extract image folder name from args.image_folder
        image_folder_name = args.image_folder.split("/")[-1]
        # create output folder for the images in args.output_dir/image_folder_name
        output_folder = os.path.join(args.output_dir, image_folder_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        transform_normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        for image_name in os.listdir(args.image_folder):
            image_path = os.path.join(args.image_folder, image_name)
            image = cv2.imread(image_path)
            image = image[..., ::-1].copy()
            image_orig = torch.from_numpy(image).to(args.device)
            image_orig = image_orig.permute(2, 0, 1)
            orig_shape = image_orig.shape[1:]
            image_orig = image_orig.unsqueeze(0)
            image_transformed = T.Resize(800)(image_orig) / 255.0
            image_tensor = transform_normalize(image_transformed)
            image_transformed_shape = image_tensor.shape[-2:]

            output = model(image_tensor)
            postprocessed = postprocessor(
                output,
                torch.as_tensor(image_transformed_shape).unsqueeze(0),
                torch.as_tensor(orig_shape).unsqueeze(0),
            )[0]
            panoptic_id = postprocessed["panoptic_id"]
            panoptic_id = panoptic_id.cpu().numpy()
            panoptic_pred, segments_info = config.create_panoptic_label(
                panoptic_id, postprocessed["segments_info"]
            )
            # resire image for plotting to have the same shape as panoptic_pred
            if args.labels:
                image_output, panoptic_map = vis_coco_reduced.vis_panoptic_seg(
                    image, panoptic_pred
                )
            else:
                image_output, panoptic_map = vis_coco_reduced.vis_seg(
                    image, panoptic_pred
                )
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, image_output)

        # panoptic_net = panoptic_models.PanopticNets.DETR(config=panoptic_models.detr.config)
        # for image_name in os.listdir(args.image_folder):
        #     image_path = os.path.join(args.image_folder, image_name)
        #     image = cv2.imread(image_path)
        #     image_processed, panoptic_image = panoptic_net(image, vis_seg=True)
        #     output_path = os.path.join(output_folder, image_name)
        #     cv2.imwrite(output_path, panoptic_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR inference script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
