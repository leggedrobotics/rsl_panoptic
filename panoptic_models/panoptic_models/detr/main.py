# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import panoptic_models
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import os
import panoptic_models.detr.util.misc as utils
from panoptic_models.detr.datasets import build_dataset, get_coco_api_from_dataset
from panoptic_models.detr.engine import evaluate, train_one_epoch
from panoptic_models.detr.models import build_model
import panoptic_models.detr.datasets as datasets
import panoptic_models.detr.datasets.build_dataset as build_dataset
import wandb
import torch
import yaml


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
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
    parser.add_argument("--threshold", default=0.85, type=float)

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

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--dataset_filename", default="coco")
    parser.add_argument("--val_dataset_file", default="coco")
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=0, type=int)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    # custom parameters
    # number of samples default is None
    parser.add_argument("--num_samples", default="0", type=str)
    parser.add_argument("--validation_epochs", default=20, type=int)
    parser.add_argument("--save_epochs", default=20, type=int)
    parser.add_argument("--run_name", default="", type=str)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--rewire", action="store_true")
    parser.add_argument("--tag", default="", type=str)
    parser.add_argument("--wandb_log", action="store_true", help="log to wandb.")
    return parser


def main(args):
    #  initialize wandb
    if args.wandb_log:
        if args.run_name is None:
            args.run_name = "detr_" + args.dataset_file
        else:
            args.run_name = "detr_" + args.run_name
        tags = ["detr"]
        tags.append(args.dataset_filename)
        tags.append(args.tag)
        wandb.init(project="vision_nav", name=args.run_name, tags=tags)
        # load variables of python config file into wandb
        package_path = os.path.dirname(panoptic_models.__file__)
        # config_path = package_path + '/detr/configs/config.yaml'
        # # load the variables in the config python file in a dict
        # config_dict = {}
        # with open(config_path, 'r') as stream:
        #     try:
        #          config_dict = yaml.safe_load(stream)
        #     except yaml.YAMLError as exc:
        #          print(exc)
        # # load the variables in the config python file in wandb
        # for key, value in config_dict.items():
        #     wandb.config.update({key: value})
        wandb.config.update(args)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print model layers that don't require gradients
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print("Skipping", name)
    print("number of params:", n_parameters)

    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    datasets = [dataset for dataset in args.dataset_file.split(",")]
    num_samples_list = [int(s) for s in args.num_samples.split(",")]
    dataset_train = build_dataset.build(
        image_set="train",
        datasets=datasets,
        masks=args.masks,
        num_samples_list=num_samples_list,
    )
    val_datasets = [dataset for dataset in args.val_dataset_file.split(",")]
    dataset_val = build_dataset.build(
        image_set="val",
        datasets=val_datasets,
        masks=args.masks,
        num_samples_list=num_samples_list,
    )

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    print("Dataloader train size:", len(data_loader_train))
    # warm up cosine lr scheduler
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )
    max_iterations = args.epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max_iterations, eta_min=0.00001
    )

    print("Dataset val is of type {}".format(type(data_loader_val)))
    # if args.dataset_file contains "panoptic"
    if args.dataset_file.find("panoptic") != -1:
        # if args.dataset_file.find("coco") != -1:
        #     # We also evaluate AP during panoptic training, on original coco DS
        #     coco_val = datasets.coco.build("val", args)
        #     print("Dataset val is of type {}".format(type(coco_val)))
        #     base_ds = get_coco_api_from_dataset(coco_val)
        #     print("base ds", base_ds)
        # else:
        base_ds = dataset_val
    else:
        print("type dataset validation is {}".format(type(dataset_val)))
        base_ds = get_coco_api_from_dataset(dataset_val)
        # print("base ds", base_ds)

    output_dir = Path(args.output_dir)

    model, checkpoint = load_model(args, model_without_ddp)

    # if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #     args.start_epoch = checkpoint['epoch']

    if args.eval:
        test_stats = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            args.output_dir,
            args.wandb_log,
        )
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
        )
        lr_scheduler.step()
        if args.output_dir and epoch % args.save_epochs == 0 and epoch != 0:
            checkpoint_name = args.run_name + "_epoch_" + str(epoch) + ".pth"
            checkpoint_paths = [output_dir / checkpoint_name]
            # save the model every args.save_epochs

            for checkpoint_path in checkpoint_paths:
                print("Saving checkpoint to: {}".format(checkpoint_path))
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )
            # save online
            # wandb.save('outputs/detr/checkpoint*')
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }
        # log stats on wandb
        if args.wandb_log:
            wandb.log(log_stats)

        if (
            epoch % args.validation_epochs == 0
            and epoch != 0
            or epoch == args.epochs - 1
        ):
            test_stats = evaluate(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                device,
                args.output_dir,
                args.wandb_log,
            )
            # add test stats to log_stats
            log_stats = {**log_stats, **{f"test_{k}": v for k, v in test_stats.items()}}
            # log stats on wandb
            if args.wandb_log:
                wandb.log(log_stats)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    if args.wandb_log:
        # save model to wandb
        model = wandb.Artifact(args.run_name, type="model")
        parent_package_path = os.path.dirname(package_path)
        model.add_file(
            parent_package_path + "/outputs/detr/detr" + args.run_name + ".pth",
            "detr.pth",
        )
        wandb.log_artifact(model)


def load_model(args, model, optimizer=None, lr_scheduler=None):
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            if args.masks:
                checkpoint = torch.load(args.resume, map_location="cpu")
                model.load_state_dict(checkpoint["model"], strict=True)
            else:
                checkpoint = torch.load(args.resume, map_location="cpu")
                checkpoint["model"] = {
                    k[5:]: v for k, v in checkpoint["model"].items() if "detr" in k
                }
                del checkpoint["model"]["class_embed.weight"]
                del checkpoint["model"]["class_embed.bias"]
                model.load_state_dict(checkpoint["model"], strict=False)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model.detr.load_state_dict(checkpoint["model"], strict=True)

    return model, checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
