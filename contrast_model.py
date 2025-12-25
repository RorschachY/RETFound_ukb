#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import time
from pathlib import Path
import warnings
import faulthandler

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from timm.data.mixup import Mixup
import timm

import util.lr_decay as lrd
import util.misc as misc
from dataset_csv import build_csv_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_finetune import train_one_epoch, evaluate

faulthandler.enable()
warnings.simplefilter(action="ignore", category=FutureWarning)

SUPPORTED_MODELS = (
    "resnet50",
    "vgg16_bn",
    "vit_base_patch16_224",
)


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Contrast models fine-tuning for image classification", add_help=False
    )

    # ---- Core training
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)

    # ---- Model parameters
    parser.add_argument(
        "--model",
        default="resnet50",
        type=str,
        choices=SUPPORTED_MODELS,
        help="Timm model name for contrast experiments",
    )
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument(
        "--no_pretrained", action="store_false", dest="pretrained"
    )
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--drop_path", type=float, default=0.0)
    parser.add_argument(
        "--adaptation",
        type=str,
        default="finetune",
        choices=["finetune", "lp"],
        help="finetune=all params, lp=linear probe",
    )

    # ---- Optimizer parameters
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None, metavar="LR")
    parser.add_argument("--blr", type=float, default=5e-3, metavar="LR")
    parser.add_argument("--layer_decay", type=float, default=1.0)
    parser.add_argument("--min_lr", type=float, default=1e-6, metavar="LR")
    parser.add_argument("--warmup_epochs", type=int, default=10, metavar="N")

    # ---- Augmentation
    parser.add_argument("--color_jitter", type=float, default=None, metavar="PCT")
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1", metavar="NAME")
    parser.add_argument("--smoothing", type=float, default=0.1)

    # ---- Random erase
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT")
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--resplit", action="store_true", default=False)

    # ---- Mixup/Cutmix
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--cutmix_minmax", type=float, nargs="+", default=None)
    parser.add_argument("--mixup_prob", type=float, default=1.0)
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5)
    parser.add_argument("--mixup_mode", type=str, default="batch")

    # ---- Finetuning
    parser.add_argument(
        "--finetune",
        default="",
        type=str,
        help="Checkpoint path for initialization (optional)",
    )
    parser.add_argument("--task", default="", type=str)

    # ---- Dataset & paths
    parser.add_argument("--data_path", default="./data/", type=str)
    parser.add_argument("--csv_path", default=None, type=str)
    parser.add_argument("--nb_classes", default=8, type=int)
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--log_dir", default="./output_logs")

    # ---- Runtime
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="Resume full state")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--eval", action="store_true", help="Evaluation only")
    parser.add_argument("--dist_eval", action="store_true", default=False)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.set_defaults(pin_mem=True)

    # ---- Distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    # ---- Misc
    parser.add_argument("--savemodel", action="store_true", default=True)
    parser.add_argument("--norm", default="IMAGENET", type=str)
    parser.add_argument("--enhance", action="store_true", default=False)
    parser.add_argument("--datasets_seed", default=2026, type=int)

    return parser


def _load_checkpoint_weights(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return cleaned


def _enable_linear_probe(model):
    for param in model.parameters():
        param.requires_grad = False

    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if isinstance(classifier, torch.nn.Module):
        for param in classifier.parameters():
            param.requires_grad = True
        return

    if isinstance(classifier, str) and classifier:
        head = getattr(model, classifier, None)
        if isinstance(head, torch.nn.Module):
            for param in head.parameters():
                param.requires_grad = True
            return

    for name, param in model.named_parameters():
        if any(key in name for key in ("head", "fc", "classifier")):
            param.requires_grad = True


def _build_optimizer(model_without_ddp, args):
    use_lrd = (
        args.layer_decay < 1.0
        and (
            hasattr(model_without_ddp, "blocks")
            or all(hasattr(model_without_ddp, name) for name in ("layer1", "layer2", "layer3", "layer4"))
        )
    )

    if use_lrd:
        no_weight_decay = (
            model_without_ddp.no_weight_decay()
            if hasattr(model_without_ddp, "no_weight_decay")
            else []
        )
        param_groups = lrd.param_groups_lrd(
            model_without_ddp,
            weight_decay=args.weight_decay,
            no_weight_decay_list=no_weight_decay,
            layer_decay=args.layer_decay,
        )
        for group in param_groups:
            group["params"] = [p for p in group["params"] if p.requires_grad]
        return torch.optim.AdamW(param_groups, lr=args.lr)

    decay, no_decay = [], []
    for name, param in model_without_ddp.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )


def main(args, criterion):
    if args.resume and not args.eval:
        resume_path = args.resume
        checkpoint = torch.load(args.resume, map_location="cpu")
        print(f"Load checkpoint (args) from: {args.resume}")
        args = checkpoint["args"]
        args.resume = resume_path

    misc.init_distributed_mode(args)

    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"{args}".replace(", ", ",\n"))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    model = timm.create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
    )

    if args.finetune and not args.eval:
        print(f"Loading checkpoint weights from: {args.finetune}")
        state_dict = _load_checkpoint_weights(args.finetune)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            print(f"Unexpected keys when loading checkpoint: {unexpected}")

    dataset_train = build_csv_dataset(is_train="train", args=args)
    dataset_val = build_csv_dataset(is_train="val", args=args)
    dataset_test = build_csv_dataset(is_train="test", args=args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if not args.eval:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print(f"Sampler_train = {sampler_train}")
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print("Warning: dist eval with dataset not divisible by #procs; results may differ slightly.")
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.dist_eval:
        if len(dataset_test) % num_tasks != 0:
            print("Warning: dist eval test set not divisible by #procs; results may differ slightly.")
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.task))
    else:
        log_writer = None

    if not args.eval:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        print(f"len of train_set: {len(data_loader_train) * args.batch_size}")

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = (args.mixup > 0) or (args.cutmix > 0.0) or (args.cutmix_minmax is not None)
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    if args.resume and args.eval:
        checkpoint = torch.load(args.resume, map_location="cpu")
        print(f"Load checkpoint for eval from: {args.resume}")
        model.load_state_dict(checkpoint["model"])

    model.to(device)
    model_without_ddp = model

    if args.adaptation == "lp":
        _enable_linear_probe(model)
        print("[Adaptation] Linear probe: training classifier head only.")
    else:
        print("[Adaptation] Full fine-tuning: training all parameters.")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable params (M): {n_parameters / 1.e6:.2f}")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"accumulate grad iterations: {args.accum_iter}")
    print(f"effective batch size: {eff_batch_size}")

    if args.distributed and torch.cuda.device_count() > 1:
        ddp_kwargs = {}
        if args.adaptation == "lp":
            ddp_kwargs["find_unused_parameters"] = True
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], **ddp_kwargs
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    optimizer = _build_optimizer(model_without_ddp, args)
    loss_scaler = NativeScaler()
    print(f"criterion = {criterion}")

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    if args.eval:
        test_stats, _ = evaluate(
            data_loader_test,
            model,
            device,
            args,
            epoch=0,
            mode="test",
            num_class=args.nb_classes,
            log_writer=log_writer,
        )
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_score = 0.0
    best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            mixup_fn,
            log_writer=log_writer,
            args=args,
        )

        val_stats, val_score = evaluate(
            data_loader_val,
            model,
            device,
            args,
            epoch,
            mode="val",
            num_class=args.nb_classes,
            log_writer=log_writer,
        )

        if max_score < val_score:
            max_score = val_score
            best_epoch = epoch
            if args.output_dir and args.savemodel:
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    mode="best",
                )
        print(f"Best epoch = {best_epoch}, Best score = {max_score:.4f}")

        if log_writer is not None:
            log_writer.add_scalar("loss/val", val_stats["loss"], epoch)
            log_writer.flush()

        log_stats = {
            **{f"train_{k}": float(v) for k, v in train_stats.items()},
            "epoch": int(epoch),
            "n_parameters": int(n_parameters),
            "val_loss": float(val_stats.get("loss", 0.0)),
            "val_score": float(val_score),
        }

        if args.output_dir and misc.is_main_process():
            with open(
                os.path.join(args.output_dir, args.task, "log.txt"),
                "a",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    ckpt_path = os.path.join(args.output_dir, args.task, "checkpoint-best.pth")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    print(f"Test with the best model, epoch = {checkpoint.get('epoch', -1)}:")
    evaluate(
        data_loader_test,
        model,
        device,
        args,
        -1,
        mode="test",
        num_class=args.nb_classes,
        log_writer=None,
    )

    if args.world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    criterion = torch.nn.CrossEntropyLoss()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, criterion)
