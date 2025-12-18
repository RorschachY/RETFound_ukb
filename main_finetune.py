#!/usr/bin/env python3

# =========================
# 标准库导入
import argparse  # 命令行参数解析
import datetime  # 日期时间处理
import json  # JSON序列化
import os  # 操作系统接口
import time  # 时间计量
from pathlib import Path  # 路径操作
import warnings  # 警告控制
import faulthandler  # 错误追踪

# =========================
# 第三方库导入
import numpy as np  # 数值计算
import torch  # PyTorch深度学习框架
import torch.backends.cudnn as cudnn  # cuDNN加速
from torch.utils.tensorboard import SummaryWriter  # TensorBoard日志记录
from timm.models.layers import trunc_normal_  # 截断正态分布初始化
from timm.data.mixup import Mixup  # 数据增强：Mixup和CutMix
from huggingface_hub import hf_hub_download, login  # Hugging Face模型下载

# =========================
# 本地模块导入
import models_vit as models  # ViT模型定义
import util.lr_decay as lrd  # 学习率衰减策略
import util.misc as misc  # 工具函数
from util.datasets import build_dataset  # 数据集构建
from util.pos_embed import interpolate_pos_embed  # 位置编码插值
from util.misc import NativeScalerWithGradNormCount as NativeScaler  # 混合精度训练缩放器
from engine_finetune import train_one_epoch, evaluate  # 训练和评估引擎

# =========================
faulthandler.enable()  # 启用错误追踪，便于调试段错误
warnings.simplefilter(action="ignore", category=FutureWarning)  # 忽略未来警告


def get_args_parser():
    """创建并返回命令行参数解析器"""
    parser = argparse.ArgumentParser(
        "MAE fine-tuning / linear probing for image classification", add_help=False
    )

    # ---- 核心训练参数
    parser.add_argument("--batch_size", default=128, type=int,
                        help="每个GPU的批次大小（有效批次大小 = batch_size * accum_iter * GPU数量）")
    parser.add_argument("--epochs", default=50, type=int)  # 训练轮数
    parser.add_argument("--accum_iter", default=1, type=int,
                        help="梯度累积步数，用于模拟更大的批次大小")

    # ---- 模型参数
    parser.add_argument("--model", default="vit_large_patch16", type=str, metavar="MODEL",
                        help="models_vit.py中的模型入口名称")
    parser.add_argument("--model_arch", default="dinov3_vits16", type=str, metavar="MODEL_ARCH",
                        help="骨干网络架构键名（如dinov2_vitl14, convnext_base等）")
    parser.add_argument("--input_size", default=256, type=int, help="输入图像尺寸")
    parser.add_argument("--drop_path", type=float, default=0.2, metavar="PCT", help="随机深度丢弃率")
    parser.add_argument("--global_pool", action="store_true"); parser.set_defaults(global_pool=True)  # 使用全局池化
    parser.add_argument("--cls_token", action="store_false", dest="global_pool",
                        help="使用类别token代替全局池化进行分类")

    # ---- 优化器参数
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM", help="梯度裁剪阈值")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="权重衰减系数")
    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="绝对学习率（覆盖blr）")
    parser.add_argument("--blr", type=float, default=5e-3, metavar="LR",
                        help="基础学习率：实际lr = blr * 总批次大小 / 256")
    parser.add_argument("--layer_decay", type=float, default=0.65, help="层级学习率衰减因子（ViT专用）")
    parser.add_argument("--min_lr", type=float, default=1e-6, metavar="LR", help="学习率下限")
    parser.add_argument("--warmup_epochs", type=int, default=10, metavar="N", help="学习率预热轮数")

    # ---- 数据增强
    parser.add_argument("--color_jitter", type=float, default=None, metavar="PCT")  # 颜色抖动强度
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1", metavar="NAME")  # AutoAugment策略
    parser.add_argument("--smoothing", type=float, default=0.1)  # 标签平滑系数

    # ---- 随机擦除
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT")  # 随机擦除概率
    parser.add_argument("--remode", type=str, default="pixel")  # 擦除模式
    parser.add_argument("--recount", type=int, default=1)  # 擦除区域数量
    parser.add_argument("--resplit", action="store_true", default=False)  # 是否重新划分

    # ---- Mixup/Cutmix数据增强
    parser.add_argument("--mixup", type=float, default=0.0)  # Mixup混合强度
    parser.add_argument("--cutmix", type=float, default=0.0)  # CutMix混合强度
    parser.add_argument("--cutmix_minmax", type=float, nargs="+", default=None)  # CutMix尺寸范围
    parser.add_argument("--mixup_prob", type=float, default=1.0)  # 应用Mixup的概率
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5)  # Mixup/CutMix切换概率
    parser.add_argument("--mixup_mode", type=str, default="batch")  # Mixup模式

    # ---- 微调与适配
    parser.add_argument("--finetune", default="", type=str, help="预训练检查点ID或路径")
    parser.add_argument("--task", default="", type=str, help="任务名称，用于日志和输出分组")
    parser.add_argument("--adaptation", default="finetune", choices=["finetune", "lp"],
                        help="适配策略：finetune=全参数微调，lp=线性探测（仅训练分类头）")

    # ---- 数据集与路径
    parser.add_argument("--data_path", default="./data/", type=str)  # 数据集根目录
    parser.add_argument("--nb_classes", default=8, type=int)  # 分类类别数
    parser.add_argument("--output_dir", default="./output_dir")  # 输出目录
    parser.add_argument("--log_dir", default="./output_logs")  # 日志目录

    # >>> 新增：训练数据效率参数 <<<
    parser.add_argument(
        "--dataratio", type=str, default="1.0",
        help=('训练数据采样比例。可以是单个浮点数(0,1]（如0.25），'
              '或逗号分隔的列表（如"1.0,0.5,0.25"）用于多比例实验')
    )
    parser.add_argument(
        "--stratified", action="store_true",
        help="启用分层采样，确保各类别按比例采样（需build_dataset支持）"
    )

    # ---- 运行时参数
    parser.add_argument("--device", default="cuda")  # 计算设备
    parser.add_argument("--seed", default=0, type=int)  # 随机种子
    parser.add_argument("--resume", default="", help="恢复完整训练状态（优化器、缩放器等）")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")  # 起始轮数
    parser.add_argument("--eval", action="store_true", help="仅评估模式")
    parser.add_argument("--dist_eval", action="store_true", default=False,
                        help="分布式评估（训练过程中更快的监控）")
    parser.add_argument("--num_workers", default=10, type=int)  # 数据加载进程数
    parser.add_argument("--pin_mem", action="store_true"); parser.set_defaults(pin_mem=True)  # 锁页内存

    # ---- 分布式训练
    parser.add_argument("--world_size", default=1, type=int)  # 总进程数
    parser.add_argument("--local_rank", default=-1, type=int)  # 本地进程序号
    parser.add_argument("--dist_on_itp", action="store_true")  # ITP集群分布式模式
    parser.add_argument("--dist_url", default="env://")  # 分布式初始化URL

    # ---- 其他参数
    parser.add_argument("--savemodel", action="store_true", default=True, help="保存最佳模型")
    parser.add_argument("--norm", default="IMAGENET", type=str)  # 归一化方式
    parser.add_argument("--enhance", action="store_true", default=False)  # 图像增强
    parser.add_argument("--datasets_seed", default=2026, type=int)  # 数据集划分种子

    return parser


# =========================
# 主函数
# =========================
def main(args, criterion):
    """主训练/评估函数"""
    # ---- 从恢复点加载参数（训练时）
    if args.resume and not args.eval:
        resume_path = args.resume
        checkpoint = torch.load(args.resume, map_location="cpu")
        print(f"从检查点加载参数: {args.resume}")
        args = checkpoint["args"]
        args.resume = resume_path

    # ---- 分布式初始化
    misc.init_distributed_mode(args)

    print(f"工作目录: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"{args}".replace(", ", ",\n"))

    device = torch.device(args.device)  # 设置计算设备

    # ---- 可复现性设置
    seed = args.seed + misc.get_rank()  # 每个进程使用不同种子
    torch.manual_seed(seed)  # PyTorch随机种子
    np.random.seed(seed)  # NumPy随机种子
    cudnn.benchmark = True  # cuDNN自动优化

    # ---- 构建模型
    if args.model == "RETFound_mae":
        model = models.__dict__[args.model](
            img_size=args.input_size,  # 输入图像尺寸
            num_classes=args.nb_classes,  # 分类类别数
            drop_path_rate=args.drop_path,  # 随机深度丢弃率
            global_pool=args.global_pool,  # 是否使用全局池化
        )
    else:
        model = models.__dict__[args.model](
            num_classes=args.nb_classes,  # 分类类别数
            drop_path_rate=args.drop_path,  # 随机深度丢弃率
            args=args,  # 其他参数
        )

    # ---- 加载预训练权重（非评估模式时）
    if args.finetune and not args.eval:
        print(f"准备加载预训练权重: {args.finetune}")

        if args.model in ["Dinov3", "Dinov2"]:
            checkpoint_path = args.finetune  # 本地路径，直接使用用户提供的权重文件
        elif args.model in ["RETFound_dinov2", "RETFound_mae"]:
            print(f"从Hugging Face Hub下载预训练权重: {args.finetune}")
            checkpoint_path = hf_hub_download(
                repo_id=f"YukunZhou/{args.finetune}",  # 仓库ID
                filename=f"{args.finetune}.pth",  # 权重文件名
            )
        else:
            raise ValueError(
                f"不支持的模型'{args.model}'。"
                f"支持的模型: Dinov3, Dinov2, RETFound_dinov2, RETFound_mae"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")  # 加载到CPU
        print(f"已加载预训练检查点: {checkpoint_path}")

        if args.model in ["Dinov3", "Dinov2"]:
            checkpoint_model = checkpoint  # 直接使用权重字典
        elif args.model == "RETFound_dinov2":
            checkpoint_model = checkpoint["teacher"]  # 提取教师模型权重
        else:  # RETFound_mae
            checkpoint_model = checkpoint["model"]  # 提取模型权重

        # -- 键名清理：移除前缀和替换不匹配的键名
        checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}

        # -- 移除形状不匹配的分类器权重
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"从预训练检查点移除键 {k}（形状不匹配）")
                del checkpoint_model[k]

        # -- 插值位置编码（适应不同输入尺寸）
        interpolate_pos_embed(model, checkpoint_model)

        # -- 加载骨干网络权重（非严格模式，允许部分键缺失）
        _ = model.load_state_dict(checkpoint_model, strict=False)

        # -- 重新初始化分类头
        if hasattr(model, "head") and hasattr(model.head, "weight"):
            trunc_normal_(model.head.weight, std=2e-5)  # 使用截断正态分布初始化

    # ---- 构建数据集和采样器
    dataset_train = build_dataset(is_train="train", args=args)  # 训练集
    dataset_val   = build_dataset(is_train="val",   args=args)  # 验证集
    dataset_test  = build_dataset(is_train="test",  args=args)  # 测试集

    num_tasks   = misc.get_world_size()  # 总进程数
    global_rank = misc.get_rank()  # 当前进程序号

    if not args.eval:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )  # 分布式训练采样器
        print(f"训练采样器 = {sampler_train}")
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print("警告: 分布式评估时验证集不能被进程数整除，结果可能略有不同")
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)  # 顺序采样

    if args.dist_eval:
        if len(dataset_test) % num_tasks != 0:
            print("警告: 分布式评估时测试集不能被进程数整除，结果可能略有不同")
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)  # 顺序采样

    # ---- TensorBoard日志记录器
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.task))  # 创建日志写入器
    else:
        log_writer = None

    # ---- 创建数据加载器
    if not args.eval:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=True,  # 丢弃最后不完整批次
        )
        print(f"训练集样本数: {len(data_loader_train) * args.batch_size}")

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=False,
        )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    # ---- Mixup/CutMix数据增强配置
    mixup_fn = None
    mixup_active = (args.mixup > 0) or (args.cutmix > 0.) or (args.cutmix_minmax is not None)
    if mixup_active:
        print("Mixup数据增强已激活！")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes
        )

    # ---- 仅评估模式：加载恢复权重
    if args.resume and args.eval:
        checkpoint = torch.load(args.resume, map_location="cpu")
        print(f"加载评估用检查点: {args.resume}")
        model.load_state_dict(checkpoint["model"])

    model.to(device)  # 模型移至指定设备
    model_without_ddp = model  # 保存非DDP包装的模型引用

    # ---- 适配策略切换
    if args.adaptation == "lp":
        for name, param in model.named_parameters():
            param.requires_grad = ("head" in name)  # 仅分类头可训练
        print("[适配策略] 线性探测：仅训练分类头参数")
    else:
        print("[适配策略] 全参数微调：训练所有参数")

    # ---- 统计可训练参数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量 (M): {n_parameters / 1.e6:.2f}")

    # ---- 根据有效批次大小缩放学习率
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()  # 有效批次大小
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256  # 线性缩放规则
    print(f"基础学习率: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"实际学习率: {args.lr:.2e}")
    print(f"梯度累积步数: {args.accum_iter}")
    print(f"有效批次大小: {eff_batch_size}")

    # ---- 分布式数据并行（多GPU）
    if args.distributed and torch.cuda.device_count() > 1:
        ddp_kwargs = {}
        if args.adaptation == "lp":
            ddp_kwargs["find_unused_parameters"] = True  # 线性探测时需要此选项
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], **ddp_kwargs
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model  # 单GPU模式

    # ---- 构建优化器参数组（冻结后）
    no_weight_decay = (model_without_ddp.no_weight_decay()
                       if hasattr(model_without_ddp, "no_weight_decay") else [])  # 获取不需要权重衰减的参数


    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        weight_decay=args.weight_decay,
        no_weight_decay_list=no_weight_decay,
        layer_decay=args.layer_decay,  # 层级学习率衰减
    )
    for g in param_groups:
        g["params"] = [p for p in g["params"] if p.requires_grad]  # 仅保留需要梯度的参数

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)  # AdamW优化器
    loss_scaler = NativeScaler()  # 混合精度训练缩放器
    print(f"损失函数 = {criterion}")

    # ---- 加载之前的完整训练状态（优化器、缩放器等）
    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    # =========================
    # 仅评估模式快捷返回
    # =========================
    if args.eval:
        if "checkpoint" in locals() and isinstance(checkpoint, dict) and ("epoch" in checkpoint):
            print(f"使用第 {checkpoint['epoch']} 轮的最佳模型进行测试")
        test_stats, auc_roc = evaluate(
            data_loader_test, model, device, args, epoch=0, mode="test",
            num_class=args.nb_classes, log_writer=log_writer
        )
        return

    # =========================
    # 训练循环
    # =========================
    print(f"开始训练，共 {args.epochs} 轮")
    start_time = time.time()  # 记录开始时间
    max_score = 0.0  # 最佳验证分数
    best_epoch = 0  # 最佳轮次

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)  # 设置采样器轮次以确保打乱

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer, args=args
        )

        val_stats, val_score = evaluate(
            data_loader_val, model, device, args, epoch, mode="val",
            num_class=args.nb_classes, log_writer=log_writer
        )

        if max_score < val_score:  # 更新最佳模型
            max_score = val_score
            best_epoch = epoch
            if args.output_dir and args.savemodel:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, mode="best"
                )
        print(f"最佳轮次 = {best_epoch}, 最佳分数 = {max_score:.4f}")

        if log_writer is not None:
            log_writer.add_scalar("loss/val", val_stats["loss"], epoch)  # 记录验证损失
            log_writer.flush()

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                     "epoch": epoch,
                     "n_parameters": n_parameters}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, args.task, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")  # 写入训练日志

    # =========================
    # 最终测试（使用最佳检查点）
    # =========================
    ckpt_path = os.path.join(args.output_dir, args.task, "checkpoint-best.pth")  # 最佳模型路径
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    print(f"使用最佳模型进行测试，轮次 = {checkpoint.get('epoch', -1)}:")
    _test_stats, _auc_roc = evaluate(
        data_loader_test, model, device, args, -1, mode="test",
        num_class=args.nb_classes, log_writer=None
    )

    total_time = time.time() - start_time  # 计算总训练时间
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"训练总耗时 {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser()  # 创建参数解析器
    args = args.parse_args()  # 解析命令行参数

    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)  # 创建输出目录

    main(args, criterion)  # 运行主函数
