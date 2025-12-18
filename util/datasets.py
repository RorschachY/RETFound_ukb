import os
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    """
    构建数据集
    
    参数:
        is_train: 数据集类型标识，'train' 表示训练集，其他表示验证/测试集
        args: 包含数据路径、输入尺寸等配置参数的对象
    
    返回:
        dataset: 构建好的数据集对象
    """
    # 根据训练/验证模式构建对应的数据变换
    transform = build_transform(is_train, args)
    
    # 拼接数据集根目录路径
    root = os.path.join(args.data_path, is_train)
    
    # 使用 ImageFolder 加载按文件夹组织的图像数据集
    dataset = datasets.ImageFolder(root, transform=transform)

    # 仅对训练集进行子采样
    if is_train == 'train':
        # 获取数据采样比例，默认为 1.0（使用全部数据）
        ratio = float(getattr(args, "dataratio", 1.0))
        # 获取随机种子，用于保证可复现性
        seed = int(getattr(args, "seed", 0))
        # 是否使用分层采样（保持各类别比例）
        stratified = bool(getattr(args, "stratified", False))

        # 当比例在 0 到 1 之间时，进行子采样
        if 0.0 < ratio < 1.0:
            if stratified:
                # 分层采样：保持各类别的样本比例
                idx = _stratified_indices(dataset.targets, ratio, seed)
            else:
                # 简单均匀采样：使用 torch.Generator 保证可复现性
                g = torch.Generator().manual_seed(seed)
                n = len(dataset)
                # 计算采样数量，至少保留 1 个样本
                k = max(1, int(n * ratio))
                # 随机打乱索引并取前 k 个
                idx = torch.randperm(n, generator=g)[:k].tolist()
            # 使用索引创建子数据集
            dataset = Subset(dataset, idx)

    return dataset


def build_transform(is_train, args):
    """
    构建图像变换/数据增强管道
    
    参数:
        is_train: 数据集类型标识，'train' 表示训练集
        args: 包含输入尺寸、数据增强参数等配置的对象
    
    返回:
        transform: 图像变换组合
    """
    # 使用 ImageNet 默认的均值和标准差进行归一化
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train == 'train':
        # 训练集使用 timm 的 create_transform，包含各种数据增强
        return create_transform(
            input_size=args.input_size,      # 输入图像尺寸
            is_training=True,                 # 启用训练模式的增强
            color_jitter=args.color_jitter,   # 颜色抖动强度
            auto_augment=args.aa,             # 自动增强策略
            interpolation='bicubic',          # 插值方法
            re_prob=args.reprob,              # 随机擦除概率
            re_mode=args.remode,              # 随机擦除模式
            re_count=args.recount,            # 随机擦除次数
            mean=mean,
            std=std,
        )

    # 验证/测试集变换：不使用数据增强
    # 计算裁剪比例：小于等于 224 时使用 224/256，否则使用 1.0
    crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
    # 计算缩放尺寸
    size = int(args.input_size / crop_pct)
    t = [
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  # 缩放图像
        transforms.CenterCrop(args.input_size),  # 中心裁剪到目标尺寸
        transforms.ToTensor(),                    # 转换为张量
        transforms.Normalize(mean, std),          # 归一化
    ]
    return transforms.Compose(t)


# ---- 辅助函数 ----

def _stratified_indices(targets, ratio: float, seed: int):
    """
    分层采样：保持各类别的样本比例
    在可能的情况下，确保每个类别至少有 1 个样本
    
    参数:
        targets: 数据集的标签列表
        ratio: 采样比例 (0.0 到 1.0 之间)
        seed: 随机种子，用于保证可复现性
    
    返回:
        keep: 采样后的索引列表
    """
    t = torch.as_tensor(targets)
    # 获取所有唯一的类别标签
    classes = torch.unique(t)
    # 创建随机数生成器
    g = torch.Generator().manual_seed(seed)

    keep = []
    for c in classes.tolist():
        # 获取当前类别的所有样本索引
        cls_idx = torch.nonzero(t == c, as_tuple=False).view(-1)
        if len(cls_idx) == 0:
            continue
        # 计算当前类别需要采样的数量，至少保留 1 个
        k = max(1, int(round(len(cls_idx) * ratio)))
        # 随机选择 k 个样本
        sel = cls_idx[torch.randperm(len(cls_idx), generator=g)[:k]]
        keep.extend(sel.tolist())

    # 打乱最终的索引列表（使用不同的种子保证稳定性）
    g2 = torch.Generator().manual_seed(seed + 1)
    keep = torch.tensor(keep)[torch.randperm(len(keep), generator=g2)].tolist()
    return keep

