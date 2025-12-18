import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score
)
from pycm import ConfusionMatrix
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(
    model: torch.nn.Module,  # 待训练的模型
    criterion: torch.nn.Module,  # 损失函数
    data_loader: Iterable,  # 数据加载器
    optimizer: torch.optim.Optimizer,  # 优化器
    device: torch.device,  # 计算设备(CPU/GPU)
    epoch: int,  # 当前训练轮次
    loss_scaler,  # 混合精度训练的损失缩放器
    max_norm: float = 0,  # 梯度裁剪的最大范数，0表示不裁剪
    mixup_fn: Optional[Mixup] = None,  # 数据增强的Mixup函数
    log_writer=None,  # TensorBoard日志写入器
    args=None  # 其他参数配置
):
    """训练模型一个epoch"""
    model.train(True)  # 设置模型为训练模式
    metric_logger = misc.MetricLogger(delimiter="  ")  # 初始化指标记录器
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 添加学习率监控
    print_freq, accum_iter = 20, args.accum_iter  # 打印频率和梯度累积步数
    optimizer.zero_grad()  # 清空梯度
    
    if log_writer:
        print(f'日志目录: {log_writer.log_dir}')
    
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)  # 动态调整学习率
        
        samples, targets = samples.to(device, non_blocking=True), targets.to(device, non_blocking=True)  # 将数据移至GPU
        if mixup_fn:
            samples, targets = mixup_fn(samples, targets)  # 应用Mixup数据增强
        
        with torch.cuda.amp.autocast():  # 启用混合精度训练
            outputs = model(samples)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
        loss_value = loss.item()  # 获取损失值
        loss /= accum_iter  # 损失除以累积步数
        
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)  # 反向传播并更新梯度
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()  # 梯度累积完成后清空梯度
        
        torch.cuda.synchronize()  # GPU同步
        metric_logger.update(loss=loss_value)  # 更新损失记录
        min_lr = 10.  # 初始化最小学习率
        max_lr = 0.  # 初始化最大学习率
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])  # 获取最小学习率
            max_lr = max(max_lr, group["lr"])  # 获取最大学习率

        metric_logger.update(lr=max_lr)  # 更新学习率记录

        loss_value_reduce = misc.all_reduce_mean(loss_value)  # 多GPU时同步损失值
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            # 使用epoch_1000x作为TensorBoard的x轴，便于不同batch size时曲线对齐
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)  # 记录训练损失
            log_writer.add_scalar('lr', max_lr, epoch_1000x)  # 记录学习率
    
    metric_logger.synchronize_between_processes()  # 多进程同步指标
    print("平均统计:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}  # 返回平均指标

@torch.no_grad()  # 禁用梯度计算以节省内存
def evaluate(data_loader, model, device, args, epoch, mode, num_class, log_writer):
    """评估模型性能"""
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    metric_logger = misc.MetricLogger(delimiter="  ")  # 初始化指标记录器
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)  # 创建输出目录
    
    model.eval()  # 设置模型为评估模式
    true_onehot, pred_onehot, true_labels, pred_labels, pred_softmax = [], [], [], [], []  # 初始化存储列表
    
    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        images, target = batch[0].to(device, non_blocking=True), batch[-1].to(device, non_blocking=True)  # 加载数据到GPU
        target_onehot = F.one_hot(target.to(torch.int64), num_classes=num_class)  # 将标签转为one-hot编码
        
        with torch.cuda.amp.autocast():  # 启用混合精度推理
            output = model(images)  # 前向传播
            loss = criterion(output, target)  # 计算损失
        output_ = nn.Softmax(dim=1)(output)  # 对输出应用Softmax得到概率分布
        output_label = output_.argmax(dim=1)  # 获取预测类别
        output_onehot = F.one_hot(output_label.to(torch.int64), num_classes=num_class)  # 预测结果转为one-hot编码
        
        metric_logger.update(loss=loss.item())  # 更新损失记录
        true_onehot.extend(target_onehot.cpu().numpy())  # 收集真实标签(one-hot)
        pred_onehot.extend(output_onehot.detach().cpu().numpy())  # 收集预测结果(one-hot)
        true_labels.extend(target.cpu().numpy())  # 收集真实标签
        pred_labels.extend(output_label.detach().cpu().numpy())  # 收集预测标签
        pred_softmax.extend(output_.detach().cpu().numpy())  # 收集预测概率
    
    accuracy = accuracy_score(true_labels, pred_labels)  # 计算准确率
    hamming = hamming_loss(true_onehot, pred_onehot)  # 计算汉明损失
    jaccard = jaccard_score(true_onehot, pred_onehot, average='macro')  # 计算Jaccard相似度
    average_precision = average_precision_score(true_onehot, pred_softmax, average='macro')  # 计算平均精度
    kappa = cohen_kappa_score(true_labels, pred_labels)  # 计算Cohen's Kappa系数
    f1 = f1_score(true_onehot, pred_onehot, zero_division=0, average='macro')  # 计算F1分数
    roc_auc = roc_auc_score(true_onehot, pred_softmax, multi_class='ovr', average='macro')  # 计算ROC AUC
    precision = precision_score(true_onehot, pred_onehot, zero_division=0, average='macro')  # 计算精确率
    recall = recall_score(true_onehot, pred_onehot, zero_division=0, average='macro')  # 计算召回率
    
    score = (f1 + roc_auc + kappa) / 3  # 计算综合得分(F1、AUC、Kappa的平均值)
    if log_writer:
        for metric_name, value in zip(['accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'average_precision', 'kappa', 'score'],
                                       [accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, kappa, score]):
            log_writer.add_scalar(f'perf/{metric_name}', value, epoch)  # 记录各项指标到TensorBoard
    
    print(f'验证损失: {metric_logger.meters["loss"].global_avg}')
    print(f'准确率: {accuracy:.4f}, F1分数: {f1:.4f}, ROC AUC: {roc_auc:.4f}, 汉明损失: {hamming:.4f},\n'
          f' Jaccard分数: {jaccard:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f},\n'
          f' 平均精度: {average_precision:.4f}, Kappa系数: {kappa:.4f}, 综合得分: {score:.4f}')
    
    metric_logger.synchronize_between_processes()  # 多进程同步指标
    
    results_path = os.path.join(args.output_dir, args.task, f'metrics_{mode}.csv')  # 结果保存路径
    file_exists = os.path.isfile(results_path)  # 检查文件是否存在
    with open(results_path, 'a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        if not file_exists:
            wf.writerow(['val_loss', 'accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'average_precision', 'kappa'])  # 写入表头
        wf.writerow([metric_logger.meters["loss"].global_avg, accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, kappa])  # 写入指标数据
    
    if mode == 'test':  # 测试模式下绘制混淆矩阵
        cm = ConfusionMatrix(actual_vector=true_labels, predict_vector=pred_labels)  # 创建混淆矩阵
        cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=True, plot_lib="matplotlib")  # 绘制归一化混淆矩阵
        plt.savefig(os.path.join(args.output_dir, args.task, 'confusion_matrix_test.jpg'), dpi=600, bbox_inches='tight')  # 保存混淆矩阵图
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, score  # 返回指标和综合得分
