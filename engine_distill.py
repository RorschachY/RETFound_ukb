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


class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy loss for soft targets (distillation)."""

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        return -(target * log_probs).sum(dim=1).mean()


def train_one_epoch_distill(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
):
    """Train one epoch using soft labels (distillation)."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.zero_grad()

    if log_writer:
        print(f'日志目录: {log_writer.log_dir}')

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if len(batch) == 3:
            samples, targets, soft_targets = batch
        else:
            samples, targets = batch
            soft_targets = None

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        soft_targets = soft_targets.to(device, non_blocking=True) if soft_targets is not None else None

        # Do not apply mixup when soft labels are used; if mixup_fn is passed and soft labels absent, allow it.
        if mixup_fn and soft_targets is None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if soft_targets is not None:
                loss = criterion(outputs, soft_targets)
            else:
                loss = criterion(outputs, targets)
        loss_value = loss.item()
        loss /= accum_iter

        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("平均统计:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_distill(data_loader, model, device, args, epoch, mode, num_class, log_writer):
    """Evaluate with soft labels (distillation)."""
    criterion = SoftTargetCrossEntropy()
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)

    model.eval()
    true_onehot, pred_onehot, true_labels, pred_labels, pred_softmax = [], [], [], [], []

    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        if len(batch) == 3:
            images, target, soft_targets = batch
        else:
            images, target = batch
            soft_targets = None

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        soft_targets = soft_targets.to(device, non_blocking=True) if soft_targets is not None else None
        target_onehot = F.one_hot(target.to(torch.int64), num_classes=num_class)

        with torch.cuda.amp.autocast():
            output = model(images)
            if soft_targets is not None:
                loss = criterion(output, soft_targets)
            else:
                loss = criterion(output, target)
        output_ = nn.Softmax(dim=1)(output)
        output_label = output_.argmax(dim=1)
        output_onehot = F.one_hot(output_label.to(torch.int64), num_classes=num_class)

        metric_logger.update(loss=loss.item())
        true_onehot.extend(target_onehot.cpu().numpy())
        pred_onehot.extend(output_onehot.detach().cpu().numpy())
        true_labels.extend(target.cpu().numpy())
        pred_labels.extend(output_label.detach().cpu().numpy())
        pred_softmax.extend(output_.detach().cpu().numpy())

    accuracy = accuracy_score(true_labels, pred_labels)
    hamming = hamming_loss(true_onehot, pred_onehot)
    jaccard = jaccard_score(true_onehot, pred_onehot, average='macro')
    average_precision = average_precision_score(true_onehot, pred_softmax, average='macro')
    kappa = cohen_kappa_score(true_labels, pred_labels)
    f1 = f1_score(true_onehot, pred_onehot, zero_division=0, average='macro')
    roc_auc = roc_auc_score(true_onehot, pred_softmax, multi_class='ovr', average='macro')
    precision = precision_score(true_onehot, pred_onehot, zero_division=0, average='macro')
    recall = recall_score(true_onehot, pred_onehot, zero_division=0, average='macro')

    score = (f1 + roc_auc + kappa) / 3
    if log_writer:
        for metric_name, value in zip(['accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'average_precision', 'kappa', 'score'],
                                       [accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, kappa, score]):
            log_writer.add_scalar(f'perf/{metric_name}', value, epoch)

    print(f'验证损失: {metric_logger.meters["loss"].global_avg}')
    print(f'准确率: {accuracy:.4f}, F1分数: {f1:.4f}, ROC AUC: {roc_auc:.4f}, 汉明损失: {hamming:.4f},\n'
          f' Jaccard分数: {jaccard:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f},\n'
          f' 平均精度: {average_precision:.4f}, Kappa系数: {kappa:.4f}, 综合得分: {score:.4f}')

    metric_logger.synchronize_between_processes()

    results_path = os.path.join(args.output_dir, args.task, f'metrics_{mode}.csv')
    file_exists = os.path.isfile(results_path)
    with open(results_path, 'a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        if not file_exists:
            wf.writerow(['val_loss', 'accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'average_precision', 'kappa'])
        wf.writerow([metric_logger.meters["loss"].global_avg, accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, kappa])

    if mode == 'test':
        cm = ConfusionMatrix(actual_vector=true_labels, predict_vector=pred_labels)
        cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=True, plot_lib="matplotlib")
        plt.savefig(os.path.join(args.output_dir, args.task, 'confusion_matrix_test.jpg'), dpi=600, bbox_inches='tight')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, score
