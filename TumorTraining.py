# TumorTraining.py (修改后)

import argparse
import datetime
import hashlib
import os
import shutil
import sys

import numpy as np
from matplotlib import pyplot

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
# 修改：导入 AdamW 和学习率调度器
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import TumorDatasets
import TumorModel

from util.util import enumerateWithEstimate
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_PRED_P_NDX = 2
METRICS_LOSS_NDX = 3
METRICS_SIZE = 4


class ClassificationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', help='设定每个训练批次的数据加载量', default=24, type=int)
        parser.add_argument('--num-workers', help='设定用于后台加载数据的工作进程数', default=4, type=int)
        parser.add_argument('--epochs', help='设定训练批次', default=20, type=int)
        parser.add_argument('--dataset', help="指定输入模型的数据集", action='store', default='LunaDataset')
        parser.add_argument('--model', help="指定用于肿瘤分类的预训练模型", action='store', default='LunaModel')
        parser.add_argument('--malignant', help="是否将把结节分类模型训练为识别良性或恶性的模型", action='store_true',
                            default=False)
        parser.add_argument('--finetune', help="启动微调模型", default='')
        parser.add_argument('--resume', help="从指定的检查点文件恢复训练", default='')
        parser.add_argument('--finetune-depth', help="指定微调的的模型层次开，从模型头开始计算", type=int, default=2)
        parser.add_argument('--tb-prefix', default='nodule_cls', help="设定供tensorboard使用的数据文件的前缀")
        parser.add_argument('comment', help="设定供tensorboard使用的数据文件的后缀", nargs='?', default='n-100')

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.augmentation_dict = {}
        if True:
            self.augmentation_dict['flip'] = True
            self.augmentation_dict['offset'] = 0.1
            self.augmentation_dict['scale'] = 0.2
            self.augmentation_dict['rotate'] = True
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer, self.scheduler = self.initOptimizer()  # 修改: 同时返回优化器和调度器

        self.start_epoch = 1

        if self.cli_args.resume:
            print(f"Resuming training from checkpoint: {self.cli_args.resume}")
            checkpoint = torch.load(self.cli_args.resume, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            # 修改: 恢复调度器状态
            if 'scheduler_state' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.totalTrainingSamples_count = checkpoint['totalTrainingSamples_count']
            print(f"Resumed. Starting from epoch {self.start_epoch}.")

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
            self.model = self.model.to(self.device)

    def initModel(self):
        model_cls = getattr(TumorModel, self.cli_args.model)
        model = model_cls()
        if self.cli_args.finetune:
            d = torch.load(self.cli_args.finetune, map_location='cpu')
            model_blocks = [n for n, subm in model.named_children() if len(list(subm.parameters())) > 0]
            finetune_blocks = model_blocks[-self.cli_args.finetune_depth:]
            log.info(f"finetuning from {self.cli_args.finetune}, blocks {' '.join(finetune_blocks)}")
            model.load_state_dict(
                {k: v for k, v in d['model_state'].items() if k.split('.')[0] not in model_blocks[-1]}, strict=False)
            for n, p in model.named_parameters():
                if n.split('.')[0] not in finetune_blocks:
                    p.requires_grad_(False)
        return model

    def initOptimizer(self):
        # 修改: 使用 AdamW 和 CosineAnnealingLR
        lr = 0.001 if self.cli_args.finetune else 0.0001
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cli_args.epochs)
        return optimizer, scheduler

    def initTrainDl(self):
        ds_cls = getattr(TumorDatasets, self.cli_args.dataset)
        train_ds = ds_cls(val_stride=10, isValSet_bool=False, ratio_int=1)
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=self.cli_args.num_workers,
                              pin_memory=self.use_cuda)
        return train_dl

    def initValDl(self):
        ds_cls = getattr(TumorDatasets, self.cli_args.dataset)
        val_ds = ds_cls(val_stride=10, isValSet_bool=True)
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=self.cli_args.num_workers,
                            pin_memory=self.use_cuda)
        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)
            self.trn_writer = SummaryWriter(log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))
        train_dl = self.initTrainDl()
        val_dl = self.initValDl()
        best_score = 0.0
        validation_cadence = 1 if not self.cli_args.finetune else 1
        for epoch_ndx in range(self.start_epoch, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx, self.cli_args.epochs, len(train_dl), len(val_dl),
                self.cli_args.batch_size, (torch.cuda.device_count() if self.use_cuda else 1),
            ))
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx % validation_cadence == 0:
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                if score > best_score:
                    self.saveModel('tumor', epoch_ndx, True)
                    best_score = score
                else:
                    self.saveModel('tumor', epoch_ndx, False)

            # 修改: 在每个 epoch 结束后更新学习率
            self.scheduler.step()

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        train_dl.dataset.shuffleSamples()
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        batch_iter = enumerateWithEstimate(train_dl, "E{} Training".format(epoch_ndx), start_ndx=train_dl.num_workers)
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g, augment=True)
            loss_var.backward()
            self.optimizer.step()
        self.totalTrainingSamples_count += len(train_dl.dataset)
        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            batch_iter = enumerateWithEstimate(val_dl, "E{} Validation ".format(epoch_ndx),
                                               start_ndx=val_dl.num_workers)
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g, augment=False)
        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g, augment=True):
        input_t, label_t, index_t, _series_list, _center_list = batch_tup
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        index_g = index_t.to(self.device, non_blocking=True)

        if augment:
            # 修改: 正确传递 augmentation_dict
            input_g = TumorModel.augment3d(input_g, augmentation_dict=self.augmentation_dict)

        logits_g, probability_g = self.model(input_g)
        loss_g = nn.functional.cross_entropy(logits_g, label_g[:, 1], reduction="none")
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)
        _, predLabel_g = torch.max(probability_g, dim=1, keepdim=False, out=None)
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = index_g
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = predLabel_g
        metrics_g[METRICS_PRED_P_NDX, start_ndx:end_ndx] = probability_g[:, 1]
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g
        return loss_g.mean()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t, classificationThreshold=0.5):
        self.initTensorboardWriters()
        log.info("E{} {}".format(epoch_ndx, type(self).__name__))
        pos = 'mal' if self.cli_args.dataset == 'MalignantLunaDataset' else 'pos'
        neg = 'ben' if self.cli_args.dataset == 'MalignantLunaDataset' else 'neg'

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] == 0
        negPred_mask = metrics_t[METRICS_PRED_NDX] == 0
        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        truePos_count = pos_correct
        falsePos_count = int((negLabel_mask & posPred_mask).sum())
        falseNeg_count = int((posLabel_mask & negPred_mask).sum())

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()
        metrics_dict['correct/all'] = (pos_correct + neg_correct) / metrics_t.shape[1] * 100
        metrics_dict['correct/neg'] = (neg_correct) / neg_count * 100 if neg_count > 0 else 0
        metrics_dict['correct/pos'] = (pos_correct) / pos_count * 100 if pos_count > 0 else 0

        precision = metrics_dict['pr/precision'] = truePos_count / np.float64(truePos_count + falsePos_count) if (
                                                                                                                             truePos_count + falsePos_count) > 0 else 0
        recall = metrics_dict['pr/recall'] = truePos_count / np.float64(truePos_count + falseNeg_count) if (
                                                                                                                       truePos_count + falseNeg_count) > 0 else 0
        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        threshold = torch.linspace(1, 0, steps=100)
        tpr = (metrics_t[None, METRICS_PRED_P_NDX, posLabel_mask] >= threshold[:, None]).sum(
            1).float() / pos_count if pos_count > 0 else torch.zeros_like(threshold)
        fpr = (metrics_t[None, METRICS_PRED_P_NDX, negLabel_mask] >= threshold[:, None]).sum(
            1).float() / neg_count if neg_count > 0 else torch.zeros_like(threshold)
        fp_diff = fpr[1:] - fpr[:-1]
        tp_avg = (tpr[1:] + tpr[:-1]) / 2
        auc = (fp_diff * tp_avg).sum()
        metrics_dict['auc'] = auc

        log.info(("E{} {:8} {loss/all:.4f} loss, {correct/all:-5.1f}% correct, {pr/precision:.4f} precision, "
                  "{pr/recall:.4f} recall, {pr/f1_score:.4f} f1 score, {auc:.4f} auc").format(epoch_ndx, mode_str,
                                                                                              **metrics_dict))
        log.info(("E{} {:8} {loss/neg:.4f} loss, {correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})")
                 .format(epoch_ndx, mode_str + '_' + neg, neg_correct=neg_correct, neg_count=neg_count, **metrics_dict))
        log.info(("E{} {:8} {loss/pos:.4f} loss, {correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})")
                 .format(epoch_ndx, mode_str + '_' + pos, pos_correct=pos_correct, pos_count=pos_count, **metrics_dict))

        writer = getattr(self, mode_str + '_writer')
        for key, value in metrics_dict.items():
            key = key.replace('pos', pos).replace('neg', neg)
            writer.add_scalar(key, value, self.totalTrainingSamples_count)
        fig = pyplot.figure()
        pyplot.plot(fpr, tpr)
        writer.add_figure('roc', fig, self.totalTrainingSamples_count)
        bins = np.linspace(0, 1)
        writer.add_histogram('label_neg', metrics_t[METRICS_PRED_P_NDX, negLabel_mask], self.totalTrainingSamples_count,
                             bins=bins)
        writer.add_histogram('label_pos', metrics_t[METRICS_PRED_P_NDX, posLabel_mask], self.totalTrainingSamples_count,
                             bins=bins)

        # 修改: 使用 f1_score 作为选择最佳模型的标准，因为它更均衡
        score = metrics_dict['pr/f1_score']
        # score = metrics_dict['auc'] # AUC 也是一个很好的选择
        return score

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'rawdata', 'tumor_checkPoint', 'models', self.cli_args.tb_prefix,
            '{}_{}_{}.{}.state'.format(type_str, self.time_str, self.cli_args.comment, self.totalTrainingSamples_count)
        )
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)
        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        # 修改: 同时保存优化器和调度器的状态
        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'scheduler_state': self.scheduler.state_dict(),  # 新增
            'scheduler_name': type(self.scheduler).__name__,  # 新增
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)
        log.info("Saved model params to {}".format(file_path))
        if isBest:
            best_path = os.path.join(
                'rawdata', 'tumor', 'models', self.cli_args.tb_prefix,
                '{}_{}_{}.{}.state'.format(type_str, self.time_str, self.cli_args.comment, 'best')
            )
            shutil.copyfile(file_path, best_path)
            log.info("Saved model params to {}".format(best_path))
        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())


if __name__ == '__main__':
    ClassificationTrainingApp().main()