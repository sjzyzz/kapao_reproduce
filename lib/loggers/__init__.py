"""
Logging utils
"""

import torch
from torch.utils.tensorboard import SummaryWriter

from lib.general import colorstr

LOGGERS = ('csv', 'tb')


class Loggers():

    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger
        self.include = include
        self.keys = [
            'train/box_loss',
            'train/obj_loss',
            'train/cls_loss',
            'train/kp_loss',  # train loss
            'metrics/precision',
            'metrics/recall',
            'metrics/mAP_0.5',
            'metrics/mAP_0.5:0.95',  # metrics
            'val/box_loss',
            'val/obj_loss',
            'val/cls_loss',
            'val/kp_loss',  # val loss
            'x/lr0',
            'x/lr1',
            'x/lr2',  # params
        ]
        for k in LOGGERS:
            setattr(self, k, None)
        self.csv = True

        s = self.save_dir
        if 'tb' in self.include and not self.opt.evolve:
            prefix = colorstr('TensorBoard: ')
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent} --bind_all', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))

    def on_pretrain_routine_end(self):
        """
        Callback runs on pre-train routine end
        """
        pass

    def on_train_batch_end(self, ni, model, imgs, targets, paths, plots, sync_bn):
        """
        Callback runs on train batch end
        """
        pass

    def on_train_epoch_end(self, epoch):
        pass

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        """
        Callback runs at the end of each fit (train+val) epoch
        """
        x = {k: v for k, v in zip(self.keys, vals)}
        if self.csv:
            file = self.save_dir / 'results.csv'
            n = len(x) + 1
            s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + self.keys)).rstrip(',') + '\n')
            # add header
            with open(file, 'a') as f:
                f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

        if self.tb:
            for k, v in x.items():
                self.tb.add_scaler(k, v, epoch)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        pass

    def on_train_end(self, last, best, plots, epoch):
        pass