
"""
Logging utils
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
assert hasattr(wandb, '__version__')

from lib.general import colorstr
from lib.loggers.wandb.wandb_utils import WandbLogger

LOGGERS = ('csv', 'tb', 'wandb')

class Loggers():
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger
        self.include = include
        self.keys = [
                    'train/box_loss', 'train/obj_loss', 'train/cls_loss', 'train/kp_loss', # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', # metrics
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss', 'val/kp_loss', # val loss
                    'x/lr0', 'x/lr1', 'x/lr2', # params
                    ]
        for k in LOGGERS:
            setattr(self, k, None)
        self.csv = True

        if not wandb:
            prefix = colorstr('Weights & Biases: ')
            s = f"{prefix}run 'pip install wandb' to automatically track and visualize YOLOv5 runs"
            print(s)
        
        s = self.save_dir
        if 'tb' in self.include and not self.opt.evolve:
            prefix = colorstr('TensorBoard: ')
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))
        
        if wandb and 'wandb' in self.include:
            wandb_artifact_resume = isinstance(self.opt.resume, str) and self.opt.resume.startswith('wandb-artifact://')
            run_id = torch.load(self.weights).get('wandb_id') if self.opt.resume and not wandb_artifact_resume else None
            self.opt.hyp = self.hyp
            self.wandb = WandbLogger(self.opt, run_id)
        else:
            self.wandb = None

    def on_pretrain_routine_end(self):
        """
        Callback runs on pre-train routine end
        """
        paths = self.save_dir.glob('*labels*.jpg')
        if self.wandb:
            self.wandb.log({"Labels": [wandb.Image(str(x), caption=x.name) for x in paths]})
        
    def on_train_batch_end(self, ni, model, imgs, targets, paths, plots, sync_bn):
        """
        Callback runs on train batch end
        """
        if plots:
            pass
    
    def on_train_epoch_end(self, epoch):
        if self.wandb:
            self.wandb.current_epoch = epoch + 1
    
    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        """
        Callback runs at the end of each fit (train+val) epoch
        """
        x = {k: v for k, v in zip(self.keys, vals)}
        if self.csv:
            pass

        if self.tb:
            for k, v in x.items():
                self.tb.add_scaler(k, v, epoch)
        
        if self.wandb:
            self.wandb.log(x)
            self.wandb.end_epoch(best_result=best_fitness == fi)
    
    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        if self.wandb:
            if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch) and self.opt.save_period != -1:
                self.wandb.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)
    
    def on_train_end(self, last, best, plots, epoch):
        pass