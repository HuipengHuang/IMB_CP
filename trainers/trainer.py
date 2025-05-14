import warnings
import torch.nn as nn
import numpy as np
import torch
import models
from dataset.utils import ImbalancedDatasetSampler
from loss.utils import get_loss_function
from .utils import get_optimizer
from predictors.get_predictor import get_predictor
from loss.losses import LDAMLoss, FocalLoss
from tqdm import tqdm
class Trainer:
    """
    Trainer class that implement all the functions regarding training.
    All the arguments are passed through args."""
    def __init__(self, args, num_classes):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = models.utils.build_model(args.model, (args.pretrained == "True"), num_classes=num_classes, device=self.device, args=args)
        self.batch_size = args.batch_size

        self.optimizer = get_optimizer(args, self.net)

        self.predictor = get_predictor(args, self.net)
        self.args = args
        self.num_classes = num_classes
        self.loss_function = get_loss_function(args, self.predictor)

    def train_batch(self, data, target):
        data = data.to(self.device)
        target = target.to(self.device)
        logits = self.net(data)
        loss = self.loss_function(logits, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_epoch(self, train_loader, epoch, val_loader=None):
        self.net.train()

        self.adjust_learning_rate(self.optimizer, epoch, self.args)
        train_dataset = train_loader.dataset
        cls_num_list = train_dataset.get_cls_num_list()
        if self.args.train_rule == 'None':
            train_sampler = None
            per_cls_weights = None
        elif self.args.train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_dataset)
            per_cls_weights = None
        elif self.args.train_rule == 'Reweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
        elif self.args.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(self.device)
        else:
            warnings.warn('Sample rule is not listed')

        if self.args.loss == 'standard':
            self.loss_function = nn.CrossEntropyLoss(weight=per_cls_weights)
        elif self.args.loss == 'LDAM':
            self.loss_function = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights)
        elif self.args.loss == 'focal':
            self.loss_function = FocalLoss(weight=per_cls_weights, gamma=1)
        else:
            warnings.warn('Loss type is not listed')
            return

        for data, target in tqdm(train_loader, desc=f"{epoch+1}"):
            self.train_batch(data, target)
        if val_loader is not None:
            self.net.eval()
            with torch.no_grad():
                top1 = 0
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    logits = self.net(data)
                    pred = torch.argmax(logits, dim=-1)
                    top1 += pred.eq(target).sum().item()
                print(f"Top1 accuracy of Epoch:{epoch+1} {top1 / len(val_loader.dataset)}")

    def train(self, train_loader, epochs, val_loader=None):
        for epoch in range(epochs):
            self.train_epoch(train_loader=train_loader, epoch=epoch, val_loader=val_loader)

        if self.args.save_model == "True":
            models.utils.save_model(self.args, self.net)

    def adjust_learning_rate(self, optimizer, epoch, args):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        epoch = epoch + 1
        if epoch <= 5:
            lr = args.learning_rate * epoch / 5
        elif epoch > 180:
            lr = args.learning_rate * 0.0001
        elif epoch > 160:
            lr = args.learning_rate * 0.01
        else:
            lr = args.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
