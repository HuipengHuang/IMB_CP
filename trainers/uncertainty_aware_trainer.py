import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from overrides import overrides
from .trainer import Trainer
from common import utils


class UncertaintyAwareTrainer():
    """
    PAPER: Training Uncertainty-Aware Classifiers with Conformalized Deep Learning
    LINK: https://arxiv.org/pdf/2205.05878
    """
    def __init__(self, args, num_classes):
        super().__init__(args, num_classes)


    def train(self, data_loader, epochs):
        """This method is modified because
         uncertainty-aware training requires randomly split the training data into two disjoint set."""
        pred_set, calibrate_set = utils.split_dataloader(data_loader, 0.5)
        self.net.train()
        if self.adapter is None:
            for epoch in range(epochs):
                pred_loader = DataLoader(pred_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
                calibrate_loader = DataLoader(calibrate_set, batch_size=self.batch_size, shuffle=True, drop_last=True)

                for pred_batch, cal_batch in tqdm(zip(pred_loader, calibrate_loader)):
                    pred_data, pred_target = pred_batch
                    cal_data, cal_target = cal_batch

                    self.train_batch(torch.cat((pred_data, cal_data), dim=0), torch.cat((pred_target, cal_target), dim=0))
        else:
            for epoch in range(epochs):
                pred_loader = DataLoader(pred_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
                calibrate_loader = DataLoader(calibrate_set, batch_size=self.batch_size, shuffle=True, drop_last=True)

                for pred_batch, cal_batch in tqdm(zip(pred_loader, calibrate_loader)):
                    pred_data, pred_target = pred_batch
                    cal_data, cal_target = cal_batch

                    self.train_batch_with_adapter(torch.cat((pred_data,cal_data), dim=0), torch.cat((pred_target,cal_target), dim=0))
