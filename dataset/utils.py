import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from .imb_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100

ROOT = "~/dataset"

def build_dataset(args):
    dataset_name = args.dataset


    if dataset_name == "cifar10":
        num_class = 10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if args.train_imb == "True":
            train_dataset = IMBALANCECIFAR10(root=ROOT, train=True, download=False, transform=train_transform,
                                             rand_number=10, imb_type=args.imb_type, imb_factor=args.imb_factor)
        else:
            train_dataset = CIFAR10(root=ROOT, train=True, download=False, transform=train_transform)
        if args.val_imb == "True":
            cal_test_dataset = IMBALANCECIFAR10(root=ROOT, train=False, download=False, transform=val_transform,
                                                rand_number=10, imb_type=args.imb_type, imb_factor=args.imb_factor)
        else:
            cal_test_dataset = CIFAR10(root=ROOT, train=False, download=False,
                                 transform=val_transform)
    elif dataset_name == "cifar100":
        num_class = 100
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if args.train_imb == "True":
            train_dataset = IMBALANCECIFAR100(root=ROOT, train=True, download=False, transform=train_transform)
        else:
            train_dataset = CIFAR100(root=ROOT, train=True, download=False, transform=train_transform)
        if args.val_imb == "True":
            cal_test_dataset = IMBALANCECIFAR100(root=ROOT, train=False, download=False, transform=val_transform)
        else:
            cal_test_dataset = CIFAR100(root=ROOT, train=False, download=False,
                                       transform=val_transform)

    elif dataset_name == "imagenet":
        num_class = 1000
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load datasets
        train_dataset = torchvision.datasets.ImageFolder(
            root="/data/dataset/imagenet/images/train",
            transform=train_transform
        )

        cal_test_dataset = torchvision.datasets.ImageFolder(
            root="/data/dataset/imagenet/images/val",
            transform=val_transform
        )

    if args.algorithm != "standard":
        cal_size = int(len(cal_test_dataset) * args.cal_ratio)
        test_size = len(cal_test_dataset) - cal_size
        cal_dataset, test_dataset = random_split(cal_test_dataset, [cal_size, test_size])
    else:
        cal_dataset, test_dataset = None, cal_test_dataset
    args.num_classes = num_class
    return train_dataset, cal_dataset, test_dataset, num_class




def split_dataloader(original_dataloader, split_ratio=0.5):
        """
        Splits a DataLoader into two Datasets

        Args:
            original_dataloader (DataLoader): The original DataLoader to split.
            split_ratio (float): The ratio of the first subset (default: 0.5).

        Returns:
            subset1: Training dataset
            subset2: Calibration dataset
        """
        dataset = original_dataloader.dataset
        total_size = len(dataset)

        split_size = int(split_ratio * total_size)

        indices = torch.randperm(total_size)
        indices_subset1 = indices[:split_size]
        indices_subset2 = indices[split_size:]

        subset1 = Subset(dataset, indices_subset1)
        subset2 = Subset(dataset, indices_subset2)

        return subset1, subset2


import torch
import shutil
import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1

        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.targets[idx]

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples


def calc_confusion_mat(val_loader, model, args):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    cf = confusion_matrix(all_targets, all_preds).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt

    print('Class Accuracy : ')
    print(cls_acc)
    classes = [str(x) for x in args.cls_num_list]
    plot_confusion_matrix(all_targets, all_preds, classes)
    plt.savefig(os.path.join(args.root_log, args.store_name, 'confusion_matrix.png'))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def prepare_folders(args):
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            print(folder, 'folder')
            os.makedirs(folder)


def save_checkpoint(args, state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print(pred.t())
        # print(target.view(1, -1))
        # print(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
