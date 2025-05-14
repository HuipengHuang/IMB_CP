import os
from models.resnet_cifar import resnet20, resnet32
import torch
import torchvision.models
def build_model(model_type, pretrained, num_classes, device, args):
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        use_norm = True if args.loss == 'LDAM' else False
        print("usenorm")
        print(use_norm)
        if model_type == "resnet20":
            net = resnet20(num_classes, use_norm)
        elif model_type == "resnet32":
            net = resnet32(num_classes, use_norm)
    else:
        if model_type == 'resnet18':
            net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_type == "resnet34":
            net = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_type == "resnet50":
            net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_type == "resnet101":
            net = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_type == "resnet152":
            net = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_type == "densenet121":
            net = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_type == "densenet161":
            net = torchvision.models.densenet161(weights=torchvision.models.DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_type == "resnext50":
            net = torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        if hasattr(net, "fc"):
            net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
        else:
                net.classifier = torch.nn.Linear(net.classifier.in_features, num_classes)

    if args.load == "True":
        load_model(args, net)

    return net.to(device)

def load_model(args, net):
        p = f"./data/{args.dataset}_{args.imb_type}_{args.imb_factor}_{args.loss_type}_{args.train_rule}_{args.model}{0}net.pth"
        net.load_state_dict(torch.load(p))

def save_model(args, net):
    i = 0
    while (True):
        p = f"./data/{args.dataset}_{args.imb_type}_{args.imb_factor}_{args.loss_type}_{args.train_rule}_{args.model}{i}net.pth"

        if os.path.exists(p):
            i += 1
            continue
        torch.save(net.state_dict(), p)
        break

