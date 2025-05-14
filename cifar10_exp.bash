CUDA_VISIBLE_DEVICES=0 python main.py --imb_type exp --imb_factor 0.5 --loss LDAM --train_rule None --model resnet20 --epochs 200 --dataset cifar10 --score thr --train_imb True --val_imb False --save True --load False

CUDA_VISIBLE_DEVICES=0 python main.py --imb_type exp --imb_factor 0.4 --loss LDAM --train_rule None --model resnet20 --epochs 200 --dataset cifar10 --score thr --train_imb True --val_imb False --save True --load False

CUDA_VISIBLE_DEVICES=0 python main.py --imb_type exp --imb_factor 0.3 --loss LDAM --train_rule None --model resnet20 --epochs 200 --dataset cifar10 --score thr --train_imb True --val_imb False --save True --load False

CUDA_VISIBLE_DEVICES=0 python main.py --imb_type exp --imb_factor 0.2 --loss LDAM --train_rule None --model resnet20 --epochs 200 --dataset cifar10 --score thr --train_imb True --val_imb False --save True --load False

CUDA_VISIBLE_DEVICES=0 python main.py --imb_type exp --imb_factor 0.1 --loss LDAM --train_rule None --model resnet20 --epochs 200 --dataset cifar10 --score thr --train_imb True --val_imb False --save True --load False

