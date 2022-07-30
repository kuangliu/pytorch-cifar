# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# Start training with: 
python main.py --net ResNet18 --train --test

# Start training for n_cls experiments:
python main_n_cls.py --net MobileNetV2 --train --test --num_class 5

# You can manually resume the training with: 
python main.py --net ResNet18 --train --test --resume --lr=0.01
```

## Testing
```
# Test only on GPU
python main.py --net ResNet18 --test

# Test only on GPU with pruning (0.3)
python main.py --net ResNet18 --test --prune --pruning_rate 0.3

# Test only on CPU
python main.py --net ResNet18 --test --select_device cpu
```

# Trained Weights
[Google Drive](https://drive.google.com/drive/folders/1DRcb7uw1goot8doydHAc0ip3us5zjilk?usp=sharing)

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      |

Pruning [Reference Link](https://github.com/ultralytics/yolov5/blob/a2a1ed201d150343a4f9912d644be2b210206984/utils/torch_utils.py#L174)
