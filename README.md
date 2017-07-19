# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Pros & cons
Pros:
- Built-in data loading and augmentation, very nice!
- Training is fast, maybe even a little bit faster.
- Very memory efficient!

Cons:
- No progress bar, sad :(
- No built-in log.

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| VGG16             | 92.64%      |
| ResNet18          | 93.02%      |
| ResNet50          | 93.62%      |
| ResNet101         | 93.75%      |
| ResNeXt29(32x4d)  | 94.73%      |
| ResNeXt29(2x64d)  | 94.82%      |
| DenseNet121       | 95.04%      |
| ResNet18(pre-act) | 94.75%      |
| DPN92             | 95.16%      |

## Learning rate adjustment
I manually change the `lr` during training:
- `0.1` for epoch `[0,150)`
- `0.01` for epoch `[150,250)`
- `0.001` for epoch `[250,350)`

Resume the training with `python main.py --resume --lr=0.01`
