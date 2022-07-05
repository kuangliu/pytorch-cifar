import os.path
import pickle
from tkinter.ttk import LabeledScale
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

# from .utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset


class CIFAR10_C(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Directory of data
        labels_root (string): Directory of corresponding labels
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``

    """

    def __init__(
        self,
        root: str,
        labels_root: str,
        transform: Optional[Callable] = None
    ) -> None:

        super().__init__(root, transform=transform)


        # self.data: Any = []
        # self.targets = []

        # now load the picked numpy arrays
        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.data = None
        self.targets = None

        self.data = np.load(root)
        self.targets = np.load(labels_root)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.data)
