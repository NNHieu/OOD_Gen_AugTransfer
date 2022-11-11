import os

from typing import Any, Callable, Optional, Tuple

from torchvision.datasets import MNIST
from torchvision.datasets.cifar import CIFAR10


class CustomMNIST(MNIST):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.data = self.data.numpy()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

class CustomCIFAR10(CIFAR10):

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
        # img = Image.fromarray(img)

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target