from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

AUG_DICT = {
    'none': None,
    'blur': A.GaussianBlur(blur_limit=(3,7), p=0.5),
    'gaussian_noise': A.GaussNoise(var_limit=(10, 50), p=0.5),
    'rotation': A.Rotate(limit=30),
    'brightness_constrast': A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
    'rgb_shift': A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    'shift': A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0, rotate_limit=0, p=0.5),
    # 'scale': A.RandomScale(scale_limit=0.1, p=0.5),
    'perspective': A.Perspective(scale=(0.05, 0.1), p=0.5),
    'flip': A.HorizontalFlip(),
    'crop': A.Compose([
        A.PadIfNeeded(36, 36),
        A.RandomCrop(32, 32, p=1.),
    ]),
}


class CustomCIFAR10(CIFAR10):

    def __getitem__(self, index: int):
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

class Cifar10DM(LightningDataModule):
    def __init__(self, data_root, train_bs, test_bs, num_workers, augmentation=None, test_augmentation=None, train_val_split=(45000, 5000), pin_memory=False, test_on_train=0.) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.transform_train = self._parse_augmentation(augmentation)
        self.transform_test = self._parse_augmentation(test_augmentation)
    
    def _parse_augmentation(self, augmentation):
        transform = []
        if augmentation is not None:
            if isinstance(augmentation, str) and augmentation != 'none':
                transform.append(AUG_DICT[augmentation])
            else:
                for a in augmentation:
                    if a != "none": transform.append(AUG_DICT[a])
        transform.append(A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform.append(ToTensorV2())
        return A.Compose(transform)

    
    def prepare_data(self) -> None:
        CustomCIFAR10(self.hparams.data_root, train=True, download=True)
        CustomCIFAR10(self.hparams.data_root, train=False, download=True)


    def setup(self, stage: str) -> None:
        self.data_train = CustomCIFAR10(self.hparams.data_root, train=True, download=False, transform=self.transform_train)
        if self.hparams.test_on_train > 0:
            val_size = int(len(self.data_train) * self.hparams.test_on_train)
            train_val_split = [len(self.data_train) - val_size, val_size]
            _ , self.data_val = random_split(self.data_train, train_val_split, 
                                            generator=torch.Generator().manual_seed(42))
            self.data_test = self.data_val
        else:
            self.data_test = CustomCIFAR10(self.hparams.data_root, train=False, download=False, transform=self.transform_test)
            self.data_val = self.data_test


    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.data_train, 
                          batch_size=self.hparams.train_bs, 
                          shuffle=True,
                          pin_memory=self.hparams.pin_memory,
                          num_workers=self.hparams.num_workers,
                          )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.data_val, 
                          batch_size=self.hparams.test_bs, 
                          shuffle=False,
                          pin_memory=self.hparams.pin_memory,
                          num_workers=self.hparams.num_workers,
                          )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.data_val, 
                          batch_size=self.hparams.test_bs, 
                          shuffle=False,
                          pin_memory=self.hparams.pin_memory,
                          num_workers=self.hparams.num_workers,
                          )
    
