from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import pytorch_lightning as pl
from .model_zoo.resnet import create_model
from .model_zoo.resnet_kuangliu import ResNet18
from torch import nn
from torchmetrics.functional import accuracy

def parse_net_config(model_name):   
    if model_name == "resnet18":
        return ResNet18()
    return create_model(model_name=model_name, num_channels=3, num_classes=10)

class LightningModel(pl.LightningModule):
    def __init__(self, 
                 net: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net'])
        self.net = net
        self.criterion = nn.CrossEntropyLoss()

    
    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        images, labels = batch
        logits = self.net(images)
        loss = self.criterion(logits, labels)
        acc = accuracy(logits, labels)
        self.log("train/loss", loss.item(), prog_bar=False)
        self.log("train/acc", acc, prog_bar=False)
        return loss
    
    def validation_step(self, batch, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        images, labels = batch
        logits = self.net(images)
        loss = self.criterion(logits, labels)
        acc = accuracy(logits, labels)
        self.log("val/loss", loss.item(), prog_bar=True, on_step=False)
        self.log("val/acc", acc, prog_bar=True, on_step=False)
        return loss
    
    def test_step(self, batch: Any, batch_idx: int):
        images, labels = batch
        logits = self.net(images)
        loss = self.criterion(logits, labels)

        # update and log metrics
        test_acc = accuracy(logits, labels)
        self.log("test/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.parameters())
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}