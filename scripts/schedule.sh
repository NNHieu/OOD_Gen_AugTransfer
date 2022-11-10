#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py trainer.max_epochs=5 logger=csv

python src/train.py trainer.max_epochs=10 logger=csv

ckpt_path=$(python -c "from src.utils import rglob_checkpoints; print(rglob_checkpoints('logs/train/runs/mnist/'))")

python src/train.py -m experiment=cifar10 'datamodule/transforms=glob(*)' test=false trainer.max_epochs=3 seed=100
