#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py trainer.max_epochs=5 logger=csv

python src/train.py trainer.max_epochs=10 logger=csv

ckpt_path=$(python -c "from src.utils import rglob_checkpoints; print(rglob_checkpoints('logs/train/runs/mnist/'))")

python src/train.py -m experiment=cifar10 'datamodule/transforms=glob(*)' test=false trainer.max_epochs=3 seed=100
python eval.py dmt/datamodule=cifar10 dmt/model=resnet32_sgd_cosine seed=41 +dmt.datamodule.test_on_train=0.4 'run_dir_pattern=outputs/cifar10_resnet32/**/train.log'