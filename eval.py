import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import pytorch_lightning as pl
import pandas as pd
import hydra
from hydra import compose, initialize
from pathlib import Path
import yaml
import omegaconf
from core.datamodules.cifar10 import AUG_DICT

def scan_checkpoint(pattern):
    run_dir = [d.parent for d in root.rglob(pattern)]
    return run_dir

def evaluate_ckpt(eval_cfg, run_dir: Path):
    print(str(run_dir))
    # Get training config
    with open(run_dir / ".hydra" / "config.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)
        train_cfg = omegaconf.DictConfig(train_cfg)

    # with initialize(version_base=None, 
    #                 config_path=run_dir.relative_to(Path.cwd()) / '.hydra'):
    #     train_cfg = compose(config_name="config", 
    #                   return_hydra_config=True, 
    #                   overrides=["paths.output_dir=notebooks/tmp"])

    # Initializing lightning model
    model = hydra.utils.instantiate(train_cfg.dmt.model)
    model = model.load_from_checkpoint(run_dir / 'lightning_logs/version_0/checkpoints/epoch=199-step=78200.ckpt', 
            net=model.net)
    
    # Initializing trainer
    trainer: pl.Trainer = hydra.utils.instantiate(eval_cfg.dmt.trainer, logger=None, callbacks=None)

    # Load metric file
    metric_file = run_dir / f"{eval_cfg.dmt.datamodule.test_on_train:0.2f}_metrics.csv"
    if metric_file.exists():
        metric_df = pd.read_csv(metric_file)
        print('Saved to ', str(metric_file))
    else:
        metric_df = pd.DataFrame(columns=["test_aug", "train_aug", "eval_seed", "train_seed"])

    for aug in AUG_DICT.keys():
        if eval_cfg.seed in metric_df[metric_df["test_aug"] == aug]["eval_seed"].unique():
            # log.info(f"Already evaluated. Skipping !!!")
            print("Already evaluated. Skipping !!!")
            continue
        if eval_cfg.get("seed"): 
            pl.seed_everything(eval_cfg.seed, workers=True)
        
        if eval_cfg.dmt.datamodule.test_on_train > 0:
            datamodule = hydra.utils.instantiate(eval_cfg.dmt.datamodule, 
                                                augmentation=aug)
        else:
            datamodule = hydra.utils.instantiate(eval_cfg.dmt.datamodule, 
                                                test_augmentation=aug)

        trainer.test(model=model, datamodule=datamodule, verbose=False)
        metric_dict = trainer.callback_metrics 
        for k,v in metric_dict.items():
            metric_dict[k] = v.item()
        metric_dict["test_aug"] = aug
        metric_dict["train_aug"] = train_cfg.dmt.datamodule.get("augmentation", "unk")
        metric_dict["eval_seed"] = eval_cfg.seed
        metric_dict["train_seed"] = train_cfg.seed
        metric_df = pd.concat([metric_df, pd.Series(metric_dict).to_frame().T], ignore_index=True)
        metric_df.to_csv(metric_file, index=False)

@hydra.main(version_base=None, config_path="configs", config_name="eval")
def eval(cfg):
    run_dirs = scan_checkpoint(cfg.run_dir_pattern)
    for run_dir in run_dirs:
        evaluate_ckpt(cfg, run_dir)

if __name__ == "__main__":
    eval()

    
