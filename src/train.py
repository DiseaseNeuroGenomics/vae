from typing import List
import numpy as np
import torch
import pickle
import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
from data import DataModule
from config import dataset_cfg, model_cfg, task_cfg, trainer_cfg
from task import LitVAE
from networks import VAE, load_model
from create_train_test_splits import Splits
import gc

torch.set_float32_matmul_precision('medium')


def main(train_idx: List[int], test_idx: List[int], cell_class: str):

    # Set seed
    pl.seed_everything(43)

    for k, v in task_cfg.items():
        print(f"{k}: {v}")

    dataset_cfg["train_idx"] = train_idx
    dataset_cfg["test_idx"] = test_idx
    dataset_cfg["cell_restrictions"] = {"subclass": cell_class}

    max_epochs = 40 if cell_class in ["Mural", "Endo", "PC", "SMC", "VMLC", "Immune"] else 20


    # Set up data module
    dm = DataModule(**dataset_cfg)
    dm.setup("train")

    task_cfg["cell_properties"] = model_cfg["cell_properties"] = dm.cell_properties
    model_cfg["n_input"] = dm.train_dataset.n_genes
    model_cfg["batch_properties"] = dm.batch_properties
    network = VAE(**model_cfg)

    task_cfg["library_mean"], task_cfg["library_var"] = dm.train_dataset.library_size_stats()
    task = LitVAE(network=network, **task_cfg)

    gc.collect()

    trainer = pl.Trainer(
        enable_checkpointing=True,
        accelerator='gpu',
        devices=trainer_cfg["n_devices"],
        max_epochs=max_epochs,
        gradient_clip_val=trainer_cfg["grad_clip_value"],
        accumulate_grad_batches=trainer_cfg["accumulate_grad_batches"],
        precision=trainer_cfg["precision"],
        strategy=DDPStrategy(find_unused_parameters=True) if trainer_cfg["n_devices"] > 1 else "auto",
        limit_train_batches=2000,
        limit_val_batches=1000,
        check_val_every_n_epoch=1 if cell_class in ["Endo", "PC", "SMC", "VMLC", "Immune"] else 1,
        max_steps=40_000,
        # callbacks=[pl.callbacks.StochasticWeightAveraging(1e-4)]
        # callbacks=[checkpoint_callback],
    )
    trainer.fit(task, dm)



if __name__ == "__main__":

    metadata_fn = "/home/masse/work/data/mssm_rush/metadata.pkl"
    save_fn = "/home/masse/work/data/mssm_rush/train_test_splits.pkl"

    #splits = Splits(metadata_fn, save_fn)

    for _ in range(1):
        #splits.create_splits()

        splits1 = pickle.load(open("/home/masse/work/data/mssm_rush/train_test_splits.pkl", "rb"))

        for cell_class in [ "Oligo", "OPC", "IN_VIP", "EN_L3_5_IT_2"]:

            for split_num in range(0, 10):

                if cell_class == "Oligo" and split_num < 4:
                    continue

                print(f"Cell class: {cell_class}, split number: {split_num}")

                gc.collect()
                train_idx = splits1[split_num]["train_idx"]
                test_idx = splits1[split_num]["test_idx"]
                main(train_idx, test_idx, cell_class)