from typing import List

import torch
import pickle
import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
from data import DataModule
from config import dataset_cfg, model_cfg, task_cfg, trainer_cfg
from task import LitVAE
from networks import VAE, load_model
import gc

torch.set_float32_matmul_precision('medium')


def main(train_idx: List[int], test_idx: List[int], cell_class: str):

    # Set seed
    pl.seed_everything(43)

    dataset_cfg["train_idx"] = train_idx
    dataset_cfg["test_idx"] = test_idx
    dataset_cfg["cell_restrictions"] = {"class": cell_class}

    max_epochs = 100 if cell_class in ["Endo", "PC", "SMC", "VMLC"] else 25

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
        limit_train_batches=1000,
        limit_val_batches=2000,
        check_val_every_n_epoch=1,
        # callbacks=[checkpoint_callback],
    )
    trainer.fit(task, dm)



if __name__ == "__main__":

    splits = pickle.load(open("/home/masse/work/data/mssm_rush/train_test_splits.pkl", "rb"))

    for cell_class in [ "Immune", "Micro", "Astro", "OPC", "Endo", "PC", "SMC", "VMLC", "EN_L2_3_IT", "EN_L3_5_IT_2", "IN_SST", "IN_VIP"]:
    #for cell_class in ["IN"]:

        # oct 8, 1am, immune starts at v_num 124
        # oct 9, Micro Femal starts at v_num 175
        # oct 11, "Immune", "Astro", "OPC", "Endo", "EN", "IN" starts at 101

        for split_num in range(0, 10):
            gc.collect()
            train_idx = splits[split_num]["train_idx"]
            test_idx = splits[split_num]["test_idx"]
            main(train_idx, test_idx, cell_class)