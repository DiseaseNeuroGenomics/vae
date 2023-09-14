import pytorch_lightning as pl
from data import DataModule
from config import dataset_cfg, model_cfg, task_cfg, trainer_cfg
from task import LitVAE
from networks import VAE, load_model

def main():

    # Set seed
    pl.seed_everything(43)

    # check_train_test_set(dataset_cfg)

    # Set up data module
    dm = DataModule(**dataset_cfg)
    dm.setup("train")

    task_cfg["cell_properties"] = model_cfg["cell_properties"] = dm.cell_properties
    model_cfg["n_input"] = dm.n_genes
    model_cfg["batch_properties"] = dm.batch_properties
    network = VAE(**model_cfg)

    #fn = "/home/masse/work/vae/lightning_logs/version_164/checkpoints/epoch=77-step=156000.ckpt"
    #network = load_model(fn, network)


    task = LitVAE(
        network=network,
        **task_cfg,
    )

    trainer = pl.Trainer(
        enable_checkpointing=True,
        accelerator='gpu',
        devices=trainer_cfg["n_devices"],
        max_epochs=300,
        gradient_clip_val=trainer_cfg["grad_clip_value"],
        accumulate_grad_batches=trainer_cfg["accumulate_grad_batches"],
        precision=trainer_cfg["precision"],
        strategy=DDPStrategy(find_unused_parameters=True) if trainer_cfg["n_devices"] > 1 else "auto",
        limit_train_batches=2_000,
        limit_val_batches=200,
    )
    trainer.fit(task, dm)

    """
    for n, p in network.named_parameters():
        if "cell_decoder" in n or "encoder" in n:
            p.requires_grad = False

    trainer = pl.Trainer(
        enable_checkpointing=True,
        accelerator='gpu',
        devices=trainer_cfg["n_devices"],
        max_epochs=100,
        gradient_clip_val=trainer_cfg["grad_clip_value"],
        accumulate_grad_batches=trainer_cfg["accumulate_grad_batches"],
        precision=trainer_cfg["precision"],
        strategy=DDPStrategy(find_unused_parameters=True) if trainer_cfg["n_devices"] > 1 else "auto",
        limit_train_batches=2_000,
        limit_val_batches=500,
    )
    trainer.fit(task, dm)
    """

if __name__ == "__main__":

    main()