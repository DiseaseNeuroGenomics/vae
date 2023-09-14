
cell_properties = {
    "AD": {"discrete": True, "values": [0, 1], "stop_grad": False},
    "Dementia": {"discrete": True, "values": [0, 1], "stop_grad": False},
    "CERAD": {"discrete": False, "values": [-1], "stop_grad": False},
    "BRAAK_AD": {"discrete": False, "values": [-1], "stop_grad": False},
    "ApoE_gt": {"discrete": True, "values": None, "stop_grad": True},
    "Sex": {"discrete": True, "values": ["Male", "Female"], "stop_grad": True},
    "Brain_bank": {"discrete": True, "values": ["MSSM", "RUSH"], "stop_grad": True},
    "class": {"discrete": True, "values": ['Astro', 'EN', 'Endo', 'IN', 'Immune', 'Mural', 'OPC', 'Oligo'], "stop_grad": True},
    "subclass": {"discrete": True, "values": None, "stop_grad": True},
    "Age": {"discrete": False, "values": [-1], "stop_grad": True},
    "PMI": {"discrete": False, "values": [-1],  "stop_grad": True},
    "SCZ": {"discrete": True, "values": [0, 1],  "stop_grad": True},
    "SubID": {"discrete": True, "values": None,  "stop_grad": True},
}

"""
batch_properties = {
    "Brain_bank": {"values": ['MSSM', 'RUSH']},
    "class": {"values": ['Astro', 'EN', 'Endo', 'IN', 'Immune', 'Mural', 'OPC', 'Oligo']},
    "subclass": {"values": None},
    "Sex": {"values": ["Male", "Female"]},
}
"""
batch_properties = None

dataset_cfg = {
    "train_data_path": "/home/masse/work/perceiver/mssm_rush_data_all_genes/train_data.dat",
    "train_metadata_path": "/home/masse/work/perceiver/mssm_rush_data_all_genes/train_metadata.pkl",
    "test_data_path": "/home/masse/work/perceiver/mssm_rush_data_all_genes/test_data.dat",
    "test_metadata_path": "/home/masse/work/perceiver/mssm_rush_data_all_genes/test_metadata.pkl",
    "cell_properties": cell_properties,
    "batch_size": 1024,
    "num_workers": 8,
    "bin_gene_count": False,
    "batch_properties": batch_properties,
}

model_cfg = {
    "n_layers": 2,
    "n_hidden": 64,
    "n_hidden_decoder": 64,
    "n_hidden_library": 64,
    "n_latent": 32,
    "n_latent_cell_decoder": 32,
    "dropout_rate": 0.5,
}

task_cfg = {
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "gene_loss_coeff": 5e-4,
    "balance_classes": True,
    "n_epochs_kl_warmup": 1,
    "batch_properties": batch_properties,
}

trainer_cfg = {
    "n_devices": 1,
    "grad_clip_value": 1.0,
    "accumulate_grad_batches": 1,
    "precision": "bf16-mixed",
}

