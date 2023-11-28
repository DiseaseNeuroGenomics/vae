
cell_properties = {
    "AD": {"discrete": True, "values": [0, 1], "stop_grad": True},
    #"Dementia_probable": {"discrete": True, "values": [0, 1], "stop_grad": False},
    "Dementia": {"discrete": True, "values": [0, 1], "stop_grad": False},
    #"Dementia_graded": {"discrete": False, "values": [0, 0.5, 1], "stop_grad": True},
    #"CERAD": {"discrete": True, "values": [1,2,3,4], "stop_grad": True},
    #"BRAAK_AD": {"discrete": True, "values": [0,1,2,3,4,5,6], "stop_grad": False},

    #"Dementia_graded": {"discrete": False, "values": [-1], "stop_grad": True},
    "CERAD": {"discrete": False, "values": [-1], "stop_grad": True},
    "BRAAK_AD": {"discrete": False, "values": [-1], "stop_grad": False},
    #"diff_CE_BR": {"discrete": False, "values": [-1], "stop_grad": True},
    #"apoe": {"discrete": True, "values": [0,1,2], "stop_grad": False},
    "Sex": {"discrete": True, "values": ["Male", "Female"], "stop_grad": False},
    "Brain_bank": {"discrete": True, "values": ["MSSM", "RUSH"], "stop_grad": False},
    #"class": {"discrete": True, "values": ['Astro', 'EN', 'Endo', 'IN', 'Immune', 'Mural', 'OPC', 'Oligo'], "stop_grad": True},
    #"subclass": {"discrete": True, "values": None, "stop_grad": True},
    "Age": {"discrete": False, "values": [-1], "stop_grad": True},
    #"Vascular": {"discrete": True, "values": [0, 1], "stop_grad": False},
    #"FTD": {"discrete": True, "values": [0, 1], "stop_grad": False},
    #"PMI": {"discrete": False, "values": [-1],  "stop_grad": True},
    #"SCZ": {"discrete": True, "values": [0, 1],  "stop_grad": False},
    #"ALS": {"discrete": True, "values": [0, 1],  "stop_grad": False},
    #"PD": {"discrete": True, "values": [0, 1],  "stop_grad": False},
    # "other_disorder": {"discrete": True, "values": [0, 1],  "stop_grad": False},
    "SubID": {"discrete": True, "values": None,  "stop_grad": False},
    #"n_counts": {"discrete": False, "values": [-1],  "stop_grad": True},
}


batch_properties = {
    "subclass": {"discrete": True, "values": None},
    "subtype": {"discrete": True, "values": None},
}

batch_properties = None

dataset_cfg = {
    "data_path": "/home/masse/work/data/mssm_rush/data.dat",
    "metadata_path": "/home/masse/work/data/mssm_rush/metadata.pkl",
    "cell_properties": cell_properties,
    "batch_size": 1024,
    "num_workers": 12,
    "batch_properties": batch_properties,
    "protein_coding_only": True,
    "cell_restrictions": {"class": "Astro"},
    "mixup": False,
    "group_balancing": "bd",
}

model_cfg = {
    "n_layers": 1,
    "n_hidden": 128,
    "n_hidden_decoder": 128,
    "n_hidden_library": 128,
    "n_latent": 8,
    "n_latent_cell_decoder": 8,
    "dropout_rate": 0.5,
    "input_dropout_rate": 0.5,
    "grad_reverse_lambda": 0.05,
    "grad_reverse_list": ["SubID", "Sex", "Brain_bank", "Age"],
}

task_cfg = {
    "learning_rate": 3e-4,
    "weight_decay": 0.0,
    "l1_lambda": 0.000,
    "gene_loss_coeff": 1e-2,
    "balance_classes": False,
    "n_epochs_kl_warmup": 1,
    "batch_properties": batch_properties,
    "save_gene_vals": False,
    "use_gdro": True,
}

trainer_cfg = {
    "n_devices": 1,
    "grad_clip_value": 0.25,
    "accumulate_grad_batches": 1,
    "precision": "bf16-mixed",
}

