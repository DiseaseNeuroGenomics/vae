from typing import Any, Dict, List, Optional, Union

import numpy as np
import pickle
import torch
import pytorch_lightning as pl
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    BatchSampler,
    SequentialSampler,
    Subset,
)


class SingleCellDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        metadata_path: str,
        cell_properties: Optional[Dict[str, Any]] = None,
        batch_properties: Optional[Dict[str, Any]] = None,
        batch_size: int = 64,
        max_cell_prop_val: float = 999,
        protein_coding_only: bool = False,
        bin_gene_count: bool = False,
        n_gene_bins: int = 16,
        restrictions: Optional[Dict[str, Any]] = {"class": "OPC"},
        max_gene_val: Optional[float] = 6.0,
        training: bool = True,
    ):

        self.metadata = pickle.load(open(metadata_path, "rb"))
        self.data_path = data_path
        self.cell_properties = cell_properties
        self.batch_properties = batch_properties
        self.n_samples = len(self.metadata["obs"]["class"])

        self._restrict_samples(restrictions)

        print(f"Number of cells {self.n_samples}")
        if "gene_name" in self.metadata["var"].keys():
            self.n_genes_original = len(self.metadata["var"]["gene_name"])
        else:
            self.n_genes_original = len(self.metadata["var"])
        self.n_cell_properties = len(cell_properties) if cell_properties is not None else 0
        self.batch_size = batch_size

        self.bin_gene_count = bin_gene_count
        self.n_gene_bins = n_gene_bins
        self.max_cell_prop_val = max_cell_prop_val
        self.protein_coding_only = protein_coding_only
        self.max_gene_val = max_gene_val
        self.training = training

        self.offset = 1 * self.n_genes_original  # UINT8 is 1 bytes

        # possibly use for embedding the gene inputs
        self.cell_classes = np.array(['Astro', 'EN', 'Endo', 'IN', 'Immune', 'Mural', 'OPC', 'Oligo'])

        # this will down-sample the number if genes if specified
        # for now, need to call this AFTER calculating offset
        # self.ad_genes = pickle.load(open("/home/masse/work/perceiver/AD_protein.pkl","rb"))
        self._get_gene_index()
        self._get_cell_prop_vals()
        self._get_batch_prop_vals()
        # self.bins = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11, 13, 16, 22, 35, 55, 9999]

    def __len__(self):
        return self.n_samples

    def _restrict_samples(self, restrictions):

        if restrictions is None:
            cond = 1
            #cond *= self.metadata["obs"]["ALS"] <= 0
            #cond *= self.metadata["obs"]["SCZ"] <= 0

            cond *= self.metadata["obs"]["ALS"] <= 99999
            cond *= self.metadata["obs"]["SCZ"] <= 99999

            self.cell_idx = np.where(cond)[0]
            self.n_samples = len(self.cell_idx)
            for k in self.metadata["obs"].keys():
                self.metadata["obs"][k] = self.metadata["obs"][k][self.cell_idx]
            print(f"Restricting samples. New number of samples: {self.n_samples}")
        else:
            cond = 1
            for k, v in restrictions.items():
                cond *= self.metadata["obs"][k] == v

            # remove ALS, SCZ
            cond *= self.metadata["obs"]["ALS"] <= 999
            cond *= self.metadata["obs"]["SCZ"] <= 999

            self.cell_idx = np.where(cond)[0]
            self.n_samples = len(self.cell_idx)
            for k in self.metadata["obs"].keys():
                self.metadata["obs"][k] = self.metadata["obs"][k][self.cell_idx]
            print(f"Restricting samples. New number of samples: {self.n_samples}")

    def _get_gene_index(self):

        # genes = pickle.load(open("/home/masse/work/mssm/living_brain/data/syn_lyso_genes", "rb"))

        if self.protein_coding_only:
            self.gene_idx = np.where(self.metadata["var"]['protein_coding'])[0]
        else:
            self.gene_idx = np.arange(self.n_genes_original)

        #self.gene_idx = [n for n, v in enumerate(self.metadata["var"]['gene_name']) if v in genes]
        self.n_genes = len(self.gene_idx)
        print(f"Sub-sampling genes. Number of genes is now {self.n_genes}")

    def _get_batch_prop_vals(self):

        if self.batch_properties is None:
            return None

        n_batch_properties = len(self.batch_properties.keys())

        self.batch_labels = np.zeros((self.n_samples, n_batch_properties), dtype=np.int64)
        self.batch_mask = np.ones((self.n_samples, n_batch_properties), dtype=np.int64)

        for n0 in range(self.n_samples):
            for n1, (k, prop) in enumerate(self.batch_properties.items()):
                cell_val = self.metadata["obs"][k][n0]
                idx = np.where(cell_val == np.array(prop["values"]))[0]
                # cell property values of -1 will imply N/A, and will be masked out
                if len(idx) == 0:
                    self.batch_labels[n0, n1] = np.nan
                    self.batch_mask[n0, n1] = 0
                else:
                    self.batch_labels[n0, n1] = idx[0]

    def _get_cell_prop_vals(self):
        """Extract the cell property value for ach entry in the batch"""
        if self.n_cell_properties == 0:
            return None

        self.labels = np.zeros((self.n_samples, self.n_cell_properties), dtype=np.float32)
        self.mask = np.ones((self.n_samples, self.n_cell_properties), dtype=np.float32)
        self.cell_class = np.zeros((self.n_samples), dtype=np.uint8)
        self.subjects = []

        for n0 in range(self.n_samples):

            self.subjects.append(self.metadata["obs"]["SubID"][n0])
            idx = np.where(self.metadata["obs"]["class"][n0] == self.cell_classes)[0]
            self.cell_class[n0] = idx[0]

            for n1, (k, cell_prop) in enumerate(self.cell_properties.items()):
                #print("AAA", k, np.unique(self.metadata["obs"][k]))
                cell_val = self.metadata["obs"][k][n0]
                if not cell_prop["discrete"]:
                    # continuous value
                    if cell_val > self.max_cell_prop_val or cell_val < -self.max_cell_prop_val or np.isnan(cell_val):
                        self.labels[n0, n1] = 0.0
                        self.mask[n0, n1] = 0.0
                    else:
                        # normalize
                        self.labels[n0, n1] = (cell_val - cell_prop["mean"]) / cell_prop["std"]
                else:
                    # discrete value
                    idx = np.where(cell_val == np.array(cell_prop["values"]))[0]
                    # cell property values of -1 will imply N/A, and will be masked out
                    if len(idx) == 0:
                        self.labels[n0, n1] = 0.0
                        self.mask[n0, n1] = 0.0
                    else:
                        self.labels[n0, n1] = idx[0]

        self.unique_subjects = np.unique(self.subjects)
        self.subjects = np.array(self.subjects)

        print("Finished creating labels")

    def _get_cell_prop_vals_batch(self, batch_idx: List[int]):

        if self.batch_properties is not None:
            return (
                self.labels[batch_idx],
                self.mask[batch_idx],
                self.batch_labels[batch_idx],
                self.batch_mask[batch_idx],
            )
        else:
            return self.labels[batch_idx], self.mask[batch_idx], None, None



    def _get_gene_vals_batch(self, batch_idx: List[int]):

        gene_vals = np.zeros((self.batch_size, self.n_genes), dtype=np.float32)
        for n, i in enumerate(batch_idx):

            j = i if self.cell_idx is None else self.cell_idx[i]
            gene_vals[n, :] = np.memmap(
                self.data_path, dtype='uint8', mode='r', shape=(self.n_genes_original,), offset=j * self.offset
            )[self.gene_idx].astype(np.float32)

        return gene_vals

    def _bin_gene_count(self, x: np.ndarray) -> np.ndarray:
        # NOT CURRENTLY USED
        return np.digitize(x, self.bins)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        # NOT CURRENTLY USED
        x = x * self.normalize_total / np.sum(x) if self.normalize_total is not None else x
        x = np.log1p(x) if self.log_normalize else x
        x = np.minimum(x, self.max_gene_val) if self.max_gene_val is not None else x
        return x

    def _prepare_data(self, batch_idx):


        #############################
        if self.training:
            pass
            #b = self.batch_labels[batch_idx[0]]
            #idx = np.where((b[0] == self.batch_labels[:, 0]) * (b[1] == self.batch_labels[:, 1]))[0]
            #N = len(idx)
            #replace = False if N >= self.batch_size else True
            #batch_idx = np.random.choice(idx, size=self.batch_size, replace=replace)


            #subject = np.random.choice(self.unique_subjects)
            #idx = np.where(self.subjects == subject)[0]
            #N = len(idx)
            #replace = False if N >= self.batch_size else True
            #batch_idx = np.random.choice(idx, size=self.batch_size, replace=replace)

        ############################



        # get input and target data, returned as numpy arrays
        gene_vals = self._get_gene_vals_batch(batch_idx)
        cell_prop_vals, cell_mask, batch_labels, batch_mask = self._get_cell_prop_vals_batch(batch_idx)

        return gene_vals, cell_prop_vals, cell_mask, batch_labels, batch_mask


    def __getitem__(self, batch_idx: Union[int, List[int]]):

        if isinstance(batch_idx, int):
            batch_idx = [batch_idx]

        if len(batch_idx) != self.batch_size:
            raise ValueError("Index length not equal to batch_size")

        gene_vals, cell_prop_vals, cell_mask, batch_labels, batch_mask = self._prepare_data(batch_idx)

        return (gene_vals, cell_prop_vals, cell_mask, batch_labels, batch_mask)


class DataModule(pl.LightningDataModule):

    # data_path: Path to directory with preprocessed data.
    # classify: Name of column from `obs` table to add classification task with. (optional)
    # Fraction of median genes to mask for prediction.
    # batch_size: Dataloader batch size
    # num_workers: Number of workers for DataLoader.

    def __init__(
        self,
        train_data_path: str,
        train_metadata_path: str,
        test_data_path: str,
        test_metadata_path: str,
        batch_size: int = 32,
        num_workers: int = 16,
        cell_properties: Optional[Dict[str, Any]] = None,
        batch_properties: Optional[Dict[str, Any]] = None,
        bin_gene_count: bool = False,
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.train_metadata_path = train_metadata_path
        self.test_data_path = test_data_path
        self.test_metadata_path = test_metadata_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cell_properties = cell_properties
        self.batch_properties = batch_properties
        self.bin_gene_count = bin_gene_count
        self._get_cell_prop_info()
        self._get_batch_prop_info()

    def _get_batch_prop_info(self):

        metadata = pickle.load(open(self.train_metadata_path, "rb"))

        if self.batch_properties is None:
            pass
        else:
            # assuming all batch keys are discrete
            for k in self.batch_properties.keys():
                cell_vals = metadata["obs"][k]

                if self.batch_properties[k]["values"] is None:
                    unique_list = np.unique(cell_vals)
                    # remove nans, negative values, or anything else suspicious
                    idx = [
                        n for n, u in enumerate(unique_list) if (
                                isinstance(u, str) or (u >= 0 and u < 999)
                        )
                    ]
                    self.batch_properties[k]["values"] = unique_list[idx]


    def _get_cell_prop_info(self, max_cell_prop_val = 999):
        """Extract the list of uniques values for each cell property (e.g. sex, cell type, etc.) to be predicted"""

        self.n_cell_properties = len(self.cell_properties) if self.cell_properties is not None else 0

        metadata = pickle.load(open(self.train_metadata_path, "rb"))

        # not a great place for this, but needed
        self.n_genes = len(metadata["var"]["gene_name"])

        if self.n_cell_properties > 0:

            for k, cell_prop in self.cell_properties.items():
                # skip if required field are already present as this function can be called multiple
                # times if using multiple GPUs

                cell_vals = metadata["obs"][k]

                if "freq" in self.cell_properties[k] or "mean" in self.cell_properties[k]:
                    continue
                if not cell_prop["discrete"]:
                    # for cell properties with continuous value, determine the mean/std for normalization
                    # remove nans, negative values, or anything else suspicious
                    if k in ["CERAD", "BRAAK_AD"]:
                        unique_list = np.unique(cell_vals)
                        print("XYYXX", k, unique_list)
                        unique_list = unique_list[unique_list > -999]
                        self.cell_properties[k]["mean"] = np.mean(unique_list)
                        self.cell_properties[k]["std"] = np.std(unique_list)
                        print("CELL PROP INFO", k, self.cell_properties[k]["mean"], self.cell_properties[k]["std"])
                    else:
                        idx = [n for n, cv in enumerate(cell_vals) if cv >= 0 and cv < max_cell_prop_val]
                        self.cell_properties[k]["mean"] = np.mean(cell_vals[idx])
                        self.cell_properties[k]["std"] = np.std(cell_vals[idx])
                        print("CELL PROP INFO", k, self.cell_properties[k]["mean"], self.cell_properties[k]["std"])

                elif cell_prop["discrete"] and cell_prop["values"] is None:
                    # for cell properties with discrete value, determine the possible values if none were supplied
                    # and find their distribution
                    unique_list, counts = np.unique(cell_vals, return_counts=True)
                    # remove nans, negative values, or anything else suspicious
                    idx = [
                        n for n, u in enumerate(unique_list) if (
                            isinstance(u, str) or (u >= 0 and u < max_cell_prop_val)
                        )
                    ]
                    self.cell_properties[k]["values"] = unique_list[idx]
                    self.cell_properties[k]["freq"] = counts[idx] / np.mean(counts[idx])
                    print("CELL PROP INFO",k, self.cell_properties[k]["freq"])

                elif cell_prop["discrete"] and cell_prop["values"] is not None:
                    unique_list, counts = np.unique(cell_vals, return_counts=True)
                    idx = [n for n, u in enumerate(unique_list) if u in cell_prop["values"]]
                    self.cell_properties[k]["freq"] = counts[idx] / np.mean(counts[idx])
                    print("CELL PROP INFO", k, self.cell_properties[k]["freq"])

        else:
            self.cell_prop_dist = None


    def setup(self, stage):

        self.train_dataset = SingleCellDataset(
            self.train_data_path,
            self.train_metadata_path,
            cell_properties=self.cell_properties,
            batch_properties=self.batch_properties,
            batch_size=self.batch_size,
            bin_gene_count=self.bin_gene_count,
            training=True,
        )
        self.val_dataset = SingleCellDataset(
            self.test_data_path,
            self.test_metadata_path,
            cell_properties=self.cell_properties,
            batch_properties=self.batch_properties,
            batch_size=self.batch_size,
            bin_gene_count=self.bin_gene_count,
            training=False,
        )

        self.n_genes = self.train_dataset.n_genes
        print(f"number of genes {self.n_genes}")


    # return the dataloader for each split
    def train_dataloader(self):
        sampler = BatchSampler(
            RandomSampler(self.train_dataset),
            batch_size=self.train_dataset.batch_size,
            drop_last=True,
        )
        dl = DataLoader(
            self.train_dataset,
            batch_size=None,
            batch_sampler=None,
            sampler=sampler,
            num_workers=self.num_workers,
        )
        return dl

    def val_dataloader(self):
        sampler = BatchSampler(
            RandomSampler(self.val_dataset),
            batch_size=self.val_dataset.batch_size,
            drop_last=True,
        )
        dl = DataLoader(
            self.val_dataset,
            batch_size=None,
            batch_sampler=None,
            sampler=sampler,
            num_workers=self.num_workers,
        )
        return dl
