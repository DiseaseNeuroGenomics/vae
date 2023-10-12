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
        cell_idx: List[int],
        cell_properties: Optional[Dict[str, Any]] = None,
        batch_properties: Optional[Dict[str, Any]] = None,
        frequency: Optional[np.array] = None,
        batch_size: int = 64,
        max_cell_prop_val: float = 999,
        protein_coding_only: bool = False,
        cell_restrictions: Optional[Dict[str, Any]] = {"class": "OPC"},
        max_gene_val: Optional[float] = 6.0,
        gene_idx: Optional[List[int]] = None,
        subject_batch_size: int = 1,
        training: bool = True,

    ):

        self.metadata = pickle.load(open(metadata_path, "rb"))
        self.data_path = data_path
        self.cell_idx = cell_idx
        self.n_samples = len(cell_idx)
        self.cell_properties = cell_properties
        self.batch_properties = batch_properties
        self.frequency = frequency
        self.subject_batch_size = subject_batch_size
        self.training = training

        self._restrict_samples(cell_restrictions)

        print(f"Number of cells {self.n_samples}")
        if "gene_name" in self.metadata["var"].keys():
            self.n_genes_original = len(self.metadata["var"]["gene_name"])
        else:
            self.n_genes_original = len(self.metadata["var"])

        self.n_cell_properties = len(cell_properties) if cell_properties is not None else 0
        self.batch_size = batch_size

        self.max_cell_prop_val = max_cell_prop_val
        self.protein_coding_only = protein_coding_only
        self.max_gene_val = max_gene_val


        # offset is needed for memmap loading
        self.offset = 1 * self.n_genes_original  # UINT8 is 1 bytes

        # possibly use for embedding the gene inputs
        self.cell_classes = np.array(['Astro', 'EN', 'Endo', 'IN', 'Immune', 'Mural', 'OPC', 'Oligo'])

        # this will down-sample the number if genes if specified
        if gene_idx is None:
            #self._gene_stats()
            self._get_gene_index()
        else:
            self.gene_idx = gene_idx
            self.n_genes = len(self.gene_idx )
        self._get_cell_prop_vals()
        self._get_batch_prop_vals()
        # self.bins = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11, 13, 16, 22, 35, 55, 9999]

    def __len__(self):
        return self.n_samples

    def _gene_stats(self):

        N = 50_000
        counts = np.zeros(self.n_genes_original, dtype=np.float32)
        idx = self.cell_idx[:N] if len(self.cell_idx) > N else self.cell_idx

        for n in idx:
            data = np.memmap(
                self.data_path, dtype='uint8', mode='r', shape=(self.n_genes_original,), offset=n * self.offset
            )
            counts += np.clip(data, 0, 1)

        self.metadata["var"]["percent_cells"] = 100 * counts / len(idx)

    def library_size_stats(self, N: int = 200_000):

        counts = []
        idx = self.cell_idx[:N] if len(self.cell_idx) > N else self.cell_idx

        for n in idx:
            data = np.memmap(
                self.data_path, dtype='uint8', mode='r', shape=(self.n_genes_original,), offset=n * self.offset
            )[self.gene_idx].astype(np.float32)
            counts.append(np.log1p(np.sum(data, axis=-1)))

        return np.mean(counts), np.var(counts)


    def _restrict_samples(self, restrictions):

        if restrictions is not None:
            cond = np.zeros(len(self.metadata["obs"]["class"]), dtype=np.uint8)
            cond[self.cell_idx] = 1
            #cond *= self.metadata["obs"]["SCZ"] != 1
            #cond *= self.metadata["obs"]["ALS"] != 1
            #cond *= self.metadata["obs"]["Sex"] == "Female"

            for k, v in restrictions.items():
                if isinstance(v, list):
                    cond *= np.sum(np.stack([self.metadata["obs"][k] == v1 for v1 in v]), axis=0)
                else:
                    cond *= self.metadata["obs"][k] == v

            self.cell_idx = np.where(cond)[0]
            self.n_samples = len(self.cell_idx)
        """
        if self.training or True:
            cond1 = self.metadata["obs"]["Dementia"] == 0
            cond = cond * cond1
            self.cell_idx = np.where(cond)[0]
            self.n_samples = len(self.cell_idx)
        """

        for k in self.metadata["obs"].keys():
            self.metadata["obs"][k] = np.array(self.metadata["obs"][k])[self.cell_idx]

        print(f"Restricting samples; number of samples: {self.n_samples}")
        print(f"Subclasses: {np.unique(self.metadata['obs']['subclass'])}")
        print(f"Subtypes: {np.unique(self.metadata['obs']['subtype'])}")
        print(f"APOE: {np.unique(self.metadata['obs']['ApoE_gt'])}")
        print(f"Dementia: {np.unique(self.metadata['obs']['Dementia_graded'])}")


    def _get_gene_index(self):

        if self.protein_coding_only:
            cond = 1
            cond *= self.metadata["var"]['percent_cells'] >= 2.0
            #cond *= self.metadata["var"]["ribosomal"]
            # cond *= self.metadata["var"]['gene_chrom'] != "X"
            # cond *= self.metadata["var"]['protein_coding']

            self.gene_idx = np.where(cond)[0]
        else:
            self.gene_idx = np.arange(self.n_genes_original)

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
                    self.batch_labels[n0, n1] = -100
                    self.batch_mask[n0, n1] = 0
                else:
                    self.batch_labels[n0, n1] = idx[0]

        print("BATCH PROP MASK", np.mean(self.batch_mask, axis=0))

    def _get_cell_prop_vals(self):
        """Extract the cell property value for ach entry in the batch"""
        if self.n_cell_properties == 0:
            return None

        self.labels = np.zeros((self.n_samples, self.n_cell_properties), dtype=np.float32)
        self.mask = np.ones((self.n_samples, self.n_cell_properties), dtype=np.float32)
        self.cell_freq = np.ones((self.n_samples,), dtype=np.float32)
        self.cell_class = np.zeros((self.n_samples), dtype=np.uint8)
        self.subjects = []

        for n0 in range(self.n_samples):

            self.subjects.append(self.metadata["obs"]["SubID"][n0])
            idx = np.where(self.metadata["obs"]["class"][n0] == self.cell_classes)[0]
            self.cell_class[n0] = idx[0]

            d = self.metadata["obs"]["Dementia"][n0]
            c = self.metadata["obs"]["CERAD"][n0] - 1
            b = self.metadata["obs"]["BRAAK_AD"][n0]
            if d >= 0 and c >= 0 and b >= 0:
                self.cell_freq[n0] = np.clip(1 / self.frequency[d, c, b], 0.1, 10.0)

            for n1, (k, cell_prop) in enumerate(self.cell_properties.items()):
                cell_val = self.metadata["obs"][k][n0]
                if not cell_prop["discrete"]:
                    # continuous value
                    if cell_val > self.max_cell_prop_val or cell_val < -self.max_cell_prop_val or np.isnan(cell_val):
                        self.labels[n0, n1] = -100
                        self.mask[n0, n1] = 0.0
                    else:
                        # normalize
                        self.labels[n0, n1] = (cell_val - cell_prop["mean"]) / cell_prop["std"]
                else:
                    # discrete value
                    idx = np.where(cell_val == np.array(cell_prop["values"]))[0]
                    # cell property values of -1 will imply N/A, and will be masked out
                    if len(idx) == 0:
                        self.labels[n0, n1] = -100
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

        gene_vals = np.zeros((len(batch_idx), self.n_genes), dtype=np.float32)
        for n, i in enumerate(batch_idx):
            j = self.cell_idx[i]
            gene_vals[n, :] = np.memmap(
                self.data_path, dtype='uint8', mode='r', shape=(self.n_genes_original,), offset=j * self.offset
            )[self.gene_idx].astype(np.float32)

        return gene_vals

    def _prepare_data(self, batch_idx):

        n_original_size = len(batch_idx)

        if self.subject_batch_size > 1:
            batch_idx = []
            while len(batch_idx) < n_original_size:
                subid = np.random.choice(self.unique_subjects)
                subid_idx = np.nonzero(self.subjects == subid)[0]
                if len(subid_idx) >= self.subject_batch_size:
                    idx = np.random.choice(subid_idx, size=self.subject_batch_size, replace=False)
                    for i in idx:
                        batch_idx.append(i)

        # get input and target data, returned as numpy arrays
        gene_vals = self._get_gene_vals_batch(batch_idx)
        cell_prop_vals, cell_mask, batch_labels, batch_mask = self._get_cell_prop_vals_batch(batch_idx)


        return gene_vals, cell_prop_vals, cell_mask, batch_labels, batch_mask


    def __getitem__(self, batch_idx: Union[int, List[int]]):

        if isinstance(batch_idx, int):
            batch_idx = [batch_idx]

        gene_vals, cell_prop_vals, cell_mask, batch_labels, batch_mask = self._prepare_data(batch_idx)
        cell_idx = self.cell_idx[batch_idx]
        freq = self.cell_freq[batch_idx]

        return (gene_vals, cell_prop_vals, cell_mask, batch_labels, batch_mask, cell_idx, freq)


class DataModule(pl.LightningDataModule):

    # data_path: Path to directory with preprocessed data.
    # classify: Name of column from `obs` table to add classification task with. (optional)
    # Fraction of median genes to mask for prediction.
    # batch_size: Dataloader batch size
    # num_workers: Number of workers for DataLoader.

    def __init__(
        self,
        data_path: str,
        metadata_path: str,
        train_idx: List[int],
        test_idx: List[int],
        batch_size: int = 32,
        num_workers: int = 16,
        cell_properties: Optional[Dict[str, Any]] = None,
        batch_properties: Optional[Dict[str, Any]] = None,
        protein_coding_only: bool = False,
        cell_restrictions: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cell_properties = cell_properties
        self.batch_properties = batch_properties
        self.protein_coding_only = protein_coding_only
        self.cell_restrictions = cell_restrictions
        self._get_cell_prop_info()
        self._get_batch_prop_info()

    def _get_batch_prop_info(self):

        metadata = pickle.load(open(self.metadata_path, "rb"))

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

        metadata = pickle.load(open(self.metadata_path, "rb"))

        # not a great place for this, but needed
        self.n_genes = len(metadata["var"]["gene_name"])

        self.frequency = np.zeros((2, 4, 7), dtype=np.float32)
        for j in range(len(metadata["obs"]["CERAD"])):
            d = metadata["obs"]["Dementia"][j]
            c = metadata["obs"]["CERAD"][j] - 1
            b = metadata["obs"]["BRAAK_AD"][j]
            if d>=0 and c>=0 and b>=0:
                self.frequency[d,c,b] += 1
        self.frequency /= np.mean(self.frequency)

        print("FREQUENCY: ", np.mean(self.frequency), np.min(self.frequency), np.max(self.frequency))

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
                        unique_list = unique_list[unique_list > -999]
                        self.cell_properties[k]["mean"] = np.mean(unique_list)
                        self.cell_properties[k]["std"] = np.std(unique_list)
                        print(f"Property: {k}, mean: {self.cell_properties[k]['mean']}, std: {self.cell_properties[k]['std']}")
                    else:
                        idx = [n for n, cv in enumerate(cell_vals) if cv >= 0 and cv < max_cell_prop_val]
                        self.cell_properties[k]["mean"] = np.mean(cell_vals[idx])
                        self.cell_properties[k]["std"] = np.std(cell_vals[idx])
                        print(f"Property: {k}, mean: {self.cell_properties[k]['mean']}, std: {self.cell_properties[k]['std']}")

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
                    print(f"Property: {k}, values: {self.cell_properties[k]['values']}")

                elif cell_prop["discrete"] and cell_prop["values"] is not None:

                    unique_list, counts = np.unique(cell_vals, return_counts=True)
                    idx = [n for n, u in enumerate(unique_list) if u in cell_prop["values"]]
                    self.cell_properties[k]["freq"] = counts[idx] / np.mean(counts[idx])
                    print(f"Property: {k}, values: {self.cell_properties[k]['values']}")

        else:
            self.cell_prop_dist = None


    def setup(self, stage):

        self.train_dataset = SingleCellDataset(
            self.data_path,
            self.metadata_path,
            self.train_idx,
            cell_properties=self.cell_properties,
            batch_properties=self.batch_properties,
            frequency=self.frequency,
            batch_size=self.batch_size,
            protein_coding_only=self.protein_coding_only,
            cell_restrictions=self.cell_restrictions,
            training=True,
        )
        self.val_dataset = SingleCellDataset(
            self.data_path,
            self.metadata_path,
            self.test_idx,
            cell_properties=self.cell_properties,
            batch_properties=self.batch_properties,
            frequency=self.frequency,
            batch_size=self.batch_size,
            protein_coding_only=self.protein_coding_only,
            cell_restrictions=self.cell_restrictions,
            gene_idx=self.train_dataset.gene_idx,
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
            SequentialSampler(self.val_dataset),
            #RandomSampler(self.val_dataset),
            batch_size=self.val_dataset.batch_size,
            drop_last=False,
        )
        dl = DataLoader(
            self.val_dataset,
            batch_size=None,
            batch_sampler=None,
            sampler=sampler,
            num_workers=self.num_workers,
        )
        return dl
