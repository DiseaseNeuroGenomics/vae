from typing import Optional
import pickle
import copy
import pandas as pd
import numpy as np
import anndata as ad
import scipy.stats as stats
import pingouin
import matplotlib.pyplot as plt


model_info = {
    "Micro": list(range(10)),
    "EN_L2_3_IT": list(range(10, 20)),
    "EN_L3_5_IT_2": list(range(20, 30)),
    "IN_SST": list(range(30, 40)),
    "IN_VIP": list(range(40, 50)),
    #"Astro": list(range(50, 60)),
}

def explained_var(x_pred, x_real):
    return 1 - np.nanvar(x_pred - x_real) / np.nanvar(x_real)

def classification_score(x_pred, x_real):
    s0 = np.sum((x_real == 0) * (x_pred < 0.5)) / np.sum(x_real == 0)
    s1 = np.sum((x_real == 1) * (x_pred >= 0.5)) / np.sum(x_real == 1)
    return (s0 + s1) / 2


class ModelResults:

    def __init__(
        self,
        data_fn: str = "/home/masse/work/data/mssm_rush/data.dat",
        meta_fn: str = "/home/masse/work/data/mssm_rush/metadata_slim.pkl",
        #gene_pathway_fn: str = "go_bp_terms_max150_top100.pkl",
        gene_pathway_fn: str = "go_bp_terms_10_150_top125.pkl",
        important_genes_fn: str = "significant_genes_0229.pkl",
        latent_dim: int = 32,
        gene_count_prior: Optional[float] = None,
        process_subclasses: bool = False,
        n_bins: int = 20,
        include_analysis_only: bool = True,
    ):

        self.data_fn = data_fn
        self.meta = pickle.load(open(meta_fn, "rb"))
        self.gene_pathways = pickle.load(open(gene_pathway_fn, "rb"))
        self.important_genes = pickle.load(open(important_genes_fn, "rb"))
        self.convert_apoe()
        self.process_subclasses = process_subclasses
        #self.obs_list = ["Dementia",  "BRAAK_AD"]
        self.obs_list = ["BRAAK_AD",  "Dementia", ]
        self.n_genes = len(self.meta["var"]["gene_name"])
        self.gene_names = self.meta["var"]["gene_name"]
        self.latent_dim = latent_dim
        self.n_bins = n_bins
        self.gene_count_prior = gene_count_prior
        self.include_analysis_only = include_analysis_only

        self.extra_obs = []
        self.obs_from_metadata = [
            "subclass", "Dementia_graded", "SubID", "include_analysis", "apoe", "Sex", "Age", "other_disorder",
            "Brain_bank", "CERAD", "prs_scaled_AD_Bellenguez", "prs_scaled_alzKunkle", "European",
        ]

        self.donor_stats = [
            "Sex", "Age", "apoe", "other_disorder", "Brain_bank", "BRAAK_AD", "Dementia", "Dementia_graded", "CERAD",
            "pred_BRAAK_AD", "pred_Dementia", "prs_scaled_AD_Bellenguez", "prs_scaled_alzKunkle", "European",
        ]

        gene_names = self.meta["var"]["gene_name"]
        self.gene_mask = np.ones((len(gene_names),))
        """
        for n, name in enumerate(gene_names):
            if name[:2] == "RP":
                self.gene_mask[n] = 0.0
        """


    def create_data(self, model_fns, subclass=None, model_average=False):

        """
        if subclass is None:
            index, cell_class, cell_subclasses = self.get_cell_index(model_fns)
            print(f"Subclasses present: {cell_subclasses}")
            print(cell_subclasses)
            assert len(cell_class) == 1, "Multiple cell classes detected!"
        """
        if model_average:
            adata = self.create_base_anndata_repeats(model_fns, subclass=subclass)
        else:
            adata = self.create_base_anndata(model_fns, subclass=subclass)

        adata = self.add_unstructured_data(adata)

        # adata = self.combine_preds_and_actual(adata)
        adata = self.add_prediction_indices(adata)

        #adata = self.add_gene_scores(adata)
        #adata = self.add_donor_stats(adata)
        adata = self.add_donor_gene_correlations(adata)
        
        #adata = self.add_pathway_means_correlations(adata)
        #adata = self.add_go_bp_pathway_scores(adata)

        #adata = self.add_pathway_donor_means(adata)



        return adata

    @staticmethod
    def normalize_data(x):
        # no normalization allows one to trest results as Poisson -> mean = variance -> supposedly better results
        x = np.float32(x)
        x = 10_000 * x / np.sum(x)
        # x = np.log1p(x)
        return x

    @staticmethod
    def subset_data_by_subclass(adata, subclass):

        return adata[adata.obs.subclass == subclass]

    @staticmethod
    def weight_probs(prob, k):
        if k == "Dementia_graded":
            w = np.array([0, 0.5, 1.0])
        elif k == "CERAD":
            w = np.array([1, 2, 3, 4])
        elif k == "BRAAK_AD":
            w = np.array([0, 1, 2, 3, 4, 5, 6])
        else:
            w = np.array([0, 1])
        w = w[None, :]

        return np.sum(prob * w, axis=1)

    def add_obs_from_metadata(self, adata):

        for k in self.obs_from_metadata:
            x = []
            for n in adata.obs["cell_idx"]:
                x.append(self.meta["obs"][k][n])
            adata.obs[k] = x

        return adata

    def combine_preds_and_actual(self, adata, alpha=0.1):

        for k in self.obs_list:
            if k == "Dementia":
                adata.obs[f"pred_{k}"] = (1-alpha) * adata.obs[f"pred_{k}"] + alpha * adata.obs["Dementia_graded"]
            else:
                adata.obs[f"pred_{k}"] = (1-alpha) * adata.obs[f"pred_{k}"] + alpha * adata.obs[k]

        return adata

    def create_single_anndata(self, z):

        n = z["Dementia"].shape[0]
        #latent = np.vstack(z["latent"])
        latent = np.zeros((n, self.latent_dim), dtype=np.uint8)
        mask = np.reshape(z["cell_mask"], (-1, z["cell_mask"].shape[-1]))

        a = ad.AnnData(latent)
        for m, k in enumerate(self.obs_list):
            try:
                a.obs[k] = z[k]
            except:
                print(f"{k} not found. Skipping.")
                continue
            idx = np.where(mask[:, m] == 0)[0]
            a.obs[k][idx] = np.nan
            if z[f"pred_{k}"].ndim == 2:
                probs = z[f"pred_{k}"]
                #print(k, probs.shape)
                # a.obs[f"pred_{k}"] = z[f"pred_{k}"][:, -1]
                a.obs[f"pred_{k}"] = self.weight_probs(probs, k)
            else:
                a.obs[f"pred_{k}"] = z[f"pred_{k}"]

        a.obs["cell_idx"] = z["cell_idx"]
        for k in self.extra_obs:
            a.obs[k] = np.array(self.meta["obs"][k])[z["cell_idx"]]
            if not isinstance(a.obs[k][0], str):
                idx = np.where(a.obs[k] < -99)[0]
                a.obs[k][idx] = np.nan

        a = self.add_obs_from_metadata(a)

        # Only include samples with include_analysis=True
        if self.include_analysis_only:
            a = a[a.obs["include_analysis"] > 0]

        return a

    def concat_arrays(self, fns):

        for n, fn in enumerate(fns):
            z = pickle.load(open(fn, "rb"))

            if n == 0:
                x = copy.deepcopy(z)
            else:
                for k in self.obs_list:
                    x[k] = np.concatenate((x[k], z[k]), axis=0)
                    x[f"pred_{k}"] = np.concatenate((x[f"pred_{k}"], z[f"pred_{k}"]), axis=0)
                for k in self.extra_obs + ["cell_idx"]:
                    x[k] = np.concatenate((x[k], z[k]), axis=0)



        return x

    def create_base_anndata_repeats(self, model_fns, subclass=None):

        x = []
        n_models = len(model_fns)
        for fns in model_fns:
            x0 = self.concat_arrays(fns)
            idx = np.argsort(x0["cell_idx"])
            for k in self.obs_list:
                x0[k] = x0[k][idx]
                x0[f"pred_{k}"] = x0[f"pred_{k}"][idx]
            for k in self.extra_obs + ["cell_idx"]:
                x0[k] = x0[k][idx]

            x.append(x0)

        x_new = copy.deepcopy(x0)
        for k in self.obs_list:
            x_new[k] = 0
            x_new[f"pred_{k}"] = 0
            for n in range(n_models):
                x_new[k] += x[n][k] / n_models
                x_new[f"pred_{k}"] += x[n][f"pred_{k}"] / n_models

        for k in x_new.keys():
            try:
                print(k, x_new[k].shape)
            except:
                continue

        adata = self.create_single_anndata(x_new)

        if subclass is not None:
            adata = adata[adata.obs.subclass == subclass]

        return adata

    def create_base_anndata(self, model_fns, subclass=None):

        for n, fn in enumerate(model_fns):
            z = pickle.load(open(fn, "rb"))
            a = self.create_single_anndata(z)
            a.obs["split_num"] = n
            if n == 0:
                adata = a.copy()
            else:
                adata = ad.concat((adata, a), axis=0)

        if subclass is not None:
            adata = adata[adata.obs.subclass == subclass]

        return adata

    def get_cell_index(self, model_fns):

        cell_idx = []
        cell_class = []
        cell_subclass = []

        for n, fn in enumerate(model_fns):
            z = pickle.load(open(fn, "rb"))
            cell_idx += z["cell_idx"].tolist()
            cell_class += self.meta["obs"]["class"][z["cell_idx"]].tolist()
            cell_subclass += self.meta["obs"]["subclass"][z["cell_idx"]].tolist()

        # assuming cell class is the same
        index = {cell_class[0]: cell_idx}
        for sc in np.unique(cell_subclass):
            idx = np.where(np.array(cell_subclass) == sc)[0]
            index[sc] = np.array(cell_idx)[idx]

        return index, np.unique(cell_class), np.unique(cell_subclass)

    def convert_apoe(self):

        self.meta["obs"]["apoe"] = np.zeros_like(self.meta["obs"]["ApoE_gt"])
        idx = np.where(self.meta["obs"]["ApoE_gt"] == 44)[0]
        self.meta["obs"]["apoe"][idx] = 2
        idx = np.where((self.meta["obs"]["ApoE_gt"] == 24) + (self.meta["obs"]["ApoE_gt"] == 34))[0]
        self.meta["obs"]["apoe"][idx] = 1
        idx = np.where(np.isnan(self.meta["obs"]["ApoE_gt"]))[0]
        self.meta["obs"]["apoe"][idx] = np.nan


    def add_gene_scores_subject(self, adata, n_bins=100):

        subjects = np.unique(adata.obs.SubID.values)
        gene_vals = np.zeros((len(subjects), self.n_genes), dtype=np.float32)
        pred_vals = np.zeros((len(subjects), len(self.obs_list)), dtype=np.float32)

        for n, s in enumerate(subjects):
            a = adata[adata.obs.SubID == s]
            for j, k in enumerate(self.obs_list):
                pred_vals[n, j] = np.mean(a.obs[f"pred_{k}"].values)

            for _, i in enumerate(a.obs["cell_idx"]):
                data = np.memmap(
                    self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                ).astype(np.float32)
                gene_vals[n, :] += data

            gene_vals[n, :] /= len(a.obs["cell_idx"])

        n_bins = 40
        percentiles = {}
        for j, k in enumerate(self.obs_list):
            v = pred_vals[:, j]
            percentiles[k] = np.percentile(v[~np.isnan(v)], np.arange(1.25, 100.5, 2.5))

        counts = {k: np.zeros(n_bins, dtype=np.float32) for k in self.obs_list}
        scores = {k: np.zeros((self.n_genes, n_bins), dtype=np.float32) for k in self.obs_list}

        for n, s in enumerate(subjects):

            for j, k in enumerate(self.obs_list):
                v = pred_vals[n, j]
                if np.isnan(v):
                    continue
                else:
                    bin_idx = np.argmin((v - percentiles[k]) ** 2)
                    counts[k][bin_idx] += 1
                    scores[k][:, bin_idx] += gene_vals[n, :]

        for k in scores.keys():
            scores[k] /= counts[k][None, :]
        name = "scores_subjects"
        adata.uns[name] = scores

        for k in self.obs_list:
            adata.uns[f"percentiles_subject_{k}"] = percentiles[k]

        return adata


    def add_gene_scores_pxr(self, adata, model_versions, n_bins=100):

        percentiles = {}
        for k in self.obs_list:
            percentiles[k] = np.percentile(adata.obs[f"pred_{k}"], np.arange(0.5, 100.5, 1.0))

        conds = [None,  {"Dementia": 0}, {"Dementia": 1}, {"Sex": "Male"}, {"Sex": "Female"}, {"apoe": 0}, {"apoe": 1}, {"apoe": 2}]
        score_names = ["", "_Dm0", "_Dm1", "_Male", "_Female", "_apoe0", "_apoe1", "_apoe2"]

        for cond, score_name in zip(conds[:3], score_names[:3]):

            print(f"Adding px_r values. Name: {score_name}")

            counts = {k: np.zeros(n_bins, dtype=np.float32) for k in self.obs_list}
            scores = {k: np.zeros((self.n_genes, n_bins), dtype=np.float32) for k in self.obs_list}

            for _, v in enumerate(model_versions):
                fn = f"{self.base_path}/version_{v}/test_results.pkl"
                z = pickle.load(open(fn, "rb"))

                adata_single = self.create_single_anndata(z)

                if cond is None:
                    a = adata_single.copy()
                else:
                    for k, v in cond.items():
                        idx = adata_single.obs[k] == v
                        a = adata_single[idx]
                        z["px_r"] = z["px_r"][idx]

                for n in range(a.X.shape[0]):

                    for j, k in enumerate(self.obs_list):
                        v = a.obs[f"pred_{k}"][n]
                        if np.isnan(v):
                            continue
                        else:
                            bin_idx = np.argmin((v - percentiles[k]) ** 2)
                            counts[k][bin_idx] += 1
                            scores[k][:, bin_idx] += z["px_r"][n, :]

            for k in scores.keys():
                scores[k] /= counts[k][None, :]
            name = "scores_pxr" + score_name
            print(f"Adding score name: {name}")
            adata.uns[name] = scores

        return adata

    def dementia_time_course(self, adata, n_bins=20):

        adata.uns["dementia_exp_var"] = np.zeros(n_bins)
        adata.uns["dementia_accuracy"] = np.zeros(n_bins)
        d = np.abs(adata.obs["pred_BRAAK_AD"].values[None, :] - adata.uns["percentiles_BRAAK_AD"][:, None])
        idx_min = np.argmin(d, axis=0)

        for n in range(n_bins):
            idx = np.where(idx_min == n)[0]
            #adata.uns["dementia_exp_var"][n] = explained_var(
            #    adata.obs[f"pred_Dementia_graded"][idx], adata.obs[f"Dementia_graded"][idx]
            #)

            adata.uns["dementia_accuracy"][n] = classification_score(
                adata.obs[f"pred_Dementia"][idx], adata.obs[f"Dementia"][idx]
            )


        return adata

    def bootstrap_dementia(self, adata, n_bins=20):

        adata.uns["BRAAK_AD_Dm_dist"] = np.zeros(n_bins)
        d = np.abs(adata.obs["pred_BRAAK_AD"].values[None, :] - adata.uns["percentiles_BRAAK_AD"][:, None])
        idx_min = np.argmin(d, axis=0)

        n_reps = 1

        for j in range(n_bins):
            for _ in range(n_reps):
                idx = np.where(idx_min == j)[0]
                n_dm0 = np.sum(adata.obs["Dementia"][idx] == 0)
                n_dm1 = np.sum(adata.obs["Dementia"][idx] == 1)
                np.random.shuffle(idx)
                idx0 = idx[: n_dm0]
                idx1 = idx[n_dm0 : n_dm0 + n_dm1]

                scores_dm0 = np.zeros(self.n_genes)
                scores_dm1 = np.zeros(self.n_genes)
                scores_sqr_dm0 = np.zeros(self.n_genes)
                scores_sqr_dm1 = np.zeros(self.n_genes)

                for _, i in enumerate(adata.obs["cell_idx"][idx0]):
                    data = np.memmap(
                        self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                    ).astype(np.float32)
                    scores_dm0 += self.normalize_data(data)
                    scores_sqr_dm0 += self.normalize_data(data)**2

                for _, i in enumerate(adata.obs["cell_idx"][idx1]):
                    data = np.memmap(
                        self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                    ).astype(np.float32)
                    scores_dm1 += self.normalize_data(data)
                    scores_sqr_dm1 += self.normalize_data(data)**2

                scores_dm0 /= n_dm0
                scores_dm1 /= n_dm1
                scores_std_dm0 = np.sqrt(scores_sqr_dm0 / n_dm0 - scores_dm0 ** 2)
                scores_std_dm1 = np.sqrt(scores_sqr_dm1 / n_dm1 - scores_dm1 ** 2)

                dist = np.abs(scores_dm0 - scores_dm1)
                z = np.sqrt(scores_std_dm0 ** 2 + scores_std_dm1 ** 2)
                adata.uns["BRAAK_AD_Dm_dist"][j] += np.sum(dist / (0.01 + z)) / n_reps

        return adata

    def combine_data(self, adata0, adata1, existing_vars=["BRAAK_AD", "CERAD"], new_var="BRAAK_CERAD"):

        assert np.all(np.array(adata0.obs.barcode) == np.array(adata1.obs.barcode)), "Barcodes don't match"

        self.obs_list += [new_var]
        s0 = np.array(adata0.obs[f"pred_{existing_vars[0]}"].values)
        s1 = np.array(adata1.obs[f"pred_{existing_vars[1]}"].values)
        adata0.obs[f"pred_{new_var}"] = (s0 / np.std(s0) + s1 / np.std(s1)) / 2

        return self.add_gene_scores(adata0)

    def add_prediction_indices(self, adata):

        for k in self.obs_list:
            preds = np.array(adata.obs[f"pred_{k}"].values)
            idx_sort = np.argsort(preds)
            idx = np.array([np.where(n == idx_sort)[0][0] for n in range(len(preds))])
            adata.obs[f"pred_idx_{k}"] = np.int64(idx * self.n_bins / len(adata))

        """
        idx_braak = np.argsort(np.argsort(adata.obs[f"pred_BRAAK_AD"].values))
        idx_dementia = np.argsort(np.argsort(adata.obs[f"pred_Dementia"].values))
        N = len(idx_braak) // 5

        good = (adata.obs[f"BRAAK_AD"].values > -99) * (adata.obs[f"Dementia"].values > -99)
        good *= ~np.isnan(np.array(adata.obs[f"BRAAK_AD"].values)) * ~np.isnan(np.array(adata.obs[f"Dementia"].values))

        y0 = np.array(adata.obs[f"pred_BRAAK_AD"].values)
        y1 = np.array(adata.obs[f"pred_Dementia"].values)
        y0 -= np.mean(y0)
        y0 /= np.std(y0)
        y1 -= np.mean(y1)
        y1 /= np.std(y1)

        adata.uns["idx_resilience"] = np.where(good * (y0 - y1 > 0.5))[0]
        adata.uns["idx_susceptible"] = np.where(good * (y1 - y0 > 0.5))[0]

        print("AAAAAAAAAAAA")
        print(len(adata.uns["idx_resilience"]))
        print(len(adata.uns["idx_susceptible"]))
        print(np.nanmean(adata.obs[f"BRAAK_AD"][adata.uns["idx_resilience"]]))
        print(np.nanmean(adata.obs[f"BRAAK_AD"][adata.uns["idx_susceptible"]]))
        print(np.nanmean(adata.obs[f"Dementia"][adata.uns["idx_resilience"]]))
        print(np.nanmean(adata.obs[f"Dementia"][adata.uns["idx_susceptible"]]))

        print("ACCURACY")
        print(explained_var(np.array(adata.obs[f"pred_BRAAK_AD"].values)[good], np.array(adata.obs[f"BRAAK_AD"].values)[good]))
        print(classification_score(np.array(adata.obs[f"pred_Dementia"].values)[good], np.array(adata.obs[f"Dementia"].values)[good]))
        1/0
        """

        return adata

    def add_gene_scores(self, adata):

        conds = [None,  {"Dementia": 0}, {"Dementia": 1}, {"Sex": "Male"}, {"Sex": "Female"}, {"apoe": 0}, {"apoe": 1}, {"apoe": 2}]
        score_names = ["", "_Dm0", "_Dm1", "_Male", "_Female", "_apoe0", "_apoe1", "_apoe2"]

        for cond, score_name in zip(conds[:3], score_names[:3]):

            if cond is None:
                a = adata.copy()
            else:
                for k, v in cond.items():
                    a = adata[adata.obs[k] == v]

            print("Condition", cond, "adata size", len(a))

            counts = {k: np.zeros(self.n_bins, dtype=np.float32) for k in self.obs_list}
            scores = {k: np.zeros((self.n_genes, self.n_bins), dtype=np.float32) for k in self.obs_list}
            scores_sqr = {k: np.zeros((self.n_genes, self.n_bins), dtype=np.float32) for k in self.obs_list}
            scores_std = {k: np.zeros((self.n_genes, self.n_bins), dtype=np.float32) for k in self.obs_list}

            print(f"Adding gene values. Name: {score_name}")

            for n, i in enumerate(a.obs["cell_idx"]):

                data = np.memmap(
                    self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                ).astype(np.float32)

                data = self.normalize_data(data)

                for j, k in enumerate(self.obs_list):
                    bin_idx = a.obs[f"pred_idx_{k}"][n]
                    counts[k][bin_idx] += 1
                    scores[k][:, bin_idx] += data
                    scores_sqr[k][:, bin_idx] += data**2

            for k in scores.keys():
                scores[k] /= counts[k][None, :]
                scores_std[k] = np.sqrt(scores_sqr[k] / counts[k][None, :] - scores[k]**2)

            adata.uns["scores" + score_name] = scores
            adata.uns["scores_std" + score_name] = scores_std

        return adata

    def add_unstructured_data(self, adata):

        adata.uns["go_bp_pathways"] = []
        adata.uns["go_bp_ids"] = []
        adata.uns["donors"] = []
        for k in self.obs_list:
            adata.uns[k] = []

        for k, v in self.gene_pathways.items():
            adata.uns["go_bp_pathways"].append(v["pathway"])
            adata.uns["go_bp_ids"].append(k)

        for subid in np.unique(adata.obs["SubID"]):
            adata.uns["donors"].append(subid)
            idx = np.where(np.array(adata.obs["SubID"].values) == subid)[0][0]
            for k in self.obs_list:
                adata.uns[k].append(adata.obs[k].values[idx])

        return adata

    def add_donor_stats(self, adata):

        n_donors = len(adata.uns["donors"])
        n_pathways = len(self.gene_pathways)
        adata.uns[f"donor_gene_means"] = np.zeros((n_donors, self.n_genes), dtype=np.float32)
        #adata.uns[f"donor_latent_means"] = np.zeros((n_donors, self.latent_dim), dtype=np.float32)
        adata.uns[f"donor_cell_count"] = np.zeros((n_donors,), dtype=np.float32)
        adata.uns[f"donor_pathway_means"] = np.zeros((n_donors, n_pathways), dtype=np.float32)


        # setting prior for pathway scores
        mean_gene_vals = np.mean(adata.uns[f"scores"]["BRAAK_AD"], axis=1)
        mask_genes = self.gene_mask
        # add a prior term
        if self.gene_count_prior is not None:
            mean_gene_vals += self.gene_count_prior

        # add a prior term
        if self.gene_count_prior is not None:
            mean_gene_vals += self.gene_count_prior

        for k in self.donor_stats:
            adata.uns[f"donor_{k}"] = np.zeros((n_donors,), dtype=np.float32)

        for m, subid in enumerate(adata.uns["donors"]):
            a = adata[adata.obs["SubID"] == subid]
            count = 1e-6
            go_scores = {}

            for n, i in enumerate(a.obs["cell_idx"]):
                data = np.memmap(
                    self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                ).astype(np.float32)

                data = self.normalize_data(data)
                adata.uns[f"donor_gene_means"][m, :] += data

                for go_id in self.gene_pathways.keys():
                    go_idx = self.gene_pathways[go_id]["gene_idx"]
                    if self.gene_count_prior is not None:
                        gene_exp = np.sum(
                            mask_genes[go_idx] * data[go_idx] / mean_gene_vals[go_idx]
                        )
                    else:
                        gene_exp = np.sum(
                            mask_genes[go_idx] * np.log1p(data[go_idx])
                        )
                    if not go_id in go_scores.keys():
                        go_scores[go_id] = [gene_exp]
                    else:
                        go_scores[go_id].append(gene_exp)

                for k in self.donor_stats:
                    if "pred_" in k:
                        adata.uns[f"donor_{k}"][m] += a.obs[f"{k}"][n]
                    elif "Sex" in k:
                        adata.uns[f"donor_{k}"][m] = float(a.obs[f"{k}"][n]=="Male")
                    elif "Brain_bank" in k:
                        adata.uns[f"donor_{k}"][m] = float(a.obs[f"{k}"][n]=="MSSM")
                    else:
                        adata.uns[f"donor_{k}"][m] = a.obs[f"{k}"][n]

                count += 1

            #adata.uns[f"donor_latent_means"][m, :] = np.mean(a.X, axis=0)
            for n, go_id in enumerate(self.gene_pathways.keys()):
                adata.uns[f"donor_pathway_means"][m, n] = np.mean(go_scores[go_id])

            adata.uns[f"donor_gene_means"][m, :] /= count
            adata.uns[f"donor_cell_count"][m] = count
            for k in self.donor_stats:
                if "pred_" in k:
                    adata.uns[f"donor_{k}"][m] /= count

        return adata

    def add_donor_gene_correlations(self, adata):

        gene_idx = self.important_genes["gene_idx"]
        gene_names = self.important_genes["gene_names"]
        adata.uns["gene_corr_names"] = gene_names

        n_donors = len(adata.uns["donors"])
        n_genes = len(gene_idx)
        adata.uns[f"donor_gene_corr"] = np.zeros((n_donors, n_genes, n_genes), dtype=np.float32)
        adata.uns[f"donor_gene_corr_counts"] = np.zeros((n_donors), dtype=np.float32)

        for m, subid in enumerate(adata.uns["donors"]):
            print(m, len(adata.uns["donors"]), subid)
            a = adata[adata.obs["SubID"] == subid]
            gene_counts = []

            for n, i in enumerate(a.obs["cell_idx"]):
                data = np.memmap(
                    self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                ).astype(np.float32)

                data = self.normalize_data(data)
                gene_counts.append(data[gene_idx])

            gene_counts = np.stack(gene_counts, axis=0)
            adata.uns[f"donor_gene_corr_counts"][m] = gene_counts.shape[0]

            for i in range(n_genes):
                for j in range(i+1, n_genes):
                    r, _ = stats.pearsonr(gene_counts[:, i], gene_counts[:, j])
                    adata.uns[f"donor_gene_corr"][m, i, j] = r
                    adata.uns[f"donor_gene_corr"][m, j, i] = r

        return adata



    def add_pathway_donor_means(self, adata, min_cells=10):

        n_pathways = len(self.gene_pathways)
        n_donors = len(adata.uns["donors"])
        max_corr_donors = 100
        count = 0
        adata.uns[f"pathway_donor_means"] = np.zeros((n_donors, n_pathways), dtype=np.float32)
        #adata.uns[f"pathway_donor_corr"] = np.zeros((max_corr_donors, n_pathways, n_pathways), dtype=np.float32)

        k = "BRAAK_AD"

        mean_gene_vals = np.mean(adata.uns[f"scores"][k], axis=1)
        # we will mask out any genes with low expression
        mask_genes = np.ones_like(mean_gene_vals)
        mask_genes = self.gene_mask
        print("XXX MEAN GENE VALS", mean_gene_vals.shape)
        # mask_genes[mean_gene_vals < gene_threshold] = 0.0

        # add a prior term
        if self.gene_count_prior is not None:
            mean_gene_vals += self.gene_count_prior

        for m, subid in enumerate(adata.uns["donors"]):
            print(m, len(adata.uns["donors"]), subid)
            a = adata[adata.obs["SubID"] == subid]
            go_scores = {}

            for n, i in enumerate(a.obs["cell_idx"]):
                data = np.memmap(
                    self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                ).astype(np.float32)

                data = self.normalize_data(data)

                for go_id in self.gene_pathways.keys():
                    go_idx = self.gene_pathways[go_id]["gene_idx"]

                    if self.gene_count_prior is not None:
                        gene_exp = np.sum(
                            mask_genes[go_idx] * data[go_idx] / mean_gene_vals[go_idx]
                        )
                    else:
                        gene_exp = np.sum(
                            mask_genes[go_idx] * np.log1p(data[go_idx])
                        )

                    if not go_id in go_scores.keys():
                        go_scores[go_id] = [gene_exp]
                    else:
                        go_scores[go_id].append(gene_exp)

            if len(go_scores[go_id]) >= min_cells:
                for n, go_id in enumerate(self.gene_pathways.keys()):
                    adata.uns[f"pathway_donor_means"][m, n] = np.mean(go_scores[go_id])
                    """
                    if count < max_corr_donors:
                        for n1, go_id1 in enumerate(self.gene_pathways.keys()):
                            if n1 <= n:
                                continue
                            r, _ = stats.pearsonr(go_scores[go_id], go_scores[go_id1])
                            adata.uns[f"pathway_donor_corr"][count, n, n1] = r
                            adata.uns[f"pathway_donor_corr"][count, n1, n] = r
                    """
                count += 1

            else:
                adata.uns[f"pathway_donor_means"][m, n] = np.nan
                #adata.uns[f"pathway_donor_corr"][m, n, :] = np.nan
                #adata.uns[f"pathway_donor_corr"][m, :, n] = np.nan

        return adata

    def add_go_bp_pathway_resilience_scores(self, adata, n_bins=20):

        for k in ["BRAAK_AD", "Dementia"]:

            n_pathways = len(self.gene_pathways)

            mean_gene_vals = np.mean(adata.uns[f"scores"][k], axis=1)
            # we will mask out any genes with low expression
            mask_genes = np.ones_like(mean_gene_vals)


            # add a prior term
            if self.gene_count_prior is not None:
                mean_gene_vals += self.gene_count_prior

            print("MEAN GENE VALS", mean_gene_vals.shape)

            adata.uns[f"pathway_mean_{k}"] = np.zeros((n_pathways, n_bins), dtype=np.float32)
            adata.uns[f"pathway_subid_mean_{k}"] = np.zeros((n_pathways, n_bins), dtype=np.float32)
            adata.uns[f"pathway_mean_resilience_{k}"] = np.zeros((n_pathways, n_bins, 2), dtype = np.float32)
            #adata.uns[f"pathway_corr_{k}"] = np.zeros((n_pathways, n_pathways, n_bins), dtype=np.float32)
            adata.uns[f"pathway_pval_{k}"] = np.ones((n_pathways, n_bins), dtype=np.float32)
            adata.uns[f"pathway_subid_pval_{k}"] = np.ones((n_pathways, n_bins), dtype=np.float32)
            adata.uns[f"pathway_ustat_{k}"] = np.ones((n_pathways, n_bins), dtype=np.float32)
            adata.uns[f"pathway_subid_ustat_{k}"] = np.ones((n_pathways, n_bins), dtype=np.float32)
            adata.uns[f"pathway_resilience_pval_{k}"] = np.ones((n_pathways, n_bins), dtype=np.float32)
            adata.uns[f"pathway_resilience_subid_pval_{k}"] = np.ones((n_pathways, n_bins), dtype=np.float32)
            adata.uns[f"pathway_resilience_ustat_{k}"] = np.ones((n_pathways, n_bins), dtype=np.float32)
            adata.uns[f"pathway_resilience_subid_ustat_{k}"] = np.ones((n_pathways, n_bins), dtype=np.float32)
            #adata.uns[f"pathway_tt_pval_{k}"] = np.ones((n_pathways, n_bins), dtype=np.float32)

            for b in range(n_bins):
                a = adata[adata.obs[f"pred_idx_{k}"] == b]
                print("Size of bin", b, len(a))

                pred_scores = []
                go_scores_subid = {}
                go_scores = {}

                for n, i in enumerate(a.obs["cell_idx"]):
                    data = np.memmap(
                        self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                    ).astype(np.float32)

                    subid = a.obs["SubID"].values[n]
                    try:
                        dementia = int(a.obs["Dementia"].values[n])
                    except:
                        dementia = -100

                    if not dementia in [0, 1]:
                        dementia = -100


                    if not dementia in go_scores_subid.keys():
                        go_scores_subid[dementia] = {}
                    if not dementia in go_scores.keys():
                        go_scores[dementia] = {}

                    data = self.normalize_data(data)
                    pred_scores.append(a.obs[f"pred_{k}"].values[n])

                    for go_id in self.gene_pathways.keys():
                        go_idx = self.gene_pathways[go_id]["gene_idx"]

                        if self.gene_count_prior is not None:
                            gene_exp = np.sum(
                                mask_genes[go_idx] * data[go_idx] / mean_gene_vals[go_idx]
                            )
                        else:
                            gene_exp = np.sum(
                                mask_genes[go_idx] * np.log1p(data[go_idx])
                            )

                        if not go_id in go_scores[dementia].keys():
                            go_scores[dementia][go_id] = [gene_exp]
                        else:
                            go_scores[dementia][go_id].append(gene_exp)

                        if not go_id in go_scores_subid[dementia].keys():
                            go_scores_subid[dementia][go_id] = {subid: [gene_exp]}
                        else:
                            if subid in go_scores_subid[dementia][go_id].keys():
                                go_scores_subid[dementia][go_id][subid].append(gene_exp)
                            else:
                                go_scores_subid[dementia][go_id][subid] = [gene_exp]

                print(f"BIN {b} mean pred score {np.mean(pred_scores)}")

                # average across subject IDs
                subject_averaged_bin_scores = {0: {}, 1: {}}
                for dementia in [0, 1]:
                    for go_id in self.gene_pathways.keys():
                        subject_averaged_bin_scores[dementia][go_id] = []
                        for subid in go_scores_subid[dementia][go_id].keys():
                            subject_averaged_bin_scores[dementia][go_id].append(np.mean(go_scores_subid[dementia][go_id][subid]))

                if b == 0:
                    first_bin_subject = {}
                    first_bin = {}
                    for go_id in self.gene_pathways.keys():
                        first_bin_subject[go_id] = (
                                subject_averaged_bin_scores[0][go_id] +
                                subject_averaged_bin_scores[1][go_id] +
                                subject_averaged_bin_scores[-100][go_id]
                        )
                        first_bin[go_id] = go_scores[0][go_id] + go_scores[1][go_id] + go_scores[-100][go_id]

                for n, go_id in enumerate(self.gene_pathways.keys()):
                    current_bin = go_scores[0][go_id] + go_scores[1][go_id] + go_scores[-100][go_id]
                    current_bin_subid = (
                            subject_averaged_bin_scores[0][go_id] +
                            subject_averaged_bin_scores[1][go_id] +
                            subject_averaged_bin_scores[-100][go_id]
                    )
                    adata.uns[f"pathway_mean_{k}"][n, b] = np.mean(current_bin)
                    adata.uns[f"pathway_subid_mean_{k}"][n, b] = np.mean(current_bin_subid)
                    if b > 0:
                        u, pval = stats.mannwhitneyu(first_bin[go_id], current_bin)
                        adata.uns[f"pathway_pval_{k}"][n, b] = pval
                        adata.uns[f"pathway_ustat_{k}"][n, b] = u / (len(first_bin[go_id]) * len(current_bin))

                        #u, pval = stats.mannwhitneyu(first_bin_subject[go_id], current_bin_subid)
                        #adata.uns[f"pathway_subid_pval_{k}"][n, b] = pval
                        #adata.uns[f"pathway_subid_ustat_{k}"][n, b] = u / (len(first_bin_subject[go_id]) * len(current_bin_subid))




                if k == "BRAAK_AD":
                    for n, go_id in enumerate(self.gene_pathways.keys()):
                        u, pval = stats.mannwhitneyu(
                            subject_averaged_bin_scores[0][go_id], subject_averaged_bin_scores[1][go_id]
                        )
                        #adata.uns[f"pathway_resilience_subid_pval_{k}"][n, b] = pval
                        #adata.uns[f"pathway_resilience_subid_ustat_{k}"][n, b] = u / (
                        #        len(subject_averaged_bin_scores[0][go_id]) * len(subject_averaged_bin_scores[1][go_id])
                        #)

                        u, pval = stats.mannwhitneyu(go_scores[0][go_id], go_scores[1][go_id])
                        adata.uns[f"pathway_resilience_pval_{k}"][n, b] = pval
                        adata.uns[f"pathway_resilience_ustat_{k}"][n, b] = u / (
                            len(go_scores[0][go_id]) * len(go_scores[1][go_id])
                        )

                    for n, go_id in enumerate(self.gene_pathways.keys()):
                        for d in [0, 1]:
                            adata.uns[f"pathway_mean_resilience_{k}"][n, b, d] = np.mean(go_scores[d][go_id])

        return adata

    def add_go_bp_pathway_scores(self, adata):

        n_pathways = len(self.gene_pathways)

        adata.uns[f"pathway_pval_resilience"] = np.ones((n_pathways, self.n_bins), dtype=np.float32)
        adata.uns[f"pathway_ustat_resilience"] = np.ones((n_pathways, self.n_bins), dtype=np.float32)

        for k in ["BRAAK_AD", "Dementia", ]:

            mean_gene_vals = np.mean(adata.uns[f"scores"][k], axis=1)
            # we will mask out any genes with low expression
            # mask_genes = np.ones_like(mean_gene_vals)
            mask_genes = self.gene_mask
            print("ADD GO BP Masking", np.mean(mask_genes))

            # add a prior term
            if self.gene_count_prior is not None:
                mean_gene_vals += self.gene_count_prior

            print("MEAN GENE VALS", mean_gene_vals.shape)

            adata.uns[f"pathway_mean_{k}"] = np.zeros((n_pathways, self.n_bins), dtype=np.float32)
            adata.uns[f"pathway_pval_{k}"] = np.ones((n_pathways, self.n_bins), dtype=np.float32)
            adata.uns[f"pathway_ustat_{k}"] = np.ones((n_pathways, self.n_bins), dtype=np.float32)


            for b in range(self.n_bins):
                a = adata[adata.obs[f"pred_idx_{k}"] == b]
                print("Size of bin", b, len(a))

                pred_scores = []
                dementia = []
                go_scores = {}

                for n, i in enumerate(a.obs["cell_idx"]):
                    data = np.memmap(
                        self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                    ).astype(np.float32)

                    data = self.normalize_data(data)
                    pred_scores.append(a.obs[f"pred_{k}"].values[n])
                    dementia.append(a.obs[f"Dementia_graded"].values[n])

                    for go_id in self.gene_pathways.keys():
                        go_idx = self.gene_pathways[go_id]["gene_idx"]

                        if self.gene_count_prior is not None:
                            gene_exp = np.sum(
                                mask_genes[go_idx] * data[go_idx] / mean_gene_vals[go_idx]
                            )
                        else:
                            gene_exp = np.sum(
                                mask_genes[go_idx] * np.log1p(data[go_idx])
                            )

                        if not go_id in go_scores.keys():
                            go_scores[go_id] = [gene_exp]
                        else:
                            go_scores[go_id].append(gene_exp)

                print(f"BIN {b} mean pred score {np.mean(pred_scores)}")

                if b == 0:
                    first_bin = {}
                    for go_id in self.gene_pathways.keys():
                        first_bin[go_id] = copy.deepcopy(go_scores[go_id])
                else:
                    for n, go_id in enumerate(self.gene_pathways.keys()):
                        adata.uns[f"pathway_mean_{k}"][n, b] = np.mean(go_scores[go_id])
                        if b > 0:
                            u, pval = stats.mannwhitneyu(first_bin[go_id], go_scores[go_id])
                            adata.uns[f"pathway_pval_{k}"][n, b] = pval
                            adata.uns[f"pathway_ustat_{k}"][n, b] = u / (len(first_bin[go_id]) * len(go_scores[go_id]))

                if k == "BRAAK_AD":
                    idx0 = np.array(dementia) == 0
                    idx1 = np.array(dementia) == 1
                    for n, go_id in enumerate(self.gene_pathways.keys()):
                        u, pval = stats.mannwhitneyu(
                            np.array(go_scores[go_id])[idx0],
                            np.array(go_scores[go_id])[idx1])
                        adata.uns[f"pathway_pval_resilience"][n, b] = pval
                        adata.uns[f"pathway_ustat_resilience"][n, b] = u / (np.sum(idx0) * np.sum(idx1))


        return adata


    def add_go_bp_pathway_scores_cell_level(self, adata):

        k = "BRAAK_AD"
        n_pathways = len(self.gene_pathways)
        n_cells = adata.shape[0]

        mean_gene_vals = np.mean(adata.uns[f"scores"][k], axis=1)
        # we will mask out any genes with low expression
        mask_genes = np.ones_like(mean_gene_vals)

        # add a prior term
        if self.gene_count_prior is not None:
            mean_gene_vals += self.gene_count_prior

        print("MEAN GENE VALS", mean_gene_vals.shape)

        pathways = {}
        for go_id in self.gene_pathways.keys():
            pathways[go_id] = np.zeros((n_cells,), dtype=np.float32)


        for n, i in enumerate(adata.obs["cell_idx"]):
            data = np.memmap(
                self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
            ).astype(np.float32)

            data = self.normalize_data(data)

            for go_id in self.gene_pathways.keys():
                go_idx = self.gene_pathways[go_id]["gene_idx"]

                if self.gene_count_prior is not None:
                    gene_exp = np.sum(
                        mask_genes[go_idx] * data[go_idx] / mean_gene_vals[go_idx]
                    )
                else:
                    gene_exp = np.sum(
                        mask_genes[go_idx] * np.log1p(data[go_idx])
                    )

                pathways[go_id] = gene_exp

        for go_id in self.gene_pathways.keys():
            adata.obs[go_id] = pathways[go_id]


        return adata


    def add_pathway_means_correlations(self, adata):

        print("Number of pathways", len(adata.uns["go_bp_pathways"]), len(self.gene_pathways))

        k = "BRAAK_AD"

        n_pathways = len(self.gene_pathways)
        mean_gene_vals = np.mean(adata.uns[f"scores"][k], axis=1)
        # we will mask out any genes with low expression
        mask_genes = self.gene_mask
        print("Mean of mask ", np.mean(mask_genes))
        #mask_genes[mean_gene_vals < self.gene_threshold] = 0.0

        # add a prior term
        if self.gene_count_prior is not None:
            mean_gene_vals += self.gene_count_prior

        print("MEAN GENE VALS", mean_gene_vals.shape)

        adata.uns[f"pathway_mean_{k}"] = np.zeros((n_pathways), dtype = np.float32)
        adata.uns[f"pathway_corr_{k}"] = np.zeros((n_pathways, n_pathways), dtype=np.float32)

        pred_scores = []
        go_scores = {}

        for n, i in enumerate(adata.obs["cell_idx"]):

            data = np.memmap(
                self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
            ).astype(np.float32)
            data = self.normalize_data(data)
            pred_scores.append(adata.obs[f"pred_{k}"].values[n])

            for go_id in self.gene_pathways.keys():
                go_idx = self.gene_pathways[go_id]["gene_idx"]
                if self.gene_count_prior is not None:
                    gene_exp = np.sum(
                        mask_genes[go_idx] * data[go_idx] / mean_gene_vals[go_idx]
                    )
                else:
                    gene_exp = np.sum(
                        mask_genes[go_idx] * np.log1p(data[go_idx])
                    )
                if not go_id in go_scores.keys():
                    go_scores[go_id] = [gene_exp]
                else:
                    go_scores[go_id].append(gene_exp)

        go_scores["model_pred"] = pred_scores

        df = pd.DataFrame(go_scores)

        for n0, go_id0 in enumerate(self.gene_pathways.keys()):
            for n1, go_id1 in enumerate(self.gene_pathways.keys()):
                if n1 <= n0:
                    continue
                s = pingouin.partial_corr(data=df, x=go_id0, y=go_id1, covar="model_pred")
                adata.uns[f"pathway_corr_{k}"][n0, n1] = s["r"].values[0]
                adata.uns[f"pathway_corr_{k}"][n1, n0] = s["r"].values[0]
            """
            for n0, go_id0 in enumerate(self.gene_pathways.keys()):
                for n1, go_id1 in enumerate(self.gene_pathways.keys()):
                    if n1 <= n0:
                        continue
                    r, _ = stats.pearsonr(go_scores[go_id0], go_scores[go_id1])
                    adata.uns[f"pathway_corr_{k}"][n0, n1, b] = r
                    adata.uns[f"pathway_corr_{k}"][n1, n0, b] = r
            """


        return adata


bad_path_words = [
    "female",
    "bone",
    "retina",
    "ureteric",
    "ear",
    "skin",
    "hair",
    "cardiac",
    "metanephros",
    "outflow",
    "sound",
    "chondrocy",
    "eye",
    "vocalization",
    "social",
    "aorta",
    "pancreas",
    "digestive",
    "cochlea",
    "optic",
    "megakaryocyte",
    "embryo",
    "ossifi",
    "anterior/posterior",
    "animal",
    "cartilage",
    "cocaine",
    "sperm",
    "blastocyst",
    "fat ",
    "mammary",
    "substantia nigra",
    "mesenchymal",
    "estrous",
    "hindbrain",
    "forebrain",
    "brain",
    "locomotory",
    "acrosome",
    "ethanol",
    "nicotine",
    "cadmium",
    "ovarian",
    "melanocyte",
    "lead",
    "thyroid",
    "dexamethasone",
    "bacterium",
    "motor",
    "lung",
    "oocyte",
    "liver",
    "odontogenesis",
    "epidermis",
    "endodermal",
    "pulmonary",
    "decidualization",
    "response to heat",
    "pituitary",
    "tissue homeo",
    "keratinocyte",
    "keratinization",
    "osteoblast",
    "epidermal",
    "sensory organ",
    "layer formation",
    "endocardial",
    "organism development",
    "embryo",
    #"killing cells",
    "protozoan",
    "smell",
    "tumor cell",
    "hemopoiesis",
    "sensory perception",
    "genitalia",
    "urine",
    "xenobiotic",
    "muscle",
]


def check_path_term(path, bad_path_words):
    for p in bad_path_words:
        if p in path:
            return True
    return False
