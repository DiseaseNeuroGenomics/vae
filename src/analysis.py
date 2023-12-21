
import pickle
import pandas as pd
import numpy as np
import anndata as ad
import scipy.stats as stats
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
            base_path: str = "/home/masse/work/vae/src/lightning_logs",
            data_fn: str = "/home/masse/work/data/mssm_rush/data.dat",
            meta_fn: str = "/home/masse/work/data/mssm_rush/metadata_slim.pkl",
            gene_pathway_fn: str = "gene_pathways.pkl",
            curated_pathways_fn: str = "ad_related_go_bp_pathways_1220.csv",
            latent_dim: int = 32,
            process_subclasses: bool = False,
    ):

        self.base_path = base_path
        self.data_fn = data_fn
        self.meta = pickle.load(open(meta_fn, "rb"))
        self.gene_pathways = pickle.load(open(gene_pathway_fn, "rb"))
        self.curated_pathways = pd.read_csv(curated_pathways_fn)
        self.convert_apoe()
        self.process_subclasses = process_subclasses
        #self.obs_list = ["Dementia",  "BRAAK_AD"]
        self.obs_list = ["Dementia", "CERAD", "BRAAK_AD", "Dementia_graded"]
        self.n_genes = len(self.meta["var"]["gene_name"])
        self.gene_names = self.meta["var"]["gene_name"]
        self.latent_dim = latent_dim
        self.extra_obs = [
            "barcode", "Dementia", "MCI", "Sex", "apoe", "Age", "r03_r04", "ethnicity",
            "ApoE_gt", "Brain_bank", "SCZ", "SubID", "subclass", "PMI", "Vascular", "ALS",
            "include_analysis",
        ]
        self.extra_obs = [
            "Sex", "apoe", "Age",
            "ApoE_gt", "Brain_bank",
            "include_analysis",
        ]
        """
        self.extra_obs = [
            "Dementia", "MCI", "Sex", "apoe",
            "ApoE_gt", "Brain_bank", "SCZ", "SubID", "subclass", "PMI", "Vascular", "ALS",
        ]
        """

        self.get_gene_idx()

    def create_data(self, model_versions, subclass=None):

        if subclass is None:
            index, cell_class, cell_subclasses = self.get_cell_index(model_versions)
            print(f"Subclasses present: {cell_subclasses}")
            print(cell_subclasses)
            assert len(cell_class) == 1, "Multiple cell classes detected!"


        adata = self.create_base_anndata(model_versions, subclass=subclass)


        #adata = self.add_gene_scores_subject(adata)
        #adata = self.add_gene_scores_pxr(adata, model_versions)
        adata = self.add_gene_scores(adata)
        #adata = self.dementia_time_course(adata)
        #adata = self.add_gene_scores_dual(adata)
        # adata = self.bootstrap_dementia(adata)
        #adata = self.add_braak_gene_scores(adata)

        adata = self.add_pathway_means_correlations(adata)

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

    def create_single_anndata(self, z):

        latent = np.concatenate(z["latent"], axis=0)
        latent = np.reshape(latent, (-1, self.latent_dim))
        mask = np.reshape(z["cell_mask"], (-1, z["cell_mask"].shape[-1]))

        a = ad.AnnData(latent)
        for m, k in enumerate(self.obs_list):
            a.obs[k] = z[k]
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


        # Only include samples with include_analysis=True
        # a = a[a.obs["include_analysis"] > 0]

        return a

    def create_base_anndata(self, model_versions, subclass=None):

        for n, v in enumerate(model_versions):
            fn = f"{self.base_path}/version_{v}/test_results.pkl"
            z = pickle.load(open(fn, "rb"))

            """
            for ep in [16, 17, 18, 19]:
                fn = f"{self.base_path}/version_{v}/test_results_ep{ep}.pkl"
                z1 = pickle.load(open(fn, "rb"))
                for k in self.obs_list:
                    z[f"pred_{k}"] += z1[f"pred_{k}"]
                    #z[f"{k}"] += z1[f"{k}"]
            for k in self.obs_list:
                z[f"pred_{k}"] /= 5.0
            
            
            for k in self.obs_list:
                z[f"pred_{k}"] /= 6.0
                if k == "BRAAK_AD":
                    v = np.array(z[f"{k}"])
                    v[np.isnan(v)] = 0.0
                    v[v<-9] = 0.0
                    z[f"pred_{k}"] = 0.75 * z[f"pred_{k}"] + 0.25 *v
            """


            a = self.create_single_anndata(z)
            a.obs["split_num"] = n
            if n == 0:
                adata = a.copy()
            else:
                adata = ad.concat((adata, a), axis=0)

        if subclass is not None:
            adata = adata[adata.obs.subclass == subclass]

        return adata

    def get_cell_index(self, model_versions):

        cell_idx = []
        cell_class = []
        cell_subclass = []

        for n, v in enumerate(model_versions):
            fn = f"{self.base_path}/version_{v}/test_results.pkl"
            z = pickle.load(open(fn, "rb"))
            cell_idx += z["cell_idx"].tolist()
            cell_class += self.meta["obs"]["class"][z["cell_idx"]].tolist()
            cell_subclass += self.meta["obs"]["subclass"][z["cell_idx"]].tolist()
            print(n, v, np.unique(self.meta["obs"]["Brain_bank"][z["cell_idx"]]))

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

    def add_gene_scores_dual(self, adata, n_bins=10):

        counts = np.zeros((n_bins, n_bins), dtype=np.float32)
        scores = np.zeros((self.n_genes, n_bins, n_bins), dtype=np.float32)

        for n, i in enumerate(adata.obs["cell_idx"]):

            data = np.memmap(
                self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
            ).astype(np.float32)

            v0 = adata.obs[f"pred_Dementia"][n]
            v1 = adata.obs[f"pred_BRAAK_AD"][n]

            if np.isnan(v0) or np.isnan(v1):
                continue
            else:
                bin0 = np.argmin((v0 - adata.uns[f"percentiles_Dementia"]) ** 2)
                bin1 = np.argmin((v1 - adata.uns[f"percentiles_BRAAK_AD"]) ** 2)
                counts[bin0, bin1] += 1
                scores[:, bin0, bin1] += self.normalize_data(data)

        adata.uns["scores_Dm_BRAAK" ] = scores / counts[None, :, :]

        return adata


    def add_gene_scores(self, adata, n_bins=100):

        k = "BRAAK_CERAD"
        self.obs_list += [k]
        adata.obs[f"pred_{k}"] = (
            adata.obs["pred_BRAAK_AD"] / np.std(adata.obs["pred_BRAAK_AD"]) + adata.obs["pred_CERAD"] / np.std(adata.obs["pred_CERAD"])
        )

        percentiles = {}
        for k in self.obs_list:
            percentiles[k] = np.percentile(adata.obs[f"pred_{k}"], np.arange(0.5, 100.5, 1.0))

        conds = [None,  {"Dementia": 0}, {"Dementia": 1}, {"Sex": "Male"}, {"Sex": "Female"}, {"apoe": 0}, {"apoe": 1}, {"apoe": 2}]
        score_names = ["", "_Dm0", "_Dm1", "_Male", "_Female", "_apoe0", "_apoe1", "_apoe2"]

        for cond, score_name in zip(conds[:3], score_names[:3]):

            if cond is None:
                a = adata.copy()
            else:
                for k, v in cond.items():
                    a = adata[adata.obs[k] == v]

            counts = {k: np.zeros(n_bins, dtype=np.float32) for k in self.obs_list}
            scores = {k: np.zeros((self.n_genes, n_bins), dtype=np.float32) for k in self.obs_list}
            scores_sqr = {k: np.zeros((self.n_genes, n_bins), dtype=np.float32) for k in self.obs_list}
            scores_std = {k: np.zeros((self.n_genes, n_bins), dtype=np.float32) for k in self.obs_list}

            print(f"Adding gene values. Name: {score_name}")

            for n, i in enumerate(a.obs["cell_idx"]):

                data = np.memmap(
                    self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                ).astype(np.float32)

                for j, k in enumerate(self.obs_list):
                    v = a.obs[f"pred_{k}"][n]
                    if np.isnan(v):
                        continue
                    else:
                        bin_idx = np.argmin((v - percentiles[k]) ** 2)
                        counts[k][bin_idx] += 1
                        scores[k][:, bin_idx] += self.normalize_data(data)
                        scores_sqr[k][:, bin_idx] += self.normalize_data(data)**2

            for k in scores.keys():
                scores[k] /= counts[k][None, :]
                scores_std[k] = np.sqrt(scores_sqr[k] / counts[k][None, :] - scores[k]**2)

            adata.uns["scores" + score_name] = scores
            adata.uns["scores_std" + score_name] = scores_std

            for k in self.obs_list:
                adata.uns[f"percentiles_{k}"] = percentiles[k]

        self.obs_list = self.obs_list[:-1]

        return adata

    def get_gene_idx(self):

        self.gene_idx = {}
        self.pathway_names = []
        for go_id, path_name in zip(self.curated_pathways.go_id.values, self.curated_pathways.pathway.values):
            genes_in_go_path = self.gene_pathways[go_id]
            self.pathway_names.append(path_name)
            self.gene_idx[go_id] = []
            for g in genes_in_go_path:
                idx = np.where(self.gene_names == g)[0]
                if len(idx) > 0:
                    self.gene_idx[go_id].append(idx[0])
            self.gene_idx[go_id] = np.stack(self.gene_idx[go_id])

        print("GENE iDX")
        c = 0
        for k, v in self.gene_idx.items():
            print(k, v)
            c += 1
            if c > 3:
                break


    def add_pathway_means_correlations(self, adata, n_bins=100, gene_threshold=0.1):

        for k in ["BRAAK_AD", "Dementia"]:

            v = np.reshape(np.array(adata.obs[f"pred_{k}"]), (-1, 1))
            pct = np.reshape(np.array(adata.uns[f"percentiles_{k}"]), (1, -1))
            n_pathways = len(self.curated_pathways)

            mean_gene_vals = np.mean(adata.uns[f"scores"][k], axis=1)
            # we will mask out any genes with low expression
            mask_genes = np.ones_like(mean_gene_vals)
            mask_genes[mean_gene_vals < gene_threshold] = 0.0

            print("MEAN GENE VALS", mean_gene_vals.shape)

            adata.uns[f"pathway_mean_{k}"] = np.zeros((n_pathways, n_bins), dtype = np.float32)
            adata.uns[f"pathway_corr_{k}"] = np.zeros((n_pathways, n_pathways, n_bins), dtype=np.float32)
            adata.uns["pathway_names"] = self.pathway_names

            d = np.abs(v - pct)
            bin_idx = np.argmin(d, axis = 1)

            for b in range(n_bins):
                idx = np.array(adata.obs["cell_idx"])[bin_idx == b]
                go_scores = []

                for _, i in enumerate(idx):
                    data = np.memmap(
                        self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                    ).astype(np.float32)

                    data = self.normalize_data(data)

                    go_scores_cell = []
                    for go_id, go_idx in self.gene_idx.items():
                        v = mask_genes[go_idx] * data[go_idx] / mean_gene_vals[go_idx]
                        go_scores_cell.append(np.mean(v))
                    go_scores.append(np.stack(go_scores_cell, axis=0))

                go_scores = np.stack(go_scores, axis=0)

                for n0 in range(n_pathways):
                    adata.uns[f"pathway_mean_{k}"][n0, b] = np.mean(go_scores[:, n0])
                    for n1 in range(n_pathways):
                        adata.uns[f"pathway_corr_{k}"][n0, n1, b], _ = stats.pearsonr(go_scores[:, n0], go_scores[:, n1])

        return adata



