from typing import Any, Dict, List, Optional

import shutil
import pickle
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.distributions import Normal, kl_divergence as kl
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torchmetrics import MetricCollection, ExplainedVariance
from torchmetrics.classification import Accuracy
from modules import one_hot
from distributions import (
    ZeroInflatedNegativeBinomial,
    NegativeBinomial,
    Poisson,
)
from losses import GroupDRO
import scipy.stats as stats

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# define the LightningModule
class LitVAE(pl.LightningModule):
    def __init__(
        self,
        network,
        cell_properties: Optional[Dict[str, Any]] = None,
        batch_properties: Optional[Dict[str, Any]] = None,
        balance_classes: bool = False,
        dispersion: str = "gene",
        reconstruction_loss: str = "zinb",
        latent_distribution: str = "normal",
        n_epochs_kl_warmup: Optional[int] = 20,
        learning_rate: float = 0.0005,
        weight_decay: float = 0.1,
        l1_lambda: float = 0.0,
        gene_loss_coeff: float = 5e-4,
        save_gene_vals: bool = True,
        library_mean: Optional[float] = None,
        library_var: Optional[float] = None,
        subject_batch_size: int = 1,
        use_gdro: bool = True,

    ):
        super().__init__()

        self.network = network
        self.cell_properties = cell_properties
        self.batch_properties = batch_properties
        self.balance_classes = balance_classes
        self.dispersion = dispersion
        self.reconstruction_loss = reconstruction_loss
        self.latent_distribution = latent_distribution
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.gene_loss_coeff = gene_loss_coeff
        self.save_gene_vals = save_gene_vals
        self.library_mean = library_mean
        self.library_var = library_var

        self.subject_batch_size = subject_batch_size

        print(f"Learning rate: {self.learning_rate}")
        print(f"Weight decay: {self.weight_decay}")

        self._cell_properties_metrics()

        self._create_results_dict()

        self.val_score = -999.0
        self.train_step = 0

        self.use_gdro = use_gdro
        if use_gdro:

            balances_dm = [
                [0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61],
            ]

            balances_braak = [
                [0, 1, 60, 61],
                [10, 11, 50, 51],
                [20, 21, 40, 41],
                [30, 31],
            ]

            self.gdro = {
                "Dementia": GroupDRO(balances=balances_dm), "BRAAK_AD": GroupDRO(balances=balances_braak)
            }


    def _create_results_dict(self):

        self.results = {"epoch": 0}
        for k in self.cell_properties.keys():
            self.results[k] = []
            self.results["pred_" + k] = []
        if self.batch_properties is not None:
            for k in self.batch_properties.keys():
                self.results[k] = []
        self.results["latent"] = []
        self.results["cell_mask"] = []
        self.results["px_r"] = []
        self.results["cell_idx"] = []

    def _cell_properties_metrics(self):

        self.cell_cross_ent = nn.ModuleDict()
        self.cell_mse = nn.ModuleDict()
        self.cell_accuracy = nn.ModuleDict()
        self.cell_explained_var = nn.ModuleDict()

        for k, cell_prop in self.cell_properties.items():
            if cell_prop["discrete"]:
                # discrete variable, set up cross entropy module
                weight = torch.from_numpy(
                    np.float32(np.clip(1 / cell_prop["freq"], 0.1, 10.0))
                ) if self.balance_classes else None
                self.cell_cross_ent[k] = nn.CrossEntropyLoss(weight=weight, reduction="none", ignore_index=-100)
                # self.cell_cross_ent[k] = FocalLoss(len(cell_prop["values"]), gamma=2.0, alpha=5.0)
                # self.cell_cross_ent[k] = Poly1FocalLoss(len(cell_prop["values"]), gamma=2.0)
                self.cell_accuracy[k] = Accuracy(
                    task="multiclass", num_classes=len(cell_prop["values"]), average="macro",
                )
            else:
                # continuous variable, set up MSE module
                self.cell_mse[k] = nn.MSELoss(reduction="none")
                self.cell_explained_var[k] = ExplainedVariance()


    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Return the reconstruction loss (for a minibatch)
        """
        # Reconstruction Loss
        if self.reconstruction_loss == "zinb":
            recon_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.reconstruction_loss == "nb":
            recon_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.reconstruction_loss == "poisson":
            recon_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return recon_loss

    def gene_loss(
        self,
        x: torch.Tensor,
        qz_m: torch.Tensor,
        qz_v: torch.Tensor,
        ql_m: torch.Tensor,
        ql_v: torch.Tensor,
        local_l_mean: torch.Tensor,
        local_l_var: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
        training: bool = True,
    ):

        # KL Divergence

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        kl_z = kl(
            Normal(qz_m, torch.sqrt(qz_v)),
            Normal(mean, scale)
        ).sum(dim=1)

        kl_l = kl(
            Normal(ql_m, torch.sqrt(ql_v)),
            Normal(local_l_mean, torch.sqrt(local_l_var)),
        ).sum(dim=1)

        recon_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        loss = self.kl_weight * (recon_loss + kl_l + 1.0 * kl_z).mean()

        if training:
            self.log("recon_loss", recon_loss.mean(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("kl_z", kl_z.mean(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            self.log("recon_loss_val", recon_loss.mean(), on_step = False, on_epoch = True, prog_bar = False, sync_dist = True)
            self.log("kl_z_val", kl_z.mean(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss, kl_z

    @property
    def kl_weight(self):
        if self.n_epochs_kl_warmup is not None:
            kl_weight = np.clip((1 + self.current_epoch) / self.n_epochs_kl_warmup, 0.0, 1.0)
        else:
            kl_weight = 1.0

        return kl_weight

    def library_stats(self, gene_vals):

        if self.library_mean is None or self.library_var is None:
            library_size_per_cell = torch.log(gene_vals.sum(dim=1))
            local_l_mean = library_size_per_cell.mean()
            local_v_mean = library_size_per_cell.var()
            local_v_mean = torch.clip(local_v_mean, 0.1, 1e3)
        else:
            local_l_mean = torch.from_numpy(np.array(self.library_mean))
            local_v_mean = torch.from_numpy(np.array(self.library_var))

        return local_l_mean, local_v_mean

    def validation_step(self, batch, batch_idx):

        gene_vals, cell_targets, cell_mask, batch_labels, batch_mask, cell_idx, _ = batch
        qz_m, qz_v, z = self.network.z_encoder(gene_vals, batch_labels, batch_mask)
        ql_m, ql_v, library = self.network.l_encoder(gene_vals, batch_labels, batch_mask)
        px_scale, _, px_rate, px_dropout = self.network.decoder(self.dispersion, z, library, batch_labels, batch_mask)

        px_r = self.network.px_r
        px_r = torch.exp(px_r)

        local_l_mean, local_v_mean = self.library_stats(gene_vals)

        gene_loss, kl_z = self.gene_loss(
            gene_vals,
            qz_m,
            qz_v,
            ql_m,
            ql_v,
            local_l_mean,
            local_v_mean,
            px_rate,
            px_r,
            px_dropout,
            training=False,
        )

        self.results["cell_idx"].append(cell_idx.detach().cpu().numpy())
        if self.save_gene_vals:
            self.results["px_r"].append(px_rate.detach().cpu().numpy().astype(np.float32))

        if self.cell_properties is not None:
            cell_pred = self.network.cell_decoder(qz_m)
            self.cell_scores(cell_pred, cell_targets, cell_mask, qz_m)

        if self.batch_properties is not None:
            for n, k in enumerate(self.batch_properties.keys()):
                self.results[k].append(batch_labels[:, n].detach().cpu().numpy())

    def on_validation_epoch_end(self):

        for k in self.cell_properties.keys():
            if len(self.results[k]) > 0:
                self.results[k] = np.concatenate(self.results[k], axis=0)
                self.results["pred_" + k] = np.concatenate(self.results["pred_" + k], axis=0)
        if self.batch_properties is not None:
            for k in self.batch_properties.keys():
                self.results[k] = np.concatenate(self.results[k], axis=0)
        if self.save_gene_vals:
            self.results["px_r"] = np.concatenate(self.results["px_r"], axis=0)
        self.results["cell_mask"] = np.concatenate(self.results["cell_mask"], axis=0)
        self.results["cell_idx"] = np.concatenate(self.results["cell_idx"], axis=0)

        if "Dementia" in self.cell_accuracy and "BRAAK_AD" in self.cell_explained_var:
            x0 = self.results["pred_Dementia"][:, 1]
            x1 = self.results["pred_BRAAK_AD"]
            idx = np.where((x0 > -9) * (x1 > -9))[0]
            r, p = stats.pearsonr(x0[idx], x1[idx])
            self.log("corr", r, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


        v = self.trainer.logger.version
        fn = f"{self.trainer.log_dir}/lightning_logs/version_{v}/test_results.pkl"
        pickle.dump(self.results, open(fn, "wb"))
        src = "/home/masse/work/vae/src/config.py"
        shutil.copyfile(src, f"{self.trainer.log_dir}/lightning_logs/version_{v}/config.py")


        if self.current_epoch == 19 or self.current_epoch == 29:
            fn = f"{self.trainer.log_dir}/lightning_logs/version_{v}/test_results_ep{self.current_epoch+1}.pkl"
            pickle.dump(self.results, open(fn, "wb"))



        self.results["epoch"] = self.current_epoch + 1
        for k in self.cell_properties.keys():
            self.results[k] = []
            self.results["pred_" + k] = []
        if self.batch_properties is not None:
            for k in self.batch_properties.keys():
                self.results[k] = []
        self.results["latent"] = []
        self.results["cell_mask"] = []
        self.results["px_r"] = []
        self.results["cell_idx"] = []

    def cell_scores(self, cell_pred, cell_targets, cell_mask, latent_mean):

        if self.subject_batch_size > 1:
            cell_mask = torch.reshape(cell_mask, (-1, self.subject_batch_size, cell_mask.size()[-1])).mean(dim=1)

        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            idx = torch.nonzero(cell_mask[:, n])

            if cell_prop["discrete"]:
                pred_idx = torch.argmax(cell_pred[k], dim=-1).to(torch.int64)
                pred_prob = F.softmax(cell_pred[k], dim=-1).to(torch.float32).detach().cpu().numpy()

                if self.subject_batch_size > 1:
                    targets = torch.reshape(cell_targets[:, n], (-1, self.subject_batch_size)).mean(dim=1).to(torch.int64)
                else:
                    targets = cell_targets[:, n].to(torch.int64)

                self.cell_accuracy[k].update(pred_idx[idx][None, :], targets[idx][None, :])
                if k != "SubID": # to save space
                    self.results[k].append(targets.detach().cpu().numpy())
                    self.results["pred_" + k].append(pred_prob)
            else:
                pred = torch.squeeze(cell_pred[k])
                if self.subject_batch_size > 1:
                    targets = torch.reshape(cell_targets[:, n], (-1, self.subject_batch_size)).mean(dim=1)
                    targets = targets[idx]
                    self.cell_explained_var[k].update(pred[idx], targets)
                else:
                    self.cell_explained_var[k].update(pred[idx], cell_targets[idx, n])

                self.results[k].append(cell_targets[:, n].detach().cpu().numpy())
                self.results["pred_" + k].append(pred.detach().cpu().to(torch.float32).numpy())
        self.results["latent"].append(latent_mean.detach().cpu().to(torch.float32).numpy())
        self.results["cell_mask"].append(cell_mask.detach().cpu().to(torch.float32).numpy())

        for k, v in self.cell_accuracy.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in self.cell_explained_var.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


    def _cell_loss(
        self,
        cell_pred: Dict[str, torch.Tensor],
        cell_prop_vals: torch.Tensor,
        cell_mask: torch.Tensor,
        group_idx: torch.Tensor,
    ):

        cell_loss = 0.0

        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            if self.use_gdro and k in self.gdro.keys() and self.train_step >= 500:
                weights = self.gdro[k].get_weights(group_idx).to(cell_pred[k].device)
            else:
                weights = 1.0

            if cell_prop["discrete"]:
                loss = self.cell_cross_ent[k](cell_pred[k], cell_prop_vals[:, n].to(torch.int64))
                cell_loss += (weights * loss * cell_mask[:, n]).mean()

            else:
                loss = self.cell_mse[k](torch.squeeze(cell_pred[k]), cell_prop_vals[:, n])
                cell_loss += (weights * loss * cell_mask[:, n]).mean()

            if self.use_gdro and k in self.gdro.keys() and self.use_gdro:
                self.gdro[k].update_batch(loss, cell_mask[:, n], group_idx)

            if self.use_gdro and k in self.gdro.keys():
                self.gdro[k].update_weights()

        return cell_loss

    def training_step(self, batch, batch_idx):

        self.train_step += 1

        gene_vals, cell_prop_vals, cell_mask, batch_labels, batch_mask, _, group_idx = batch

        # qz_m is the mean of the latent, qz_v is the variance, and z is the sampled latent
        qz_m, qz_v, z = self.network.z_encoder(gene_vals, batch_labels, batch_mask)

        # same as above, but for library size
        ql_m, ql_v, library = self.network.l_encoder(gene_vals, batch_labels, batch_mask)

        px_scale, px_r, px_rate, px_dropout = self.network.decoder(
            self.dispersion, z, library, batch_labels, batch_mask,
        )

        if self.cell_properties is not None:
            cell_pred = self.network.cell_decoder(z)
            cell_loss = self._cell_loss(cell_pred, cell_prop_vals, cell_mask, group_idx)
        else:
            cell_loss = 0.0

        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.network.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(dec_batch_index, self.n_batch), self.network.px_r)
        elif self.dispersion == "gene":
            px_r = self.network.px_r
        px_r = torch.exp(px_r)

        local_l_mean, local_v_mean = self.library_stats(gene_vals)

        # Loss
        gene_loss, kl_z = self.gene_loss(gene_vals, qz_m, qz_v, ql_m, ql_v, local_l_mean, local_v_mean, px_rate, px_r, px_dropout)
        loss = self.gene_loss_coeff * gene_loss + cell_loss

        if self.l1_lambda > 0:
            l1_regularization = 0.0
            for name, param in self.network.named_parameters():
                if 'z_encoder' in name and "bias" not in name:
                    l1_regularization += torch.norm(param, p=1)
            loss += self.l1_lambda * l1_regularization


        self.log("gene_loss", gene_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("cell_loss", cell_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("kl_weight", self.kl_weight, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)


        return loss

    def get_reconstruction_loss(
            self,
            x,
            px_rate,
            px_r,
            px_dropout,
            **kwargs,
    ) -> torch.Tensor:
        """Return the reconstruction loss (for a minibatch)"""

        # Reconstruction Loss
        if self.reconstruction_loss == "zinb":
            reconst_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.reconstruction_loss == "nb":
            reconst_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.reconstruction_loss == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return reconst_loss


    def configure_optimizers(self):

        encoder_params = []
        other_params = []
        for n, p in self.network.named_parameters():
            if "z_encoder" in n or "cell_decoder" in n:
                encoder_params.append(p)
            else:
                other_params.append(p)



        opt = torch.optim.AdamW(
            [
                {'params': encoder_params, "weight_decay": self.weight_decay},
                {'params': other_params, "weight_decay":0.0},
            ],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.998),
            eps=1e-7,
        )

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3, verbose=True)
        lr_scheduler = {
            'scheduler': WarmupConstantAndDecaySchedule(opt, warmup_steps=1000),
            'interval': 'step',
        }

        return {
           'optimizer': opt,
           'lr_scheduler': lr_scheduler, # Changed scheduler to lr_scheduler
           'interval': 'step',
       }

class WarmupConstantAndDecaySchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.alpha = np.sqrt(warmup_steps)
        super(WarmupConstantAndDecaySchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        arg1 = (step + 1) ** (-0.5)
        arg2 = (step + 1) * (self.warmup_steps ** -1.5)
        return self.alpha * np.minimum(arg1, arg2)








