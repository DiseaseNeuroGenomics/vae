import collections
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


def identity(x):
    return x

def one_hot(index, n_cat):
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)


class ResidualLayer(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_hidden: int,
        dropout_rate: float = 0.1,
    ):

        super().__init__()
        self.ff0 = nn.Linear(n_in, n_hidden)
        self.ff1 = nn.Linear(n_hidden, n_in)
        self.ln = nn.LayerNorm(n_in, elementwise_affine=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=float(dropout_rate))
        self.soft = nn.Softmax(dim=-1)

    def forward(self, input: torch.Tensor):

        x = self.ff0(input)
        x = self.act(x)
        x = self.drop(x)
        x = self.ff1(x)


        return self.ln(input + x)


class FCLayers(nn.Module):
    r"""A helper class to build fully-connected layers for a neural network.

    :param n_in: The dimensionality of the input
    :param n_out: The dimensionality of the output
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    :param use_batch_norm: Whether to have `BatchNorm` layers or not
    :param use_relu: Whether to have `ReLU` layers or not
    :param bias: Whether to learn bias in linear layers or not

    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        batch_properties: Optional[Dict[str, Any]] = None,
    ):

        # we will inject the batch variables into the first hidden layer

        super().__init__()

        self.batch_dims = 0 if batch_properties is None else [
            len(batch_properties[k]["values"]) for k in batch_properties.keys()
        ]

        self.act = nn.GELU()
        n = n_hidden if n_layers > 1 else n_hidden - np.sum(self.batch_dims)
        self.ff0 = nn.Linear(n_in, n)
        self.drop = nn.Dropout(p=float(dropout_rate))
        self.ln = nn.LayerNorm(n, elementwise_affine=True)
        layers_dim = [n_hidden + np.sum(self.batch_dims)] + (n_layers - 1) * [n_hidden] + [n_out]

        layers = []
        for n in range(n_layers - 1):
            layers.append(nn.Linear(layers_dim[n], layers_dim[n + 1]))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(n_hidden, elementwise_affine=True))
            layers.append(nn.Dropout(p=float(dropout_rate)))
            #layers.append(ResidualLayer(n_hidden, 2 * n_hidden, dropout_rate=dropout_rate))

        self.fc_layers = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor, batch_vals: torch.Tensor, batch_mask: torch.Tensor):

        x = self.ff0(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.ln(x)

        if np.sum(self.batch_dims) > 0:
            batch_vals[batch_vals < 0] = 0
            batch_vars = []
            for n in range(len(self.batch_dims)):
                b = F.one_hot(batch_vals[:, n], num_classes=self.batch_dims[n])
                batch_vars.append(b * batch_mask[:, n: n+1])

            x = torch.cat((x, *batch_vars), dim=-1)

        return self.fc_layers(x)

class GroupedEncoder(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    :param distribution: Distribution of z
    """

    def __init__(
        self,
        n_input: int,
        n_latent: int,
        cell_properties: Optional[Dict[str, Any]] = None,
        batch_properties: Optional[Dict[str, Any]] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        input_dropout_rate: float = 0.0,
        latent_distribution: str = "normal"
    ):
        super().__init__()

        self.z_enc = nn.ModuleDict()
        self.cell_properties = cell_properties
        self.cell_predict = nn.ModuleDict()

        for k in cell_properties.keys():

            self.z_enc[k] = Encoder(
                n_input,
                n_latent,
                batch_properties=batch_properties,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                input_dropout_rate=input_dropout_rate,
                distribution=latent_distribution,
            )

    def forward(self, x: torch.Tensor, batch_labels: torch.Tensor, batch_mask: torch.Tensor):

        q_m = []
        q_v = []
        latent = []
        for k in self.cell_properties.keys():
            q_m_k, q_v_k, latent_k = self.z_enc[k](x, batch_labels, batch_mask)
            q_m.append(q_m_k)
            q_v.append(q_v_k)
            latent.append(latent_k)

        return torch.cat(q_m, dim=-1), torch.cat(q_v, dim=-1), torch.cat(latent, dim=-1)

# Encoder
class Encoder(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    :param distribution: Distribution of z
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        batch_properties: Optional[Dict[str, Any]] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        input_dropout_rate: float = 0.0,
        distribution: str = "normal",
    ):
        super().__init__()

        self.distribution = distribution
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            batch_properties=batch_properties,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        #self.bn = nn.BatchNorm1d(n_input, momentum=0.05, eps=1.0)

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity

        self.drop = nn.Dropout(p=input_dropout_rate)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, batch_labels: torch.Tensor, batch_mask: torch.Tensor):

        x = x - x.mean(dim=-1, keepdim=True)
        x = self.drop(x)   # 0.54 is the approximate mean of x
        #x = self.bn(x)
        #x = self.drop(x)
        # Parameters for latent distribution
        q = self.encoder(x, batch_labels, batch_mask)
        q_m = self.mean_encoder(q)
        q_v = self.softplus(self.var_encoder(q)) + 1e-4
        latent = self.z_transformation(reparameterize_gaussian(q_m, q_v))
        return q_m, q_v, latent


class CellDecoder(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        cell_properties: Dict[str, Any],
        grouped: bool = False,
        subject_batch_size: int =1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.cell_properties = cell_properties
        self.grouped = grouped
        self.subject_batch_size = subject_batch_size
        self.cell_mlp = nn.ModuleDict()

        for k, cell_prop in cell_properties.items():
            # the output size of the cell property prediction MLP will be 1 if the property is continuous;
            # if it is discrete, then it will be the length of the possible values
            n_targets = 1 if not cell_prop["discrete"] else len(cell_prop["values"])
            self.cell_mlp[k] = nn.Linear(latent_dim, n_targets, bias=True)


    def forward(self, latent: torch.Tensor):

        # Predict cell properties
        output = {}
        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):
            u = latent[:, n * self.latent_dim: (n + 1) * self.latent_dim] if self.grouped else latent
            x = u.detach() if cell_prop["stop_grad"] else u
            if self.subject_batch_size > 1:
                x = torch.reshape(x, (-1, 8, x.size()[-1])).mean(dim=1)

            output[k] = self.cell_mlp[k](x)

        return output


# Decoder
class DecoderSCVI(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        batch_properties: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            batch_properties=batch_properties,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_labels: torch.Tensor,
        batch_mask: torch.Tensor,
    ):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, batch_labels, batch_mask)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout

