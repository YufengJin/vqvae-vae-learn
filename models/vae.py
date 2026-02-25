"""
VAE with same Encoder/Decoder structure as VQ-VAE.
Objective: reconstruction + beta * KL divergence.
"""

import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder
from models.decoder import Decoder


class VAE(nn.Module):
    """
    Variational Autoencoder with VQ-VAE architecture.
    Encoder outputs mu, logvar; reparameterize z = mu + sigma * epsilon; decode.
    """

    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VAE, self).__init__()
        # n_embeddings unused for VAE, kept for interface compatibility
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        # output mu and logvar: 2 * embedding_dim channels
        self.to_latent = nn.Conv2d(h_dim, 2 * embedding_dim, kernel_size=1, stride=1)
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, verbose=False):
        h = self.encoder(x)
        h = self.to_latent(h)  # (B, 2*emb_dim, H, W)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)

        # KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        if verbose:
            print('original data shape:', x.shape)
            print('mu shape:', mu.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        # return (reg_term, x_hat, aux_metric) to match VQ-VAE interface
        return self.beta * kl_loss, x_hat, kl_loss
