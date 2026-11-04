"""Sparse autoencoder for the sae2 concept decomposition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tgcl.device_utils import get_device_config


class SparseAutoencoder(nn.Module):
    """One-layer sparse autoencoder: ReLU code, linear decoder (sae2 concepts).

    The decoder columns are the concepts. Sparsity comes from a hard top-k on
    the code when top_k is set, else from an L1 penalty (sparsity_lambda).

    Args:
        input_dim: Feature dimension.
        hidden_dim: Number of concepts (code size).
        sparsity_lambda: L1 weight on the code (without top_k).
        top_k: Code entries kept per sample (None: all).
    """

    def __init__(self, input_dim, hidden_dim, sparsity_lambda=5.0, top_k=None):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_lambda = sparsity_lambda
        # Hard top-k activation: keep only the top_k largest ReLU codes per
        # sample and zero the rest, guaranteeing an exact sparsity level by
        # construction. A soft L1 penalty alone cannot reliably hit a target
        # zero-fraction (it plateaus well short of e.g. 99% no matter how high
        # sparsity_lambda is set), since exact-zero only happens when training
        # happens to push a pre-activation below zero.
        self.top_k = top_k

    def forward(self, x):
        """(reconstruction, code) of a batch."""
        z = F.relu(self.encoder(x))
        if self.top_k is not None and self.top_k < z.shape[1]:
            _, topk_idx = torch.topk(z, self.top_k, dim=1)
            mask = torch.zeros_like(z)
            mask.scatter_(1, topk_idx, 1.0)
            z = z * mask
        x_hat = self.decoder(z)
        return x_hat, z

    def loss(self, x, x_hat, z):
        """Reconstruction MSE, plus the L1 penalty when top_k is not set."""
        recon_loss = F.mse_loss(x_hat, x)
        if self.top_k is not None:
            # Sparsity is already guaranteed structurally by top-k masking;
            # an additional L1 penalty would only shrink the surviving
            # coefficients and hurt reconstruction for no sparsity benefit.
            return recon_loss
        sparsity_loss = torch.mean(torch.abs(z))
        return recon_loss + self.sparsity_lambda * sparsity_loss


def l2_normalize_rows(x, eps=1e-8):
    """Rows scaled to unit L2 norm."""
    norm = torch.norm(x, dim=1, keepdim=True) + eps
    return x / norm


def train_sae(
    model,
    data,
    epochs=100,
    lr=1e-3,
    batch_size=512,
    device=None,
    normalize_input=True,
    lr_cooldown=True,
):
    """Train the autoencoder on the feature rows with Adam.

    Args:
        model: The SparseAutoencoder.
        data: (samples x dim) features, numpy or torch.
        epochs: Passes over the data.
        lr: Learning rate.
        batch_size: Rows per step.
        device: Training device (default: DEVICE via tgcl.device_utils).
        normalize_input: Train on unit-norm rows.
        lr_cooldown: Decay the learning rate linearly to 0 over the last 20 %
            of the epochs.
    """
    if device is None:
        device = get_device_config(None).primary_device
    model = model.to(device)
    model.train()

    if not isinstance(data, torch.Tensor):
        data_tensor = torch.from_numpy(data).float()
    else:
        data_tensor = data.float()

    if normalize_input:
        data_tensor = l2_normalize_rows(data_tensor)

    data_tensor = data_tensor.to(device)
    dataloader = torch.utils.data.DataLoader(
        data_tensor, batch_size=batch_size, shuffle=True
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.9999))

    for epoch in range(epochs):
        total_loss = 0.0
        if lr_cooldown and epoch > 0.8 * epochs:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * (
                    1 - (epoch - 0.8 * epochs) / (0.2 * epochs)
                )

        for batch in dataloader:
            batch = batch.to(device)
            if normalize_input:
                batch = l2_normalize_rows(batch)

            optimizer.zero_grad()
            x_hat, z = model(batch)
            loss = model.loss(batch, x_hat, z)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(data_tensor)
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(
                f"[SAE] Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.6f}"
            )
