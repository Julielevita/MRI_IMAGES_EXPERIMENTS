import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# Meme normalisation que les autres scripts generation
from mri_image_normalizer import CTImageNormalizer


@dataclass
class SRVAEConfig:
    pairs_manifest_path: str = "Healthy_SR_pairs/scale_2/manifest.jsonl"
    output_dir: str = "ImageGeneration/sr_vae_outputs"

    # Reconstituer high (taille fixe par defaut)
    high_size: int = 64
    low_size: int = 32

    img_channels: int = 1
    latent_dim: int = 64
    cond_emb_dim: int = 128

    base_channels: int = 64

    # Training
    batch_size: int = 32
    num_epochs: int = 50
    lr: float = 2e-4
    beta_kl: float = 0.1

    # Logging
    sample_interval: int = 500  # en itérations
    num_sample_images: int = 16

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _load_manifest(manifest_path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


class SRPairsDataset(Dataset):
    """
    Charge des paires (low, high) preparees par `downsample_images.py`.
    """

    def __init__(self, manifest_path: str, high_size: int, low_size: int, normalize_pixels: bool = True):
        self.manifest_path = manifest_path
        self.items = _load_manifest(manifest_path)
        self.high_size = high_size
        self.low_size = low_size

        self.high_normalizer = CTImageNormalizer(target_size=(high_size, high_size), normalize_pixels=normalize_pixels)
        self.low_normalizer = CTImageNormalizer(target_size=(low_size, low_size), normalize_pixels=normalize_pixels)

        print(f"\nSRPairsDataset: {len(self.items)} paires")

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_path(self, p: str) -> str:
        """
        Résout un chemin depuis `manifest.jsonl`.

        Le manifest peut contenir des chemins :
        - absolus
        - relatifs au dossier où le script a été lancé (souvent depuis le repo)
        - relatifs avec un préfixe du type `Healthy_SR_pairs/scale_2/...` (ce qui casse
          une résolution simple via `base_dir + p`).

        On essaye plusieurs candidats et on renvoie le premier qui existe.
        """
        if os.path.isabs(p):
            return p

        base_dir = os.path.dirname(self.manifest_path)
        parent_dir = os.path.dirname(base_dir)

        candidates = [
            # 1) relatif au cwd (comportement attendu si le manifest stocke un chemin repo-relative)
            os.path.abspath(p),
            # 2) relatif au dossier du manifest
            os.path.abspath(os.path.join(base_dir, p)),
            # 3) relatif au parent du dossier du manifest (au cas où base_dir serait trop profond)
            os.path.abspath(os.path.join(parent_dir, p)),
        ]

        for c in candidates:
            if os.path.exists(c):
                return c

        # fallback: on renvoie la résolution "base_dir + p" (pour ne pas casser l'appelant),
        # mais on laisse l'erreur explicite ensuite.
        return os.path.abspath(os.path.join(base_dir, p))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.items[idx]
        low_path = self._resolve_path(rec["low_path"])
        high_path = self._resolve_path(rec["high_path"])

        low = self.low_normalizer.process_single_image(low_path)
        if low is None:
            low = np.zeros((self.low_size, self.low_size, 1), dtype=np.float32)
        low_t = torch.from_numpy(low).permute(2, 0, 1)  # (1, lowH, lowW)

        high = self.high_normalizer.process_single_image(high_path)
        if high is None:
            high = np.zeros((self.high_size, self.high_size, 1), dtype=np.float32)
        high_t = torch.from_numpy(high).permute(2, 0, 1)  # (1, highH, highW)

        return low_t, high_t


class ConvConditionalVAE(nn.Module):
    """
    Conditional VAE pour super-resolution (low -> high).

    Idee:
    - cond_up = upsample(low) -> (1, highH, highW)
    - Encoder: q(z | x_high, cond_up) prend concat([x_high, cond_up]) (2 canaux)
    - Cond encoder: phi(cond_up) -> cond_emb
    - Decoder: p(x_high | z, cond_emb)
    """

    def __init__(self, img_channels: int, high_size: int, latent_dim: int, cond_emb_dim: int, base_channels: int = 64):
        super().__init__()

        if high_size != 64:
            raise ValueError("ConvConditionalVAE est implemente pour high_size=64 (pour un decodage 4x4->64x64).")

        self.img_channels = img_channels
        self.high_size = high_size
        self.latent_dim = latent_dim
        self.cond_emb_dim = cond_emb_dim

        in_ch_enc = img_channels * 2  # x_high + cond_up
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # Encoder -> mu/logvar
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch_enc, c1, 4, 2, 1),  # 64->32
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(c1, c2, 4, 2, 1),  # 32->16
            nn.GroupNorm(8, c2),
            nn.SiLU(),
            nn.Conv2d(c2, c3, 4, 2, 1),  # 16->8
            nn.GroupNorm(8, c3),
            nn.SiLU(),
            nn.Conv2d(c3, c4, 4, 2, 1),  # 8->4
            nn.GroupNorm(8, c4),
            nn.SiLU(),
        )
        enc_feat_dim = c4 * 4 * 4
        self.fc_mu = nn.Linear(enc_feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_feat_dim, latent_dim)

        # Encoder de la condition -> cond_emb
        self.cond_enc = nn.Sequential(
            nn.Conv2d(img_channels, c1, 4, 2, 1),  # 64->32
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(c1, c2, 4, 2, 1),  # 32->16
            nn.GroupNorm(8, c2),
            nn.SiLU(),
            nn.Conv2d(c2, c3, 4, 2, 1),  # 16->8
            nn.GroupNorm(8, c3),
            nn.SiLU(),
            nn.Conv2d(c3, c4, 4, 2, 1),  # 8->4
            nn.GroupNorm(8, c4),
            nn.SiLU(),
        )
        cond_feat_dim = c4 * 4 * 4
        self.cond_to_emb = nn.Linear(cond_feat_dim, cond_emb_dim)

        # Decoder
        dec_in = latent_dim + cond_emb_dim
        dec_feat_dim = c4 * 4 * 4
        self.fc_dec = nn.Linear(dec_in, dec_feat_dim)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, 4, 2, 1),  # 4->8
            nn.GroupNorm(8, c3),
            nn.SiLU(),
            nn.ConvTranspose2d(c3, c2, 4, 2, 1),  # 8->16
            nn.GroupNorm(8, c2),
            nn.SiLU(),
            nn.ConvTranspose2d(c2, c1, 4, 2, 1),  # 16->32
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.ConvTranspose2d(c1, img_channels, 4, 2, 1),  # 32->64
            nn.Sigmoid(),
        )

    def encode(self, x_high: torch.Tensor, cond_up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(torch.cat([x_high, cond_up], dim=1))
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def encode_cond(self, cond_up: torch.Tensor) -> torch.Tensor:
        h = self.cond_enc(cond_up)
        h = h.view(h.size(0), -1)
        return self.cond_to_emb(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(torch.cat([z, cond_emb], dim=1))
        h = h.view(h.size(0), -1, 4, 4)
        return self.dec(h)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # low: (B,1,lowH,lowW)  high: (B,1,64,64)
        cond_up = F.interpolate(low, size=(self.high_size, self.high_size), mode="bilinear", align_corners=False)
        mu, logvar = self.encode(high, cond_up)
        z = self.reparameterize(mu, logvar)
        cond_emb = self.encode_cond(cond_up)
        recon = self.decode(z, cond_emb)
        return recon, mu, logvar

    @torch.no_grad()
    def sample_from_low(self, low: torch.Tensor) -> torch.Tensor:
        cond_up = F.interpolate(low, size=(self.high_size, self.high_size), mode="bilinear", align_corners=False)
        cond_emb = self.encode_cond(cond_up)
        z = torch.randn(low.size(0), self.latent_dim, device=low.device)
        return self.decode(z, cond_emb)


def sr_vae_loss(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta_kl: float):
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    # KL moyenne par batch
    b = target.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
    total = recon_loss + beta_kl * kl
    return total, recon_loss, kl


@torch.no_grad()
def save_sr_samples(
    model: ConvConditionalVAE,
    low_batch: torch.Tensor,
    high_batch: torch.Tensor,
    step: int,
    epoch: int,
    output_dir: str,
    prefix: str = "recon",
    max_items: int = 16,
):
    model.eval()
    low = low_batch[:max_items].to(next(model.parameters()).device)
    high = high_batch[:max_items].to(next(model.parameters()).device)

    # Reconstruction "moyenne" (z = mu) n'est pas exposee directement; on utilise forward complet pour garder simple.
    # Pour un rendu stable, on peut generer un sample depuis le prior aussi.
    recon, _, _ = model(low, high)

    # low up pour visualisation
    cond_up = F.interpolate(low, size=(model.high_size, model.high_size), mode="bilinear", align_corners=False)

    # Grille: low_up | high | recon
    grid = torch.cat([cond_up.cpu(), high.cpu(), recon.cpu()], dim=0)
    nrow = max_items
    path = os.path.join(output_dir, f"{prefix}_epoch_{epoch:03d}_step_{step:06d}.png")
    save_image(grid, path, nrow=nrow, normalize=False)

    # Samples "stochastiques" depuis prior -> pour voir la generativite
    samples = model.sample_from_low(low).cpu()
    grid2 = torch.cat([cond_up.cpu(), samples], dim=0)
    path2 = os.path.join(output_dir, f"samples_epoch_{epoch:03d}_step_{step:06d}.png")
    save_image(grid2, path2, nrow=max_items, normalize=False)


@torch.no_grad()
def save_epoch_comparison_low(
    model: ConvConditionalVAE,
    low_batch: torch.Tensor,
    high_batch: torch.Tensor,
    epoch: int,
    output_dir: str,
    max_items: int = 16,
    filename_prefix: str = "compare_low",
):
    """
    Sauvegarde une comparaison en resolution de `low` :
    - low
    - high downsampled en low_size
    - pseudo_high (recon) downsampled en low_size
    """
    model.eval()
    device = next(model.parameters()).device

    low = low_batch[:max_items].to(device)
    high = high_batch[:max_items].to(device)

    # Reconstruit en high_size, puis on downsample pour comparer a la taille low
    recon, _, _ = model(low, high)

    low_h, low_w = int(low.shape[-2]), int(low.shape[-1])
    high_down = F.interpolate(high, size=(low_h, low_w), mode="bilinear", align_corners=False)
    recon_down = F.interpolate(recon, size=(low_h, low_w), mode="bilinear", align_corners=False)

    grid = torch.cat([low.cpu(), high_down.cpu(), recon_down.cpu()], dim=0)
    nrow = grid.shape[0] // 3 if grid.shape[0] >= 3 else 1
    out_path = os.path.join(output_dir, f"{filename_prefix}_epoch_{epoch:03d}.png")
    save_image(grid, out_path, nrow=nrow, normalize=False)


def train_sr_vae(config: SRVAEConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device(config.device)
    print("=" * 70)
    print("TRAINING CONDITIONAL VAE - SUPER RESOLUTION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Manifest: {config.pairs_manifest_path}")

    dataset = SRPairsDataset(
        manifest_path=config.pairs_manifest_path,
        high_size=config.high_size,
        low_size=config.low_size,
        normalize_pixels=True,
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset vide. Verifie manifest et chemins low/high.")

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = ConvConditionalVAE(
        img_channels=config.img_channels,
        high_size=config.high_size,
        latent_dim=config.latent_dim,
        cond_emb_dim=config.cond_emb_dim,
        base_channels=config.base_channels,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    step = 0
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}", unit="batch")

        avg_total = 0.0
        avg_recon = 0.0
        avg_kl = 0.0
        n_batches = 0
        last_low_cpu = None
        last_high_cpu = None

        for low, high in pbar:
            low = low.to(device)
            high = high.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(low, high)

            total, recon_loss, kl = sr_vae_loss(
                recon=recon,
                target=high,
                mu=mu,
                logvar=logvar,
                beta_kl=config.beta_kl,
            )

            total.backward()
            optimizer.step()

            avg_total += float(total.item())
            avg_recon += float(recon_loss.item())
            avg_kl += float(kl.item())
            n_batches += 1

            pbar.set_postfix(
                {
                    "Loss": f"{total.item():.4f}",
                    "Recon": f"{recon_loss.item():.4f}",
                    "KL": f"{kl.item():.4f}",
                }
            )

            if step % config.sample_interval == 0:
                save_sr_samples(
                    model=model,
                    low_batch=low.detach().cpu(),
                    high_batch=high.detach().cpu(),
                    step=step,
                    epoch=epoch,
                    output_dir=config.output_dir,
                    prefix="recon",
                    max_items=config.num_sample_images,
                )

            # Conserve le dernier batch pour une comparaison epoch-friendly.
            # (On garde sur CPU pour ne pas exploser la RAM GPU.)
            last_low_cpu = low.detach().cpu()
            last_high_cpu = high.detach().cpu()

            step += 1

        # Checkpoint fin d'époque
        torch.save(model.state_dict(), os.path.join(config.output_dir, f"sr_vae_epoch_{epoch:03d}.pth"))

        # Comparaison low/high/recon a la resolution low (1 PNG par époque)
        if last_low_cpu is not None and last_high_cpu is not None:
            save_epoch_comparison_low(
                model=model,
                low_batch=last_low_cpu,
                high_batch=last_high_cpu,
                epoch=epoch,
                output_dir=config.output_dir,
                max_items=config.num_sample_images,
                filename_prefix="compare_low",
            )

        # mini-log
        avg_total /= max(1, n_batches)
        avg_recon /= max(1, n_batches)
        avg_kl /= max(1, n_batches)
        print(f"Epoch {epoch} done. Avg Loss={avg_total:.4f} Recon={avg_recon:.4f} KL={avg_kl:.4f}")

    print("Training done.")


def main() -> None:
    config = SRVAEConfig()
    train_sr_vae(config)


if __name__ == "__main__":
    main()

