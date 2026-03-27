import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

# Même normaliseur que dans first_try.py
from mri_image_normalizer import MRIImageNormalizer


@dataclass
class VAEConfig:
    """
    Configuration complète de l'entraînement VAE.
    """

    healthy_dir: str = "../Brain Tumor MRI images/Healthy"
    output_dir: str = "ImageGeneration/vae_outputs"

    # Dimensions des images (64x64 pour une architecture conv simple et stable)
    img_size: int = 64
    img_channels: int = 1

    # Hyperparamètres
    latent_dim: int = 128
    batch_size: int = 64
    num_epochs: int = 50
    lr: float = 2e-4
    beta_kl: float = 1.0  # beta-VAE (1.0 = VAE standard)

    # Logging / sauvegarde
    sample_interval: int = 500  # en itérations
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class HealthyMRIDataset(Dataset):
    """
    Dataset PyTorch pour les images MRI saines uniquement.
    Retourne des tensors (1, H, W) normalisés en [0, 1].
    """

    def __init__(self, healthy_dir: str, target_size: int = 64, normalize_pixels: bool = True):
        self.healthy_dir = healthy_dir
        self.target_size = (target_size, target_size)
        self.normalize_pixels = normalize_pixels

        self.normalizer = MRIImageNormalizer(
            target_size=self.target_size,
            normalize_pixels=self.normalize_pixels,
        )

        self.image_paths: List[str] = []
        self._load_images()

        print(f"\nDataset VAE (Healthy MRI) créé:")
        print(f"  - Dossier: {self.healthy_dir}")
        print(f"  - Nombre d'images: {len(self.image_paths)}")
        print(f"  - Taille cible: {self.target_size[0]}x{self.target_size[1]}")

    def _load_images(self) -> None:
        import glob

        exts = [".jpg", ".jpeg", ".png"]
        for ext in exts:
            pattern = os.path.join(self.healthy_dir, f"*{ext}")
            self.image_paths.extend(sorted(glob.glob(pattern)))
            pattern_upper = os.path.join(self.healthy_dir, f"*{ext.upper()}")
            self.image_paths.extend(sorted(glob.glob(pattern_upper)))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        img_array = self.normalizer.process_single_image(img_path)

        if img_array is None:
            img_array = np.zeros((self.target_size[1], self.target_size[0], 1), dtype=np.float32)

        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (C, H, W)
        return img_tensor


class ConvVAE(nn.Module):
    """
    VAE convolutionnel pour images 1 canal 64x64.
    - Encodeur: conv -> conv -> conv -> conv -> flatten -> (mu, logvar)
    - Décodeur: z -> fc -> reshape -> convT -> ... -> sigmoid
    """

    def __init__(self, img_channels: int, img_size: int, latent_dim: int):
        super().__init__()

        if img_size != 64:
            raise ValueError("ConvVAE est configuré pour img_size=64 (facile à étendre ensuite).")

        self.img_channels = img_channels
        self.img_size = img_size
        self.latent_dim = latent_dim

        # Encodeur (64 -> 32 -> 16 -> 8 -> 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),  # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc_feat_dim = 512 * 4 * 4
        self.fc_mu = nn.Linear(self.enc_feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_feat_dim, latent_dim)

        # Décodeur
        self.fc_dec = nn.Linear(latent_dim, self.enc_feat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1),  # 64x64
            nn.Sigmoid(),  # recon en [0, 1]
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(h.size(0), 512, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta_kl: float):
    """
    Perte VAE = reconstruction + beta * KL.
    Reconstruction en BCE (images en [0, 1]).
    """
    # BCE par pixel, puis somme par batch (plus stable que mean selon datasets)
    recon_loss = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
    # KL divergence entre q(z|x) et N(0, I)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta_kl * kl
    return total, recon_loss, kl


@torch.no_grad()
def save_samples(
    model: ConvVAE,
    device: torch.device,
    output_dir: str,
    epoch: int,
    step: int,
    fixed_z: torch.Tensor,
    real_batch: torch.Tensor,
) -> None:
    model.eval()

    # 1) Reconstructions (quelques exemples)
    real = real_batch[:32].to(device)
    recon, _, _ = model(real)

    # grille: ligne 1 = réel, ligne 2 = recon
    grid = torch.cat([real.cpu(), recon.cpu()], dim=0)
    recon_path = os.path.join(output_dir, f"recon_epoch_{epoch:03d}_step_{step:06d}.png")
    save_image(grid, recon_path, nrow=16, normalize=False)

    # 2) Samples depuis z fixe
    samples = model.decode(fixed_z.to(device)).cpu()
    sample_path = os.path.join(output_dir, f"samples_epoch_{epoch:03d}_step_{step:06d}.png")
    save_image(samples, sample_path, nrow=8, normalize=False)


def train_vae(config: VAEConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device(config.device)
    print("=" * 70)
    print("ENTRAÎNEMENT D'UN VAE SUR LES IRM SAINES")
    print("=" * 70)
    print(f"\nDevice utilisé: {device}")

    # =========================================================
    # 1. Dataset & DataLoader
    # =========================================================
    print("\n" + "=" * 70)
    print("ÉTAPE 1: Chargement du dataset (Healthy uniquement)")
    print("=" * 70)

    dataset = HealthyMRIDataset(
        healthy_dir=config.healthy_dir,
        target_size=config.img_size,
        normalize_pixels=True,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"Aucune image trouvée dans {config.healthy_dir}. "
            f"Vérifie le chemin et les extensions (.jpg/.jpeg/.png)."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # =========================================================
    # 2. Modèle / Optimiseur
    # =========================================================
    print("\n" + "=" * 70)
    print("ÉTAPE 2: Initialisation du VAE")
    print("=" * 70)

    model = ConvVAE(
        img_channels=config.img_channels,
        img_size=config.img_size,
        latent_dim=config.latent_dim,
    ).to(device)

    print(f"\nVAE:\n{model}")

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    fixed_z = torch.randn(64, config.latent_dim)

    # =========================================================
    # 3. Entraînement
    # =========================================================
    print("\n" + "=" * 70)
    print("ÉTAPE 3: Entraînement du VAE")
    print("=" * 70)

    step = 0
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}", unit="batch")

        for batch in pbar:
            x = batch.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(x)
            loss, recon_loss, kl = vae_loss(recon, x, mu, logvar, beta_kl=config.beta_kl)

            loss.backward()
            optimizer.step()

            # Normalisation des métriques par batch_size pour lecture facile
            bsz = x.size(0)
            pbar.set_postfix(
                {
                    "Loss": f"{(loss.item() / bsz):.4f}",
                    "Recon": f"{(recon_loss.item() / bsz):.4f}",
                    "KL": f"{(kl.item() / bsz):.4f}",
                }
            )

            if step % config.sample_interval == 0:
                save_samples(
                    model=model,
                    device=device,
                    output_dir=config.output_dir,
                    epoch=epoch,
                    step=step,
                    fixed_z=fixed_z,
                    real_batch=batch,
                )

            step += 1

        # Checkpoint fin d'époque
        torch.save(model.state_dict(), os.path.join(config.output_dir, f"vae_epoch_{epoch:03d}.pth"))

    print("\n" + "=" * 70)
    print("ENTRAÎNEMENT DU VAE TERMINÉ")
    print("=" * 70)


def main() -> None:
    config = VAEConfig()
    train_vae(config)


if __name__ == "__main__":
    main()

