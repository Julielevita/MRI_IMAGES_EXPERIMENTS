import math
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

# Même normaliseur que dans first_try.py / model_VAE.py
from mri_image_normalizer import CTImageNormalizer


@dataclass
class DiffusionConfig:
    """
    Configuration complète de l'entraînement DDPM (diffusion).
    """

    healthy_dir: str = "../Brain Tumor MRI images/Healthy"
    output_dir: str = "ImageGeneration/diffusion_outputs"

    img_size: int = 64
    img_channels: int = 1

    # Entraînement
    batch_size: int = 64
    num_epochs: int = 50
    lr: float = 2e-4

    # Diffusion
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # Logging / sauvegarde
    sample_interval: int = 1000  # en itérations (sampling un peu coûteux)
    num_sample_images: int = 64
    ddim_steps: int = 50  # sampling plus rapide (DDIM)
    ddim_eta: float = 0.0  # 0.0 = déterministe (souvent plus net)
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

        self.normalizer = CTImageNormalizer(
            target_size=self.target_size,
            normalize_pixels=self.normalize_pixels,
        )

        self.image_paths: List[str] = []
        self._load_images()

        print(f"\nDataset Diffusion (Healthy MRI) créé:")
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

        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (C, H, W) en [0,1]
        return img_tensor


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Embedding sinusoidal des timesteps (style Transformer / DDPM).
    timesteps: (B,) int64
    return: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=timesteps.device).float() / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    """
    UNet léger pour prédire le bruit epsilon.
    Conçu pour 64x64 en niveaux de gris.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64, time_dim: int = 256):
        super().__init__()

        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Down
        self.down1 = ResidualBlock(base_channels, base_channels, time_dim)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_dim)
        self.downsample1 = nn.Conv2d(base_channels * 2, base_channels * 2, 4, 2, 1)  # 64->32

        self.down3 = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)
        self.down4 = ResidualBlock(base_channels * 2, base_channels * 4, time_dim)
        self.downsample2 = nn.Conv2d(base_channels * 4, base_channels * 4, 4, 2, 1)  # 32->16

        # Middle
        self.mid1 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid2 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)

        # Up
        self.upsample1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, 2, 1)  # 16->32
        self.up1 = ResidualBlock(base_channels * 8, base_channels * 2, time_dim)
        self.up2 = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)

        self.upsample2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, 2, 1)  # 32->64
        self.up3 = ResidualBlock(base_channels * 4, base_channels, time_dim)
        self.up4 = ResidualBlock(base_channels, base_channels, time_dim)

        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x0 = self.in_conv(x)

        d1 = self.down1(x0, t_emb)
        d2 = self.down2(d1, t_emb)
        x = self.downsample1(d2)

        d3 = self.down3(x, t_emb)
        d4 = self.down4(d3, t_emb)
        x = self.downsample2(d4)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        x = self.upsample1(x)
        x = torch.cat([x, d4], dim=1)
        x = self.up1(x, t_emb)
        x = self.up2(x, t_emb)

        x = self.upsample2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.up3(x, t_emb)
        x = self.up4(x, t_emb)

        x = self.out_conv(self.out_act(self.out_norm(x)))
        return x


class DDPMScheduler:
    """
    Pré-calcul des coefficients DDPM (beta linéaire).
    """

    def __init__(self, timesteps: int, beta_start: float, beta_end: float, device: torch.device):
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # Variance du posterior q(x_{t-1} | x_t, x0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise
        """
        b = x0.size(0)
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(b, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alphas_cumprod[t].view(b, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_omab * noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        Échantillonner x_{t-1} depuis x_t.
        """
        b = x.size(0)
        t_batch = torch.full((b,), t, device=x.device, dtype=torch.long)
        eps_pred = model(x, t_batch)

        beta_t = self.betas[t]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
        sqrt_omab_t = self.sqrt_one_minus_alphas_cumprod[t]

        # mean = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t)*eps)
        mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_omab_t) * eps_pred)

        if t == 0:
            return mean

        var = self.posterior_variance[t]
        noise = torch.randn_like(x)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        Génère des images depuis du bruit (x_T ~ N(0, I)).
        Retourne des images en [-1, 1].
        """
        x = torch.randn(shape, device=self.device)
        for t in range(self.timesteps - 1, -1, -1):
            x = self.p_sample(model, x, t)
        return x

    @torch.no_grad()
    def sample_ddim(
        self,
        model: nn.Module,
        shape: Tuple[int, int, int, int],
        steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Sampling DDIM: beaucoup plus rapide que DDPM.
        - steps: nombre d'itérations (ex: 25/50/100)
        - eta: 0 = déterministe, >0 ajoute du bruit (plus de diversité)

        Retourne des images en [-1, 1].
        """
        if steps < 2:
            raise ValueError("steps doit être >= 2")

        # Sous-sampling des timesteps (de T-1 à 0)
        times = torch.linspace(self.timesteps - 1, 0, steps, device=self.device).long()
        x = torch.randn(shape, device=self.device)

        for i in range(steps):
            t = times[i].item()
            t_prev = times[i + 1].item() if i < steps - 1 else 0

            b = x.size(0)
            t_batch = torch.full((b,), t, device=self.device, dtype=torch.long)
            eps = model(x, t_batch)

            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_prev = self.alphas_cumprod[t_prev]

            # x0_pred = (x_t - sqrt(1-a_bar)*eps) / sqrt(a_bar)
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

            # DDIM sigma (contrôle du bruit)
            sigma = (
                eta
                * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
                * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
            )

            # direction vers x_{t-1}
            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0)) * eps

            noise = torch.randn_like(x) if (eta > 0.0 and i < steps - 1) else torch.zeros_like(x)
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

        return x


@torch.no_grad()
def save_diffusion_samples(
    model: nn.Module,
    scheduler: DDPMScheduler,
    device: torch.device,
    output_dir: str,
    epoch: int,
    step: int,
    num_images: int,
    img_channels: int,
    img_size: int,
    ddim_steps: int,
    ddim_eta: float,
) -> None:
    model.eval()
    x = scheduler.sample_ddim(
        model,
        (num_images, img_channels, img_size, img_size),
        steps=ddim_steps,
        eta=ddim_eta,
    ).detach().cpu()
    # [-1,1] -> [0,1]
    x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    out_path = os.path.join(output_dir, f"samples_epoch_{epoch:03d}_step_{step:06d}.png")
    save_image(x, out_path, nrow=int(math.sqrt(num_images)), normalize=False)


def train_diffusion(config: DiffusionConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device(config.device)
    print("=" * 70)
    print("ENTRAÎNEMENT D'UN MODÈLE DE DIFFUSION (DDPM) SUR LES IRM SAINES")
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
    # 2. Modèle / Scheduler / Optimiseur
    # =========================================================
    print("\n" + "=" * 70)
    print("ÉTAPE 2: Initialisation du modèle de bruit (UNet) + scheduler")
    print("=" * 70)

    model = SimpleUNet(in_channels=config.img_channels, base_channels=64, time_dim=256).to(device)
    scheduler = DDPMScheduler(
        timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
    )

    print(f"\nUNet:\n{model}")
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()

    # =========================================================
    # 3. Entraînement
    # =========================================================
    print("\n" + "=" * 70)
    print("ÉTAPE 3: Entraînement DDPM (prédiction du bruit)")
    print("=" * 70)

    step = 0
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}", unit="batch")

        for batch in pbar:
            x0 = batch.to(device)  # [0,1]
            x0 = x0 * 2.0 - 1.0  # -> [-1,1]

            b = x0.size(0)
            t = torch.randint(0, config.timesteps, (b,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = scheduler.q_sample(x0, t, noise)

            optimizer.zero_grad()
            noise_pred = model(x_t, t)
            loss = mse(noise_pred, noise)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"MSE": f"{loss.item():.4f}", "t": int(t.float().mean().item())})

            if step % config.sample_interval == 0:
                save_diffusion_samples(
                    model=model,
                    scheduler=scheduler,
                    device=device,
                    output_dir=config.output_dir,
                    epoch=epoch,
                    step=step,
                    num_images=config.num_sample_images,
                    img_channels=config.img_channels,
                    img_size=config.img_size,
                    ddim_steps=config.ddim_steps,
                    ddim_eta=config.ddim_eta,
                )

            step += 1

        # Checkpoint fin d'époque
        torch.save(model.state_dict(), os.path.join(config.output_dir, f"diffusion_unet_epoch_{epoch:03d}.pth"))

    print("\n" + "=" * 70)
    print("ENTRAÎNEMENT DIFFUSION TERMINÉ")
    print("=" * 70)


def main() -> None:
    config = DiffusionConfig()
    train_diffusion(config)


if __name__ == "__main__":
    main()

