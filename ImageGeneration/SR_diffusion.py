"""
Super-résolution par diffusion conditionnelle (DDPM) : low -> high.
Même jeu de paires (manifest) que SR_VAE ; la low est injectée à plusieurs échelles dans l'UNet.
"""
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from model_diffusion import DDPMScheduler, ResidualBlock, timestep_embedding
from SR_VAE import SRPairsDataset


def cond_pyramid_from_low(low: torch.Tensor, high_size: int):
    """
    low: (B, 1, low_h, low_w) en [0, 1]
    Retourne cond_64, cond_32, cond_16 en [-1, 1], alignés sur l'UNet 64->32->16.
    """
    cond_64 = F.interpolate(low, size=(high_size, high_size), mode="bilinear", align_corners=False)
    cond_64 = cond_64 * 2.0 - 1.0
    cond_32 = F.avg_pool2d(cond_64, kernel_size=2, stride=2)
    cond_16 = F.avg_pool2d(cond_32, kernel_size=2, stride=2)
    return cond_64, cond_32, cond_16


class ConditionalSRUNet(nn.Module):
    """
    UNet bruit-prédicteur avec injection multi-échelle de la condition (low upsamplée + pools).
    Même squelette que SimpleUNet, avec fusions 1x1 après concat de la cond à 32 et 16.
    """

    def __init__(
        self,
        img_channels: int = 1,
        cond_channels: int = 1,
        high_size: int = 64,
        base_channels: int = 64,
        time_dim: int = 256,
    ):
        super().__init__()
        if high_size != 64:
            raise ValueError("ConditionalSRUNet est calibré pour high_size=64 (grille UNet).")

        self.high_size = high_size
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(img_channels + cond_channels, base_channels, 3, padding=1)

        self.down1 = ResidualBlock(base_channels, base_channels, time_dim)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_dim)
        self.downsample1 = nn.Conv2d(base_channels * 2, base_channels * 2, 4, 2, 1)

        self.fuse_32 = nn.Conv2d(base_channels * 2 + cond_channels, base_channels * 2, kernel_size=1)

        self.down3 = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)
        self.down4 = ResidualBlock(base_channels * 2, base_channels * 4, time_dim)
        self.downsample2 = nn.Conv2d(base_channels * 4, base_channels * 4, 4, 2, 1)

        self.fuse_16 = nn.Conv2d(base_channels * 4 + cond_channels, base_channels * 4, kernel_size=1)

        self.mid1 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid2 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)

        self.upsample1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, 2, 1)
        self.up1 = ResidualBlock(base_channels * 8, base_channels * 2, time_dim)
        self.up2 = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)

        self.upsample2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, 2, 1)
        self.up3 = ResidualBlock(base_channels * 4, base_channels, time_dim)
        self.up4 = ResidualBlock(base_channels, base_channels, time_dim)

        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, H) bruité [-1, 1]
        t: (B,) timesteps
        low: (B, 1, low_h, low_w) en [0, 1]
        """
        c64, c32, c16 = cond_pyramid_from_low(low, self.high_size)

        t_emb = timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x0 = self.in_conv(torch.cat([x, c64], dim=1))

        d1 = self.down1(x0, t_emb)
        d2 = self.down2(d1, t_emb)
        x = self.downsample1(d2)
        x = self.fuse_32(torch.cat([x, c32], dim=1))

        d3 = self.down3(x, t_emb)
        d4 = self.down4(d3, t_emb)
        x = self.downsample2(d4)
        x = self.fuse_16(torch.cat([x, c16], dim=1))

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


@dataclass
class SRDiffusionConfig:
    pairs_manifest_path: str = "Healthy_SR_pairs/scale_2/manifest.jsonl"
    output_dir: str = "ImageGeneration/sr_diffusion_outputs"

    high_size: int = 64
    low_size: int = 32
    img_channels: int = 1
    base_channels: int = 64

    batch_size: int = 32
    num_epochs: int = 50
    lr: float = 2e-4

    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    sample_interval: int = 500
    num_sample_images: int = 16
    ddim_steps: int = 50
    ddim_eta: float = 0.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def sample_ddim_cond(
    scheduler: DDPMScheduler,
    model: nn.Module,
    low: torch.Tensor,
    img_channels: int,
    high_size: int,
    steps: int,
    eta: float,
) -> torch.Tensor:
    """DDIM conditionné sur low ; sortie [-1, 1]."""
    if steps < 2:
        raise ValueError("steps doit être >= 2")

    device = low.device
    b = low.size(0)
    times = torch.linspace(scheduler.timesteps - 1, 0, steps, device=device).long()
    x = torch.randn(b, img_channels, high_size, high_size, device=device)

    for i in range(steps):
        t = times[i].item()
        t_prev = times[i + 1].item() if i < steps - 1 else 0

        t_batch = torch.full((b,), t, device=device, dtype=torch.long)
        eps = model(x, t_batch, low)

        alpha_bar_t = scheduler.alphas_cumprod[t]
        alpha_bar_prev = scheduler.alphas_cumprod[t_prev]

        x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

        sigma = (
            eta
            * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
            * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
        )

        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0)) * eps
        noise = torch.randn_like(x) if (eta > 0.0 and i < steps - 1) else torch.zeros_like(x)
        x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

    return x


def _to_display(x_neg1_1: torch.Tensor) -> torch.Tensor:
    x = (x_neg1_1 + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)


@torch.no_grad()
def save_sr_diffusion_samples(
    model: nn.Module,
    scheduler: DDPMScheduler,
    low_batch: torch.Tensor,
    high_batch: torch.Tensor,
    step: int,
    epoch: int,
    output_dir: str,
    config: SRDiffusionConfig,
    prefix: str = "recon",
    max_items: int = 16,
):
    model.eval()
    device = next(model.parameters()).device
    low = low_batch[:max_items].to(device)
    high = high_batch[:max_items].to(device)

    pseudo = sample_ddim_cond(
        scheduler,
        model,
        low,
        img_channels=config.img_channels,
        high_size=config.high_size,
        steps=config.ddim_steps,
        eta=config.ddim_eta,
    )

    low_up = F.interpolate(low, size=(config.high_size, config.high_size), mode="bilinear", align_corners=False)
    grid = torch.cat([low_up.cpu(), high.cpu(), _to_display(pseudo).cpu()], dim=0)
    path = os.path.join(output_dir, f"{prefix}_epoch_{epoch:03d}_step_{step:06d}.png")
    save_image(grid, path, nrow=max_items, normalize=False)


@torch.no_grad()
def save_epoch_comparison_low(
    model: nn.Module,
    scheduler: DDPMScheduler,
    low_batch: torch.Tensor,
    high_batch: torch.Tensor,
    epoch: int,
    output_dir: str,
    config: SRDiffusionConfig,
    max_items: int = 16,
    filename_prefix: str = "compare_low",
):
    """
    Comparaison à la résolution low : low | high réduite | pseudo_high réduite.
    """
    model.eval()
    device = next(model.parameters()).device

    low = low_batch[:max_items].to(device)
    high = high_batch[:max_items].to(device)

    pseudo = sample_ddim_cond(
        scheduler,
        model,
        low,
        img_channels=config.img_channels,
        high_size=config.high_size,
        steps=config.ddim_steps,
        eta=config.ddim_eta,
    )
    pseudo = _to_display(pseudo)

    low_h, low_w = int(low.shape[-2]), int(low.shape[-1])
    high_down = F.interpolate(high, size=(low_h, low_w), mode="bilinear", align_corners=False)
    recon_down = F.interpolate(pseudo, size=(low_h, low_w), mode="bilinear", align_corners=False)

    grid = torch.cat([low.cpu(), high_down.cpu(), recon_down.cpu()], dim=0)
    nrow = max_items
    out_path = os.path.join(output_dir, f"{filename_prefix}_epoch_{epoch:03d}.png")
    save_image(grid, out_path, nrow=nrow, normalize=False)


def train_sr_diffusion(config: SRDiffusionConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device(config.device)

    print("=" * 70)
    print("TRAINING CONDITIONAL DIFFUSION - SUPER RESOLUTION (low -> high)")
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
        raise RuntimeError("Dataset vide. Vérifie manifest et chemins low/high.")

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = ConditionalSRUNet(
        img_channels=config.img_channels,
        cond_channels=1,
        high_size=config.high_size,
        base_channels=config.base_channels,
        time_dim=256,
    ).to(device)

    scheduler = DDPMScheduler(
        timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()

    step = 0
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}", unit="batch")

        avg_loss = 0.0
        n_batches = 0
        last_low_cpu = None
        last_high_cpu = None

        for low, high in pbar:
            low = low.to(device)
            high = high.to(device)

            x0 = high * 2.0 - 1.0

            b = x0.size(0)
            t = torch.randint(0, config.timesteps, (b,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = scheduler.q_sample(x0, t, noise)

            optimizer.zero_grad()
            noise_pred = model(x_t, t, low)
            loss = mse(noise_pred, noise)
            loss.backward()
            optimizer.step()

            avg_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix({"MSE": f"{loss.item():.4f}"})

            if step % config.sample_interval == 0:
                save_sr_diffusion_samples(
                    model=model,
                    scheduler=scheduler,
                    low_batch=low.detach().cpu(),
                    high_batch=high.detach().cpu(),
                    step=step,
                    epoch=epoch,
                    output_dir=config.output_dir,
                    config=config,
                    prefix="recon",
                    max_items=config.num_sample_images,
                )

            last_low_cpu = low.detach().cpu()
            last_high_cpu = high.detach().cpu()
            step += 1

        torch.save(model.state_dict(), os.path.join(config.output_dir, f"sr_diffusion_epoch_{epoch:03d}.pth"))

        if last_low_cpu is not None and last_high_cpu is not None:
            save_epoch_comparison_low(
                model=model,
                scheduler=scheduler,
                low_batch=last_low_cpu,
                high_batch=last_high_cpu,
                epoch=epoch,
                output_dir=config.output_dir,
                config=config,
                max_items=config.num_sample_images,
                filename_prefix="compare_low",
            )

        avg_loss /= max(1, n_batches)
        print(f"Epoch {epoch} done. Avg noise MSE={avg_loss:.4f}")

    print("Training done.")


def main() -> None:
    cfg = SRDiffusionConfig()
    train_sr_diffusion(cfg)


if __name__ == "__main__":
    main()
