import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

# Normalisation cohérente avec le reste du projet
from mri_image_normalizer import CTImageNormalizer


@dataclass
class GANConfig:
    """
    Configuration complète de l'entraînement GAN.
    """

    healthy_dir: str = "../Brain Tumor MRI images/Healthy"
    output_dir: str = "ImageGeneration/gan_outputs"

    # Dimensions des images générées (on réduit à 64x64 pour stabiliser le GAN)
    img_size: int = 64
    img_channels: int = 1

    # Hyperparamètres GAN
    latent_dim: int = 100
    batch_size: int = 64
    num_epochs: int = 50
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999

    # Logging / sauvegarde
    sample_interval: int = 500  # en itérations
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class HealthyMRIDataset(Dataset):
    """
    Dataset PyTorch pour les images MRI saines uniquement.
    Utilise CTImageNormalizer pour obtenir des tensors (1, H, W) normalisés.
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

        print(f"\nDataset GAN (Healthy MRI) créé:")
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

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img_array = self.normalizer.process_single_image(img_path)

        if img_array is None:
            # Image noire de secours
            img_array = np.zeros((self.target_size[1], self.target_size[0], 1), dtype=np.float32)

        # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        return img_tensor


class Generator(nn.Module):
    """
    Générateur de type DCGAN pour images 1 canal 64x64.
    Entrée: bruit z de dimension latent_dim
    Sortie: image (1, 64, 64) avec tanh
    """

    def __init__(self, latent_dim: int, img_channels: int):
        super().__init__()

        self.net = nn.Sequential(
            # input: (latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # (512, 4, 4)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # (256, 8, 8)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # (128, 16, 16)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # (64, 32, 32)
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # (img_channels, 64, 64)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """
    Discriminateur de type DCGAN pour images 1 canal 64x64.
    Sortie: score scalaire (probabilité "réel") par image.
    """

    def __init__(self, img_channels: int):
        super().__init__()

        self.net = nn.Sequential(
            # (img_channels, 64, 64)
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 32, 32)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 16, 16)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 8, 8)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # (512, 4, 4)
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(-1)


def weights_init(module: nn.Module) -> None:
    """
    Initialisation recommandée pour DCGAN.
    """
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def train_gan(config: GANConfig) -> None:
    """
    Pipeline complète d'entraînement GAN (type DCGAN) sur les MRI saines.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device(config.device)
    print("=" * 70)
    print("ENTRAÎNEMENT D'UN GAN (DCGAN) SUR LES IRM SAINES")
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

    # Mise à l'échelle des vraies images en [-1, 1] pour matcher tanh
    def preprocess_real(batch: torch.Tensor) -> torch.Tensor:
        return batch * 2.0 - 1.0

    # =========================================================
    # 2. Initialisation du GAN
    # =========================================================
    print("\n" + "=" * 70)
    print("ÉTAPE 2: Initialisation du générateur et du discriminateur")
    print("=" * 70)

    generator = Generator(config.latent_dim, config.img_channels).to(device)
    discriminator = Discriminator(config.img_channels).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    print(f"\nGénérateur:\n{generator}")
    print(f"\nDiscriminateur:\n{discriminator}")

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

    # Bruit fixe pour suivre visuellement la progression
    fixed_noise = torch.randn(64, config.latent_dim, 1, 1, device=device)

    # =========================================================
    # 3. Boucle d'entraînement
    # =========================================================
    print("\n" + "=" * 70)
    print("ÉTAPE 3: Entraînement du GAN")
    print("=" * 70)

    step = 0
    for epoch in range(config.num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}", unit="batch")

        for real_images in pbar:
            real_images = real_images.to(device)
            real_images = preprocess_real(real_images)

            batch_size = real_images.size(0)
            real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
            fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

            # -------------------------------------------------
            # (1) Mise à jour du Discriminateur: maximise log(D(x)) + log(1 - D(G(z)))
            # -------------------------------------------------
            discriminator.zero_grad()

            # Vrai
            out_real = discriminator(real_images)
            loss_D_real = criterion(out_real, real_labels)

            # Faux
            noise = torch.randn(batch_size, config.latent_dim, 1, 1, device=device)
            fake_images = generator(noise).detach()
            out_fake = discriminator(fake_images)
            loss_D_fake = criterion(out_fake, fake_labels)

            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_D.step()

            # -------------------------------------------------
            # (2) Mise à jour du Générateur: maximise log(D(G(z)))
            # -------------------------------------------------
            generator.zero_grad()

            noise = torch.randn(batch_size, config.latent_dim, 1, 1, device=device)
            gen_images = generator(noise)
            out_gen = discriminator(gen_images)
            loss_G = criterion(out_gen, real_labels)  # on "trompe" le D

            loss_G.backward()
            optimizer_G.step()

            pbar.set_postfix(
                {
                    "Loss_D": f"{loss_D.item():.4f}",
                    "Loss_G": f"{loss_G.item():.4f}",
                    "D(x)": f"{out_real.mean().item():.3f}",
                    "D(G(z))": f"{out_fake.mean().item():.3f}",
                }
            )

            # Sauvegarde régulière d'images générées (avec bruit fixe)
            if step % config.sample_interval == 0:
                with torch.no_grad():
                    fake_fixed = generator(fixed_noise).detach().cpu()
                # Remet en [0, 1] pour sauvegarde
                fake_fixed = (fake_fixed + 1.0) / 2.0
                sample_path = os.path.join(config.output_dir, f"epoch_{epoch + 1:03d}_step_{step:06d}.png")
                save_image(fake_fixed, sample_path, nrow=8, normalize=False)
            step += 1

        # Sauvegarde des poids en fin d'époque
        torch.save(generator.state_dict(), os.path.join(config.output_dir, f"generator_epoch_{epoch + 1:03d}.pth"))
        torch.save(
            discriminator.state_dict(),
            os.path.join(config.output_dir, f"discriminator_epoch_{epoch + 1:03d}.pth"),
        )

    print("\n" + "=" * 70)
    print("ENTRAÎNEMENT DU GAN TERMINÉ")
    print("=" * 70)


def main() -> None:
    """
    Point d'entrée principal de la pipeline GAN.
    """

    config = GANConfig()
    train_gan(config)


if __name__ == "__main__":
    main()

