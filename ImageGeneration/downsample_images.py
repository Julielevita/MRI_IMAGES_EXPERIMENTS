import argparse
import json
import os
from typing import List, Tuple

from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def iter_image_paths(input_dir: str, exts: Tuple[str, ...] = IMAGE_EXTENSIONS) -> List[str]:
    paths: List[str] = []
    for name in os.listdir(input_dir):
        lower = name.lower()
        if any(lower.endswith(e) for e in exts):
            paths.append(os.path.join(input_dir, name))
    paths.sort()
    return paths


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def downsample_healthy_images(
    input_dir: str,
    output_root: str,
    high_size: int = 64,
    low_size: int = 32,
    low_jpeg_quality: int = 25,
    high_format: str = "png",
    copy_originals: bool = False,
    overwrite: bool = False,
) -> str:
    """
    Genere des paires (low, high) a partir de chaque image dans `input_dir`.

    - high: image resizee a `high_size` (qualite preservee via PNG par defaut)
    - low : image resizee a `low_size` + reduction de qualite via JPEG `quality`
    - trace: `manifest.jsonl` contient original_path, high_path, low_path

    Retourne le chemin vers `manifest.jsonl`.
    """
    if high_size <= 0 or low_size <= 0:
        raise ValueError("high_size et low_size doivent etre > 0")

    high_dir = os.path.join(output_root, "high")
    low_dir = os.path.join(output_root, "low")
    originals_dir = os.path.join(output_root, "originals")

    ensure_dir(high_dir)
    ensure_dir(low_dir)
    if copy_originals:
        ensure_dir(originals_dir)

    manifest_path = os.path.join(output_root, "manifest.jsonl")
    if os.path.exists(manifest_path) and not overwrite:
        # Re-utilisable directement pour l'entraînement
        return manifest_path

    image_paths = iter_image_paths(input_dir)
    if not image_paths:
        raise RuntimeError(f"Aucune image trouvee dans: {input_dir}")

    if high_format.lower() not in ("png", "jpg", "jpeg"):
        raise ValueError("high_format doit etre 'png' ou 'jpg'/'jpeg'")

    with open(manifest_path, "w", encoding="utf-8") as f:
        for idx, original_path in enumerate(image_paths):
            base = os.path.splitext(os.path.basename(original_path))[0]
            # EVITE collision si meme base, on ajoute idx.
            pair_id = f"{base}_{idx:06d}"

            out_high_path = os.path.join(high_dir, f"{pair_id}.{high_format.lower()}")
            out_low_path = os.path.join(low_dir, f"{pair_id}.jpg")  # low en JPEG pour simuler reduction qualite

            if (not overwrite) and os.path.exists(out_high_path) and os.path.exists(out_low_path):
                high_path = out_high_path
                low_path = out_low_path
            else:
                with Image.open(original_path) as img:
                    img = img.convert("L")

                    high_img = img.resize((high_size, high_size), Image.LANCZOS)
                    low_img = img.resize((low_size, low_size), Image.LANCZOS)

                    # Sauvegarde high
                    if high_format.lower() == "png":
                        high_img.save(out_high_path, format="PNG")
                    else:
                        # JPEG pour high si specifie
                        high_img.save(out_high_path, format="JPEG", quality=95)

                    # Sauvegarde low (JPEG avec compression)
                    low_img.save(out_low_path, format="JPEG", quality=int(low_jpeg_quality), optimize=True)

                    high_path = out_high_path
                    low_path = out_low_path

                if copy_originals:
                    # Petite trace en local (optionnel)
                    safe_original_path = os.path.join(originals_dir, f"{pair_id}{os.path.splitext(original_path)[1].lower()}")
                    if not os.path.exists(safe_original_path):
                        # copie par lecture/écriture simple (sans dépendances)
                        with Image.open(original_path) as img:
                            img.save(safe_original_path)

            rec = {
                "id": pair_id,
                "original_path": original_path,
                "high_path": high_path,
                "low_path": low_path,
                "high_size": high_size,
                "low_size": low_size,
                "low_jpeg_quality": low_jpeg_quality,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Pairs generees dans: {output_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Nombre d'images: {len(image_paths)}")
    return manifest_path


# Alias : même pipeline pour tout dossier d'images (Healthy, Tumor, etc.)
downsample_sr_pairs = downsample_healthy_images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère des paires low/high (SR) depuis un dossier d'images MRI."
    )
    parser.add_argument(
        "--input_dir",
        default=os.path.join("..", "Brain Tumor MRI images", "Healthy"),
        help="Dossier source (jpg/png/jpeg), ex. ../Brain Tumor MRI images/Tumor",
    )
    parser.add_argument(
        "--output_root",
        default=os.path.join("Healthy_SR_pairs", "scale_2"),
        help="Sortie : high/, low/, manifest.jsonl (chemins relatifs au CWD)",
    )
    parser.add_argument("--high_size", type=int, default=64)
    parser.add_argument("--low_size", type=int, default=32)
    parser.add_argument("--low_jpeg_quality", type=int, default=25)
    parser.add_argument("--high_format", default="png", choices=("png", "jpg", "jpeg"))
    parser.add_argument(
        "--copy_originals",
        action="store_true",
        help="Copier aussi les originaux sous output_root/originals",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Régénérer même si manifest.jsonl existe déjà",
    )
    args = parser.parse_args()

    downsample_sr_pairs(
        input_dir=args.input_dir,
        output_root=args.output_root,
        high_size=args.high_size,
        low_size=args.low_size,
        low_jpeg_quality=args.low_jpeg_quality,
        high_format=args.high_format,
        copy_originals=args.copy_originals,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

