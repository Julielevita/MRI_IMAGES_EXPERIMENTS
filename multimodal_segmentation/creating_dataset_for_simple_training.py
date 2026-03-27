import os
import glob
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import numpy as np

# Importe la classe de normalisation depuis ImageNormalizer.py
from ImageNormalizer import CTImageNormalizer


class BrainTumorMRIDataset(Dataset):
    """
    Dataset PyTorch pour les images MRI de tumeurs cérébrales.
    Charge les images des dossiers Healthy et Tumor, les normalise et les convertit en tenseurs.
    """
    
    def __init__(self, 
                 healthy_dir: str,
                 tumor_dir: str,
                 target_size: Tuple[int, int] = (224, 224),
                 normalize_pixels: bool = True,
                 file_extensions: List[str] = None,
                 auto_analyze: bool = True):
        """
        Initialise le dataset.
        
        Args:
            healthy_dir: Chemin vers le dossier contenant les images saines
            tumor_dir: Chemin vers le dossier contenant les images avec tumeur
            target_size: Taille cible pour toutes les images (largeur, hauteur)
            normalize_pixels: Si True, normalise les pixels entre 0 et 1
            file_extensions: Liste des extensions de fichiers à charger. 
                            Par défaut: ['.jpg', '.jpeg', '.png']
            auto_analyze: Si True, analyse automatiquement les dimensions et recommande une taille
        """
        self.healthy_dir = healthy_dir
        self.tumor_dir = tumor_dir
        self.target_size = target_size
        self.normalize_pixels = normalize_pixels
        
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png']
        self.file_extensions = file_extensions
        
        # Analyse automatique des dimensions si demandé
        if auto_analyze:
            print("Analyse automatique des dimensions...")
            # Analyse le dossier Healthy
            healthy_analysis = CTImageNormalizer.analyze_dimensions(
                healthy_dir, 
                file_extensions=file_extensions,
                max_samples=200,
                verbose=False
            )
            # Analyse le dossier Tumor
            tumor_analysis = CTImageNormalizer.analyze_dimensions(
                tumor_dir,
                file_extensions=file_extensions,
                max_samples=200,
                verbose=False
            )
            
            # Utilise la taille recommandée si elle est disponible
            if healthy_analysis['recommended_size']:
                recommended = healthy_analysis['recommended_size']
                print(f"Taille recommandée détectée: {recommended[0]}x{recommended[1]}")
                if self.target_size == (224, 224):  # Si taille par défaut, utilise la recommandation
                    self.target_size = recommended
                    print(f"Utilisation de la taille recommandée: {self.target_size}")
        
        # Initialise le normaliseur
        self.normalizer = CTImageNormalizer(
            target_size=self.target_size,
            normalize_pixels=self.normalize_pixels
        )
        
        # Charge les chemins des images
        self.image_paths = []
        self.labels = []
        
        # Charge les images saines (label = 0)
        self._load_images_from_directory(healthy_dir, label=0)
        
        # Charge les images avec tumeur (label = 1)
        self._load_images_from_directory(tumor_dir, label=1)
        
        print(f"\nDataset créé avec succès:")
        print(f"  - Images saines (label 0): {self.labels.count(0)}")
        print(f"  - Images avec tumeur (label 1): {self.labels.count(1)}")
        print(f"  - Total: {len(self.image_paths)} images")
        print(f"  - Taille cible: {self.target_size[0]}x{self.target_size[1]}")
    
    def _load_images_from_directory(self, directory: str, label: int):
        """
        Charge tous les chemins d'images d'un répertoire et leur associe un label.
        
        Args:
            directory: Chemin vers le répertoire
            label: Label à associer aux images (0 pour Healthy, 1 pour Tumor)
        """
        image_paths = []
        
        for ext in self.file_extensions:
            # Cherche les fichiers avec l'extension en minuscules
            pattern = os.path.join(directory, f'*{ext}')
            image_paths.extend(glob.glob(pattern))
            
            # Cherche aussi les extensions en majuscules
            pattern = os.path.join(directory, f'*{ext.upper()}')
            image_paths.extend(glob.glob(pattern))
        
        # Trie pour un ordre cohérent
        image_paths.sort()
        
        # Ajoute les chemins et labels
        self.image_paths.extend(image_paths)
        self.labels.extend([label] * len(image_paths))
    
    def __len__(self) -> int:
        """
        Retourne le nombre d'images dans le dataset.
        
        Returns:
            Nombre total d'images
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Récupère une image et son label à l'index donné.
        
        Args:
            idx: Index de l'image à récupérer
            
        Returns:
            Tuple (image_tensor, label_tensor) où:
            - image_tensor: Tenseur PyTorch de shape (1, height, width) pour niveaux de gris
            - label_tensor: Tenseur PyTorch contenant le label (0 ou 1)
        """
        # Récupère le chemin et le label
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Traite l'image avec le normaliseur
        # process_single_image retourne un array numpy de shape (height, width, channels)
        image_array = self.normalizer.process_single_image(image_path)
        
        if image_array is None:
            # Si l'image n'a pas pu être chargée, retourne une image noire
            print(f"Attention: Impossible de charger l'image {image_path}")
            image_array = np.zeros((self.target_size[1], self.target_size[0], 1), dtype=np.float32)
        
        # Convertit en tenseur PyTorch
        # PyTorch attend (channels, height, width), donc on transpose
        # image_array est (height, width, channels=1)
        # On veut (channels=1, height, width)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        
        # Convertit le label en tenseur
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_tensor, label_tensor
    
    def get_class_distribution(self) -> dict:
        """
        Retourne la distribution des classes dans le dataset.
        
        Returns:
            Dictionnaire avec le nombre d'images par classe
        """
        from collections import Counter
        label_counts = Counter(self.labels)
        return {
            'Healthy (0)': label_counts.get(0, 0),
            'Tumor (1)': label_counts.get(1, 0),
            'Total': len(self.labels)
        }
    
    def get_statistics(self) -> dict:
        """
        Calcule des statistiques sur le dataset.
        
        Returns:
            Dictionnaire contenant des statistiques sur le dataset
        """
        stats = {
            'total_images': len(self.image_paths),
            'class_distribution': self.get_class_distribution(),
            'target_size': self.target_size,
            'normalize_pixels': self.normalize_pixels,
            'healthy_dir': self.healthy_dir,
            'tumor_dir': self.tumor_dir
        }
        return stats


# Exemple d'utilisation
if __name__ == "__main__":
    # Chemins vers les dossiers
    healthy_dir = "Brain Tumor MRI images/Healthy"
    tumor_dir = "Brain Tumor MRI images/Tumor"
    
    # Crée le dataset
    print("Création du dataset...")
    dataset = BrainTumorMRIDataset(
        healthy_dir=healthy_dir,
        tumor_dir=tumor_dir,
        target_size=(224, 224),  # Peut être changé ou laissé pour auto-analyse
        normalize_pixels=True,
        auto_analyze=True  # Analyse automatique des dimensions
    )
    
    # Affiche les statistiques
    print("\n" + "="*60)
    print("STATISTIQUES DU DATASET")
    print("="*60)
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test: récupère quelques échantillons
    print("\n" + "="*60)
    print("TEST: Récupération d'échantillons")
    print("="*60)
    
    if len(dataset) > 0:
        # Récupère le premier échantillon
        image, label = dataset[0]
        print(f"\nPremier échantillon:")
        print(f"  Shape de l'image: {image.shape}")  # Devrait être (1, height, width)
        print(f"  Label: {label.item()} ({'Healthy' if label.item() == 0 else 'Tumor'})")
        print(f"  Type: {type(image)}")
        print(f"  Valeurs min/max: {image.min():.4f} / {image.max():.4f}")
        
        # Récupère quelques échantillons supplémentaires
        print(f"\nTest de plusieurs échantillons:")
        for i in [0, len(dataset)//4, len(dataset)//2, len(dataset)-1]:
            if i < len(dataset):
                img, lbl = dataset[i]
                print(f"  Index {i}: Label {lbl.item()} - Shape {img.shape}")
        
        # Test avec DataLoader PyTorch
        print(f"\n" + "="*60)
        print("TEST: Utilisation avec DataLoader")
        print("="*60)
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Récupère un batch
        for batch_idx, (images, labels) in enumerate(dataloader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Shape des images: {images.shape}")  # (batch_size, channels, height, width)
            print(f"  Shape des labels: {labels.shape}")  # (batch_size,)
            print(f"  Labels: {labels.tolist()}")
            if batch_idx >= 2:  # Limite à 3 batches pour l'exemple
                break
