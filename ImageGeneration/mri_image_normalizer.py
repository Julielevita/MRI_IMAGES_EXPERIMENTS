import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
import glob
from collections import Counter


class CTImageNormalizer:
    """
    Classe pour normaliser les images CT scan pour la classification CNN.
    Normalise la taille et les échelles de gris des images.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), normalize_pixels: bool = True):
        """
        Initialise le normaliseur d'images CT.
        
        Args:
            target_size: Taille cible pour toutes les images (largeur, hauteur). Par défaut (224, 224).
            normalize_pixels: Si True, normalise les pixels entre 0 et 1. Si False, garde les valeurs 0-255.
        """
        self.target_size = target_size
        self.normalize_pixels = normalize_pixels
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Charge une image depuis un chemin de fichier.
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Array numpy de l'image en niveaux de gris
        """
        try:
            # Charge l'image avec PIL
            img = Image.open(image_path)
            
            # Convertit en niveaux de gris si nécessaire
            if img.mode != 'L':
                img = img.convert('L')
            
            # Convertit en array numpy
            img_array = np.array(img, dtype=np.float32)
            
            return img_array
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {image_path}: {e}")
            return None
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Redimensionne une image à la taille cible.
        
        Args:
            image: Array numpy de l'image
            
        Returns:
            Image redimensionnée
        """
        # Convertit l'array numpy en PIL Image pour le redimensionnement
        img_pil = Image.fromarray(image.astype(np.uint8))
        
        # Redimensionne avec interpolation LANCZOS (haute qualité)
        img_resized = img_pil.resize(self.target_size, Image.LANCZOS)
        
        # Convertit de nouveau en array numpy
        img_array = np.array(img_resized, dtype=np.float32)
        
        return img_array
    
    def normalize_pixel_values(self, image: np.ndarray) -> np.ndarray:
        """
        Normalise les valeurs de pixels entre 0 et 1.
        
        Args:
            image: Array numpy de l'image
            
        Returns:
            Image avec valeurs normalisées entre 0 et 1
        """
        if self.normalize_pixels:
            # Normalise entre 0 et 1
            image = image / 255.0
        else:
            # Garde les valeurs entre 0 et 255
            image = np.clip(image, 0, 255)
        
        return image
    
    def process_single_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Traite une seule image: charge, redimensionne et normalise.
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Array numpy normalisé de l'image, ou None en cas d'erreur
        """
        # Charge l'image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Redimensionne
        image = self.resize_image(image)
        
        # Normalise les valeurs de pixels
        image = self.normalize_pixel_values(image)
        
        # Ajoute une dimension de canal pour la compatibilité CNN (height, width, channels)
        image = np.expand_dims(image, axis=-1)
        
        return image
    
    def process_directory(self, directory_path: str, file_extensions: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Traite toutes les images d'un répertoire.
        
        Args:
            directory_path: Chemin vers le répertoire contenant les images
            file_extensions: Liste des extensions de fichiers à traiter. 
                           Par défaut: ['.jpg', '.jpeg', '.png']
        
        Returns:
            Tuple contenant:
            - Array numpy de toutes les images (n_images, height, width, channels)
            - Liste des chemins des images traitées avec succès
        """
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png']
        
        # Trouve tous les fichiers d'images dans le répertoire
        image_paths = []
        for ext in file_extensions:
            pattern = os.path.join(directory_path, f'*{ext}')
            image_paths.extend(glob.glob(pattern))
            # Cherche aussi les extensions en majuscules
            pattern = os.path.join(directory_path, f'*{ext.upper()}')
            image_paths.extend(glob.glob(pattern))
        
        # Trie les chemins pour un ordre cohérent
        image_paths.sort()
        
        print(f"Trouvé {len(image_paths)} images dans {directory_path}")
        
        # Traite chaque image
        processed_images = []
        successful_paths = []
        
        for img_path in image_paths:
            processed_img = self.process_single_image(img_path)
            if processed_img is not None:
                processed_images.append(processed_img)
                successful_paths.append(img_path)
        
        if len(processed_images) == 0:
            print("Aucune image n'a pu être traitée.")
            return np.array([]), []
        
        # Convertit la liste en array numpy
        images_array = np.array(processed_images)
        
        print(f"Traitement terminé: {len(successful_paths)} images traitées avec succès")
        print(f"Shape du tableau final: {images_array.shape}")
        
        return images_array, successful_paths
    
    @staticmethod
    def analyze_dimensions(directory_path: str, 
                          file_extensions: List[str] = None,
                          max_samples: Optional[int] = None,
                          verbose: bool = True) -> Dict:
        """
        Analyse les dimensions des images dans un répertoire et recommande une taille optimale.
        
        Args:
            directory_path: Chemin vers le répertoire contenant les images
            file_extensions: Liste des extensions de fichiers à analyser. 
                           Par défaut: ['.jpg', '.jpeg', '.png']
            max_samples: Nombre maximum d'images à analyser. Si None, analyse toutes les images.
            verbose: Si True, affiche les résultats de l'analyse
            
        Returns:
            Dictionnaire contenant:
            - 'dimensions': liste de toutes les dimensions trouvées (width, height)
            - 'size_distribution': Counter des dimensions les plus fréquentes
            - 'statistics': statistiques (min, max, moyenne pour width et height)
            - 'recommended_size': taille recommandée (width, height)
            - 'recommendation_reason': raison de la recommandation
        """
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png']
        
        # Trouve tous les fichiers d'images dans le répertoire
        image_paths = []
        for ext in file_extensions:
            pattern = os.path.join(directory_path, f'*{ext}')
            image_paths.extend(glob.glob(pattern))
            pattern = os.path.join(directory_path, f'*{ext.upper()}')
            image_paths.extend(glob.glob(pattern))
        
        image_paths.sort()
        
        # Limite le nombre d'échantillons si demandé
        if max_samples and len(image_paths) > max_samples:
            import random
            image_paths = random.sample(image_paths, max_samples)
        
        if verbose:
            print(f"Analyse des dimensions de {len(image_paths)} images...")
        
        # Analyse les dimensions
        dimensions = []
        failed = 0
        
        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    dimensions.append(img.size)  # (width, height)
            except Exception as e:
                failed += 1
                if verbose and failed <= 5:
                    print(f"  Erreur sur {os.path.basename(img_path)}: {e}")
        
        if len(dimensions) == 0:
            if verbose:
                print("Aucune image n'a pu être analysée.")
            return {
                'dimensions': [],
                'size_distribution': Counter(),
                'statistics': {},
                'recommended_size': (224, 224),
                'recommendation_reason': 'Aucune image analysée, utilisation de la taille par défaut'
            }
        
        # Calcule les statistiques
        widths = [d[0] for d in dimensions]
        heights = [d[1] for d in dimensions]
        size_distribution = Counter(dimensions)
        
        stats = {
            'width': {
                'min': min(widths),
                'max': max(widths),
                'mean': int(np.mean(widths)),
                'median': int(np.median(widths))
            },
            'height': {
                'min': min(heights),
                'max': max(heights),
                'mean': int(np.mean(heights)),
                'median': int(np.median(heights))
            },
            'total_images': len(dimensions),
            'unique_sizes': len(size_distribution),
            'failed': failed
        }
        
        # Calcule les ratios largeur/hauteur
        ratios = [w/h for w, h in dimensions]
        stats['aspect_ratio'] = {
            'min': min(ratios),
            'max': max(ratios),
            'mean': np.mean(ratios)
        }
        
        # Recommandation de taille
        recommended_size, reason = CTImageNormalizer._recommend_size(
            size_distribution, stats, dimensions
        )
        
        result = {
            'dimensions': dimensions,
            'size_distribution': size_distribution,
            'statistics': stats,
            'recommended_size': recommended_size,
            'recommendation_reason': reason
        }
        
        # Affiche les résultats si demandé
        if verbose:
            CTImageNormalizer._print_analysis_results(result)
        
        return result
    
    @staticmethod
    def _recommend_size(size_distribution: Counter, stats: Dict, dimensions: List[Tuple]) -> Tuple[Tuple[int, int], str]:
        """
        Recommande une taille optimale basée sur l'analyse des dimensions.
        
        Args:
            size_distribution: Distribution des tailles (Counter)
            stats: Statistiques calculées
            dimensions: Liste de toutes les dimensions
            
        Returns:
            Tuple (taille_recommandée, raison)
        """
        # Tailles standards pour CNN
        standard_sizes = [128, 224, 256, 320, 384, 512]
        
        # Si une taille domine (>80% des images)
        most_common = size_distribution.most_common(1)[0]
        most_common_size, count = most_common
        percentage = (count / len(dimensions)) * 100
        
        if percentage >= 80:
            # La majorité des images ont la même taille
            width, height = most_common_size
            
            # Si c'est carré, recommande une taille standard proche
            if width == height:
                # Trouve la taille standard la plus proche
                closest_standard = min(standard_sizes, key=lambda x: abs(x - width))
                if abs(closest_standard - width) / width < 0.3:  # Si proche (<30% de différence)
                    return (closest_standard, closest_standard), \
                           f"{percentage:.1f}% des images sont en {width}x{height}, " \
                           f"recommandation: {closest_standard}x{closest_standard} (taille standard CNN)"
                else:
                    return (width, height), \
                           f"{percentage:.1f}% des images sont en {width}x{height}, " \
                           f"recommandation: garder cette taille"
            else:
                # Image rectangulaire, recommande une taille carrée standard
                avg_dim = int((width + height) / 2)
                closest_standard = min(standard_sizes, key=lambda x: abs(x - avg_dim))
                return (closest_standard, closest_standard), \
                       f"Images rectangulaires ({width}x{height} majoritaire), " \
                       f"recommandation: {closest_standard}x{closest_standard} (carré standard)"
        
        # Si pas de taille dominante, utilise la médiane
        median_width = stats['width']['median']
        median_height = stats['height']['median']
        
        # Si les images sont globalement carrées
        if abs(median_width - median_height) / max(median_width, median_height) < 0.1:
            # Trouve la taille standard la plus proche de la médiane
            closest_standard = min(standard_sizes, key=lambda x: abs(x - median_width))
            return (closest_standard, closest_standard), \
                   f"Médiane: {median_width}x{median_height}, " \
                   f"recommandation: {closest_standard}x{closest_standard} (taille standard CNN)"
        else:
            # Images rectangulaires variées, recommande une taille carrée standard
            avg_dim = int((median_width + median_height) / 2)
            closest_standard = min(standard_sizes, key=lambda x: abs(x - avg_dim))
            return (closest_standard, closest_standard), \
                   f"Dimensions variées (médiane: {median_width}x{median_height}), " \
                   f"recommandation: {closest_standard}x{closest_standard} (taille standard CNN)"
    
    @staticmethod
    def _print_analysis_results(result: Dict):
        """Affiche les résultats de l'analyse de manière lisible."""
        print("\n" + "="*60)
        print("ANALYSE DES DIMENSIONS D'IMAGES")
        print("="*60)
        
        stats = result['statistics']
        print(f"\nNombre d'images analysées: {stats['total_images']}")
        if stats['failed'] > 0:
            print(f"Images non analysées: {stats['failed']}")
        
        print(f"\nStatistiques de largeur:")
        print(f"  Min: {stats['width']['min']}px")
        print(f"  Max: {stats['width']['max']}px")
        print(f"  Moyenne: {stats['width']['mean']}px")
        print(f"  Médiane: {stats['width']['median']}px")
        
        print(f"\nStatistiques de hauteur:")
        print(f"  Min: {stats['height']['min']}px")
        print(f"  Max: {stats['height']['max']}px")
        print(f"  Moyenne: {stats['height']['mean']}px")
        print(f"  Médiane: {stats['height']['median']}px")
        
        print(f"\nRatio largeur/hauteur:")
        print(f"  Min: {stats['aspect_ratio']['min']:.2f}")
        print(f"  Max: {stats['aspect_ratio']['max']:.2f}")
        print(f"  Moyenne: {stats['aspect_ratio']['mean']:.2f}")
        
        print(f"\nTailles uniques: {stats['unique_sizes']}")
        print(f"\nTop 5 dimensions les plus fréquentes:")
        for (width, height), count in result['size_distribution'].most_common(5):
            percentage = (count / stats['total_images']) * 100
            print(f"  {width}x{height}: {count} images ({percentage:.1f}%)")
        
        print(f"\n{'='*60}")
        print("RECOMMANDATION:")
        print(f"{'='*60}")
        recommended = result['recommended_size']
        print(f"Taille recommandée: {recommended[0]}x{recommended[1]}")
        print(f"Raison: {result['recommendation_reason']}")
        print("="*60 + "\n")
    
    def get_statistics(self, images: np.ndarray) -> dict:
        """
        Calcule des statistiques sur les images normalisées.
        
        Args:
            images: Array numpy des images (n_images, height, width, channels)
            
        Returns:
            Dictionnaire contenant les statistiques (mean, std, min, max)
        """
        stats = {
            'mean': np.mean(images),
            'std': np.std(images),
            'min': np.min(images),
            'max': np.max(images),
            'shape': images.shape
        }
        return stats


# Exemple d'utilisation
if __name__ == "__main__":
    # Chemin vers le dossier des images saines
    healthy_dir = "Brain Tumor CT scan Images/Healthy"
    
    # ÉTAPE 1: Analyse automatique des dimensions et recommandation
    print("ÉTAPE 1: Analyse des dimensions...")
    analysis = CTImageNormalizer.analyze_dimensions(
        healthy_dir, 
        max_samples=500,  # Analyse jusqu'à 500 images pour être rapide
        verbose=True
    )
    
    # Récupère la taille recommandée
    recommended_size = analysis['recommended_size']
    print(f"\n✓ Taille recommandée: {recommended_size[0]}x{recommended_size[1]}")
    
    # ÉTAPE 2: Initialise le normaliseur avec la taille recommandée
    print(f"\nÉTAPE 2: Initialisation avec la taille recommandée...")
    normalizer = CTImageNormalizer(
        target_size=recommended_size, 
        normalize_pixels=True
    )
    
    # ÉTAPE 3: Traite toutes les images du dossier
    print(f"\nÉTAPE 3: Traitement des images...")
    images, paths = normalizer.process_directory(healthy_dir)
    
    # ÉTAPE 4: Affiche les statistiques des images normalisées
    if len(images) > 0:
        stats = normalizer.get_statistics(images)
        print("\nStatistiques des images normalisées:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Exemple: traiter une seule image
        example_path = "Brain Tumor CT scan Images/Tumor/ct_tumor (1).jpg"
        if os.path.exists(example_path):
            example_img = normalizer.process_single_image(example_path)
            if example_img is not None:
                print(f"\nExemple d'image traitée - Shape: {example_img.shape}")
                print(f"Valeurs min/max: {np.min(example_img):.4f} / {np.max(example_img):.4f}")