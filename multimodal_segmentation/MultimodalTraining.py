import os
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Importe la classe de normalisation depuis draft_1
from ImageNormalizer import CTImageNormalizer


class MultimodalBrainTumorDataset(Dataset):
    """
    Dataset PyTorch multimodal pour les images CT et MRI de tumeurs cérébrales.
    Charge les paires d'images CT/MRI correspondantes du même patient.
    """
    
    def __init__(self, 
                 ct_healthy_dir: str,
                 ct_tumor_dir: str,
                 mri_healthy_dir: str,
                 mri_tumor_dir: str,
                 target_size: tuple = (224, 224),
                 normalize_pixels: bool = True,
                 file_extensions: list = None,
                 auto_analyze: bool = True):
        """
        Initialise le dataset multimodal.
        
        Args:
            ct_healthy_dir: Chemin vers le dossier CT sain
            ct_tumor_dir: Chemin vers le dossier CT avec tumeur
            mri_healthy_dir: Chemin vers le dossier MRI sain
            mri_tumor_dir: Chemin vers le dossier MRI avec tumeur
            target_size: Taille cible pour toutes les images
            normalize_pixels: Si True, normalise les pixels entre 0 et 1
            file_extensions: Liste des extensions de fichiers à charger
            auto_analyze: Si True, analyse automatiquement les dimensions
        """
        self.target_size = target_size
        self.normalize_pixels = normalize_pixels
        
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png']
        self.file_extensions = file_extensions
        
        # Initialise le normaliseur
        self.normalizer = CTImageNormalizer(
            target_size=self.target_size,
            normalize_pixels=self.normalize_pixels
        )
        
        # Trouve les paires d'images correspondantes
        self.pairs = []
        self.labels = []
        
        # Charge les paires saines (Healthy)
        self._load_pairs(ct_healthy_dir, mri_healthy_dir, label=0)
        
        # Charge les paires avec tumeur (Tumor)
        self._load_pairs(ct_tumor_dir, mri_tumor_dir, label=1)
        
        print(f"\nDataset multimodal créé avec succès:")
        print(f"  - Paires saines (label 0): {self.labels.count(0)}")
        print(f"  - Paires avec tumeur (label 1): {self.labels.count(1)}")
        print(f"  - Total: {len(self.pairs)} paires d'images")
        print(f"  - Taille cible: {self.target_size[0]}x{self.target_size[1]}")
    
    def _extract_number(self, filename):
        """Extrait le numéro du nom de fichier."""
        match = re.search(r'\((\d+)\)', filename)
        return int(match.group(1)) if match else None
    
    def _load_pairs(self, ct_dir: str, mri_dir: str, label: int):
        """
        Charge les paires d'images CT/MRI correspondantes.
        
        Args:
            ct_dir: Dossier contenant les images CT
            mri_dir: Dossier contenant les images MRI
            label: Label à associer (0 pour Healthy, 1 pour Tumor)
        """
        # Charge les fichiers CT
        ct_files = {}
        for ext in self.file_extensions:
            pattern = os.path.join(ct_dir, f'*{ext}')
            for filepath in glob.glob(pattern):
                filename = os.path.basename(filepath)
                num = self._extract_number(filename)
                if num is not None:
                    ct_files[num] = filepath
        
        # Charge les fichiers MRI
        mri_files = {}
        for ext in self.file_extensions:
            pattern = os.path.join(mri_dir, f'*{ext}')
            for filepath in glob.glob(pattern):
                filename = os.path.basename(filepath)
                num = self._extract_number(filename)
                if num is not None:
                    mri_files[num] = filepath
        
        # Trouve les paires correspondantes (même numéro)
        common_numbers = set(ct_files.keys()) & set(mri_files.keys())
        
        for num in sorted(common_numbers):
            self.pairs.append((ct_files[num], mri_files[num]))
            self.labels.append(label)
    
    def __len__(self) -> int:
        """Retourne le nombre de paires d'images."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Récupère une paire d'images CT/MRI et son label.
        
        Args:
            idx: Index de la paire à récupérer
            
        Returns:
            Tuple (ct_tensor, mri_tensor, label_tensor)
        """
        ct_path, mri_path = self.pairs[idx]
        label = self.labels[idx]
        
        # Traite l'image CT
        ct_array = self.normalizer.process_single_image(ct_path)
        if ct_array is None:
            ct_array = np.zeros((self.target_size[1], self.target_size[0], 1), dtype=np.float32)
        
        # Traite l'image MRI
        mri_array = self.normalizer.process_single_image(mri_path)
        if mri_array is None:
            mri_array = np.zeros((self.target_size[1], self.target_size[0], 1), dtype=np.float32)
        
        # Convertit en tenseurs PyTorch (channels, height, width)
        ct_tensor = torch.from_numpy(ct_array).permute(2, 0, 1)
        mri_tensor = torch.from_numpy(mri_array).permute(2, 0, 1)
        
        # Convertit le label en tenseur
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return ct_tensor, mri_tensor, label_tensor


class FeatureExtractorCNN(nn.Module):
    """
    CNN pour extraire des features d'une image (CT ou MRI).
    Retourne une représentation vectorielle au lieu de la classification directe.
    """
    
    def __init__(self, input_size=(224, 224), feature_dim=512):
        """
        Initialise l'extracteur de features.
        
        Args:
            input_size: Taille d'entrée des images (height, width)
            feature_dim: Dimension de la représentation finale
        """
        super(FeatureExtractorCNN, self).__init__()
        
        # Première couche de convolution
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Deuxième couche de convolution
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Troisième couche de convolution
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calcule la taille après les convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size[0], input_size[1])
            dummy_output = self._forward_features(dummy_input)
            self.feature_size = dummy_output.view(1, -1).size(1)
        
        # Couche fully connected pour la représentation finale
        self.fc = nn.Linear(self.feature_size, feature_dim)
        self.relu4 = nn.ReLU()
    
    def _forward_features(self, x):
        """Passe avant jusqu'aux features (avant fully connected)."""
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        return x
    
    def forward(self, x):
        """
        Extrait les features d'une image.
        
        Args:
            x: Tenseur d'images de shape (batch_size, 1, height, width)
            
        Returns:
            Représentation vectorielle de shape (batch_size, feature_dim)
        """
        # Convolutions et poolings
        x = self._forward_features(x)
        
        # Aplatit
        x = x.view(x.size(0), -1)
        
        # Fully connected pour obtenir la représentation finale
        x = self.relu4(self.fc(x))
        
        return x


class MultimodalFusionModel(nn.Module):
    """
    Modèle multimodal qui fusionne les représentations CT et MRI.
    Architecture: CT_CNN -> Features_CT | MRI_CNN -> Features_MRI -> Fusion -> Classifier
    """
    
    def __init__(self, 
                 ct_input_size=(224, 224),
                 mri_input_size=(224, 224),
                 feature_dim=512,
                 num_classes=2,
                 fusion_method='concat'):
        """
        Initialise le modèle multimodal.
        
        Args:
            ct_input_size: Taille d'entrée des images CT
            mri_input_size: Taille d'entrée des images MRI
            feature_dim: Dimension des features extraites par chaque CNN
            num_classes: Nombre de classes (2: Healthy, Tumor)
            fusion_method: Méthode de fusion ('concat', 'add', 'multiply')
        """
        super(MultimodalFusionModel, self).__init__()
        
        self.fusion_method = fusion_method
        
        # Deux extracteurs de features (un pour CT, un pour MRI)
        self.ct_extractor = FeatureExtractorCNN(ct_input_size, feature_dim)
        self.mri_extractor = FeatureExtractorCNN(mri_input_size, feature_dim)
        
        # Module de fusion
        if fusion_method == 'concat':
            fusion_input_dim = feature_dim * 2
        elif fusion_method in ['add', 'multiply']:
            fusion_input_dim = feature_dim
        else:
            raise ValueError(f"Méthode de fusion inconnue: {fusion_method}")
        
        # Couches de classification après fusion
        self.fusion_fc1 = nn.Linear(fusion_input_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        
        self.fusion_fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        
        # Couche de sortie
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, ct_images, mri_images):
        """
        Passe avant du modèle multimodal.
        
        Args:
            ct_images: Tenseur d'images CT (batch_size, 1, height, width)
            mri_images: Tenseur d'images MRI (batch_size, 1, height, width)
            
        Returns:
            Logits de shape (batch_size, num_classes)
        """
        # Extrait les features de chaque modalité
        ct_features = self.ct_extractor(ct_images)  # (batch_size, feature_dim)
        mri_features = self.mri_extractor(mri_images)  # (batch_size, feature_dim)
        
        # Fusion des features
        if self.fusion_method == 'concat':
            fused_features = torch.cat([ct_features, mri_features], dim=1)
        elif self.fusion_method == 'add':
            fused_features = ct_features + mri_features
        elif self.fusion_method == 'multiply':
            fused_features = ct_features * mri_features
        else:
            raise ValueError(f"Méthode de fusion inconnue: {self.fusion_method}")
        
        # Classification
        x = self.relu1(self.fusion_fc1(fused_features))
        x = self.dropout1(x)
        x = self.relu2(self.fusion_fc2(x))
        x = self.dropout2(x)
        x = self.classifier(x)
        
        return x


def train_epoch(model, dataloader, criterion, optimizer, device, epoch=None, total_epochs=None):
    """Entraîne le modèle sur une époque."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    desc = f"Train Epoch {epoch}/{total_epochs}" if epoch else "Training"
    pbar = tqdm(dataloader, desc=desc, unit="batch", leave=False)
    
    for ct_images, mri_images, labels in pbar:
        ct_images = ct_images.to(device)
        mri_images = mri_images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(ct_images, mri_images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        current_loss = running_loss / (total / labels.size(0))
        current_acc = 100 * correct / total
        pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch=None, total_epochs=None):
    """Valide le modèle sur un dataset de validation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    desc = f"Val Epoch {epoch}/{total_epochs}" if epoch else "Validating"
    pbar = tqdm(dataloader, desc=desc, unit="batch", leave=False)
    
    with torch.no_grad():
        for ct_images, mri_images, labels in pbar:
            ct_images = ct_images.to(device)
            mri_images = mri_images.to(device)
            labels = labels.to(device)
            
            outputs = model(ct_images, mri_images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            current_loss = running_loss / (total / labels.size(0))
            current_acc = 100 * correct / total
            pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def test_model(model, dataloader, criterion, device):
    """Teste le modèle sur le dataset de test."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Testing", unit="batch")
    
    with torch.no_grad():
        for ct_images, mri_images, labels in pbar:
            ct_images = ct_images.to(device)
            mri_images = mri_images.to(device)
            labels = labels.to(device)
            
            outputs = model(ct_images, mri_images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            current_loss = running_loss / (total / labels.size(0))
            current_acc = 100 * correct / total
            pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_labels


def main():
    """Fonction principale pour l'entraînement du modèle multimodal."""
    print("="*70)
    print("ENTRAÎNEMENT DU MODÈLE MULTIMODAL CT/MRI POUR CLASSIFICATION")
    print("="*70)
    
    # Configuration
    BATCH_SIZE = 16  # Plus petit car on charge 2 images par échantillon
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.001
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    FEATURE_DIM = 512
    FUSION_METHOD = 'concat'  # 'concat', 'add', ou 'multiply'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice utilisé: {device}")
    
    # ==========================================
    # 1. CHARGEMENT ET DIVISION DU DATASET
    # ==========================================
    print("\n" + "="*70)
    print("ÉTAPE 1: Chargement et division du dataset multimodal")
    print("="*70)
    
    ct_healthy_dir = "Brain Tumor CT scan Images/Healthy"
    ct_tumor_dir = "Brain Tumor CT scan Images/Tumor"
    mri_healthy_dir = "Brain Tumor MRI images/Healthy"
    mri_tumor_dir = "Brain Tumor MRI images/Tumor"
    
    # Crée le dataset complet
    print("\nChargement du dataset multimodal complet...")
    full_dataset = MultimodalBrainTumorDataset(
        ct_healthy_dir=ct_healthy_dir,
        ct_tumor_dir=ct_tumor_dir,
        mri_healthy_dir=mri_healthy_dir,
        mri_tumor_dir=mri_tumor_dir,
        target_size=(224, 224),
        normalize_pixels=True,
        auto_analyze=False
    )
    
    print(f"\nDataset complet chargé: {len(full_dataset)} paires d'images")
    
    # Divise le dataset
    indices = list(range(len(full_dataset)))
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=42,
        stratify=[full_dataset.labels[i] for i in indices]
    )
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),
        random_state=42,
        stratify=[full_dataset.labels[i] for i in temp_indices]
    )
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"\nDivision du dataset:")
    print(f"  - Entraînement: {len(train_dataset)} paires ({len(train_dataset)/len(full_dataset)*100:.1f}%)")
    print(f"  - Validation: {len(val_dataset)} paires ({len(val_dataset)/len(full_dataset)*100:.1f}%)")
    print(f"  - Test: {len(test_dataset)} paires ({len(test_dataset)/len(full_dataset)*100:.1f}%)")
    
    # Crée les DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ==========================================
    # 2. INITIALISATION DU MODÈLE
    # ==========================================
    print("\n" + "="*70)
    print("ÉTAPE 2: Initialisation du modèle multimodal")
    print("="*70)
    
    # Récupère la taille des images
    sample_ct, sample_mri, _ = full_dataset[0]
    ct_size = (sample_ct.shape[1], sample_ct.shape[2])
    mri_size = (sample_mri.shape[1], sample_mri.shape[2])
    
    print(f"\nTaille des images CT: {ct_size[0]}x{ct_size[1]}")
    print(f"Taille des images MRI: {mri_size[0]}x{mri_size[1]}")
    
    model = MultimodalFusionModel(
        ct_input_size=ct_size,
        mri_input_size=mri_size,
        feature_dim=FEATURE_DIM,
        num_classes=2,
        fusion_method=FUSION_METHOD
    ).to(device)
    
    # Affiche le nombre de paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModèle multimodal créé:")
    print(f"  - Paramètres totaux: {total_params:,}")
    print(f"  - Paramètres entraînables: {trainable_params:,}")
    print(f"  - Méthode de fusion: {FUSION_METHOD}")
    
    # Fonction de perte et optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nConfiguration:")
    print(f"  - Fonction de perte: CrossEntropyLoss")
    print(f"  - Optimiseur: Adam (lr={LEARNING_RATE})")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Nombre d'époques: {NUM_EPOCHS}")
    
    # ==========================================
    # 3. ENTRAÎNEMENT
    # ==========================================
    print("\n" + "="*70)
    print("ÉTAPE 3: Entraînement du modèle multimodal")
    print("="*70)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Époque {epoch + 1}/{NUM_EPOCHS} ---")
        
        # Entraînement
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch + 1, total_epochs=NUM_EPOCHS
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        val_loss, val_acc = validate(
            model, val_loader, criterion, device,
            epoch=epoch + 1, total_epochs=NUM_EPOCHS
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ✓ Nouveau meilleur modèle (val_acc: {best_val_acc:.2f}%)")
    
    # ==========================================
    # 4. TEST FINAL
    # ==========================================
    print("\n" + "="*70)
    print("ÉTAPE 4: Test final sur le dataset de test")
    print("="*70)
    
    test_loss, test_acc, test_predictions, test_labels = test_model(
        model, test_loader, criterion, device
    )
    
    # ==========================================
    # 5. AFFICHAGE DES RÉSULTATS
    # ==========================================
    print("\n" + "="*70)
    print("RÉSULTATS FINAUX")
    print("="*70)
    
    print(f"\n--- Résultats d'entraînement ---")
    print(f"Loss finale (train): {train_losses[-1]:.4f}")
    print(f"Accuracy finale (train): {train_accs[-1]:.2f}%")
    
    print(f"\n--- Résultats de validation ---")
    print(f"Loss finale (val): {val_losses[-1]:.4f}")
    print(f"Accuracy finale (val): {val_accs[-1]:.2f}%")
    print(f"Meilleure accuracy (val): {best_val_acc:.2f}%")
    
    print(f"\n--- Résultats de test ---")
    print(f"Loss (test): {test_loss:.4f}")
    print(f"Accuracy (test): {test_acc:.2f}%")
    
    from collections import Counter
    correct_predictions = sum(1 for p, l in zip(test_predictions, test_labels) if p == l)
    total_predictions = len(test_predictions)
    
    print(f"\n--- Détails du test ---")
    print(f"Prédictions correctes: {correct_predictions}/{total_predictions}")
    print(f"Prédictions incorrectes: {total_predictions - correct_predictions}/{total_predictions}")
    
    pred_dist = Counter(test_predictions)
    label_dist = Counter(test_labels)
    print(f"\nDistribution des labels réels:")
    print(f"  - Healthy (0): {label_dist.get(0, 0)}")
    print(f"  - Tumor (1): {label_dist.get(1, 0)}")
    print(f"\nDistribution des prédictions:")
    print(f"  - Healthy (0): {pred_dist.get(0, 0)}")
    print(f"  - Tumor (1): {pred_dist.get(1, 0)}")
    
    print(f"\n--- Évolution de l'accuracy ---")
    print("Époque | Train Acc | Val Acc")
    print("-" * 30)
    for i in range(NUM_EPOCHS):
        print(f"  {i+1}    |   {train_accs[i]:5.2f}%  | {val_accs[i]:5.2f}%")
    
    print("\n" + "="*70)
    print("ENTRAÎNEMENT TERMINÉ")
    print("="*70)


if __name__ == "__main__":
    main()
