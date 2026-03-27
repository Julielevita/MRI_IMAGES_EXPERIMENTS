import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Importe le dataset depuis draft_2
from creating_dataset_for_simple_training import BrainTumorMRIDataset


class SimpleBrainTumorCNN(nn.Module):
    """
    CNN simple pour la classification binaire d'images de tumeurs cérébrales.
    Architecture: Conv -> Pool -> Conv -> Pool -> Conv -> Pool -> FC -> FC -> Output
    """
    
    def __init__(self, num_classes=2, input_size=(224, 224)):
        """
        Initialise le modèle CNN.
        
        Args:
            num_classes: Nombre de classes (2: Healthy, Tumor)
            input_size: Taille d'entrée des images (height, width)
        """
        super(SimpleBrainTumorCNN, self).__init__()
        
        # Première couche de convolution
        # Input: (batch, 1, height, width) - 1 canal car images en niveaux de gris
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
        
        # Calcule la taille après les convolutions et poolings
        # Après 3 poolings (chaque pooling divise par 2), on a height/8 x width/8
        # Avec 128 canaux
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size[0], input_size[1])
            dummy_output = self._forward_features(dummy_input)
            self.feature_size = dummy_output.view(1, -1).size(1)
        
        # Couches fully connected
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Couche de sortie
        self.fc2 = nn.Linear(512, num_classes)
    
    def _forward_features(self, x):
        """
        Passe avant jusqu'aux features (avant fully connected).
        Utilisé pour calculer la taille des features.
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        return x
    
    def forward(self, x):
        """
        Passe avant du modèle.
        
        Args:
            x: Tenseur d'images de shape (batch_size, 1, height, width)
            
        Returns:
            Logits de shape (batch_size, num_classes)
        """
        # Convolutions et poolings
        x = self._forward_features(x)
        
        # Aplatit pour les couches fully connected
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def train_epoch(model, dataloader, criterion, optimizer, device, epoch=None, total_epochs=None):
    """
    Entraîne le modèle sur une époque.
    
    Args:
        model: Modèle CNN
        dataloader: DataLoader pour les données d'entraînement
        criterion: Fonction de perte
        optimizer: Optimiseur
        device: Device (CPU ou GPU)
        epoch: Numéro de l'époque (pour l'affichage)
        total_epochs: Nombre total d'époques (pour l'affichage)
        
    Returns:
        Tuple (loss moyenne, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Crée la barre de progression
    if epoch is not None and total_epochs is not None:
        desc = f"Train Epoch {epoch}/{total_epochs}"
    else:
        desc = "Training"
    
    pbar = tqdm(dataloader, desc=desc, unit="batch", leave=False)
    
    for images, labels in pbar:
        # Déplace les données sur le device
        images = images.to(device)
        labels = labels.to(device)
        
        # Remet à zéro les gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Met à jour la barre de progression avec les statistiques en temps réel
        current_loss = running_loss / (total / labels.size(0))
        current_acc = 100 * correct / total
        pbar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Acc': f'{current_acc:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch=None, total_epochs=None):
    """
    Valide le modèle sur un dataset de validation.
    
    Args:
        model: Modèle CNN
        dataloader: DataLoader pour les données de validation
        criterion: Fonction de perte
        device: Device (CPU ou GPU)
        epoch: Numéro de l'époque (pour l'affichage)
        total_epochs: Nombre total d'époques (pour l'affichage)
        
    Returns:
        Tuple (loss moyenne, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Crée la barre de progression
    if epoch is not None and total_epochs is not None:
        desc = f"Val Epoch {epoch}/{total_epochs}"
    else:
        desc = "Validating"
    
    pbar = tqdm(dataloader, desc=desc, unit="batch", leave=False)
    
    with torch.no_grad():
        for images, labels in pbar:
            # Déplace les données sur le device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistiques
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Met à jour la barre de progression avec les statistiques en temps réel
            current_loss = running_loss / (total / labels.size(0))
            current_acc = 100 * correct / total
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def test_model(model, dataloader, criterion, device):
    """
    Teste le modèle sur le dataset de test.
    
    Args:
        model: Modèle CNN
        dataloader: DataLoader pour les données de test
        criterion: Fonction de perte
        device: Device (CPU ou GPU)
        
    Returns:
        Tuple (loss moyenne, accuracy, prédictions, labels)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    # Crée la barre de progression
    pbar = tqdm(dataloader, desc="Testing", unit="batch")
    
    with torch.no_grad():
        for images, labels in pbar:
            # Déplace les données sur le device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistiques
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Stocke les prédictions et labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Met à jour la barre de progression avec les statistiques en temps réel
            current_loss = running_loss / (total / labels.size(0))
            current_acc = 100 * correct / total
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_labels


def main():
    """
    Fonction principale pour l'entraînement du modèle.
    """
    print("="*70)
    print("ENTRAÎNEMENT DU MODÈLE CNN POUR CLASSIFICATION DE TUMEURS CÉRÉBRALES")
    print("="*70)
    
    # Configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.001
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Device (GPU si disponible, sinon CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice utilisé: {device}")
    
    # ==========================================
    # 1. CHARGEMENT ET DIVISION DU DATASET
    # ==========================================
    print("\n" + "="*70)
    print("ÉTAPE 1: Chargement et division du dataset")
    print("="*70)
    
    healthy_dir = "Brain Tumor MRI images/Healthy"
    tumor_dir = "Brain Tumor MRI images/Tumor"
    
    # Crée le dataset complet
    print("\nChargement du dataset complet...")
    full_dataset = BrainTumorMRIDataset(
        healthy_dir=healthy_dir,
        tumor_dir=tumor_dir,
        target_size=(224, 224),
        normalize_pixels=True,
        auto_analyze=True
    )
    
    print(f"\nDataset complet chargé: {len(full_dataset)} images")
    
    # Divise le dataset en train/val/test
    # D'abord, sépare train et (val+test)
    indices = list(range(len(full_dataset)))
    train_indices, temp_indices = train_test_split(
        indices, 
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=42,
        stratify=[full_dataset.labels[i] for i in indices]  # Stratification pour équilibrer les classes
    )
    
    # Ensuite, sépare val et test
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),
        random_state=42,
        stratify=[full_dataset.labels[i] for i in temp_indices]
    )
    
    # Crée les sous-datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"\nDivision du dataset:")
    print(f"  - Entraînement: {len(train_dataset)} images ({len(train_dataset)/len(full_dataset)*100:.1f}%)")
    print(f"  - Validation: {len(val_dataset)} images ({len(val_dataset)/len(full_dataset)*100:.1f}%)")
    print(f"  - Test: {len(test_dataset)} images ({len(test_dataset)/len(full_dataset)*100:.1f}%)")
    
    # Crée les DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ==========================================
    # 2. INITIALISATION DU MODÈLE
    # ==========================================
    print("\n" + "="*70)
    print("ÉTAPE 2: Initialisation du modèle")
    print("="*70)
    
    # Récupère la taille des images depuis le dataset
    sample_image, _ = full_dataset[0]
    image_height, image_width = sample_image.shape[1], sample_image.shape[2]
    print(f"\nTaille des images détectée: {image_height}x{image_width}")
    
    model = SimpleBrainTumorCNN(num_classes=2, input_size=(image_height, image_width)).to(device)
    
    # Affiche le nombre de paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModèle créé:")
    print(f"  - Paramètres totaux: {total_params:,}")
    print(f"  - Paramètres entraînables: {trainable_params:,}")
    
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
    print("ÉTAPE 3: Entraînement du modèle")
    print("="*70)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Époque {epoch + 1}/{NUM_EPOCHS} ---")
        
        # Entraînement avec barre de progression
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            epoch=epoch + 1, total_epochs=NUM_EPOCHS
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation avec barre de progression
        val_loss, val_acc = validate(
            model, val_loader, criterion, device,
            epoch=epoch + 1, total_epochs=NUM_EPOCHS
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Affiche les résultats de l'époque
        print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        # Sauvegarde le meilleur modèle
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
    
    # Matrice de confusion simple
    from collections import Counter
    correct_predictions = sum(1 for p, l in zip(test_predictions, test_labels) if p == l)
    total_predictions = len(test_predictions)
    
    print(f"\n--- Détails du test ---")
    print(f"Prédictions correctes: {correct_predictions}/{total_predictions}")
    print(f"Prédictions incorrectes: {total_predictions - correct_predictions}/{total_predictions}")
    
    # Distribution des prédictions
    pred_dist = Counter(test_predictions)
    label_dist = Counter(test_labels)
    print(f"\nDistribution des labels réels:")
    print(f"  - Healthy (0): {label_dist.get(0, 0)}")
    print(f"  - Tumor (1): {label_dist.get(1, 0)}")
    print(f"\nDistribution des prédictions:")
    print(f"  - Healthy (0): {pred_dist.get(0, 0)}")
    print(f"  - Tumor (1): {pred_dist.get(1, 0)}")
    
    # Évolution de l'accuracy
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
