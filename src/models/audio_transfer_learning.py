"""
Transfer Learning for Audio Classification using Pretrained Image Models.
Uses Mel-spectrograms as input to pretrained CNN models like ResNet18.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- Dataset with Image-like Processing ---
class MelSpectrogramDataset(Dataset):
    def __init__(self, df, audio_col, label_col, sr=16000, n_mels=224, duration=3.0, 
                 root_dir=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.audio_col = audio_col
        self.label_col = label_col
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.root_dir = root_dir
        self.transform = transform
        self.samples = int(sr * duration)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row[self.audio_col]
        if self.root_dir and not os.path.isabs(audio_path):
            audio_path = os.path.join(self.root_dir, audio_path)
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
        
        # Pad or trim
        if len(y) < self.samples:
            y = np.pad(y, (0, self.samples - len(y)))
        else:
            y = y[:self.samples]
        
        # Mel-spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Resize to square image (224x224 for ResNet)
        mel_db = librosa.util.fix_length(mel_db, size=224, axis=1)
        
        # Convert to PIL Image for transforms
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min()) * 255
        mel_db = mel_db.astype(np.uint8)
        
        # Convert to 3-channel image (repeat single channel)
        mel_image = np.stack([mel_db, mel_db, mel_db], axis=2)
        
        # Apply transforms
        if self.transform:
            mel_image = self.transform(mel_image)
        
        label = int(row[self.label_col])
        return mel_image, label

# --- Transfer Learning Models ---
class AudioTransferModel(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True, num_classes=2, freeze_backbone=False):
        super().__init__()
        self.model_name = model_name
        
        # Load pretrained model
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        elif model_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace classifier
        if model_name.startswith('resnet'):
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        elif model_name == 'efficientnet_b0':
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        elif model_name == 'mobilenet_v2':
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)

# --- Training Functions ---
def train_model(model, train_loader, val_loader, device, epochs=30, lr=1e-4, 
                scheduler_step_size=10, scheduler_gamma=0.1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    
    best_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * xb.size(0)
            preds = out.argmax(1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                
                preds = out.argmax(1)
                correct += (preds == yb).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                   f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, f'best_{model.model_name}_audio.pth')
    
    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Evaluating"):
            xb = xb.to(device)
            out = model(xb)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
            all_probs.extend(probs)
    
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['Male', 'Female'])
    
    return acc, auc, cm, report, all_probs

# --- Visualization Functions ---
def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training/Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training/Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, output_dir, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def get_model(model_name, pretrained=True, freeze_backbone=False):
    model = AudioTransferModel(model_name, pretrained, 2, freeze_backbone)
    return model

def evaluate_model(model, dataloader, device):
    """Evaluates the model on a given dataset and returns metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, f1, precision, recall, cm

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, model_save_path):
    # ... existing code ...
    return model, history

def plot_training_history(history, save_dir):
    """Plots and saves the training and validation loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    # Use the standardized path for the report
    save_path = os.path.join(save_dir, "audio_transfer_learning_loss.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Training history plot saved to {save_path}")

# --- Main Execution ---
def main():
    # Configuration
    config = {
        'features_csv': r"C:\Users\Asael\PycharmProjects\wonderful_mission\data\processed\timit_features.csv",
        'audio_root': r"C:\Users\Asael\PycharmProjects\wonderful_mission\data\raw\full_timit\data",
        'output_dir': r"results/deep_learning",
        'model_name': 'resnet18',  # Options: resnet18, resnet34, resnet50, efficientnet_b0, mobilenet_v2
        'batch_size': 16,
        'epochs': 30,
        'lr': 1e-4,
        'freeze_backbone': False,  # Set to True for feature extraction only
        'n_mels': 224,
        'duration': 3.0
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(config['features_csv'])
    
    # Create train/val/test splits
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_ds = MelSpectrogramDataset(train_df, 'file_path', 'label', 
                                    n_mels=config['n_mels'], duration=config['duration'],
                                    root_dir=config['audio_root'], transform=transform)
    val_ds = MelSpectrogramDataset(val_df, 'file_path', 'label',
                                  n_mels=config['n_mels'], duration=config['duration'],
                                  root_dir=config['audio_root'], transform=transform)
    test_ds = MelSpectrogramDataset(test_df, 'file_path', 'label',
                                   n_mels=config['n_mels'], duration=config['duration'],
                                   root_dir=config['audio_root'], transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = get_model(config['model_name'], True, config['freeze_backbone']).to(device)
    
    logger.info(f"Model: {config['model_name']}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training
    logger.info("Starting training...")
    model_save_path = os.path.join(config['output_dir'], f"{config['model_name']}_best.pth")
    model, history = train_model(model, train_loader, val_loader, device, 
                                 epochs=config['epochs'], lr=config['lr'])
    
    # Plot training history
    plot_training_history(history, config['output_dir'])

    # --- Final Evaluation ---
    logger.info("Performing final evaluation on the test set...")
    # Load the best performing model
    model.load_state_dict(torch.load(model_save_path)['model_state_dict'])

    test_acc, test_f1, test_precision, test_recall, test_cm = evaluate_model(model, test_loader, device)

    logger.info(f"Test Set Metrics for {config['model_name']}:")
    logger.info(f"  Accuracy:  {test_acc:.4f}")
    logger.info(f"  F1-Score:  {test_f1:.4f}")
    logger.info(f"  Precision: {test_precision:.4f}")
    logger.info(f"  Recall:    {test_recall:.4f}")

    # Plot and save the confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'])
    plt.title(f'Confusion Matrix - {config["model_name"]} (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_path = os.path.join(config['output_dir'], f'{config["model_name"]}_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")

    # Save results
    results = {
        'model_name': config['model_name'],
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'best_val_accuracy': torch.load(model_save_path)['best_acc'],
        'epochs': config['epochs'],
        'learning_rate': config['lr'],
        'freeze_backbone': config['freeze_backbone']
    }
    
    pd.DataFrame([results]).to_csv(
        os.path.join(config['output_dir'], f'results_{config["model_name"]}.csv'), index=False
    )
    
    logger.info(f"Results saved to {config['output_dir']}")

if __name__ == "__main__":
    main() 