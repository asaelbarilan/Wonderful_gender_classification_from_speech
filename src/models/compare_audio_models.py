"""
Compare different audio classification models:
1. Simple CNN (from audio_cnn.py)
2. Transfer Learning with ResNet18
3. Transfer Learning with EfficientNet-B0
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import our models
from src.models.audio_cnn import AudioCNN, MelSpectrogramDataset as SimpleMelDataset, train_model as train_simple, evaluate_model as eval_simple
from src.models.audio_transfer_learning import AudioTransferModel, MelSpectrogramDataset as TransferMelDataset, train_model as train_transfer, evaluate_model as eval_transfer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_simple_cnn(df, audio_root, output_dir):
    """Run simple CNN model"""
    logger.info("Running Simple CNN...")
    
    # Create train/test splits
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    # Create datasets
    train_ds = SimpleMelDataset(train_df, 'file_path', 'label', root_dir=audio_root)
    test_ds = SimpleMelDataset(test_df, 'file_path', 'label', root_dir=audio_root)
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    
    # Model
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioCNN(n_mels=64, n_classes=2).to(device)
    
    # Train
    train_losses, val_losses = train_simple(model, train_loader, test_loader, device, epochs=20, lr=1e-3)
    
    # Load best model
    model.load_state_dict(torch.load('best_audio_cnn.pth'))
    
    # Evaluate
    acc, auc, cm, report = eval_simple(model, test_loader, device)
    
    return {
        'model_name': 'Simple CNN',
        'test_accuracy': acc,
        'test_auc': auc,
        'confusion_matrix': cm,
        'classification_report': report,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

def run_transfer_learning(df, audio_root, output_dir, model_name='resnet18'):
    """Run transfer learning model"""
    logger.info(f"Running Transfer Learning with {model_name}...")
    
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create train/val/test splits
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    # Create datasets
    train_ds = TransferMelDataset(train_df, 'file_path', 'label', 
                                 n_mels=224, duration=3.0,
                                 root_dir=audio_root, transform=transform)
    val_ds = TransferMelDataset(val_df, 'file_path', 'label',
                               n_mels=224, duration=3.0,
                               root_dir=audio_root, transform=transform)
    test_ds = TransferMelDataset(test_df, 'file_path', 'label',
                                n_mels=224, duration=3.0,
                                root_dir=audio_root, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioTransferModel(
        model_name=model_name,
        pretrained=True,
        num_classes=2,
        freeze_backbone=False
    ).to(device)
    
    # Train
    train_losses, val_losses, train_accs, val_accs = train_transfer(
        model, train_loader, val_loader, device, epochs=30, lr=1e-4
    )
    
    # Load best model
    checkpoint = torch.load(f'best_{model_name}_audio.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    acc, auc, cm, report, probs = eval_transfer(model, test_loader, device)
    
    return {
        'model_name': f'Transfer Learning ({model_name})',
        'test_accuracy': acc,
        'test_auc': auc,
        'best_val_accuracy': checkpoint['best_acc'],
        'confusion_matrix': cm,
        'classification_report': report,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

def plot_comparison(results, output_dir):
    """Plot comparison of all models"""
    
    # Accuracy comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Bar plot of accuracies
    model_names = [r['model_name'] for r in results]
    accuracies = [r['test_accuracy'] for r in results]
    aucs = [r['test_auc'] for r in results]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
    ax1.bar(x + width/2, aucs, width, label='ROC AUC', color='lightcoral')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (acc, auc) in enumerate(zip(accuracies, aucs)):
        ax1.text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        ax1.text(i + width/2, auc + 0.01, f'{auc:.3f}', ha='center', va='bottom')
    
    # Confusion matrices
    for i, result in enumerate(results):
        if i < 3:  # Only plot first 3 models
            row = i // 2
            col = i % 2
            ax = ax3 if row == 1 else ax2
            
            if col == 0:
                ax = ax2 if row == 0 else ax3
            else:
                ax = ax4 if row == 1 else ax3
            
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'],
                       ax=ax)
            ax.set_title(f'{result["model_name"]}\nAccuracy: {result["test_accuracy"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        if i < 4:  # Only plot first 4 models
            ax = axes[i]
            train_losses = result['train_losses']
            val_losses = result['val_losses']
            
            ax.plot(train_losses, label='Train Loss', color='blue')
            ax.plot(val_losses, label='Val Loss', color='red')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'{result["model_name"]} - Training Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Configuration
    config = {
        'features_csv': r"C:\Users\Asael\PycharmProjects\wonderful_mission\data\processed\timit_features.csv",
        'audio_root': r"C:\Users\Asael\PycharmProjects\wonderful_mission\data\raw\full_timit\data",
        'output_dir': r"C:\Users\Asael\PycharmProjects\wonderful_mission\data\processed\model_comparison_results"
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(config['features_csv'])
    logger.info(f"Loaded {len(df)} samples")
    
    results = []
    
    # Run Simple CNN
    try:
        simple_result = run_simple_cnn(df, config['audio_root'], config['output_dir'])
        results.append(simple_result)
        logger.info(f"Simple CNN - Accuracy: {simple_result['test_accuracy']:.4f}, AUC: {simple_result['test_auc']:.4f}")
    except Exception as e:
        logger.error(f"Error running Simple CNN: {e}")
    
    # Run Transfer Learning models
    transfer_models = ['resnet18', 'efficientnet_b0']
    
    for model_name in transfer_models:
        try:
            transfer_result = run_transfer_learning(df, config['audio_root'], config['output_dir'], model_name)
            results.append(transfer_result)
            logger.info(f"{transfer_result['model_name']} - Accuracy: {transfer_result['test_accuracy']:.4f}, AUC: {transfer_result['test_auc']:.4f}")
        except Exception as e:
            logger.error(f"Error running {model_name}: {e}")
    
    # Save results summary
    summary = []
    for result in results:
        summary.append({
            'model_name': result['model_name'],
            'test_accuracy': result['test_accuracy'],
            'test_auc': result['test_auc'],
            'best_val_accuracy': result.get('best_val_accuracy', 'N/A')
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(config['output_dir'], 'model_comparison_summary.csv'), index=False)
    
    logger.info("Model Comparison Summary:")
    logger.info(summary_df.to_string(index=False))
    
    # Plot comparison
    if len(results) > 1:
        plot_comparison(results, config['output_dir'])
        logger.info(f"Comparison plots saved to {config['output_dir']}")
    
    # Find best model
    if results:
        best_model = max(results, key=lambda x: x['test_accuracy'])
        logger.info(f"\nBest Model: {best_model['model_name']}")
        logger.info(f"Best Accuracy: {best_model['test_accuracy']:.4f}")
        logger.info(f"Best AUC: {best_model['test_auc']:.4f}")

if __name__ == "__main__":
    main() 