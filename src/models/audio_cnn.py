"""
Deep learning audio classification using Mel-spectrograms and PyTorch.
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
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm
import seaborn as sns

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

# --- Vocabulary and Phoneme Processing ---
# NOTE: This is a placeholder vocabulary. You should generate a full one
# by iterating through all .PHN files in your dataset.
PHONEME_VOCAB = [
    '<pad>', 'h#', 'd', 'ow', 'q', 'ux', 'tcl', 'ch', 'ae', 'l', 'iy', 'z',
    'dx', 'er', 'ih', 'sh', 's', 't', 'ax', 'kcl', 'k', 'r', 'n', 'p', 'b', 'g',
    'ah', 'm', 'f', 'v', 'dh', 'w', 'y', 'ng', 'jh', 'pcl', 'gcl', 'epi'
]
PHONE_TO_ID = {ph: i for i, ph in enumerate(PHONEME_VOCAB)}
ID_TO_PHONE = {i: ph for i, ph in enumerate(PHONEME_VOCAB)}

def phonemes_to_ids(phonemes, max_len=None):
    """Convert a list of phonemes to a tensor of integer IDs."""
    ids = [PHONE_TO_ID.get(p, PHONE_TO_ID['<pad>']) for p in phonemes]
    if max_len:
        ids = ids[:max_len]
        ids += [PHONE_TO_ID['<pad>']] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

# --- Dataset ---
class MelSpectrogramDataset(Dataset):
    def __init__(self, df, audio_col, label_col, sr=16000, n_mels=64, duration=2.0, root_dir=None, include_phonemes=False):
        self.df = df.reset_index(drop=True)
        self.audio_col = audio_col
        self.label_col = label_col
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.root_dir = root_dir
        self.samples = int(sr * duration)
        self.include_phonemes = include_phonemes

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
        # Normalize
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        # To tensor
        mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, time)
        label = int(row[self.label_col])
        
        if not self.include_phonemes:
            return mel_tensor, label
        else:
            # --- Phoneme Loading ---
            phn_path = audio_path.replace('.WAV', '.PHN')
            try:
                with open(phn_path, 'r') as f:
                    # We only care about the phoneme symbols, not the timings for now
                    phonemes = [line.strip().split()[-1] for line in f]
                phoneme_ids = phonemes_to_ids(phonemes)
            except FileNotFoundError:
                # If a phoneme file is missing, return an empty sequence
                phonemes = []
                phoneme_ids = torch.tensor([], dtype=torch.long)
            
            return mel_tensor, phoneme_ids, label

# --- Collate function for batching variable-length phoneme sequences ---
def collate_fn_multimodal(batch):
    """Pads phoneme sequences to the max length in a batch."""
    mel_tensors, phoneme_ids, labels = zip(*batch)
    
    # Pad phoneme tensors
    phoneme_ids_padded = nn.utils.rnn.pad_sequence(phoneme_ids, batch_first=True, padding_value=PHONE_TO_ID['<pad>'])
    
    # Stack mel tensors and labels
    mel_tensors_stacked = torch.stack(mel_tensors, 0)
    labels_stacked = torch.tensor(labels, dtype=torch.long)
    
    return mel_tensors_stacked, phoneme_ids_padded, labels_stacked

# --- Model ---
class AudioCNN(nn.Module):
    def __init__(self, n_mels=64, n_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # The input size is calculated as: 64_channels * (64_mels / 8) * (63_time_frames / 8)
            # which is 64 * 8 * 7 = 3584. The original value was based on an
            # incorrect time dimension.
            nn.Linear(3584, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x

# --- Multi-Modal Model ---
class MultiModalAudioCNN(nn.Module):
    def __init__(self, n_mels=64, n_classes=2, vocab_size=len(PHONEME_VOCAB), embedding_dim=64, lstm_hidden_dim=64):
        super().__init__()
        # 1. Acoustic Branch (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.acoustic_fc = nn.Linear(3584, 128) # Same as original CNN output

        # 2. Linguistic Branch (Phoneme LSTM)
        self.phoneme_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PHONE_TO_ID['<pad>'])
        self.phoneme_lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.linguistic_fc = nn.Linear(lstm_hidden_dim * 2, 128) # *2 for bidirectional

        # 3. Fusion and Classifier
        self.fusion_dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128 + 128, n_classes) # 128 from acoustic + 128 from linguistic

    def forward(self, mel_input, phoneme_input):
        # Process acoustic input
        acoustic_out = self.cnn(mel_input)
        acoustic_out = acoustic_out.view(acoustic_out.size(0), -1) # Flatten
        acoustic_features = nn.functional.relu(self.acoustic_fc(acoustic_out))

        # Process linguistic input
        phoneme_embedded = self.phoneme_embedding(phoneme_input)
        lstm_out, (h_n, c_n) = self.phoneme_lstm(phoneme_embedded)
        # We can use the final hidden state of the LSTM
        # Concatenate the final forward and backward hidden states
        linguistic_features = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        linguistic_features = nn.functional.relu(self.linguistic_fc(linguistic_features))

        # Fuse the features
        combined_features = torch.cat((acoustic_features, linguistic_features), dim=1)
        combined_features = self.fusion_dropout(combined_features)
        
        # Final classification
        output = self.classifier(combined_features)
        return output

# --- Training & Evaluation ---
def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-3, is_multimodal=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            if is_multimodal:
                mel, phonemes, labels = batch
                mel, phonemes, labels = mel.to(device), phonemes.to(device), labels.to(device)
                out = model(mel, phonemes)
                loss = criterion(out, labels)
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item() * mel.size(0) if is_multimodal else loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                if is_multimodal:
                    mel, phonemes, labels = batch
                    mel, phonemes, labels = mel.to(device), phonemes.to(device), labels.to(device)
                    out = model(mel, phonemes)
                    all_labels.extend(labels.cpu().numpy())
                    val_loss += criterion(out, labels).item() * mel.size(0)
                else:
                    xb, yb = batch
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    all_labels.extend(yb.cpu().numpy())
                    val_loss += criterion(out, yb).item() * xb.size(0)

                preds = out.argmax(1)
                correct += (preds == (labels if is_multimodal else yb)).sum().item()
                all_preds.extend(preds.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_acc = correct / len(val_loader.dataset)
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_audio_cnn.pth')
    return train_losses, val_losses

def evaluate_model(model, test_loader, device, is_multimodal=False):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if is_multimodal:
                mel, phonemes, labels = batch
                mel, phonemes, labels = mel.to(device), phonemes.to(device), labels.to(device)
                outputs = model(mel, phonemes)
                all_labels.extend(labels.cpu().numpy())
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                all_labels.extend(labels.cpu().numpy())

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    return acc, f1, precision, recall, cm

# --- Main ---
if __name__ == "__main__":
    # Paths (update as needed)
    features_csv = r"C:\Users\Asael\PycharmProjects\wonderful_mission\data\processed\timit_features.csv"
    audio_root = r"C:\Users\Asael\PycharmProjects\wonderful_mission\data\raw\full_timit\data"
    # Standardize output directory for the report
    output_dir = r"results/deep_learning"
    os.makedirs(output_dir, exist_ok=True)

    # --- CHOOSE WHICH MODEL TO RUN ---
    USE_MULTIMODAL = True # <-- SET THIS TO True TO RUN THE NEW MODEL

    # Load features and splits
    df = pd.read_csv(features_csv)

    # Assume 'split' column exists (train/test) or use your split logic
    if 'split' in df.columns:
        train_df = df[df['split'] == 'TRAIN']
        test_df = df[df['split'] == 'TEST']
    else:
        # Fallback: 80/20 split
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    # Dataset/Loader
    if USE_MULTIMODAL:
        logger.info("Using Multi-Modal model with phonemes.")
        train_ds = MelSpectrogramDataset(train_df, audio_col='file_path', label_col='label', root_dir=audio_root, include_phonemes=True)
        test_ds = MelSpectrogramDataset(test_df, audio_col='file_path', label_col='label', root_dir=audio_root, include_phonemes=True)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, collate_fn=collate_fn_multimodal)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate_fn_multimodal)
    else:
        logger.info("Using standard Audio-only CNN model.")
        train_ds = MelSpectrogramDataset(train_df, audio_col='file_path', label_col='label', root_dir=audio_root)
        test_ds = MelSpectrogramDataset(test_df, audio_col='file_path', label_col='label', root_dir=audio_root)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)


    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if USE_MULTIMODAL:
        model = MultiModalAudioCNN().to(device)
    else:
        model = AudioCNN(n_mels=64, n_classes=2).to(device)

    # Train
    train_losses, val_losses = train_model(model, train_loader, test_loader, device, epochs=20, lr=1e-3, is_multimodal=USE_MULTIMODAL)

    # Load best model
    model.load_state_dict(torch.load('best_audio_cnn.pth'))

    # Evaluate
    acc, f1, precision, recall, cm = evaluate_model(model, test_loader, device, is_multimodal=USE_MULTIMODAL)
    logger.info(f"Test Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Plot loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss (CNN)')
    plt.legend()
    # Save with a unique name
    plt.savefig(os.path.join(output_dir, 'cnn_loss_curve.png'))
    plt.close()

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - CNN (Test Set)')
    # Save with a unique name
    plt.savefig(os.path.join(output_dir, 'cnn_confusion_matrix.png'))
    plt.close() 