import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
import multiprocessing as mp
import hashlib
from tqdm import tqdm
from sklearn.metrics import classification_report
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter  # <-- Added TensorBoard Import
from pathlib import Path

# --- CONFIGURATION ---
DATASET_PATH = r'C:\Users\Jense\Documents\CODING-C\ViolenceDetection\ViolenceDetection\TRAINING\training_datasets'  # UPDATE THIS PATH
SEQUENCE_LENGTH = 20                 # Number of frames to analyze per clip
K_PEOPLE = 2                         # The K-Rule: Track Top-2 people
NUM_KEYPOINTS = 17                   # YOLOv8 standard
FEATURES_PER_PERSON = NUM_KEYPOINTS * 3  # (x, y, conf)
INPUT_SIZE = K_PEOPLE * FEATURES_PER_PERSON # 2 * 17 * 3 = 102 features per frame
BATCH_SIZE = 32
NUM_EPOCHS = 100

_TRAINING_DIR = Path(__file__).resolve().parent
_CACHES_DIR = _TRAINING_DIR / 'caches'
CACHE_DIR = _CACHES_DIR / f'pose_cache_lstm_seq{SEQUENCE_LENGTH}_k{K_PEOPLE}'

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = _PROJECT_ROOT / 'MODELS'
BEST_MODEL_PATH = _MODELS_DIR / 'best_violence_lstm.pt'

# Set multiprocessing start method
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

class PoseLSTM(nn.Module):
    """
    The Temporal Brain: Analyzes motion over time using LSTM.
    Input Shape: (Batch, Sequence_Length, Features) -> (Batch, 30, 102)
    """
    def __init__(self, input_size=INPUT_SIZE, hidden_size=256, num_layers=2, num_classes=1):
        super(PoseLSTM, self).__init__()
        
        # 1. Feature Extractor (Linear projection of raw coordinates)
        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 2. LSTM Layers (The "Time" component)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # 3. Attention Mechanism (Focus on the most violent frame in the sequence)
        self.attention = nn.Linear(hidden_size, 1)
        
        # 4. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        # Embed features
        x = self.embedding(x)
        
        # Run LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # Attention: Calculate weight for each frame
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        
        # Context vector: Weighted sum of all frames
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Classify
        out = self.classifier(context)
        return out

def extract_sequence_features(video_path, model):
    """
    Extracts Top-K skeletons from a video file into a (30, 102) tensor.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return torch.zeros(SEQUENCE_LENGTH, INPUT_SIZE)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Adaptive stride: If video is long, skip frames to cover the whole action
    stride = max(1, total_frames // SEQUENCE_LENGTH)
    
    frames_data = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % stride == 0 and len(frames_data) < SEQUENCE_LENGTH:
            # Run YOLO Pose
            results = model(frame, verbose=False, stream=False)
            
            frame_features = []
            if results[0].keypoints is not None:
                # Get data: (N, 17, 2) and (N,)
                kps = results[0].keypoints.xyn.cpu().numpy()  # Normalized coordinates
                confs = results[0].boxes.conf.cpu().numpy()   # Confidence scores
                
                # --- THE K-RULE: SORT BY CONFIDENCE ---
                # This naturally handles crowds by picking the 2 clearest people
                sorted_idx = np.argsort(confs)[::-1]
                
                for k in range(K_PEOPLE):
                    if k < len(sorted_idx):
                        idx = sorted_idx[k]
                        # Combine (x, y) with confidence score -> (17, 3)
                        kp_conf = results[0].keypoints.conf[idx].cpu().numpy().reshape(-1, 1)
                        person_data = np.hstack([kps[idx], kp_conf]).flatten()
                    else:
                        # Zero padding if fewer than K people
                        person_data = np.zeros(FEATURES_PER_PERSON)
                    frame_features.extend(person_data)
            else:
                # No one detected -> Zero frame
                frame_features = [0] * INPUT_SIZE
                
            frames_data.append(frame_features)
            
        frame_idx += 1
        if len(frames_data) >= SEQUENCE_LENGTH:
            break
            
    cap.release()
    
    # Pad if video was too short
    while len(frames_data) < SEQUENCE_LENGTH:
        frames_data.append([0] * INPUT_SIZE)
        
    return torch.tensor(frames_data, dtype=torch.float32)

def precompute_dataset(dataset_path, cache_dir=CACHE_DIR):
    """
    Scans folders, runs YOLO, and caches the LSTM-ready tensors.
    """
    os.makedirs(str(cache_dir), exist_ok=True)
    yolo_model = YOLO('yolo11m-pose.pt') # Use 'n' for speed, 'm' for accuracy
    
    data_map = [] # Stores (cache_path, label)
    
    print(f"Scanning {dataset_path}...")
    categories = {'non_violence': 0, 'violence': 1}
    
    for category, label in categories.items():
        folder = os.path.join(dataset_path, category)
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found.")
            continue
            
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        print(f"Processing {len(files)} videos in {category}...")
        
        for vid_file in tqdm(files):
            vid_path = os.path.join(folder, vid_file)
            
            # Unique ID for caching
            file_hash = hashlib.md5(vid_path.encode()).hexdigest()
            cache_path = str(Path(cache_dir) / f"{file_hash}.pt")
            
            if not os.path.exists(cache_path):
                # Extract and save if not cached
                features = extract_sequence_features(vid_path, yolo_model)
                torch.save(features, cache_path)
            
            data_map.append({'path': cache_path, 'label': label})
            
    return data_map

class ViolenceSequenceDataset(Dataset):
    def __init__(self, data_map, augment=False):
        self.data_map = data_map
        self.augment = augment

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        cache_path = self.data_map[idx]['path']
        label = self.data_map[idx]['label']
        
        # Load tensor: (30, 102)
        features = torch.load(cache_path)
        
        # --- DATA AUGMENTATION (Coordinate Flip) ---
        if self.augment and np.random.random() > 0.5:
            # Flip X coordinates (indices 0, 3, 6... are x)
            # x_new = 1.0 - x_old
            features[:, 0::3] = 1.0 - features[:, 0::3]
            
        return features, torch.tensor(label, dtype=torch.float32)

class FocalLoss(nn.Module):
    """Handles class imbalance nicely."""
    def __init__(self, alpha=0.25, gamma=1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # --- Initialize TensorBoard ---
    writer = SummaryWriter(log_dir='runs/violence_lstm_experiment')
    print("TensorBoard initialized. Run 'tensorboard --logdir=runs' in another terminal.")

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Data
    data_map = precompute_dataset(DATASET_PATH)
    if not data_map:
        print("No data found! Check paths.")
        return

    # Split
    labels = [d['label'] for d in data_map]
    train_data, val_data = train_test_split(data_map, test_size=0.2, stratify=labels, random_state=42)
    
    # 2. Datasets & Loaders
    train_ds = ViolenceSequenceDataset(train_data, augment=True)
    val_ds = ViolenceSequenceDataset(val_data, augment=False)
    
    # Weighted Sampler for balance
    train_labels = [d['label'] for d in train_data]
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Model Setup
    model = PoseLSTM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = FocalLoss()
    scaler = GradScaler('cuda') # Mixed precision
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        threshold=0.01,
        threshold_mode='rel',
        min_lr=1e-6,
    )

    # 4. Training Loop
    print("\nStarting Training...")
    best_acc = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            with autocast('cuda'):
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        val_preds_list = []
        val_labels_list = []
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device).unsqueeze(1)
                preds = model(X_val)
                loss = criterion(preds, y_val)
                predicted = (torch.sigmoid(preds) > 0.5).float()
                
                correct += (predicted == y_val).sum().item()
                total += y_val.size(0)
                val_loss += float(loss.item())
                
                val_preds_list.extend(predicted.cpu().numpy())
                val_labels_list.extend(y_val.cpu().numpy())
                
        acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        lr = float(optimizer.param_groups[0]['lr'])
        print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}% | LR: {lr:.6g}")
        
        # --- Log to TensorBoard ---
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', acc, epoch)
        writer.add_scalar('LR', lr, epoch)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), str(BEST_MODEL_PATH))
            print("--> Model Saved!")

    print("\nFinal Classification Report:")
    print(classification_report(val_labels_list, val_preds_list, target_names=['Non-Violence', 'Violence']))

    # --- Close TensorBoard ---
    writer.close()

if __name__ == "__main__":
    main()