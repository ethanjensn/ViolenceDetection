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
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# --- CONFIGURATION ---
DATASET_PATH = r'C:\Users\Jense\Documents\CODING-C\ViolenceDetection\ViolenceDetection\TRAINING\training_datasets_V2'

# FROM OLD SCRIPT: simpler features = less overfitting on ~2000 videos
SEQUENCE_LENGTH = 20          # 20 outperformed 30 for this dataset size
K_PEOPLE = 2                  # top-2 only — fights are 2-person events
NUM_KEYPOINTS = 17
FEATURES_PER_PERSON = NUM_KEYPOINTS * 3   # (x, y, conf) — no velocity/accel
INPUT_SIZE = K_PEOPLE * FEATURES_PER_PERSON  # 102

BATCH_SIZE = 32
NUM_EPOCHS = 150
EARLY_STOP_PATIENCE = 30  # was 20 — model was still improving when it stopped

_TRAINING_DIR = Path(__file__).resolve().parent
_CACHES_DIR = _TRAINING_DIR / 'caches'
CACHE_DIR = _CACHES_DIR / f'pose_cache_lstm_seq{SEQUENCE_LENGTH}_k{K_PEOPLE}_v2'

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = _PROJECT_ROOT / 'MODELS'
BEST_MODEL_PATH = _MODELS_DIR / 'best_violence_lstm.pt'

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


class PoseLSTM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=256, num_layers=2, num_classes=1):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.attention = nn.Linear(hidden_size, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.classifier(context)


def extract_sequence_features(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return torch.zeros(SEQUENCE_LENGTH, INPUT_SIZE)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, total_frames // SEQUENCE_LENGTH)

    frames_data = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0 and len(frames_data) < SEQUENCE_LENGTH:
            results = model(frame, verbose=False, stream=False)

            frame_features = []
            if results[0].keypoints is not None:
                kps = results[0].keypoints.xyn.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                sorted_idx = np.argsort(confs)[::-1]

                for k in range(K_PEOPLE):
                    if k < len(sorted_idx):
                        idx = sorted_idx[k]
                        kp_conf = results[0].keypoints.conf[idx].cpu().numpy().reshape(-1, 1)
                        person_data = np.hstack([kps[idx], kp_conf]).flatten()
                    else:
                        person_data = np.zeros(FEATURES_PER_PERSON)
                    frame_features.extend(person_data)
            else:
                frame_features = [0] * INPUT_SIZE

            frames_data.append(frame_features)

        frame_idx += 1
        if len(frames_data) >= SEQUENCE_LENGTH:
            break

    cap.release()

    while len(frames_data) < SEQUENCE_LENGTH:
        frames_data.append([0] * INPUT_SIZE)

    return torch.tensor(frames_data, dtype=torch.float32)


def precompute_dataset(dataset_path, cache_dir=CACHE_DIR):
    os.makedirs(str(cache_dir), exist_ok=True)
    yolo_model = YOLO('yolo11m-pose.pt')

    data_map = []
    categories = {'non_violence': 0, 'violence': 1}

    for category, label in categories.items():
        folder = os.path.join(dataset_path, category)
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found.")
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

        for vid_file in tqdm(files, desc=f'Caching {category}'):
            vid_path = os.path.join(folder, vid_file)

            # Hash file contents to avoid stale cache on rename/move
            with open(vid_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            cache_path = str(Path(cache_dir) / f"{file_hash}.pt")

            if not os.path.exists(cache_path):
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
        features = torch.load(self.data_map[idx]['path'], weights_only=True)
        label = self.data_map[idx]['label']

        if self.augment:
            features = features.clone()

            # Horizontal flip: X coords at indices 0, 3, 6, ...
            if np.random.random() > 0.5:
                features[:, 0::3] = 1.0 - features[:, 0::3]

            # Small gaussian noise
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.01, features.shape).astype(np.float32)
                features += torch.from_numpy(noise)

            # Random frame dropout
            if np.random.random() > 0.7:
                drop_idx = np.random.randint(0, SEQUENCE_LENGTH, size=2)
                features[drop_idx] = 0

        return features, torch.tensor(label, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            scores = torch.sigmoid(model(X)).cpu().numpy().flatten()
            all_scores.extend(scores.tolist())
            all_preds.extend((scores > threshold).astype(int).tolist())
            all_labels.extend(y.int().numpy().tolist())

    acc = 100.0 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    report = classification_report(
        all_labels, all_preds,
        target_names=['non_violence', 'violence'],
        output_dict=True, zero_division=0
    )
    return acc, report, np.array(all_scores), np.array(all_labels)


def find_best_threshold(scores, labels, max_fp_rate=0.20):
    best_thresh, best_f1, best_stats = 0.5, 0.0, {}
    fallback_thresh, fallback_f1, fallback_stats = 0.5, 0.0, {}

    for thresh in np.arange(0.30, 0.85, 0.01):
        preds = (scores > thresh).astype(int)
        report = classification_report(
            labels, preds,
            target_names=['non_violence', 'violence'],
            output_dict=True, zero_division=0
        )
        fp_rate = 1.0 - report['non_violence']['precision']
        v_f1 = report['violence']['f1-score']
        stats = {
            'precision': report['violence']['precision'],
            'recall': report['violence']['recall'],
            'f1': v_f1, 'fp_rate': fp_rate,
            'acc': 100.0 * sum(preds == labels) / len(labels)
        }

        if v_f1 > fallback_f1:
            fallback_f1, fallback_thresh, fallback_stats = v_f1, thresh, stats

        if v_f1 > best_f1 and fp_rate <= max_fp_rate:
            best_f1, best_thresh, best_stats = v_f1, thresh, stats

    if not best_stats:
        print(f"  Warning: no threshold met FP rate <= {max_fp_rate:.0%}. Using unconstrained best.")
        return fallback_thresh, fallback_f1, fallback_stats

    return best_thresh, best_f1, best_stats


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    writer = SummaryWriter(log_dir='runs/violence_lstm_experiment')
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data_map = precompute_dataset(DATASET_PATH)
    if not data_map:
        print("No data found! Check paths.")
        return

    labels = [d['label'] for d in data_map]
    train_data, val_data = train_test_split(
        data_map, test_size=0.2, stratify=labels, random_state=42
    )

    train_ds = ViolenceSequenceDataset(train_data, augment=True)
    val_ds = ViolenceSequenceDataset(val_data, augment=False)

    train_labels = [d['label'] for d in train_data]
    class_weights = 1. / np.bincount(train_labels)
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    model = PoseLSTM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-3)
    criterion = FocalLoss()
    scaler = GradScaler('cuda')

    # patience=10 so LR doesn't drop on normal val loss fluctuation (was 5, too aggressive)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10,
        threshold=0.005, min_lr=1e-6
    )

    best_f1 = 0.0
    epochs_no_improve = 0

    print("\nStarting Training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()

            with autocast('cuda'):
                preds = model(X)
                loss = criterion(preds, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Compute val loss for scheduler
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).unsqueeze(1)
                with autocast('cuda'):
                    loss = criterion(model(X), y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        acc, report, _, _ = evaluate(model, val_loader, device, threshold=0.5)
        violence_precision = report['violence']['precision']
        violence_recall    = report['violence']['recall']
        violence_f1        = report['violence']['f1-score']
        fp_rate = 1.0 - report['non_violence']['precision']
        lr = float(optimizer.param_groups[0]['lr'])

        print(
            f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
            f"Loss: {avg_train_loss:.4f} | ValLoss: {avg_val_loss:.4f} | "
            f"Acc: {acc:.2f}% | "
            f"Violence — P: {violence_precision:.3f}  R: {violence_recall:.3f}  F1: {violence_f1:.3f} | "
            f"FP: {fp_rate:.3f} | LR: {lr:.6f}"
        )

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Val', acc, epoch)
        writer.add_scalar('Violence/Precision', violence_precision, epoch)
        writer.add_scalar('Violence/Recall', violence_recall, epoch)
        writer.add_scalar('Violence/F1', violence_f1, epoch)
        writer.add_scalar('FalsePositiveRate', fp_rate, epoch)
        writer.add_scalar('LR', lr, epoch)

        # Save on F1 — balances precision and recall
        if violence_f1 > best_f1:
            best_f1 = violence_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> Saved (F1: {violence_f1:.3f}, P: {violence_precision:.3f}, R: {violence_recall:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Post-training threshold sweep
    print("\n--- Threshold sweep on validation set ---")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    _, _, val_scores, val_labels = evaluate(model, val_loader, device, threshold=0.5)
    best_thresh, _, stats = find_best_threshold(val_scores, val_labels, max_fp_rate=0.20)

    print(f"\nOptimal threshold: {best_thresh:.2f}")
    print(f"  Acc: {stats.get('acc', 0):.2f}%")
    print(f"  Violence — P: {stats.get('precision', 0):.3f}  R: {stats.get('recall', 0):.3f}  F1: {stats.get('f1', 0):.3f}")
    print(f"  FP rate: {stats.get('fp_rate', 0):.3f}")

    threshold_path = _MODELS_DIR / 'inference_threshold.txt'
    with open(threshold_path, 'w') as f:
        f.write(str(round(float(best_thresh), 4)))

    print(f"\nModel: {BEST_MODEL_PATH}")
    print(f"Threshold file: {threshold_path}")
    print(f"Best val F1: {best_f1:.3f}")
    writer.close()


if __name__ == "__main__":
    main()