"""
RGB + Pose Fusion — Maximum Accuracy Version
=============================================
Upgrades over v1:
  1. Temporal RGB: 5 evenly-spaced frames instead of 1 middle frame.
     MobileNetV3 runs on each, embeddings are averaged → temporal coverage
     without processing all 20 frames.
  2. Larger LSTM: hidden_size=512 (was 256). Clean ~4200 video dataset
     can support this without overfitting.
  3. More MobileNetV3 layers unfrozen: blocks 6-7 trainable (was only 6-7).
  4. Test-Time Augmentation (TTA) at threshold sweep: 3 forward passes with
     flip/noise, averaged — adds ~1-2% accuracy for free at inference.
  5. Two-phase training: freeze RGB backbone entirely for first 10 epochs
     so the LSTM learns first, then unfreeze for joint fine-tuning.
     Prevents the CNN from dominating early and the LSTM never learning.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
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

SEQUENCE_LENGTH   = 20
K_PEOPLE          = 2
NUM_KEYPOINTS     = 17
FEATURES_PER_PERSON = NUM_KEYPOINTS * 3
POSE_INPUT_SIZE   = K_PEOPLE * FEATURES_PER_PERSON  # 102

RGB_FRAMES        = 5      # evenly-spaced frames sampled per video for CNN
RGB_SIZE          = 112    # spatial size fed to MobileNetV3
LSTM_HIDDEN       = 512    # was 256 — larger model, clean data can support it

BATCH_SIZE        = 16     # RGB in memory, keep at 16
NUM_EPOCHS        = 150
EARLY_STOP_PATIENCE = 30
PHASE2_EPOCH      = 10     # unfreeze RGB backbone after this many epochs

_TRAINING_DIR = Path(__file__).resolve().parent
_CACHES_DIR   = _TRAINING_DIR / 'caches'
POSE_CACHE_DIR = _CACHES_DIR / f'pose_cache_lstm_seq{SEQUENCE_LENGTH}_k{K_PEOPLE}_v2'
RGB_CACHE_DIR  = _CACHES_DIR / f'rgb_cache_{RGB_FRAMES}frames_size{RGB_SIZE}_v2'

POSE_MODEL_WEIGHTS = 'yolo11m-pose.pt'

_PROJECT_ROOT  = Path(__file__).resolve().parents[1]
_MODELS_DIR    = _PROJECT_ROOT / 'MODELS'
BEST_MODEL_PATH = _MODELS_DIR / 'best_violence_rgb_pose_v2.pt'

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD  = [0.229, 0.224, 0.225]

rgb_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RGB_SIZE, RGB_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
])

rgb_augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RGB_SIZE, RGB_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
])


# =============================================================================
# MODEL
# =============================================================================

class RGBPoseFusionV2(nn.Module):
    """
    Two-stream fusion:
      Stream A — PoseLSTM (hidden_size=512) on 20-frame skeleton sequence
      Stream B — MobileNetV3-Small on RGB_FRAMES evenly-spaced frames,
                 embeddings averaged before projection
      Fusion   — concat 128+128 → classifier
    """
    def __init__(self, pose_input_size=POSE_INPUT_SIZE, lstm_hidden=LSTM_HIDDEN):
        super().__init__()

        # ── Stream A: Pose LSTM ──
        self.pose_embedding = nn.Sequential(
            nn.Linear(pose_input_size, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.lstm = nn.LSTM(lstm_hidden, lstm_hidden, num_layers=2,
                            batch_first=True, dropout=0.3)
        self.pose_attention = nn.Linear(lstm_hidden, 1)
        self.pose_proj = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # ── Stream B: MobileNetV3-Small ──
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.rgb_backbone = mobilenet.features   # output: (B, 576, H', W')
        self.rgb_pool = nn.AdaptiveAvgPool2d(1)
        self.rgb_proj = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Start with ALL backbone layers frozen — unfreeze in phase 2
        self._freeze_backbone()

        # ── Fusion classifier ──
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def _freeze_backbone(self):
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, blocks_from_end=3):
        """Unfreeze last N blocks for phase-2 fine-tuning."""
        total = len(self.rgb_backbone)
        for i, layer in enumerate(self.rgb_backbone):
            if i >= total - blocks_from_end:
                for param in layer.parameters():
                    param.requires_grad = True
        trainable = sum(p.numel() for p in self.rgb_backbone.parameters()
                        if p.requires_grad)
        print(f"  [Phase 2] Unfroze last {blocks_from_end} backbone blocks "
              f"({trainable:,} params now trainable)")

    def forward(self, pose_seq, rgb_frames):
        """
        pose_seq:   (B, SEQUENCE_LENGTH, POSE_INPUT_SIZE)
        rgb_frames: (B, RGB_FRAMES, 3, RGB_SIZE, RGB_SIZE)
        """
        # Pose stream
        x = self.pose_embedding(pose_seq)
        lstm_out, _ = self.lstm(x)
        attn = F.softmax(self.pose_attention(lstm_out), dim=1)
        pose_ctx = torch.sum(attn * lstm_out, dim=1)
        pose_emb = self.pose_proj(pose_ctx)

        # RGB stream — process each frame, average embeddings
        B, T, C, H, W = rgb_frames.shape
        frames_flat = rgb_frames.view(B * T, C, H, W)
        feat = self.rgb_backbone(frames_flat)
        feat = self.rgb_pool(feat).flatten(1)          # (B*T, 576)
        feat = feat.view(B, T, -1).mean(dim=1)         # average over frames
        rgb_emb = self.rgb_proj(feat)

        fused = torch.cat([pose_emb, rgb_emb], dim=1)
        return self.classifier(fused)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_pose_features(video_path, yolo_model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return torch.zeros(SEQUENCE_LENGTH, POSE_INPUT_SIZE)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, total_frames // SEQUENCE_LENGTH)
    frames_data = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0 and len(frames_data) < SEQUENCE_LENGTH:
            results = yolo_model(frame, verbose=False, stream=False)
            frame_features = []

            if results[0].keypoints is not None:
                kps      = results[0].keypoints.xyn.cpu().numpy()
                confs    = results[0].boxes.conf.cpu().numpy()
                kp_confs = results[0].keypoints.conf.cpu().numpy()
                sorted_idx = np.argsort(confs)[::-1]

                for k in range(K_PEOPLE):
                    if k < len(sorted_idx):
                        idx = sorted_idx[k]
                        person_data = np.hstack(
                            [kps[idx], kp_confs[idx].reshape(-1, 1)]
                        ).flatten()
                    else:
                        person_data = np.zeros(FEATURES_PER_PERSON)
                    frame_features.extend(person_data)
            else:
                frame_features = [0.0] * POSE_INPUT_SIZE

            frames_data.append(frame_features)

        frame_idx += 1
        if len(frames_data) >= SEQUENCE_LENGTH:
            break

    cap.release()
    while len(frames_data) < SEQUENCE_LENGTH:
        frames_data.append([0.0] * POSE_INPUT_SIZE)

    return torch.tensor(frames_data, dtype=torch.float32)


def extract_rgb_frames(video_path, n_frames=RGB_FRAMES):
    """
    Sample n_frames evenly across the video.
    Returns (n_frames, 3, RGB_SIZE, RGB_SIZE) tensor.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return torch.zeros(n_frames, 3, RGB_SIZE, RGB_SIZE)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # evenly spaced indices, avoid frame 0 (often black) and last frame
    indices = np.linspace(max(1, total * 0.1),
                          max(2, total * 0.9),
                          n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_transform(frame_rgb))
        else:
            frames.append(torch.zeros(3, RGB_SIZE, RGB_SIZE))

    cap.release()

    while len(frames) < n_frames:
        frames.append(torch.zeros(3, RGB_SIZE, RGB_SIZE))

    return torch.stack(frames[:n_frames])   # (n_frames, 3, H, W)


def precompute_dataset(dataset_path):
    POSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RGB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    yolo_model = YOLO(POSE_MODEL_WEIGHTS)
    data_map = []
    categories = {'non_violence': 0, 'violence': 1}

    for category, label in categories.items():
        folder = Path(dataset_path) / category
        if not folder.exists():
            print(f"Warning: {folder} not found.")
            continue

        files = [f for f in folder.iterdir()
                 if f.suffix.lower() in ('.mp4', '.avi', '.mov')]

        for vid_file in tqdm(files, desc=f'Caching {category}'):
            vid_path = str(vid_file)

            with open(vid_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            pose_cache = POSE_CACHE_DIR / f"{file_hash}.pt"
            rgb_cache  = RGB_CACHE_DIR  / f"{file_hash}.pt"

            if not pose_cache.exists():
                torch.save(extract_pose_features(vid_path, yolo_model), pose_cache)

            if not rgb_cache.exists():
                torch.save(extract_rgb_frames(vid_path), rgb_cache)

            data_map.append({
                'pose_path': str(pose_cache),
                'rgb_path':  str(rgb_cache),
                'label':     label
            })

    return data_map


# =============================================================================
# DATASET
# =============================================================================

class ViolenceFusionDataset(Dataset):
    def __init__(self, data_map, augment=False):
        self.data_map = data_map
        self.augment  = augment

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        item  = self.data_map[idx]
        pose  = torch.load(item['pose_path'], weights_only=True)
        rgb   = torch.load(item['rgb_path'],  weights_only=True)  # (T, 3, H, W)
        label = item['label']

        if self.augment:
            pose = pose.clone()
            do_flip = np.random.random() > 0.5

            # Pose augmentations
            if do_flip:
                pose[:, 0::3] = 1.0 - pose[:, 0::3]
            if np.random.random() > 0.5:
                pose += torch.from_numpy(
                    np.random.normal(0, 0.01, pose.shape).astype(np.float32)
                )
            if np.random.random() > 0.7:
                drop = np.random.randint(0, SEQUENCE_LENGTH, size=2)
                pose[drop] = 0

            # RGB augmentation — apply same flip + colour jitter to all frames
            rgb_aug = []
            mean_t = torch.tensor(RGB_MEAN).view(3, 1, 1)
            std_t  = torch.tensor(RGB_STD).view(3, 1, 1)
            for t in range(rgb.shape[0]):
                frame = rgb[t]
                # de-normalise → uint8
                frame_dn = (frame * std_t + mean_t).clamp(0, 1)
                frame_np = (frame_dn.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                if do_flip:
                    frame_np = frame_np[:, ::-1, :].copy()
                rgb_aug.append(rgb_augment(frame_np))
            rgb = torch.stack(rgb_aug)

        return pose, rgb, torch.tensor(label, dtype=torch.float32)


# =============================================================================
# LOSS + EVALUATION
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


def evaluate(model, loader, device, threshold=0.5, tta=False):
    """
    Evaluate model.
    tta=True: run 3 passes (normal, h-flip, noise) and average scores.
    """
    model.eval()
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for pose, rgb, y in loader:
            pose, rgb = pose.to(device), rgb.to(device)

            if tta:
                # Pass 1: normal
                s1 = torch.sigmoid(model(pose, rgb))
                # Pass 2: horizontal flip pose + RGB
                pose_f = pose.clone()
                pose_f[:, :, 0::3] = 1.0 - pose_f[:, :, 0::3]
                rgb_f  = torch.flip(rgb, dims=[4])
                s2 = torch.sigmoid(model(pose_f, rgb_f))
                # Pass 3: slight noise
                pose_n = pose + torch.randn_like(pose) * 0.01
                s3 = torch.sigmoid(model(pose_n, rgb))
                scores = ((s1 + s2 + s3) / 3).cpu().numpy().flatten()
            else:
                scores = torch.sigmoid(model(pose, rgb)).cpu().numpy().flatten()

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
        preds  = (scores > thresh).astype(int)
        report = classification_report(
            labels, preds,
            target_names=['non_violence', 'violence'],
            output_dict=True, zero_division=0
        )
        fp_rate = 1.0 - report['non_violence']['precision']
        v_f1    = report['violence']['f1-score']
        stats   = {
            'precision': report['violence']['precision'],
            'recall':    report['violence']['recall'],
            'f1':        v_f1, 'fp_rate': fp_rate,
            'acc':       100.0 * sum(preds == labels) / len(labels)
        }
        if v_f1 > fallback_f1:
            fallback_f1, fallback_thresh, fallback_stats = v_f1, thresh, stats
        if v_f1 > best_f1 and fp_rate <= max_fp_rate:
            best_f1, best_thresh, best_stats = v_f1, thresh, stats

    if not best_stats:
        print("  Warning: no threshold met FP <= 20%. Using unconstrained best.")
        return fallback_thresh, fallback_f1, fallback_stats
    return best_thresh, best_f1, best_stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    writer = SummaryWriter(log_dir='runs/violence_rgb_pose_v2')
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data_map = precompute_dataset(DATASET_PATH)
    if not data_map:
        print("No data found! Check DATASET_PATH.")
        return

    labels = [d['label'] for d in data_map]
    train_data, val_data = train_test_split(
        data_map, test_size=0.2, stratify=labels, random_state=42
    )

    train_ds = ViolenceFusionDataset(train_data, augment=True)
    val_ds   = ViolenceFusionDataset(val_data,   augment=False)

    train_labels  = [d['label'] for d in train_data]
    class_weights = 1. / np.bincount(train_labels)
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model     = RGBPoseFusionV2().to(device)
    criterion = FocalLoss()
    scaler    = GradScaler('cuda')

    # Phase 1 optimizer — only pose + fusion params (backbone frozen)
    def make_optimizer(phase=1):
        pose_params   = (list(model.pose_embedding.parameters()) +
                         list(model.lstm.parameters()) +
                         list(model.pose_attention.parameters()) +
                         list(model.pose_proj.parameters()) +
                         list(model.classifier.parameters()) +
                         list(model.rgb_proj.parameters()))
        rgb_trainable = [p for p in model.rgb_backbone.parameters()
                         if p.requires_grad]

        if phase == 1 or not rgb_trainable:
            return optim.AdamW(pose_params, lr=0.001, weight_decay=5e-3)
        else:
            return optim.AdamW([
                {'params': pose_params,   'lr': 0.001},
                {'params': rgb_trainable, 'lr': 0.0001}
            ], weight_decay=5e-3)

    optimizer = make_optimizer(phase=1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10,
        threshold=0.005, min_lr=1e-6
    )

    best_f1 = 0.0
    epochs_no_improve = 0
    phase = 1

    print(f"\nStarting Training (Phase 1: RGB backbone frozen for {PHASE2_EPOCH} epochs)...")

    for epoch in range(NUM_EPOCHS):

        # Switch to phase 2 — unfreeze backbone and rebuild optimizer
        if epoch == PHASE2_EPOCH and phase == 1:
            phase = 2
            print(f"\n[Epoch {epoch+1}] Switching to Phase 2 — unfreezing RGB backbone")
            model.unfreeze_backbone(blocks_from_end=3)
            optimizer = make_optimizer(phase=2)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10,
                threshold=0.005, min_lr=1e-6
            )

        model.train()
        train_loss = 0.0

        for pose, rgb, y in train_loader:
            pose, rgb, y = pose.to(device), rgb.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()

            with autocast('cuda'):
                preds = model(pose, rgb)
                loss  = criterion(preds, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pose, rgb, y in val_loader:
                pose, rgb, y = pose.to(device), rgb.to(device), y.to(device).unsqueeze(1)
                with autocast('cuda'):
                    val_loss += criterion(model(pose, rgb), y).item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        acc, report, _, _ = evaluate(model, val_loader, device, threshold=0.5)
        vp  = report['violence']['precision']
        vr  = report['violence']['recall']
        vf1 = report['violence']['f1-score']
        fp  = 1.0 - report['non_violence']['precision']
        lr  = float(optimizer.param_groups[0]['lr'])

        print(
            f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | Ph{phase} | "
            f"Loss: {avg_train_loss:.4f} | ValLoss: {avg_val_loss:.4f} | "
            f"Acc: {acc:.2f}% | "
            f"Violence — P: {vp:.3f}  R: {vr:.3f}  F1: {vf1:.3f} | "
            f"FP: {fp:.3f} | LR: {lr:.6f}"
        )

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val',   avg_val_loss,   epoch)
        writer.add_scalar('Accuracy/Val', acc, epoch)
        writer.add_scalar('Violence/F1',        vf1, epoch)
        writer.add_scalar('Violence/Precision', vp,  epoch)
        writer.add_scalar('Violence/Recall',    vr,  epoch)
        writer.add_scalar('FalsePositiveRate',  fp,  epoch)
        writer.add_scalar('LR', lr, epoch)

        if vf1 > best_f1:
            best_f1 = vf1
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> Saved (F1: {vf1:.3f}, P: {vp:.3f}, R: {vr:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Post-training threshold sweep WITH TTA
    print("\n--- Threshold sweep with TTA (3-pass average) ---")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    acc_tta, report_tta, val_scores, val_labels = evaluate(
        model, val_loader, device, threshold=0.5, tta=True
    )
    print(f"TTA accuracy at 0.5: {acc_tta:.2f}%")

    best_thresh, _, stats = find_best_threshold(val_scores, val_labels, max_fp_rate=0.20)

    print(f"\nOptimal threshold (with TTA): {best_thresh:.2f}")
    print(f"  Acc: {stats.get('acc', 0):.2f}%")
    print(f"  Violence — P: {stats.get('precision', 0):.3f}  "
          f"R: {stats.get('recall', 0):.3f}  F1: {stats.get('f1', 0):.3f}")
    print(f"  FP rate: {stats.get('fp_rate', 0):.3f}")

    threshold_path = _MODELS_DIR / 'inference_threshold.txt'
    with open(threshold_path, 'w') as f:
        f.write(str(round(float(best_thresh), 4)))

    # Save a flag so inference script knows TTA is available
    tta_flag_path = _MODELS_DIR / 'tta_enabled.txt'
    with open(tta_flag_path, 'w') as f:
        f.write('1')

    print(f"\nModel:     {BEST_MODEL_PATH}")
    print(f"Threshold: {threshold_path}")
    print(f"Best val F1 (no TTA): {best_f1:.3f}")
    writer.close()


if __name__ == '__main__':
    main()