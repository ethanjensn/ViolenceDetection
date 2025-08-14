import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn
import multiprocessing as mp
from functools import partial
import hashlib
from tqdm import tqdm
import timm  # For pretrained vision models
from sklearn.metrics import classification_report, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)

# Enable cuDNN benchmarking for faster training
cudnn.benchmark = True

def precompute_pose_features(image_paths, cache_dir='pose_cache', batch_size=32):
    """Pre-compute all pose features with enhanced keypoint processing"""
    print("Pre-computing pose features...")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load YOLO model once
    yolo_model = YOLO('yolo11m-pose.pt')
    
    # Process images in batches
    pose_features_list = []
    batch_pbar = tqdm(range(0, len(image_paths), batch_size), desc="Computing pose features")
    
    for i in batch_pbar:
        batch_paths = image_paths[i:i + batch_size]
        batch_features = []
        
        for img_path in batch_paths:
            # Check cache first
            cache_key = hashlib.md5(img_path.encode()).hexdigest()
            cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        features = torch.load(f)
                except:
                    features = None
            else:
                features = None
            
            if features is None:
                # Compute features
                try:
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = yolo_model(image, verbose=False)
                    features = extract_enhanced_pose_features(results, image.shape[:2])
                    
                    # Cache features
                    try:
                        with open(cache_path, 'wb') as f:
                            torch.save(features, f)
                    except:
                        pass
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    features = torch.zeros(325, dtype=torch.float32)  # Updated feature size
            
            batch_features.append(features)
        
        pose_features_list.extend(batch_features)
        batch_pbar.set_postfix({'Processed': i + len(batch_paths), 'Total': len(image_paths)})
    
    return pose_features_list

def extract_enhanced_pose_features(results, image_shape):
    """Extract enhanced pose features with spatial relationships and violence-specific metrics"""
    MAX_PEOPLE = 4
    features = []
    h, w = image_shape
    
    if results and len(results) > 0:
        result = results[0]
        
        # Basic keypoints and boxes
        keypoints_data = []
        boxes_data = []
        
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy()
            num_people = min(len(keypoints), MAX_PEOPLE)
            
            for i in range(num_people):
                # Normalize keypoints by image dimensions
                kp = keypoints[i].copy()
                kp[:, 0] = kp[:, 0] / w  # Normalize x
                kp[:, 1] = kp[:, 1] / h  # Normalize y
                keypoints_data.append(kp)
                features.extend(kp.flatten())
            
            # Pad with zeros
            features.extend([0] * ((MAX_PEOPLE - num_people) * 17 * 3))
        else:
            features.extend([0] * (MAX_PEOPLE * 17 * 3))
        
        # Enhanced bounding boxes with area and aspect ratio
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.data.cpu().numpy()
            num_boxes = min(len(boxes), MAX_PEOPLE)
            
            for i in range(num_boxes):
                box = boxes[i][:4]
                # Normalize coordinates
                box[0] /= w  # x1
                box[1] /= h  # y1
                box[2] /= w  # x2
                box[3] /= h  # y2
                
                # Calculate area and aspect ratio
                width = box[2] - box[0]
                height = box[3] - box[1]
                area = width * height
                aspect_ratio = width / height if height > 0 else 0
                
                boxes_data.append(box)
                features.extend([*box, area, aspect_ratio])  # 6 features per box
            
            # Pad with zeros
            features.extend([0] * ((MAX_PEOPLE - num_boxes) * 6))
        else:
            features.extend([0] * (MAX_PEOPLE * 6))
        
        # Violence-specific features
        violence_features = calculate_violence_indicators(keypoints_data, boxes_data)
        features.extend(violence_features)
        
    else:
        # No detections
        features.extend([0] * (MAX_PEOPLE * 17 * 3 + MAX_PEOPLE * 6))
        features.extend([0] * 13)  # Violence features
    
    # Number of people detected
    num_people = min(len(result.boxes.data) if results and len(results) > 0 and hasattr(result, 'boxes') and result.boxes is not None else 0, MAX_PEOPLE)
    features.append(num_people)
    
    # Ensure consistent feature size (325 features total)
    expected_size = 325
    if len(features) < expected_size:
        features.extend([0] * (expected_size - len(features)))
    elif len(features) > expected_size:
        features = features[:expected_size]
    
    return torch.tensor(features, dtype=torch.float32)

def calculate_violence_indicators(keypoints_data, boxes_data):
    """Calculate violence-specific indicators from pose data"""
    features = []
    
    if len(keypoints_data) < 2:
        # Not enough people for interaction analysis
        return [0] * 13
    
    # Inter-person distances
    distances = []
    for i in range(len(keypoints_data)):
        for j in range(i + 1, len(keypoints_data)):
            kp1, kp2 = keypoints_data[i], keypoints_data[j]
            # Distance between torso centers (average of shoulders and hips)
            torso1 = np.mean([kp1[5], kp1[6], kp1[11], kp1[12]], axis=0)[:2]  # shoulders + hips
            torso2 = np.mean([kp2[5], kp2[6], kp2[11], kp2[12]], axis=0)[:2]
            if all(torso1 > 0) and all(torso2 > 0):  # Valid keypoints
                dist = np.linalg.norm(torso1 - torso2)
                distances.append(dist)
    
    # Statistical features of distances
    if distances:
        features.extend([np.mean(distances), np.std(distances), np.min(distances)])
    else:
        features.extend([0, 0, 1])  # Default values
    
    # Pose dynamics (approximated by limb angles)
    limb_angles = []
    for kp in keypoints_data:
        # Right arm angle (shoulder-elbow-wrist)
        if all(kp[[6, 8, 10], 2] > 0.5):  # Confidence check
            shoulder, elbow, wrist = kp[6][:2], kp[8][:2], kp[10][:2]
            angle = calculate_angle(shoulder, elbow, wrist)
            limb_angles.append(angle)
        
        # Left arm angle
        if all(kp[[5, 7, 9], 2] > 0.5):
            shoulder, elbow, wrist = kp[5][:2], kp[7][:2], kp[9][:2]
            angle = calculate_angle(shoulder, elbow, wrist)
            limb_angles.append(angle)
    
    if limb_angles:
        features.extend([np.mean(limb_angles), np.std(limb_angles)])
    else:
        features.extend([90, 0])  # Default neutral pose
    
    # Box overlap (potential for physical contact)
    overlaps = []
    for i in range(len(boxes_data)):
        for j in range(i + 1, len(boxes_data)):
            overlap = calculate_box_overlap(boxes_data[i], boxes_data[j])
            overlaps.append(overlap)
    
    if overlaps:
        features.extend([np.mean(overlaps), np.max(overlaps)])
    else:
        features.extend([0, 0])
    
    # Crowd density
    total_area = sum([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes_data])
    features.append(total_area)
    
    # Average person size (may indicate children vs adults)
    avg_height = np.mean([box[3] - box[1] for box in boxes_data]) if boxes_data else 0
    features.append(avg_height)
    
    # Pose spread (how spread out are the keypoints)
    pose_spreads = []
    for kp in keypoints_data:
        valid_kp = kp[kp[:, 2] > 0.5][:, :2]  # Only confident keypoints
        if len(valid_kp) > 2:
            spread = np.std(valid_kp, axis=0).mean()
            pose_spreads.append(spread)
    
    if pose_spreads:
        features.append(np.mean(pose_spreads))
    else:
        features.append(0)
    
    # Arm extension (raised arms might indicate aggression or defense)
    arm_extensions = []
    for kp in keypoints_data:
        # Check if wrists are above shoulders
        for wrist_idx, shoulder_idx in [(9, 5), (10, 6)]:  # left and right
            if kp[wrist_idx, 2] > 0.5 and kp[shoulder_idx, 2] > 0.5:
                extension = kp[shoulder_idx, 1] - kp[wrist_idx, 1]  # Positive if wrist above shoulder
                arm_extensions.append(max(0, extension))
    
    features.append(np.mean(arm_extensions) if arm_extensions else 0)
    
    return features

def calculate_angle(p1, p2, p3):
    """Calculate angle at p2 formed by p1-p2-p3"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

def calculate_box_overlap(box1, box2):
    """Calculate intersection over union of two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EnhancedViolenceDataset(Dataset):
    def __init__(self, image_paths, labels, pose_features, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.pose_features = pose_features
        self.transform = transform
        self.is_training = is_training
        
        # Heavy augmentation for training
        if is_training:
            self.aug_transform = A.Compose([
                A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.CLAHE(p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.aug_transform = A.Compose([
                A.Resize(height=224, width=224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply albumentations
        augmented = self.aug_transform(image=image)
        image = augmented['image']
        
        return {'image': image, 'pose_features': self.pose_features[idx], 'label': self.labels[idx]}

class EnhancedViolenceClassifier(nn.Module):
    def __init__(self, pose_feature_dim=325, use_pretrained=True):
        super().__init__()
        
        # Enhanced pose encoder with attention
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_feature_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2)
        )
        
        # Self-attention for pose features
        self.pose_attention = nn.MultiheadAttention(128, 8, dropout=0.1)
        
        # Use pretrained CNN backbone
        if use_pretrained:
            self.backbone = timm.create_model('efficientnet_b2', pretrained=True, num_classes=0)
            image_feature_dim = self.backbone.num_features
        else:
            # Fallback to custom CNN
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
            )
            image_feature_dim = 512
        
        # Feature fusion with cross-attention
        self.cross_attention = nn.MultiheadAttention(128, 8, dropout=0.1)
        
        # Enhanced classifier with residual connections
        self.fusion_layer = nn.Linear(128 + image_feature_dim, 512)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
        
        # Residual connection
        self.residual_layer = nn.Linear(512, 2)
    
    def forward(self, pose_features, image):
        # Encode pose features
        pose_encoded = self.pose_encoder(pose_features)
        
        # Self-attention on pose features
        pose_encoded = pose_encoded.unsqueeze(0)  # Add sequence dimension
        pose_attended, _ = self.pose_attention(pose_encoded, pose_encoded, pose_encoded)
        pose_attended = pose_attended.squeeze(0)
        
        # Extract image features
        image_features = self.backbone(image)
        
        # Feature fusion
        combined = torch.cat([pose_attended, image_features], dim=1)
        fused = F.relu(self.fusion_layer(combined))
        
        # Main classification path
        main_output = self.classifier(fused)
        
        # Residual connection
        residual_output = self.residual_layer(fused)
        
        # Combine outputs
        output = main_output + 0.1 * residual_output  # Weighted residual
        
        return output

def extract_frames_parallel(video_path, output_dir, target_frames=6):
    """Extract frames from video with adaptive sampling based on video length"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frame interval to get target_frames
        if total_frames <= target_frames:
            # Short video: extract all frames
            frame_interval = 1
        else:
            # Longer video: sample evenly
            frame_interval = max(1, total_frames // target_frames)
        
        frame_count = saved_count = 0
        frame_paths = []
        
        while cap.isOpened() and saved_count < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_frame{saved_count}.jpg")
                if cv2.imwrite(frame_path, frame):
                    frame_paths.append(frame_path)
                    saved_count += 1
            
            frame_count += 1
        
        cap.release()
        
        # Debug info for very short videos
        if len(frame_paths) < 2 and duration < 2.0:
            print(f"    Warning: Short video ({duration:.1f}s) extracted {len(frame_paths)} frames")
        
        return frame_paths
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return []

def process_combined_dataset(dataset_path, output_dir):
    """Process the combined dataset with both videos and images"""
    image_paths, labels = [], []
    
    print(f"Processing combined dataset: {dataset_path}")
    
    # Expected folder structure: dataset_path/non_violence and dataset_path/violence
    for label, folder in [(0, 'non_violence'), (1, 'violence')]:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            print(f"  Warning: {folder} folder not found at {folder_path}")
            continue
            
        print(f"  Processing {folder} folder...")
        temp_dir = os.path.join(output_dir, f'temp_{folder}')
        
        # Get all files in the folder
        files = os.listdir(folder_path)
        video_files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"    Found {len(video_files)} videos and {len(image_files)} images")
        
        # Process videos
        if video_files:
            print(f"    Processing {len(video_files)} videos...")
            video_paths = [os.path.join(folder_path, f) for f in video_files]
            
            # Use multiprocessing to extract frames
            with mp.Pool(processes=mp.cpu_count()) as pool:
                extract_func = partial(extract_frames_parallel, output_dir=temp_dir, target_frames=6)
                results = list(tqdm(
                    pool.imap(extract_func, video_paths),
                    total=len(video_paths),
                    desc=f"      Extracting frames from videos",
                    unit="video"
                ))
            
            # Add extracted frame paths
            for frame_paths in results:
                image_paths.extend(frame_paths)
                labels.extend([label] * len(frame_paths))
        
        # Process images directly
        if image_files:
            print(f"    Processing {len(image_files)} images...")
            for filename in image_files:
                image_path = os.path.join(folder_path, filename)
                image_paths.append(image_path)
                labels.append(label)
    
    return image_paths, labels

def create_weighted_sampler(labels):
    """Create weighted sampler to handle class imbalance"""
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def train_model_enhanced(model, train_loader, val_loader, num_epochs=20, device='cuda', output_dir='training_outputs'):
    """Enhanced training with advanced techniques"""
    # Use Focal Loss for better handling of class imbalance
    criterion = FocalLoss(alpha=1, gamma=2)
    
    # Advanced optimizer with different learning rates for different parts
    backbone_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 0.0001, 'weight_decay': 0.01},  # Lower LR for pretrained
        {'params': other_params, 'lr': 0.001, 'weight_decay': 0.01}
    ])
    
    # Advanced scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[0.0001, 0.001], 
        epochs=num_epochs, 
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        div_factor=10,
        final_div_factor=100
    )
    
    scaler = GradScaler('cuda')
    
    os.makedirs(output_dir, exist_ok=True)
    best_val_accuracy = 0.0
    patience = 8
    patience_counter = 0
    
    # Training loop with enhanced monitoring
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = train_correct = train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        
        for batch_idx, batch in enumerate(train_pbar):
            images = batch['image'].to(device, non_blocking=True)
            pose_features = batch['pose_features'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(pose_features, images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.4f}',
                'Acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        # Validation phase with detailed metrics
        model.eval()
        val_loss = val_correct = val_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(device, non_blocking=True)
                pose_features = batch['pose_features'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                
                with autocast('cuda'):
                    outputs = model(pose_features, images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_accuracy = 100 * val_correct / val_total
        
        # Print detailed metrics
        if epoch % 5 == 0:  # Every 5 epochs
            print("\nDetailed Classification Report:")
            print(classification_report(all_labels, all_predictions, 
                                      target_names=['Non-Violence', 'Violence']))
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'accuracy': val_accuracy
            }, os.path.join(output_dir, 'violence_detection_model_best.pt'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train: {train_loss/len(train_loader):.4f}, {100*train_correct/train_total:.2f}% - Val: {val_loss/len(val_loader):.4f}, {val_accuracy:.2f}%')

# Update the main function to use enhanced components
def main():
    # Configuration
    dataset_path = r'Combined Datasets'
    batch_size = 256  # Slightly reduced for more stable training
    num_epochs = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'training_outputs_enhanced'
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        print("Please update the dataset_path variable to point to your dataset location.")
        # Try some common alternative paths
        alternative_paths = [
            'Combined Datasets',
            '../Combined Datasets',
            './DATASETS/Combined Datasets',
            '/data/Combined Datasets'
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"Found alternative dataset path: {alt_path}")
                dataset_path = alt_path
                break
        else:
            print("Please check the dataset path and try again.")
            return
    
    # Load and process dataset
    image_paths, labels = process_combined_dataset(dataset_path, output_dir)
    
    if len(image_paths) == 0:
        print("Error: No images found in the dataset!")
        print("Please ensure your dataset has the following structure:")
        print("  dataset_path/")
        print("    non_violence/  (images and videos)")
        print("    violence/      (images and videos)")
        return
    
    print(f"Dataset loaded: {len(image_paths)} samples (Non-violence: {labels.count(0)}, Violence: {labels.count(1)})")
    
    # Enhanced pose features
    pose_features = precompute_pose_features(image_paths, cache_dir='pose_cache_enhanced')
    
    # Stratified split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Split pose features
    train_features = [pose_features[i] for i, path in enumerate(image_paths) if path in train_paths]
    val_features = [pose_features[i] for i, path in enumerate(image_paths) if path in val_paths]
    
    print(f"Train samples: {len(train_paths)}, Validation samples: {len(val_paths)}")
    
    # Create enhanced datasets
    train_dataset = EnhancedViolenceDataset(train_paths, train_labels, train_features, is_training=True)
    val_dataset = EnhancedViolenceDataset(val_paths, val_labels, val_features, is_training=False)
    
    # Create weighted sampler for balanced training
    sampler = create_weighted_sampler(train_labels)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True)
    
    # Enhanced model
    model = EnhancedViolenceClassifier(pose_feature_dim=325, use_pretrained=True).to(device)
    print(f"Enhanced model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train with enhanced techniques
    train_model_enhanced(model, train_loader, val_loader, num_epochs, device, output_dir)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'EnhancedViolenceClassifier',
        'pose_feature_dim': 325
    }, os.path.join(output_dir, 'violence_detection_model_final_enhanced.pt'))
    
    print("Enhanced training complete!")

if __name__ == "__main__":
    main()