# Simple video processor using your trained violence detection model
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
import timm
import time

# Configuration
VIDEO_PATH = r"<INSERT VIDEO PATH HERE>"
FRAME_SKIP = 1  # Process every Nth frame
YOLO_MODEL_PATH = '<INSERT YOLO MODEL PATH HERE>'
VIOLENCE_MODEL_PATH = '<INSERT VIOLENCE DETECTION MODEL PATH HERE>'

# Global models
detection_model = None
violence_model = None
device = None
transform = None

# Alert tracking
alert_start_time = None
alert_threshold = 0.8
alert_duration_threshold = 0.5  # 0.5 seconds
total_alerts = 0
total_alert_duration = 0.0
current_alert_start = None

class EnhancedViolenceClassifier(nn.Module):
    def __init__(self, pose_feature_dim=325, use_pretrained=True):
        super().__init__()
        
        # Pose encoder
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
        
        # Feature fusion
        self.cross_attention = nn.MultiheadAttention(128, 8, dropout=0.1)
        self.fusion_layer = nn.Linear(128 + image_feature_dim, 512)
        
        # Classifier
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
        pose_encoded = pose_encoded.unsqueeze(0)
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
        output = main_output + 0.1 * residual_output
        
        return output

def load_models():
    """Load YOLO and violence detection models"""
    global detection_model, violence_model, device, transform
    
    print("[INFO] Loading models...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Load YOLO pose detection model
    detection_model = YOLO(YOLO_MODEL_PATH)
    detection_model.to(device)
    print(f"[INFO] YOLO pose model loaded on {device}")
    
    # Load violence detection model
    violence_model = EnhancedViolenceClassifier(pose_feature_dim=325, use_pretrained=True).to(device)
    checkpoint = torch.load(VIOLENCE_MODEL_PATH, map_location=device)
    
    # Handle different save formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        violence_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        violence_model.load_state_dict(checkpoint)
        
    violence_model.eval()
    print(f"[INFO] Violence detection model loaded on {device}")
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("[INFO] All models loaded successfully!")

def extract_pose_features(results, image_shape):
    """Extract pose features for violence detection"""
    features = []
    
    if not results or len(results) == 0:
        return torch.zeros(325, dtype=torch.float32)
    
    result = results[0]
    
    # Extract keypoints and boxes
    keypoints_data = []
    boxes_data = []
    
    if result.keypoints is not None:
        if hasattr(result.keypoints, 'xy'):
            keypoints = result.keypoints.xy.cpu().numpy()
        elif hasattr(result.keypoints, 'cpu'):
            keypoints = result.keypoints.cpu().numpy()
        else:
            keypoints = result.keypoints.numpy() if hasattr(result.keypoints, 'numpy') else result.keypoints
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        else:
            boxes = []
        
        for i in range(min(len(keypoints), len(boxes))):
            if i < len(keypoints):
                keypoints_data.append(keypoints[i])
            if i < len(boxes):
                boxes_data.append(boxes[i])
    
    # Flatten keypoints (17 keypoints × 3 coordinates × max 4 people)
    max_people = 4
    keypoint_features = []
    
    for i in range(max_people):
        if i < len(keypoints_data):
            person_keypoints = keypoints_data[i]
            if person_keypoints is not None and len(person_keypoints) >= 17:
                for kp in person_keypoints[:17]:
                    if kp is not None and len(kp) >= 3:
                        keypoint_features.extend([float(kp[0]), float(kp[1]), float(kp[2])])
                    else:
                        keypoint_features.extend([0.0, 0.0, 0.0])
            else:
                keypoint_features.extend([0.0] * 51)
        else:
            keypoint_features.extend([0.0] * 51)
    
    # Flatten bounding boxes (max 4 people × 6 features)
    box_features = []
    for i in range(max_people):
        if i < len(boxes_data):
            box = boxes_data[i]
            width = box[2] - box[0]
            height = box[3] - box[1]
            box_features.extend([float(box[0]), float(box[1]), float(box[2]), float(box[3]), width, height])
        else:
            box_features.extend([0.0] * 6)
    
    # Add placeholder zeros for removed features
    placeholder_features = [0.0] * 13
    
    # Combine all features
    features = keypoint_features + box_features + placeholder_features
    
    # Ensure exact size (325 to match saved model)
    expected_size = 325
    if len(features) < expected_size:
        features.extend([0] * (expected_size - len(features)))
    elif len(features) > expected_size:
        features = features[:expected_size]
    
    return torch.tensor(features, dtype=torch.float32)

def process_video():
    """Process video and save output with violence detection"""
    
    global total_alerts, total_alert_duration
    
    # Load video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    print(f"Processing every {FRAME_SKIP} frame(s)")
    
    # Setup video writer
    output_video_path = "violence_detection_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    processed_count = 0
    
    try:
        print("Processing frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames based on FRAME_SKIP
            if frame_count % FRAME_SKIP != 0:
                frame_count += 1
                continue
            
            # Process frame
            frame_result = process_frame(frame, frame_count)
            
            # Create annotated frame
            annotated_frame = create_annotated_frame(frame, frame_result)
            
            # Write frame to output video
            out.write(annotated_frame)
            
            processed_count += 1
            frame_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} frames...")
                
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    finally:
        # Handle any ongoing alert when video ends
        if current_alert_start is not None:
            total_alert_duration += time.time() - current_alert_start
        
        cap.release()
        out.release()
    
    print(f"\nProcessing complete!")
    print(f"Frames processed: {processed_count}/{total_frames}")
    print(f"Total alerts: {total_alerts}")
    print(f"Total alert duration: {total_alert_duration:.2f} seconds")
    print(f"Output video saved to: {output_video_path}")

def process_frame(frame, frame_number):
    """Process a single frame and return results"""
    
    try:
        # 1. Pose detection
        results = detection_model(frame, verbose=False, conf=0.3, iou=0.3, agnostic_nms=True, max_det=10)
        
        # 2. Extract pose features
        pose_features = extract_pose_features(results, frame.shape[:2]).unsqueeze(0).to(device)
        
        # 3. Prepare image for violence detection
        image_tensor = transform(frame).unsqueeze(0).to(device)
        
        # 4. Get violence prediction
        with torch.no_grad():
            violence_pred = violence_model(pose_features, image_tensor)
            violence_probs = torch.softmax(violence_pred, dim=1)
            violence_score = float(violence_probs[0][1])
        
        # 5. Extract detection data
        detected_poses_data = []
        
        if results[0].keypoints is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) if results[0].boxes is not None else []
            
            if hasattr(results[0].keypoints, 'xy'):
                keypoints = results[0].keypoints.xy.cpu().numpy()
            elif hasattr(results[0].keypoints, 'cpu'):
                keypoints = results[0].keypoints.cpu().numpy()
            else:
                keypoints = results[0].keypoints.numpy() if hasattr(results[0].keypoints, 'numpy') else results[0].keypoints
            
            confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []
            
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                if i < len(keypoints) and i < len(confidences):
                    try:
                        if isinstance(keypoints[i], np.ndarray):
                            pose_keypoints = keypoints[i].astype(float).tolist()
                        elif hasattr(keypoints[i], 'tolist'):
                            pose_keypoints = keypoints[i].tolist()
                        elif hasattr(keypoints[i], 'detach'):
                            pose_keypoints = keypoints[i].detach().cpu().numpy().tolist()
                        else:
                            pose_keypoints = np.array(keypoints[i]).astype(float).tolist()
                    except Exception as e:
                        pose_keypoints = []
                    
                    confidence = float(confidences[i])
                    
                    detected_poses_data.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "keypoints": pose_keypoints,
                        "confidence": confidence
                    })
        
        return {
            "frame_number": frame_number,
            "violence_score": violence_score,
            "detections": detected_poses_data,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "frame_number": frame_number,
            "status": "exception",
            "error_message": str(e)
        }

def create_annotated_frame(frame, frame_result):
    """Create frame with bounding boxes and violence score"""
    
    global alert_start_time, total_alerts, total_alert_duration, current_alert_start
    
    annotated_frame = frame.copy()
    
    if frame_result["status"] != "success":
        cv2.putText(annotated_frame, f"Error: {frame_result.get('error_message', 'Unknown')}", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return annotated_frame
    
    # Add frame info
    frame_num = frame_result["frame_number"]
    violence_score = frame_result["violence_score"]
    
    cv2.putText(annotated_frame, f"Frame: {frame_num}", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display violence score with color coding
    if violence_score < 0.4:
        score_color = (0, 255, 0)  # Green - Low risk
        risk_level = "LOW"
    elif violence_score < 0.7:
        score_color = (0, 165, 255)  # Orange - Medium risk
        risk_level = "MEDIUM"
    else:
        score_color = (0, 0, 255)  # Red - High risk
        risk_level = "HIGH"
    
    cv2.putText(annotated_frame, f"Violence Score: {violence_score:.3f} ({risk_level})", 
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
    
    # Alert system
    current_time = time.time()
    
    if violence_score >= alert_threshold:
        if alert_start_time is None:
            alert_start_time = current_time
            current_alert_start = current_time
        elif current_time - alert_start_time >= alert_duration_threshold:
            # Alert triggered - violence above threshold for 5+ seconds
            cv2.putText(annotated_frame, "ALERT", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            
            # Track alert duration
            if current_alert_start is not None:
                total_alert_duration += current_time - current_alert_start
                current_alert_start = current_time
        else:
            # Violence above threshold but not long enough yet
            remaining_time = alert_duration_threshold - (current_time - alert_start_time)
            cv2.putText(annotated_frame, f"Alert in: {remaining_time:.1f}s", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        # Violence below threshold - reset alert timer
        if alert_start_time is not None and current_time - alert_start_time >= alert_duration_threshold:
            # This was a completed alert
            total_alerts += 1
        
        alert_start_time = None
        current_alert_start = None
        cv2.putText(annotated_frame, "No alerts yet", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display alert statistics
    cv2.putText(annotated_frame, f"Total Alerts: {total_alerts}", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Alert Duration: {total_alert_duration:.1f}s", (20, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw bounding boxes for all detected people
    for i, detection in enumerate(frame_result["detections"]):
        bbox = detection["bbox"]
        confidence = detection["confidence"]
        x1, y1, x2, y2 = bbox
        
        # Use violence score to color boxes
        if violence_score < 0.4:
            box_color = (0, 255, 0)  # Green
        elif violence_score < 0.7:
            box_color = (0, 165, 255)  # Orange
        else:
            box_color = (0, 0, 255)  # Red
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Add person label and confidence
        cv2.putText(annotated_frame, f"Person {i+1}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Conf: {confidence:.2f}", (x1, y2+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw keypoints if available
        if "keypoints" in detection and detection["keypoints"]:
            keypoints = detection["keypoints"]
            for kp in keypoints:
                if len(kp) >= 2:
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(annotated_frame, (x, y), 3, (0, 255, 255), -1)

    return annotated_frame

if __name__ == "__main__":
    print("=== Simple Violence Detection Processor ===")
    print(f"Video: {VIDEO_PATH}")
    print(f"Frame skip: {FRAME_SKIP}")
    print("=" * 50)
    
    # Load models first
    load_models()
    
    # Process video
    process_video()
