# Violence Detection V2

A deep learning-based system for detecting violent behavior in video streams using pose estimation and computer vision techniques.

## Project Overview

This project implements a real-time violence detection system that combines:
- **YOLO11M-Pose**: For human pose estimation and keypoint detection
- **Custom CNN + Transformer**: For violence classification using pose features and image analysis
- **Real-time Processing**: Video stream analysis with configurable frame rates

## Project Structure

```
Violence Detection V2/
├── CLIENT/                          # Runtime detection system
│   ├── detection_runner.py         # Main video processing script
│   └── requirements.txt            # Runtime dependencies
├── MODELS/                          # Pre-trained models
│   ├── violence_model.pt           # Trained violence classifier
│   └── yolo11m-pose.pt            # YOLO pose estimation model
├── TRAINING/                        # Model training scripts
│   ├── train_violence_detection.py # Training pipeline
│   └── requirements.txt            # Training dependencies
└── testing_files/                   # Test videos and validation data
    ├── crowd*.mp4                  # Crowd scene test videos
    └── validation/                 # Fight scene validation videos
```

## Quick Start

### Prerequisites

#### For Runtime Detection (CLIENT)
```bash
cd CLIENT
pip install -r requirements.txt
```

#### For Training (TRAINING)
```bash
cd TRAINING
pip install -r requirements.txt
```

#### Manual Installation (Alternative)
```bash
pip install torch>=2.0.0 torchvision>=0.15.0 opencv-python>=4.8.0 numpy>=1.24.0 ultralytics>=8.0.0 timm>=0.9.0
```

### Running Detection
1. **Configure paths** in `CLIENT/detection_runner.py`:
   ```python
   VIDEO_PATH = "path/to/your/video.mp4"
   YOLO_MODEL_PATH = "MODELS/yolo11m-pose.pt"
   VIOLENCE_MODEL_PATH = "MODELS/violence_model.pt"
   ```

2. **Run detection**:
   ```bash
   cd CLIENT
   python detection_runner.py
   ```
3. Here is a link to test videos:
    https://www.dropbox.com/scl/fo/1bwsmg8a882nqemau9wro/AEEeT9scdAgfX0rBOnd3TB4?rlkey=klhgmtioxg5sw432w0qtkju34&st=c422vn85&dl=0


## Key Features

### Detection System
- **Real-time processing** with configurable frame skip rates
- **Multi-person pose analysis** (up to 4 people simultaneously)
- **Violence probability scoring** with configurable thresholds
- **Advanced alert system** with duration tracking and statistics

### Alert System
The system implements a sophisticated multi-level alert mechanism:

#### **Alert Triggers**
- **Violence Score Threshold**: Default 0.8 (80% confidence)
- **Duration Threshold**: Default 0.5 seconds (minimum sustained violence)
- **Real-time Monitoring**: Continuous frame-by-frame analysis

#### **Alert States**
1. **No Alert** (Green): Violence score < 0.4
2. **Warning** (Orange): Violence score 0.4-0.7
3. **Alert Pending** (Orange): Score ≥ 0.8, counting down to trigger
4. **ALERT ACTIVE** (Red): Score ≥ 0.8 for ≥ 0.5 seconds

#### **Alert Logic**
- **Activation**: Violence score must remain above threshold for specified duration
- **Deactivation**: Automatically resets when score drops below threshold
- **Statistics Tracking**: Counts total alerts and cumulative alert duration
- **Visual Feedback**: Color-coded bounding boxes and on-screen alerts

#### **Configuration Options**
```python
alert_threshold = 0.8              # Violence probability to trigger alert
alert_duration_threshold = 0.5     # Seconds above threshold to activate
```

### Model Architecture
- **EnhancedViolenceClassifier**: Combines pose features with image analysis
- **Pose encoder**: Processes 325-dimensional pose features
- **Self-attention mechanisms**: For pose and cross-modal feature fusion
- **Residual connections**: Improves gradient flow and performance

### Training Pipeline
- **Data augmentation**: Albumentations for robust training
- **Class balancing**: Weighted sampling for imbalanced datasets
- **Multi-GPU support**: Distributed training capabilities
- **Pose feature caching**: Pre-computed features for faster training

## Model Performance

The system processes video frames and outputs:
- **Violence probability** (0-1 scale)
- **Alert tracking** with configurable thresholds
- **Real-time statistics** including total alerts and duration
- **Visual overlays** on processed video frames

## Test Videos

- **Crowd scenes**: 8 test videos for general crowd behavior analysis
- **Validation set**: 60+ fight scene videos for model evaluation
- **Real-world scenarios**: Various lighting and crowd density conditions

## Configuration

### Detection Parameters
- `FRAME_SKIP`: Process every Nth frame (default: 1)
- `alert_threshold`: Violence probability threshold (default: 0.8)
- `alert_duration_threshold`: Minimum alert duration (default: 0.5s)

### Alert System Parameters
- **`alert_threshold`**: Violence probability (0.0-1.0) that triggers alert system
- **`alert_duration_threshold`**: Minimum time (seconds) above threshold to activate alert
- **Risk Level Thresholds**:
  - **Low Risk**: 0.0-0.4 (Green indicators)
  - **Medium Risk**: 0.4-0.7 (Orange indicators)  
  - **High Risk**: 0.7-1.0 (Red indicators)

### Model Parameters
- **Pose features**: 325 dimensions including keypoints, spatial relationships
- **Image backbone**: EfficientNet-B2 (pretrained) or custom CNN
- **Attention heads**: 8-head multi-head attention
- **Dropout rates**: Progressive dropout (0.2-0.4) for regularization

## Training

### Data Preparation
1. Organize training images in appropriate directories
2. Ensure pose features are pre-computed and cached
3. Configure data augmentation parameters

### Training Command
```bash
cd TRAINING
python train_violence_detection.py
```

## Usage Examples

### Basic Video Analysis
```python
# Process a single video file
python detection_runner.py

# Output includes:
# - Real-time violence probability
# - Alert notifications
# - Processed video with overlays
# - Statistical summary
```

### Custom Configuration
```python
# Modify detection parameters
FRAME_SKIP = 2          # Process every 2nd frame
alert_threshold = 0.7   # Lower sensitivity
alert_duration_threshold = 1.0  # Longer alert duration

# Alert system examples
alert_threshold = 0.6        # More sensitive (triggers at 60% violence)
alert_threshold = 0.9        # Less sensitive (triggers at 90% violence)
alert_duration_threshold = 0.3  # Faster alerts (0.3 seconds)
alert_duration_threshold = 2.0  # Slower alerts (2.0 seconds)
```

## Technical Details

### Pose Features (325 dimensions)
- **Keypoint coordinates**: 17 keypoints × 3 (x, y, confidence)
- **Spatial relationships**: Distances between keypoints
- **Movement patterns**: Velocity and acceleration features
- **Violence indicators**: Specific pose combinations

### Model Architecture
- **Input**: Pose features + RGB images
- **Processing**: Multi-modal fusion with attention
- **Output**: Binary classification (violent/non-violent)
- **Optimization**: Mixed precision training, gradient scaling

## Requirements

### System Requirements
- **Python**: 3.8+
- **PyTorch**: 2.0+ (CUDA 11.8+ recommended for GPU acceleration)
- **Memory**: 8GB+ RAM, 4GB+ VRAM recommended

### Runtime Dependencies (CLIENT)
- opencv-python>=4.8.0
- numpy>=1.24.0
- torch>=2.0.0
- torchvision>=0.15.0
- ultralytics>=8.0.0
- timm>=0.9.0

### Training Dependencies (TRAINING)
- All runtime dependencies plus:
- scikit-learn>=1.3.0
- tqdm>=4.65.0
- albumentations>=1.3.0


## Support

For issues or questions:
1. Check the configuration parameters
2. Verify model file paths
3. Ensure sufficient system resources
4. Review video format compatibility

---

**Note**: This system is designed for research and educational purposes. Always ensure compliance with privacy laws and ethical guidelines when processing video content.
