# Violence Detection

<table>
  <thead>
    <tr>
      <th>Fight Detection</th>
      <th>Fight Detection</th>
      <th>Non-violent Scene</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <video controls autoplay muted loop playsinline
               src="https://github.com/user-attachments/assets/11671d14-5f63-44a0-9491-071c9a82d811"
               style="width:100%; max-width:320px;"></video>
      </td>
      <td>
        <video controls autoplay muted loop playsinline
               src="https://github.com/user-attachments/assets/d8e95cc2-35f1-4e4e-8012-dc10e8d77c96"
               style="width:100%; max-width:320px;"></video>
      </td>
      <td>
        <video controls autoplay muted loop playsinline
               src="https://github.com/user-attachments/assets/cf662126-b0cd-4ea5-ab15-2005e34a2e66"
               style="width:100%; max-width:320px;"></video>
      </td>
    </tr>
  </tbody>
</table>

A deep learning-based system for detecting violent behavior in video streams using pose estimation and computer vision techniques.

## Project Overview

This project implements a real-time violence detection system that combines:
- **YOLO11M-Pose**: For human pose estimation and keypoint detection
- **Custom CNN + Transformer**: For violence classification using pose features and image analysis
- **Real-time Processing**: Video stream analysis with configurable frame rates

## Project Structure

```
Violence Detection/
├── CLIENT/                          # Runtime detection system
│   ├── detection_runner.py         # Main video processing script / runner
│   └── requirements.txt            # Runtime dependencies
├── asset_videos/                    # Embedded demo clips referenced in README
│   ├── Bike_Edited.mp4
│   ├── fightone_edited.mp4
│   └── Non_Violence.mp4
├── MODELS/                          # Pre-trained models
│   ├── violence_model.pt           # Trained violence classifier
│   └── yolo11m-pose.pt            # YOLO pose estimation model
├── TRAINING/                        # Model training scripts
│   ├── train_rgb_pose.py           # RGB + pose fusion training pipeline
│   └── requirements.txt            # Training dependencies
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

#### Slicing Long Videos into 5-Second Chunks (FFmpeg)

This repo includes a helper script to split long videos into fixed-length chunks for dataset prep:

```bash
cd TRAINING
python slice_video_ffmpeg.py "path/to/video.mp4"
```

By default, chunks are 5 seconds and are written to:

`./slices/<video_stem>/`

#### Common options

```bash
# Choose output directory
python slice_video_ffmpeg.py "path/to/video.mp4" --out-dir "path/to/output_chunks"

# Change chunk size
python slice_video_ffmpeg.py "path/to/video.mp4" --seconds 5

# Overwrite existing output files
python slice_video_ffmpeg.py "path/to/video.mp4" --overwrite

# More accurate cuts (slower): re-encode instead of stream copy
python slice_video_ffmpeg.py "path/to/video.mp4" --reencode

# If ffmpeg is not on PATH
python slice_video_ffmpeg.py "path/to/video.mp4" --ffmpeg-path "C:\\path\\to\\ffmpeg.exe"
```

Note: Without `--reencode`, FFmpeg uses stream copy (`-c copy`), which is very fast but segment boundaries may align to existing keyframes.

#### Manual Installation (Alternative)
```bash
pip install torch>=2.0.0 torchvision>=0.15.0 opencv-python>=4.8.0 numpy>=1.24.0 ultralytics>=8.0.0 timm>=0.9.0
```

### Running Detection (Local Realtime)

1. **Install runtime deps** (inside the repo root):
   ```bash
   cd CLIENT
   pip install -r requirements.txt
   ```

2. **Webcam stream (default source 0):**
   ```bash
   python detection_runner.py
   ```
   - Press `q` to exit.

3. **Analyze a specific video file:**
   ```bash
   python detection_runner.py --video "path/to/video.mp4"
   ```

4. **Save the annotated output:**
   ```bash
   python detection_runner.py --source 0 --output "output.mp4"
   # or for a file
   python detection_runner.py --video "path/to/video.mp4" --output "output.mp4"
   ```

5. **Tune performance:** use `--frame-skip N` to process every Nth frame (default 1).

### Detection Runner Details (CLIENT/detection_runner.py)

`detection_runner.py` is the single entry point for realtime inference. It:

- Loads the YOLO11M-Pose backbone and the fusion classifier from `MODELS/`
- Accepts webcam (`--source`), file (`--video`) or RTSP (`--url`) inputs
- Exposes CLI flags for frame skipping, output recording, alert thresholds, and probability smoothing
- Streams annotated frames while tracking alert state transitions and emitting console stats

Common runner flags:

```bash
# Custom camera index and frame skipping
python detection_runner.py --source 1 --frame-skip 2

# Override alert sensitivity
python detection_runner.py --alert-threshold 0.75 --alert-duration 0.3

# Read from RTSP and record output
python detection_runner.py --url "rtsp://<camera>" --output recorded.mp4
```

All arguments are optional; defaults target a standard webcam feed with conservative alerting. Review `CLIENT/detection_runner.py` for additional toggles (smoothing window, max people tracked, visualization colors, etc.).


## Key Features

### Detection System
- **Real-time processing** with configurable frame skip rates
- **Multi-person pose analysis** (up to 4 people simultaneously)
- **Violence probability scoring** with configurable thresholds
- **Advanced alert system** with duration tracking and statistics

### Alert System
The real-time runner now mirrors the production logic implemented in `CLIENT/detection_runner.py`:

- **Adaptive thresholding**: `alert_threshold` defaults to the value stored in `MODELS/inference_threshold.txt`, which is produced by the training threshold sweep. If the file is absent, it falls back to `0.5`.
- **Temporal smoothing**: A rolling window (`SMOOTHING_WINDOW = 8`) averages logits before comparing them to the alert threshold to suppress single-frame spikes.
- **Duration gating**: An alert is only declared if the smoothed score remains above `alert_threshold` for `alert_duration_threshold` seconds (default `0.5`).
- **Statistics**: The runner tracks `total_alerts`, cumulative `total_alert_duration`, and surfaces “Alert in …” countdowns while the timer is arming.
- **States on HUD**: When the score is below threshold the overlay shows “No alerts yet.” Above threshold it displays either a countdown or an “ALERT” banner once the dwell time is satisfied.

To tweak behavior at runtime use flags such as:

```bash
python detection_runner.py --alert-threshold 0.7 --alert-duration 0.3 --frame-skip 2
```

Internally, the runner exposes additional knobs (`SMOOTHING_WINDOW`, webcam source, RTSP URL) so power users can tailor responsiveness to their deployment scenario.

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

### Training Pipelines

#### `train_rgb_pose.py`

The production trainer (`TRAINING/train_rgb_pose.py`) builds the exact RGBPoseFusionV2 weights consumed by the runner. Highlights:

1. **Caching + preprocessing**
   - `precompute_dataset()` runs YOLO11M-Pose over every clip to store skeleton tensors in `TRAINING/caches/pose_cache_lstm_seq20_k2_v2/` and RGB crops in `TRAINING/caches/rgb_cache_5frames_size112_v2/`.
   - Sampling grabs 20 evenly spaced pose frames per video and 5 RGB frames (10–90% span) resized to 112×112.

2. **Model + optimizer schedule**
   - Dual-stream architecture: Pose LSTM (hidden 512, 2 layers + attention) fused with MobileNetV3-Small embeddings.
   - Phase 1 (epochs 0–9): MobileNet stays frozen; only pose/fusion heads train with AdamW (`lr=1e-3`).
   - Phase 2 (epoch 10+): Last three MobileNet blocks unfreeze; optimizer adds a low `1e-4` LR group for backbone fine-tuning.
   - Mixed precision via `torch.amp.GradScaler` and ReduceLROnPlateau scheduler.

3. **Losses + sampling**
   - WeightedRandomSampler balances violence/non-violence clips.
   - Custom `FocalLoss(alpha=0.75, gamma=2)` combats class imbalance.

4. **Validation + artifacts**
   - After each epoch, validation metrics (precision/recall/F1, FP rate) are logged to TensorBoard.
   - Best F1 checkpoint saved to `MODELS/best_violence_rgb_pose_v2.pt`.
   - Post-training, a 3-pass TTA evaluation (normal, flip, noise) runs a threshold sweep (0.30–0.84) capped at 20% FP rate. The chosen threshold is written to `MODELS/inference_threshold.txt` and a `MODELS/tta_enabled.txt` flag is set so the runner knows TTA-calibrated weights are in use.

Before training, update the constants at the top of the script (`DATASET_PATH`, cache dirs, batch size) to match your environment, then run:

```bash
cd TRAINING
python train_rgb_pose.py
```

Expected dataset layout:

```
training_datasets/
├── violence/
└── non_violence/
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
