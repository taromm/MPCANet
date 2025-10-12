# TMSOD: Thermal-Visible Multimodal Salient Object Detection Network

## 📋 Model Introduction

TMSOD (Thermo-Modal Salient Object Detection) is a deep learning-based dual-modal salient object detection model specifically designed to fuse RGB visible light images and thermal infrared images for accurate salient object detection.

## 🎯 Application Scenarios

This model is suitable for the following application scenarios:

- **Night Target Detection and Tracking**: Combining thermal infrared images to improve target detection accuracy in low-light or nighttime environments
- **Complex Environment Monitoring**: Target recognition in low-visibility environments such as smoke and haze
- **Autonomous Driving**: Multimodal sensor fusion to enhance pedestrian and vehicle detection under various lighting and weather conditions
- **Security Monitoring**: All-weather intrusion detection and abnormal behavior recognition
- **Search and Rescue Missions**: Quickly locating vital signs using thermal infrared features at disaster sites
- **Industrial Inspection**: Equipment anomaly detection and quality control combining visible light and thermal infrared information
- **Medical Imaging**: Lesion area segmentation in multimodal medical images

## 🏗️ Model Architecture

TMSOD adopts an advanced dual-branch encoder-decoder architecture, mainly including the following core modules:

### Core Components

1. **Dual-Branch Encoder**
   - RGB Branch: Swin Transformer-based visible light feature extraction
   - Thermal Branch: Feature extraction adapted for single-channel thermal infrared images
   - Multi-scale Feature Pyramid: Extracting semantic information at different resolutions

2. **TPMA: Thermal Physical Modulation Attention**
   - Physical Prior Encoding: Extracting thermal edge, diffusion, inertia, and emissivity physical descriptors
   - Physical-Guided Asymmetric Attention: Enhancing feature representation using thermal physical priors

3. **TSM-CWI: Thermal Saliency Modulated Cross-Window Interaction**
   - Saliency-Aware Dynamic Windows: Adaptively adjusting attention windows based on saliency maps
   - Semantic-Guided Deformable Alignment: Precisely aligning RGB and thermal infrared features

4. **BS-CCD: Boundary-Semantic Coupled Cascade Decoder**
   - Edge-Aware Skip Connection Fusion: Predicting edge maps and using them as attention to purify features
   - Semantic Gating: Re-weighting decoder features using saliency priors
   - Multi-scale Cascade Decoding: Progressively recovering high-resolution saliency maps

5. **MOCO: Multi-Objective Consistency Optimization**
   - Edge Alignment Loss: Ensuring consistency between edge predictions and final saliency maps
   - Cross-Modal Alignment Consistency Loss: Supervising aligned feature quality

### Network Features

- ✅ End-to-end training
- ✅ Multi-GPU parallel training support
- ✅ Mixed precision training (AMP) acceleration
- ✅ Physics-guided cross-modal feature fusion
- ✅ Boundary-aware decoder design

## 📦 Environment Dependencies

```bash
torch>=1.10.0
torchvision>=0.11.0
numpy
Pillow
tqdm
timm
```

Install dependencies:
```bash
pip install torch torchvision numpy Pillow tqdm timm
```

## 📁 Data Preparation

Training and testing data should be organized as follows:

```
dataset/
├── RGB/                # RGB visible light images
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── Thermal/            # Thermal infrared images
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── GT/                 # Ground truth annotations (required for training only)
    ├── 1.png
    ├── 2.png
    └── ...
```

## 🚀 Training the Model

### 1. Configure Training Parameters

Edit the `train.py` file and set the following paths and parameters:

```python
train_root = '/path/to/RGB/'           # RGB image path
gt_root = '/path/to/GT/'               # Ground truth annotation path
thermal_root = '/path/to/Thermal/'     # Thermal infrared image path
save_path = '/path/to/save/checkpoints/'  # Model save path

trainsize = 384        # Training image size
batchsize = 8          # Batch size
base_lr = 1e-5         # Base learning rate
num_epochs = 200       # Number of epochs
```

### 2. Download Pre-trained Weights

Download Swin Transformer pre-trained weights:
```bash
# Download swin_base_patch4_window12_384_22k.pth
# Place in the project root directory
```

### 3. Start Training

```bash
python train.py
```

**Training Features:**
- Automatic detection and use of multiple GPUs (if available)
- Mixed precision training acceleration
- Cosine annealing learning rate scheduling
- Automatic saving of training logs (CSV format)
- Model checkpoint saving every epoch
- Real-time display of training loss, validation metrics (MAE, F-measure, S-measure, E-measure)

### 4. Training Output

During training, the following will be generated:
- `best_model_{epoch}.pth`: Model weights for each epoch
- `training_log_1.csv`: Detailed training logs

Training logs include:
- Epoch number
- Training loss
- Validation loss
- MAE (Mean Absolute Error)
- F-measure (F-score)
- S-measure (Structural Similarity)
- E-measure (Enhanced Alignment)
- Consistency loss

## 🔍 Testing the Model

### 1. Configure Test Parameters

Edit the `test.py` file and set the following parameters:

```python
RGB_ROOT = '/path/to/test/RGB/'        # Test RGB image path
THERMAL_ROOT = '/path/to/test/Thermal/'  # Test thermal infrared image path
WEIGHTS_PATH = '/path/to/best_model.pth'  # Trained model weights
SAVE_DIR = '/path/to/save/results/'    # Prediction result save path

TEST_SIZE = 384        # Test image size
THRESHOLD = 0.5        # Binarization threshold
```

### 2. Run Inference

```bash
python test.py
```

### 3. Output Results

- Predicted saliency maps will be saved in PNG format (0-255 grayscale)
- File naming format: `1.png`, `2.png`, ...
- Terminal displays inference progress and completion information

## 📊 Model Performance

Model performance on RGB-T salient object detection datasets:

- **MAE (Mean Absolute Error)**: Lower is better
- **F-measure**: Comprehensive consideration of precision and recall
- **S-measure**: Structural similarity measure
- **E-measure**: Enhanced alignment measure

Please refer to the training log files for specific performance metrics.

## 🔧 Advanced Usage

### Multi-GPU Training

The model automatically detects the number of available GPUs:
- 2 or more GPUs: Automatically enables DataParallel training
- 1 GPU: Single GPU training
- No GPU: CPU training (not recommended, slow)

### Custom Window Size

Window size of TSM-CWI module can be adjusted before training or testing:

```python
# In train.py or test.py
model.MSA4_r.window_size2 = 4  # Adjust RGB branch window
model.MSA4_t.window_size2 = 4  # Adjust Thermal branch window
```

### Loss Function Weight Adjustment

Different loss weights can be adjusted in `train.py`:

```python
# Main loss
criterion = CombinedLoss(weight_dice=0.5, weight_bce=0.5)

# Consistency loss weight
total_loss = loss + 0.1 * consistency_loss  # 0.1 is the consistency loss weight
```

## 📝 Notes

1. **Memory Requirements**: It is recommended to use a GPU with at least 16GB VRAM for training (batchsize=8)
2. **Image Pairing**: Ensure RGB and Thermal images are strictly paired and have the same quantity
3. **Image Format**: Supports `.jpg` and `.png` formats
4. **Thermal Infrared Images**: Model accepts single-channel grayscale thermal infrared images
5. **Numerical Stability**: Random seed (seed=42) is set during training to ensure reproducibility

## 🐛 Common Issues

**Q: Out of memory during training?**  
A: Reduce the `batchsize` parameter, for example, from 8 to 4 or 2.

**Q: RGB and Thermal image counts don't match?**  
A: Check the dataset to ensure images in both folders are named in the same order and have the same quantity.

**Q: Cannot find pre-trained weights?**  
A: Download `swin_base_patch4_window12_384_22k.pth` and place it in the project root directory, or modify the path in the code.

**Q: Test results are all black or all white?**  
A: Adjust the `THRESHOLD` parameter in `test.py` (default 0.5), try values between 0.3-0.7.

## 📄 Citation

If this model is helpful for your research, please cite the related work.

## 📧 Contact

If you have any questions or suggestions, please feel free to submit an Issue or Pull Request.

Author contact: amdusia@outlook.com
---
