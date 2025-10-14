# MPCANet: Multi-Physical Prior Guided Cross-Modal Attention Network for RGB-T Salient Object Detection

## Abstract

RGB-T salient object detection (SOD) fuses the fine details of RGB with the robustness of thermal infrared (TIR) to segment salient objects across diverse scenes and lighting conditions. However, existing approaches face three main challenges: (i) thermal noise and registration errors contaminate attention at early fusion, (ii) cross-modal misalignment and detail preservation are difficult to balance, and (iii) thermophysical properties (e.g., heat diffusion and inertia) are under-modeled, leading to instability in low-contrast and dynamic settings. Thermophysical priors could provide interpretable cues on boundaries, diffusion, inertia, and material differences, suppressing noise at the source and strengthening structural reliability. When combined with high-level semantics that provide global task relevance and region-level attention, interaction is reinforced only in semantically relevant and physically reliable areas. Building on this principle, we propose a semantics–physics integrated cross-modal framework. The key idea is to inject thermophysical priors into attention logits before normalization to suppress upstream thermal noise, while semantic saliency dynamically controls the receptive field and sampling for bi-directional alignment, balancing global consistency with local sharpness. During decoding process, a boundary–semantic coupling restores fine structures. Extensive experiments on public benchmarks demonstrate state-of-the-art performance with sharper boundaries and more coherent regions.

## 📋 Model Introduction

MPCANet (Multi-Physical Prior Guided Cross-Modal Attention Network) is a deep learning-based dual-modal salient object detection model specifically designed to fuse RGB visible light images and thermal infrared images for accurate salient object detection.

## 🏗️ Model Architecture

MPCANet adopts an advanced dual-branch encoder-decoder architecture, mainly including the following core modules:

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

## 🚀 Quick Start

For detailed training and testing instructions, please refer to the **[Training Guide](TRAINING_GUIDE.md)**.

### Quick Training
```bash
# 1. Configure paths in train.py
# 2. Download pre-trained weights
# 3. Start training
python train.py
```

### Quick Testing
```bash
# 1. Configure paths in test.py
# 2. Run inference
python test.py
```

## 📄 Citation

If this model is helpful for your research, please cite the related work.

## 📧 Contact

If you have any questions or suggestions, please feel free to submit an Issue or Pull Request.

Author contact: amdusia@outlook.com
---
