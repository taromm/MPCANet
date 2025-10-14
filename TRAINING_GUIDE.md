# MPCANet Training and Testing Guide

## 🚀 Training the Model

1. **Configure paths** in `train.py`:
   ```python
   train_root = '/path/to/RGB/'           # RGB images
   gt_root = '/path/to/GT/'               # Ground truth
   thermal_root = '/path/to/Thermal/'     # Thermal images
   save_path = '/path/to/save/checkpoints/'
   ```

2. **Download pre-trained weights**:
   ```bash
   # Download swin_base_patch4_window12_384_22k.pth to project root
   ```

3. **Start training**:
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

### Training Output

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
```

## 🐛 Common Issues

**Q: Out of memory during training?**  
A: Reduce the `batchsize` parameter, for example, from 8 to 4 or 2.

**Q: RGB and Thermal image counts don't match?**  
A: Check the dataset to ensure images in both folders are named in the same order and have the same quantity.

**Q: Cannot find pre-trained weights?**  
A: Download `swin_base_patch4_window12_384_22k.pth` and place it in the project root directory, or modify the path in the code.

**Q: Test results are all black or all white?**  
A: Adjust the `THRESHOLD` parameter in `test.py` (default 0.5), try values between 0.3-0.7.

## 📝 Notes

1. **Memory Requirements**: It is recommended to use a GPU with at least 16GB VRAM for training (batchsize=8)
2. **Image Pairing**: Ensure RGB and Thermal images are strictly paired and have the same quantity
3. **Image Format**: Supports `.jpg` and `.png` formats
4. **Thermal Infrared Images**: Model accepts single-channel grayscale thermal infrared images
5. **Numerical Stability**: Random seed (seed=42) is set during training to ensure reproducibility
