# TMSOD: 热-可见光多模态显著目标检测网络

## 📋 模型简介

TMSOD (Thermo-Modal Salient Object Detection) 是一个基于深度学习的双模态显著目标检测模型，专门设计用于融合RGB可见光图像和热红外图像进行精确的显著目标检测。

## 🎯 应用场景

本模型适用于以下应用场景：

- **夜间目标检测与追踪**：在低光照或夜间环境下，结合热红外图像提高目标检测准确率
- **复杂环境监控**：在烟雾、雾霾等能见度低的环境中进行目标识别
- **自动驾驶**：多模态传感器融合，提升在各种光照和天气条件下的行人和车辆检测
- **安防监控**：全天候的入侵检测和异常行为识别
- **搜救任务**：在灾难现场利用热红外特征快速定位生命体征
- **工业检测**：结合可见光和热红外信息进行设备异常检测和质量控制
- **医学影像**：多模态医学图像的病灶区域分割

## 🏗️ 模型架构

TMSOD采用先进的双分支编码器-解码器架构，主要包含以下核心模块：

### 核心组件

1. **双分支编码器 (Dual-Branch Encoder)**
   - RGB分支：基于Swin Transformer的可见光特征提取
   - Thermal分支：适配单通道热红外图像的特征提取
   - 多尺度特征金字塔：提取不同分辨率的语义信息

2. **TPMA: 热物理调制注意力机制**
   - 物理先验编码：提取热边缘、扩散、惯性和发射率等物理描述子
   - 物理引导非对称注意力：利用热物理先验增强特征表达

3. **TSM-CWI: 热显著性调制交叉窗口交互**
   - 显著性感知动态窗口：根据显著性图自适应调整注意力窗口
   - 语义引导可变形对齐：精确对齐RGB和热红外特征

4. **BS-CCD: 边界-语义耦合级联解码器**
   - 边缘感知跳跃连接融合：预测边缘图并作为注意力净化特征
   - 语义门控：利用显著性先验重加权解码器特征
   - 多尺度级联解码：逐步恢复高分辨率显著性图

5. **MOCO: 多目标一致性优化**
   - 边缘对齐损失：确保边缘预测与最终显著性图一致
   - 跨模态对齐一致性损失：监督对齐特征质量

### 网络特点

- ✅ 端到端训练
- ✅ 支持多GPU并行训练
- ✅ 混合精度训练（AMP）加速
- ✅ 物理引导的跨模态特征融合
- ✅ 边界感知的解码器设计

## 📦 环境依赖

```bash
torch>=1.10.0
torchvision>=0.11.0
numpy
Pillow
tqdm
timm
```

安装依赖：
```bash
pip install torch torchvision numpy Pillow tqdm timm
```

## 📁 数据准备

训练和测试数据应按以下结构组织：

```
dataset/
├── RGB/                # RGB可见光图像
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── Thermal/            # 热红外图像
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── GT/                 # 真值标注（仅训练时需要）
    ├── 1.png
    ├── 2.png
    └── ...
```

## 🚀 训练模型

### 1. 配置训练参数

编辑 `train.py` 文件，设置以下路径和参数：

```python
train_root = '/path/to/RGB/'           # RGB图像路径
gt_root = '/path/to/GT/'               # 真值标注路径
thermal_root = '/path/to/Thermal/'     # 热红外图像路径
save_path = '/path/to/save/checkpoints/'  # 模型保存路径

trainsize = 384        # 训练图像尺寸
batchsize = 8          # 批次大小
base_lr = 1e-5         # 基础学习率
num_epochs = 200       # 训练轮数
```

### 2. 下载预训练权重

下载Swin Transformer预训练权重：
```bash
# 下载 swin_base_patch4_window12_384_22k.pth
# 放置在项目根目录
```

### 3. 开始训练

```bash
python train.py
```

**训练特性：**
- 自动检测并使用多GPU（如果可用）
- 混合精度训练加速
- 余弦退火学习率调度
- 自动保存训练日志（CSV格式）
- 每个epoch保存模型检查点
- 实时显示训练损失、验证指标（MAE、F-measure、S-measure、E-measure）

### 4. 训练输出

训练过程中会生成：
- `best_model_{epoch}.pth`：每个epoch的模型权重
- `training_log_1.csv`：详细的训练日志

训练日志包含：
- Epoch编号
- 训练损失
- 验证损失
- MAE（平均绝对误差）
- F-measure（F值）
- S-measure（结构相似度）
- E-measure（增强对齐度）
- 一致性损失

## 🔍 测试模型

### 1. 配置测试参数

编辑 `test.py` 文件，设置以下参数：

```python
RGB_ROOT = '/path/to/test/RGB/'        # 测试RGB图像路径
THERMAL_ROOT = '/path/to/test/Thermal/'  # 测试热红外图像路径
WEIGHTS_PATH = '/path/to/best_model.pth'  # 训练好的模型权重
SAVE_DIR = '/path/to/save/results/'    # 预测结果保存路径

TEST_SIZE = 384        # 测试图像尺寸
THRESHOLD = 0.5        # 二值化阈值
```

### 2. 运行推理

```bash
python test.py
```

### 3. 输出结果

- 预测的显著性图将保存为PNG格式（0-255灰度图）
- 文件命名格式：`1.png`, `2.png`, ...
- 终端显示推理进度和完成信息

## 📊 模型性能

模型在RGB-T显著目标检测数据集上的表现：

- **MAE (Mean Absolute Error)**：越低越好
- **F-measure**：综合考虑精确率和召回率
- **S-measure**：结构相似度度量
- **E-measure**：增强对齐度度量

具体性能指标请参考训练日志文件。

## 🔧 高级用法

### 多GPU训练

模型会自动检测可用GPU数量：
- 2个或以上GPU：自动启用DataParallel并行训练
- 1个GPU：单GPU训练
- 无GPU：CPU训练（不推荐，速度慢）

### 自定义窗口大小

在训练或测试前可以调整TSM-CWI模块的窗口大小：

```python
# 在train.py或test.py中
model.MSA4_r.window_size2 = 4  # 调整RGB分支窗口
model.MSA4_t.window_size2 = 4  # 调整Thermal分支窗口
```

### 损失函数权重调整

在 `train.py` 中可以调整不同损失的权重：

```python
# 主损失
criterion = CombinedLoss(weight_dice=0.5, weight_bce=0.5)

# 一致性损失权重
total_loss = loss + 0.1 * consistency_loss  # 0.1为一致性损失权重
```

## 📝 注意事项

1. **内存要求**：建议使用至少16GB显存的GPU进行训练（batchsize=8）
2. **图像配对**：确保RGB和Thermal图像严格配对且数量一致
3. **图像格式**：支持`.jpg`和`.png`格式
4. **热红外图像**：模型接受单通道灰度热红外图像
5. **数值稳定性**：训练时已设置随机种子（seed=42）保证可复现性

## 🐛 常见问题

**Q: 训练时显存不足怎么办？**  
A: 减小`batchsize`参数，例如从8改为4或2。

**Q: RGB和Thermal图像数量不匹配？**  
A: 检查数据集，确保两个文件夹中的图像按相同顺序命名且数量一致。

**Q: 找不到预训练权重？**  
A: 下载`swin_base_patch4_window12_384_22k.pth`并放在项目根目录，或在代码中修改路径。

**Q: 测试结果全黑或全白？**  
A: 调整`test.py`中的`THRESHOLD`参数（默认0.5），尝试0.3-0.7之间的值。

## 📄 引用

如果本模型对您的研究有帮助，欢迎引用相关工作。

## 📧 联系方式

如有问题或建议，欢迎提出Issue或Pull Request。

作者联系方式：amdusia@outlook.com
---

**祝您使用愉快！** 🎉

