# 噪声空间图像融合

这个功能允许您在DDPM的可编辑噪声空间中融合两张图像，创建兼具两图特征的混合效果。相比传统图像融合方法，噪声空间融合能够保留更多的语义结构和视觉特性。

## 原理

传统图像融合通常在像素空间或特征空间中进行，而本工具利用论文"An Edit Friendly DDPM Noise Space"提出的可编辑噪声空间进行融合。这种方法具有以下优势：

1. **保留语义内容**：融合过程考虑图像的语义层次结构
2. **自然过渡**：避免传统混合中可能出现的伪影
3. **多样融合模式**：支持从简单线性混合到复杂的特征选择性融合

## 使用方法

### 基本用法

```bash
python main_run.py --mode="image_fusion" --dataset_yaml="fusion_examples.yaml" --second_image="/example_images/second_image.jpg" --fusion_ratio=0.5 --fusion_mode="linear"
```

### 必需参数

- `--mode="image_fusion"`: 指定使用图像融合模式
- `--dataset_yaml`: YAML配置文件，指定第一张图像和提示词
- `--second_image`: 第二张图像的路径

### 可选参数

- `--fusion_ratio`: 融合比例（0.0-1.0），默认为0.5
  - 0.0: 完全是第一张图像
  - 0.5: 均等混合
  - 1.0: 完全是第二张图像
- `--fusion_mode`: 融合模式，可选值:
  - `linear`: 简单线性混合
  - `cross_fade`: 渐变混合，不同时间步使用不同权重
  - `feature_selective`: 特征选择性融合，选择更显著的特征
  - `layer_selective`: 层选择性融合，不同层次使用不同策略
- `--skip`: 跳过的时间步数，默认为36

## 融合模式详解

### linear（线性混合）

最简单的混合方式，在所有时间步上使用相同的融合比例直接混合两个噪声向量。

```bash
python main_run.py --mode="image_fusion" --dataset_yaml="fusion_examples.yaml" --second_image="/example_images/second_image.jpg" --fusion_ratio=0.3 --fusion_mode="linear"
```

### cross_fade（渐变混合）

在不同的时间步上使用不同的融合比例，创建渐变过渡效果。早期时间步偏向第一张图像的结构，后期时间步偏向第二张图像的细节。

```bash
python main_run.py --mode="image_fusion" --dataset_yaml="fusion_examples.yaml" --second_image="/example_images/second_image.jpg" --fusion_ratio=0.6 --fusion_mode="cross_fade"
```

### feature_selective（特征选择性融合）

根据特征的显著性选择保留哪张图像的特征。对于每个位置，选择强度更高的特征，然后根据融合比例进行调整。这种模式能够创造更清晰、更有定义的融合效果。

```bash
python main_run.py --mode="image_fusion" --dataset_yaml="fusion_examples.yaml" --second_image="/example_images/second_image.jpg" --fusion_ratio=0.5 --fusion_mode="feature_selective"
```

### layer_selective（层选择性融合）

针对噪声空间的不同层次使用不同的融合策略：
- 早期层（结构层）：根据fusion_ratio决定主要使用哪张图像的结构
- 中期层（混合层）：线性融合
- 后期层（纹理层）：特征选择性融合

这种模式可以创建既保留一张图像主要结构，又混合两张图像纹理的效果。

```bash
python main_run.py --mode="image_fusion" --dataset_yaml="fusion_examples.yaml" --second_image="/example_images/second_image.jpg" --fusion_ratio=0.7 --fusion_mode="layer_selective"
```

## 融合比例的影响

- **0.0-0.3**: 结果主要偏向第一张图像，但带有第二张图像的一些特征
- **0.4-0.6**: 两张图像均等混合
- **0.7-1.0**: 结果主要偏向第二张图像，但保留第一张图像的一些特征

## 实用示例

### 人物混合

```bash
# 混合两个人物形象
python main_run.py --mode="image_fusion" --dataset_yaml="my_config.yaml" --second_image="/path/to/person2.jpg" --fusion_ratio=0.5 --fusion_mode="feature_selective"
```

### 风格转换

```bash
# 将内容图像与风格图像混合
python main_run.py --mode="image_fusion" --dataset_yaml="my_config.yaml" --second_image="/path/to/style_image.jpg" --fusion_ratio=0.3 --fusion_mode="layer_selective"
```

### 背景替换

```bash
# 混合前景和背景
python main_run.py --mode="image_fusion" --dataset_yaml="my_config.yaml" --second_image="/path/to/background.jpg" --fusion_ratio=0.7 --fusion_mode="cross_fade"
```

## 提示和建议

1. **相似图像效果最佳**：融合相似类型的图像通常效果最好（如两张人脸照片）
2. **融合比例实验**：尝试不同的融合比例以获得最佳效果
3. **尝试不同模式**：每种融合模式都有其独特效果，值得都尝试一下
4. **跳过参数**：如果想保留更多原始图像特征，可以增加skip值（如40-50）

## 高级使用：多图融合

您还可以通过多次应用融合操作来实现三张或更多图像的融合：

```bash
# 先融合图像1和图像2
python main_run.py --mode="image_fusion" --dataset_yaml="config1.yaml" --second_image="image2.jpg" --fusion_ratio=0.5 --fusion_mode="linear"

# 然后将融合结果与图像3融合
python main_run.py --mode="image_fusion" --dataset_yaml="config_fused.yaml" --second_image="image3.jpg" --fusion_ratio=0.3 --fusion_mode="feature_selective"
```

## 原理进阶：噪声空间的优势

本工具所用的DDPM噪声空间相比传统方法有几个关键优势：
- **分层表示**：噪声空间自然包含从粗糙结构到精细细节的层次结构
- **语义理解**：潜在表示捕获了图像的语义信息，而不仅仅是像素值
- **统计独立性**：不同时间步的噪声向量相对独立，可以选择性地混合

这些特性使得在噪声空间中进行的图像融合能够产生更有意义、更自然的结果。 