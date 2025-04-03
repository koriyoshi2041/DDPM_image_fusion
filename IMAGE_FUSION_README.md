# 图像融合功能使用指南

这个功能允许你在DDPM可编辑噪声空间中融合两张图像，创建具有两张图像特征的新图像。与传统的像素级融合方法不同，噪声空间融合能够产生更自然、更有语义理解的结果。

## 优势

1. **保留语义内容**: 融合过程考虑了图像的语义层次结构
2. **自然过渡**: 避免了传统混合可能产生的伪影
3. **多样化融合模式**: 支持从简单的线性混合到复杂的特征选择性融合的各种技术

## 使用方法

### 基本命令

```bash
python main_run.py --mode="image_fusion" --dataset_yaml="fusion_examples.yaml" --second_image="/example_images/second_image.jpg" --fusion_ratio=0.5 --fusion_mode="linear"
```

### 必需参数

- `--mode`: 设置为 "image_fusion" 以使用融合功能
- `--dataset_yaml`: 第一张图像的配置文件路径
- `--second_image`: 第二张图像的路径

### 可选参数

- `--fusion_ratio`: 融合比例（0.0到1.0之间），决定两个图像的混合程度（默认: 0.5）
- `--fusion_mode`: 融合模式，可选 "linear", "cross_fade", "feature_selective", "layer_selective"（默认: "linear"）

## 关于提示词的说明

本实现使用通用空提示词 "a photo" 进行图像处理。你无需提供特定的图像描述提示词，系统会自动使用这个通用提示词。这使得融合过程完全依赖于图像的噪声空间表示，而不是文本描述。

## 融合模式

1. **线性融合 (linear)**: 在所有时间步使用相同的融合比例进行简单混合
2. **渐变混合 (cross_fade)**: 不同时间步使用不同的融合比例，创造渐变效果
3. **特征选择性融合 (feature_selective)**: 根据特征的显著性选择每个图像的最重要部分
4. **层选择性融合 (layer_selective)**: 在不同噪声层次应用不同策略，保留结构特征的同时混合纹理

## 融合比例的影响

融合比例决定了两个图像在最终结果中的权重：
- 0.0: 完全保留第一张图像的特征
- 0.5: 两张图像特征平均混合
- 1.0: 完全保留第二张图像的特征

## 使用实例

### 人物融合

```bash
python main_run.py --mode="image_fusion" --dataset_yaml="examples.yaml" --second_image="/example_images/person2.jpg" --fusion_ratio=0.4 --fusion_mode="feature_selective"
```

### 风格转换

```bash
python main_run.py --mode="image_fusion" --dataset_yaml="examples.yaml" --second_image="/example_images/style_reference.jpg" --fusion_ratio=0.7 --fusion_mode="layer_selective"
```

### 背景替换

```bash
python main_run.py --mode="image_fusion" --dataset_yaml="examples.yaml" --second_image="/example_images/new_background.jpg" --fusion_ratio=0.6 --fusion_mode="cross_fade"
```

## 提示与建议

- 尝试使用视觉上相似的图像进行融合可获得更好的结果
- 调整融合比例以获得最佳效果
- 尝试不同的融合模式，观察结果的差异
- 使用"layer_selective"模式可以保留第一张图像的主要结构，同时采用第二张图像的风格或纹理

## 高级用法：多图像融合

要融合超过两张图像，可以按顺序应用融合过程：

```bash
# 第一次融合：图像1 + 图像2
python main_run.py --mode="image_fusion" --dataset_yaml="examples.yaml" --second_image="/example_images/image2.jpg" --fusion_ratio=0.4 --fusion_mode="linear"

# 使用第一次融合的结果 + 图像3
python main_run.py --mode="image_fusion" --dataset_yaml="fusion_result.yaml" --second_image="/example_images/image3.jpg" --fusion_ratio=0.3 --fusion_mode="feature_selective"
```

## 为什么DDPM噪声空间适合图像融合？

DDPM噪声空间提供了图像的分层表示：
1. **分层表示**: 不同时间步捕获不同级别的抽象
2. **语义理解**: 与像素级融合相比，保留更多语义信息
3. **统计独立性**: 不同时间步之间的相对独立性允许精细控制 