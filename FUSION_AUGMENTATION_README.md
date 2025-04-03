# 基于相似度的图像融合数据扩增

本功能通过自动检测训练集中相似图像并融合它们，为DDPM训练生成多样化的扩增样本。该方法在可编辑的DDPM噪声空间中进行图像融合，保留语义内容的同时引入足够变化。

## 功能特点

- **自动相似度检测**：使用预训练ResNet50提取图像特征并计算余弦相似度
- **智能图像融合**：仅融合足够相似的图像，避免生成混乱无意义的结果
- **多种融合模式**：支持线性融合、渐变融合、特征选择性融合和层选择性融合
- **批量处理**：自动处理整个训练集并生成扩增数据
- **可配置参数**：提供多种参数调整融合行为和扩增数量

## 使用方法

### 基本命令

```bash
python augment_dataset_fusion.py --input_dir="输入训练集目录" --output_dir="输出扩增数据集目录"
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_dir` | 输入训练集目录（必需） | - |
| `--output_dir` | 输出扩增数据集目录（必需） | - |
| `--similarity_threshold` | 图像相似度阈值，大于该值的图像对被认为足够相似 | 0.7 |
| `--max_pairs` | 最大融合对数，防止生成过多图像 | 100 |
| `--fusion_modes` | 要使用的融合模式，逗号分隔 | linear,cross_fade,feature_selective,layer_selective |
| `--device` | 使用的设备 | cuda(如可用)或cpu |
| `--batch_size` | 特征提取批次大小 | 16 |
| `--copy_original` | 是否将原始图像复制到输出目录 | False |
| `--temp_yaml` | 临时配置文件路径 | temp_fusion.yaml |

### 使用示例

1. **基本使用**：
   ```bash
   python augment_dataset_fusion.py --input_dir="原始训练集" --output_dir="扩增数据集"
   ```

2. **调整相似度阈值**：
   ```bash
   python augment_dataset_fusion.py --input_dir="原始训练集" --output_dir="扩增数据集" --similarity_threshold=0.8
   ```

3. **生成更多融合图像**：
   ```bash
   python augment_dataset_fusion.py --input_dir="原始训练集" --output_dir="扩增数据集" --max_pairs=500
   ```

4. **只使用特定融合模式**：
   ```bash
   python augment_dataset_fusion.py --input_dir="原始训练集" --output_dir="扩增数据集" --fusion_modes="linear,feature_selective"
   ```

5. **包含原始图像**：
   ```bash
   python augment_dataset_fusion.py --input_dir="原始训练集" --output_dir="扩增数据集" --copy_original
   ```

## 工作原理

### 图像相似度检测

1. **特征提取**：使用预训练的ResNet50模型从每张图像提取特征向量
2. **相似度计算**：计算所有图像对之间的余弦相似度
3. **筛选相似对**：根据相似度阈值选择足够相似的图像对

### 图像融合过程

1. **准备图像**：将识别出的相似图像对读入系统
2. **噪声空间映射**：使用DDPM的反向过程将图像映射到噪声空间
3. **执行融合**：在噪声空间中执行选定的融合操作
4. **生成融合图像**：使用融合后的噪声向量通过前向过程生成新图像
5. **保存结果**：将融合图像保存到输出目录

### 融合模式说明

1. **线性融合 (linear)**：简单地按比例混合两个噪声向量
   ```
   zs_fused = ratio * zs1 + (1 - ratio) * zs2
   ```

2. **渐变融合 (cross_fade)**：不同时间步使用不同的融合比例
   ```
   fade_ratio = ratio + (1 - ratio) * (t / T)
   zs_fused[t] = fade_ratio * zs1[t] + (1 - fade_ratio) * zs2[t]
   ```

3. **特征选择性融合 (feature_selective)**：基于特征强度选择性地保留每个图像的关键特征
   ```
   mask = |zs1| > |zs2|
   zs_fused = mask * zs1 + (~mask) * zs2
   ```

4. **层选择性融合 (layer_selective)**：不同语义层次使用不同的融合策略
   ```
   对于结构层：使用权重偏向第一张图像
   对于纹理层：使用权重偏向第二张图像
   对于中间层：使用平衡的混合
   ```

## 使用建议

- **数据集相似性**：对于包含相似对象类别的数据集效果最佳（如人脸数据集、同类动物数据集等）
- **阈值选择**：相似度阈值通常设置在0.6-0.8之间，视具体数据集而定
- **融合数量**：适度使用融合扩增，建议融合图像数量不超过原始数据集大小
- **融合模式选择**：对于精细结构，建议使用layer_selective模式；对于一般场景，linear或cross_fade通常效果良好

## 技术要求

- Python 3.7+
- PyTorch 1.7+
- torchvision
- scikit-learn
- PIL (Pillow)
- NumPy
- tqdm
- PyYAML

## 注意事项

- 此功能需要较大的内存和GPU资源，特别是处理大量图像时
- 处理高分辨率图像时可能需要调整批处理大小以适应GPU内存
- 图像相似度检测基于视觉特征，可能不能完全捕捉语义相似性 