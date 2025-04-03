# 噪声空间编辑

这个功能允许您直接编辑图像的噪声空间，而不需要知道图像的具体内容。通过对潜在空间的深度操作，您可以实现从轻微调整到彻底重构的各种变换效果。

## 编辑类型

### 基础编辑
这些编辑主要影响图像的视觉属性，保留原始图像的主要内容：

- **brightness**: 调整图像亮度
- **contrast**: 调整图像对比度
- **saturation**: 调整图像饱和度
- **style**: 添加随机风格化效果
- **smooth**: 使图像更平滑

### 深度变换
这些编辑会对图像进行根本性的变换，产生显著不同的结果：

- **transform**: 应用几何变换，包括旋转、缩放和扭曲
- **restructure**: 重新组织噪声空间的不同层次，彻底改变图像结构
- **abstract**: 创建抽象化效果，通过放大某些特征而抑制其他特征
- **dream**: 产生梦境般的扭曲效果，混合不同时间步的特征
- **random**: 在保留一些基础结构的同时，引入大量随机变化，实现最彻底的变换

### 训练数据扩增
专为DDPM训练数据扩增设计的编辑模式：

- **data_augment**: 产生与原始图像高度相似但有细微变化的图像，适合用作训练数据扩增。这种模式会保留主要结构和语义内容，同时添加足够的变化以防止模型过拟合/记忆原始样本。

## 使用方法

1. 将要编辑的图像放到 `example_images` 目录中
2. 使用提供的 `test_noise_edit.yaml` 文件或创建您自己的配置文件
3. 运行以下命令:

```bash
python main_run.py --mode="noise_edit" --dataset_yaml="test_noise_edit.yaml" --edit_type="restructure" --edit_strength=1.0
```

## 参数说明

- `--mode="noise_edit"`: 指定使用噪声空间编辑模式
- `--dataset_yaml`: YAML配置文件路径，指定输入图像和基本提示词
- `--edit_type`: 编辑类型，可选值见上面列出的各种编辑类型
- `--edit_strength`: 编辑强度，推荐范围: 0.1-2.0（不同编辑类型的最佳范围可能有所不同）
- `--skip`: 控制对原始图像的保留程度，范围: 0-40

## 深度变换示例

### 几何变换
```bash
python main_run.py --mode="noise_edit" --dataset_yaml="test_noise_edit.yaml" --edit_type="transform" --edit_strength=1.2
```

### 结构重组
```bash
python main_run.py --mode="noise_edit" --dataset_yaml="test_noise_edit.yaml" --edit_type="restructure" --edit_strength=1.0
```

### 抽象化
```bash
python main_run.py --mode="noise_edit" --dataset_yaml="test_noise_edit.yaml" --edit_type="abstract" --edit_strength=0.8
```

### 梦境效果
```bash
python main_run.py --mode="noise_edit" --dataset_yaml="test_noise_edit.yaml" --edit_type="dream" --edit_strength=1.5
```

### 完全随机变换
```bash
python main_run.py --mode="noise_edit" --dataset_yaml="test_noise_edit.yaml" --edit_type="random" --edit_strength=0.7
```

### 训练数据扩增
```bash
python main_run.py --mode="noise_edit" --dataset_yaml="test_noise_edit.yaml" --edit_type="data_augment" --edit_strength=0.3
```

## 数据扩增专用工具

我们还提供了一个专门用于数据扩增的批处理工具 `augment_dataset.py`，可以自动处理整个训练数据集并生成多个变体。详情请参阅 `DATA_AUGMENTATION_README.md`。

## 编辑强度的影响

- **低强度**（0.1-0.5）: 保留更多原始图像的特征，变化较为温和
- **中等强度**（0.5-1.0）: 在保留一些可识别特征的同时引入显著变化
- **高强度**（1.0-2.0）: 产生最激进的变换，可能创建与原始图像完全不同的结果

对于 **data_augment** 模式，推荐使用较低的强度（0.2-0.5），以确保生成的图像与原始图像足够相似，同时又有足够的差异以防止模型记忆。

## 推荐组合

以下是一些推荐的组合方式，可以产生非常有趣的效果：

1. **抽象艺术风格**：先应用`abstract`编辑（强度0.8），然后应用`transform`（强度1.0）
   ```bash
   python main_run.py --mode="noise_edit" --dataset_yaml="test_noise_edit.yaml" --edit_type="abstract" --edit_strength=0.8
   # 然后对结果图像再次运行：
   python main_run.py --mode="noise_edit" --dataset_yaml="test_noise_edit.yaml" --edit_type="transform" --edit_strength=1.0
   ```

2. **超现实梦境**：先应用`dream`编辑（强度1.2），然后应用`restructure`（强度0.6）
   ```bash
   python main_run.py --mode="noise_edit" --dataset_yaml="test_noise_edit.yaml" --edit_type="dream" --edit_strength=1.2
   # 然后对结果图像再次运行：
   python main_run.py --mode="noise_edit" --dataset_yaml="test_noise_edit.yaml" --edit_type="restructure" --edit_strength=0.6
   ```

3. **完全重构**：直接应用`random`编辑（强度1.5）
   ```bash
   python main_run.py --mode="noise_edit" --dataset_yaml="test_noise_edit.yaml" --edit_type="random" --edit_strength=1.5
   ```

4. **训练数据多样化**：为训练集创建多个变体
   ```bash
   # 使用不同的随机种子生成多个变体
   for i in {1..5}; do
     python main_run.py --mode="noise_edit" --dataset_yaml="test_noise_edit.yaml" --edit_type="data_augment" --edit_strength=0.3
   done
   ```

## 高级使用

### 自定义组合编辑

您可以创建自定义的组合编辑函数，将多种效果整合为一个操作。例如，在`main_run.py`中添加：

```python
elif edit_type == "custom_combo":
    # 先应用结构重组
    edited_zs = edit_noise_space(zs, "restructure", strength * 0.8)
    # 然后应用梦境效果
    edited_zs = edit_noise_space(edited_zs, "dream", strength * 1.2)
    # 最后添加一些抽象元素
    edited_zs = edit_noise_space(edited_zs, "abstract", strength * 0.5)
```

### 噪声空间探索

每次使用相同的参数运行相同的编辑类型（尤其是含有随机性的类型），您会得到不同的结果。这使您可以探索潜在空间的多样性，找到最符合您创意的结果。

### 批量生成

您可以编写脚本批量生成多种变体：

```bash
for strength in 0.5 1.0 1.5; do
  for type in transform restructure abstract dream random; do
    python main_run.py --mode="noise_edit" --dataset_yaml="test_noise_edit.yaml" --edit_type=$type --edit_strength=$strength
  done
done
```

## 训练数据扩增特别说明

如果您的目标是扩充训练数据集以避免模型记忆，我们强烈推荐：

1. 使用 `data_augment` 编辑类型，它专为训练数据扩增设计，能创建具有微妙变化但保留主要特征的图像
2. 使用较低的编辑强度（0.2-0.4）以确保生成的图像与原始图像足够相似
3. 为每张图像生成多个变体（3-5个）以显著扩充训练集

对于大规模数据集处理，建议使用我们提供的 `augment_dataset.py` 脚本，它可以自动批量处理整个数据集。

## 自定义编辑类型

如果您想要创建自己的编辑类型，可以修改 `main_run.py` 文件中的 `edit_noise_space` 函数，添加新的编辑类型逻辑。 