# DDPM训练数据扩增工具

这个工具专为扩充DDPM（去噪扩散概率模型）的训练数据集而设计。它通过操作潜在噪声空间生成与原始图像相似但又有细微差异的变体，帮助减少模型在训练过程中的记忆化问题。

## 原理

传统的DDPM模型在训练过程中可能会记忆训练样本，导致生成结果缺乏多样性。本工具通过以下方式解决这个问题：

1. **保留语义结构**：生成的图像在视觉上与原始图像非常相似
2. **引入可控变化**：添加细微的几何、纹理和颜色变化
3. **增加数据多样性**：每次生成的变体都略有不同，即使使用相同的参数

通过扩充训练集，可以降低模型直接记忆原始样本的概率，从而提高生成的多样性和质量。

## 安装要求

确保您已经安装了以下依赖：
```
pip install pyyaml tqdm torch
```

## 使用方法

### 基本用法

```bash
python augment_dataset.py --input_dir="原始训练图像目录" --output_dir="输出目录"
```

默认情况下，工具会为每张图像生成3个变体，变换强度在0.2到0.8之间随机选择。

### 高级参数

```bash
python augment_dataset.py \
  --input_dir="原始训练图像目录" \
  --output_dir="输出目录" \
  --variants=5 \
  --prompt="通用描述文本" \
  --min_strength=0.3 \
  --max_strength=0.6
```

参数说明：
- `--input_dir`: 包含原始训练图像的目录
- `--output_dir`: 存放生成的增强图像的目录
- `--variants`: 每张原始图像生成的变体数量（默认：3）
- `--prompt`: 可选的源提示词（可以留空）
- `--min_strength`: 最小变换强度（默认：0.2）
- `--max_strength`: 最大变换强度（默认：0.8）

## 变换强度说明

- **0.1-0.3**: 极轻微变化，几乎不可见但足以区分不同样本
- **0.3-0.5**: 轻微变化，保留所有主要特征但细节有差异
- **0.5-0.8**: 中等变化，总体结构相似但有明显差异
- **>0.8**: 较大变化，可能改变一些关键特征

## 数据扩增原理

该工具对噪声空间进行以下操作：

1. **轻微的几何变形**：应用小幅度的旋转、缩放等变换
2. **纹理变化**：在高频细节部分添加平滑的随机变化
3. **颜色调整**：通过改变通道权重模拟颜色和材质变化
4. **极小随机噪声**：添加几乎不可见的噪声以确保每个像素值都是唯一的

## 推荐的训练流程

1. **创建基础数据集**：准备原始训练图像
2. **生成扩增数据**：使用本工具生成3-5倍量的变体
3. **合并数据集**：将原始图像和变体合并为一个大数据集
4. **训练模型**：使用扩增后的数据集训练DDPM模型

示例：
```bash
# 扩充训练集
python augment_dataset.py --input_dir="original_dataset" --output_dir="augmented_dataset" --variants=4

# 将原始数据复制到扩充集
cp original_dataset/* augmented_dataset/

# 使用扩充后的数据集训练模型
python train_ddpm.py --data_path="augmented_dataset"
```

## 注意事项

- 建议保持原始图像在训练集中，并将变体作为补充
- 如果数据集已经非常大，可以使用较少的变体数量（如1-2个）
- 对于小型数据集，可以生成更多变体（5-10个）以显著增加数据量
- 变换强度应根据您的具体任务调整，如果需要保持高度的图像细节，请使用较低的强度

## 高级用法：针对特定域的定制

如果您的训练数据属于特定领域（如人脸、风景等），可以通过修改`main_run.py`中的`data_augment`函数来定制变换。例如，对于人脸图像，可能需要保持关键面部特征不变；对于风景图像，可能希望引入更多的光照和季节变化。 