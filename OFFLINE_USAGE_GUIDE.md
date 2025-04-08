# 离线环境使用指南

本文档提供在没有网络连接的环境（如租借的GPU服务器）上运行本项目的完整指南。

## 准备工作（在有网络的环境中完成）

### 1. 下载代码仓库

```bash
git clone https://github.com/koriyoshi2041/DDPM_image_fusion.git
cd DDPM_image_fusion
```

### 2. 下载预训练模型

Stable Diffusion v1.4 是本项目使用的基础模型，大小约4-5GB。

#### 方法1：使用Python脚本下载

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='CompVis/stable-diffusion-v1-4', local_dir='./stable_diff_local')"
```

#### 方法2：使用Hugging Face CLI

```bash
pip install huggingface-cli
huggingface-cli download CompVis/stable-diffusion-v1-4 --local-dir ./stable_diff_local
```

> **注意**：您需要先在Hugging Face网站上接受Stable Diffusion的使用条款。

### 3. 安装依赖

为了在离线环境中安装依赖，您需要预先下载所有必要的Python包：

```bash
pip download -r requirements.txt -d ./pip_packages
```

这会将所有依赖包下载到`pip_packages`目录中。

### 4. 准备示例图像

将您想要处理的图像放入`example_images`目录：

```bash
mkdir -p example_images
# 复制您的图像到这个目录
cp /path/to/your/images/*.jpg example_images/
```

### 5. 打包所有文件

```bash
tar -czvf ddpm_offline_package.tar.gz DDPM_image_fusion stable_diff_local pip_packages example_images
```

## 在离线环境中设置

### 1. 解压文件

将打包好的文件传输到离线环境后解压：

```bash
tar -xzvf ddpm_offline_package.tar.gz
cd DDPM_image_fusion
```

### 2. 安装依赖

从下载好的包中安装依赖：

```bash
pip install --no-index --find-links=../pip_packages -r requirements.txt
```

### 3. 运行验证脚本

我们提供了一个脚本来验证您的设置：

```bash
python setup_offline.py
```

这将检查所有必要的模型文件和依赖是否正确安装。

## 运行图像融合

### 基本使用

```bash
python main_run.py --mode=image_fusion --dataset_yaml=fusion_examples.yaml --second_image=./example_images/your_second_image.jpg --fusion_ratio=0.5 --fusion_mode=linear
```

### 参数说明

- `--mode`: 设置为"image_fusion"以使用融合功能
- `--dataset_yaml`: 配置文件路径，包含第一张图像信息
- `--second_image`: 第二张图像的路径
- `--fusion_ratio`: 融合比例(0.0-1.0)，值越大第二张图像占比越高
- `--fusion_mode`: 融合模式，可选:
  - `linear`: 线性混合
  - `cross_fade`: 渐变混合
  - `feature_selective`: 特征选择性融合
  - `layer_selective`: 层选择性融合

### 自定义配置文件

您可以创建自己的配置文件来指定第一张图像：

```yaml
example1:
  init_image: ./example_images/your_first_image.jpg
  source_prompt: a photo
  target_prompts:
    - a photo
```

## 常见问题解决

### CUDA相关错误

确保您的CUDA版本与PyTorch兼容：

```bash
# 检查CUDA版本
nvidia-smi

# 检查PyTorch是否能使用CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### 内存不足错误

如果遇到内存不足错误，尝试减小批处理大小或使用较小的图像：

```bash
# 使用较低的扩散步数
python main_run.py --mode=image_fusion --num_diffusion_steps=50 ...
```

### 模型文件缺失

如果`setup_offline.py`报告模型文件缺失，确保您已完整下载模型：

```bash
# 检查模型目录结构
ls -la ./stable_diff_local
```

所需的文件结构应该包括:
- model_index.json
- scheduler/scheduler_config.json
- text_encoder/pytorch_model.bin
- tokenizer/vocab.json, merges.txt
- unet/diffusion_pytorch_model.bin
- vae/diffusion_pytorch_model.bin

## 其他资源

如需进一步帮助，请参考项目的其他文档：
- IMAGE_FUSION_README.md: 详细的图像融合使用指南
- README.md: 项目总体说明 