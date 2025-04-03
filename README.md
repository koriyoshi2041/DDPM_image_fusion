<!-- [![DDPM inversion](https://img.shields.io/badge/single%20image-generative%20model-yellow)](https://github.com/topics/single-image-generation) -->
[![Python 3.8](https://img.shields.io/badge/python-3.812+-blue)](https://www.python.org/downloads/release/python-38/)
[![torch](https://img.shields.io/badge/torch-2.0.0+-green)](https://pytorch.org/)


# DDPM inversion, CVPR 2024

[Project page](https://inbarhub.github.io/DDPM_inversion/) | [Arxiv](https://arxiv.org/abs/2304.06140) | [Supplementary materials](https://inbarhub.github.io/DDPM_inversion/resources/inversion_supp.pdf) | [Hugging Face Demo](https://huggingface.co/spaces/LinoyTsaban/edit_friendly_ddpm_inversion)
### Official pytorch implementation of the paper: <br>"An Edit Friendly DDPM Noise Space: Inversion and Manipulations"
#### Inbar Huberman-Spiegelglas, Vladimir Kulikov and Tomer Michaeli 
<br>

![](imgs/teaser.jpg)
Our inversion can be used for text-based **editing of real images**, either by itself or in combination with other editing methods.
Due to the stochastic nature of our method, we can generate **diverse outputs**, a feature that is not naturally available with methods relying on the DDIM inversion.

In this repository we support editing using our inversion, prompt-to-prompt (p2p)+our inversion, ddim or [p2p](https://github.com/google/prompt-to-prompt) (with ddim inversion).<br>
**our inversion**: our ddpm inversion followed by generating an image conditioned on the target prompt. 

**prompt-to-prompt (p2p) + our inversion**: p2p method using our ddpm inversion. 

**ddim**: ddim inversion followed by generating an image conditioned on the target prompt.

**p2p**: p2p method using ddim inversion (original paper).

## Table of Contents
* [Requirements](#Requirements)
* [Repository Structure](#Repository-Structure)
* [Algorithm Inputs and Parameters](#Algorithm-Inputs-and-Parameters)
* [Usage Example](#Usage-Example)

* [Citation](#Citation)

## Requirements 

```
python -m pip install -r requirements.txt
```
This code was tested with python 3.8 and torch 2.0.0. 

## Repository Structure 
```
├── ddm_inversion - folder contains inversions in order to work on real images: ddim inversion as well as ddpm inversion (our method).
├── example_images - folder of input images to be edited
├── imgs - images used in this repository readme.md file
├── prompt_to_prompt - p2p code
├── main_run.py - main python file for real image editing
└── test.yaml - yaml file contains images and prompts to test on
```

A folder named 'results' will be automatically created and all the results will be saved to this folder. We also add a timestamp to the saved images in this folder.

## Algorithm Inputs and Parameters
Method's inputs: 
```
init_img - the path to the input images
source_prompt - a prompt describing the input image
target_prompts - the edit prompt (creates several images if multiple prompts are given)
```
These three inputs are supplied through a YAML file (please use the provided 'test.yaml' file as a reference).

<br>
Method's parameters are:

```
skip - controlling the adherence to the input image
cfg_tar - classifier free guidance strengths
```
These two parameters have default values, as descibed in the paper.

## Usage Example 
```
python3 main_run.py --mode="our_inv" --dataset_yaml="test.yaml" --skip=36 --cfg_tar=15 
python3 main_run.py --mode="p2pinv" --dataset_yaml="test.yaml" --skip=12 --cfg_tar=9 

```
The ```mode``` argument can also be: ```ddim``` or ```p2p```.

In ```our_inv``` and ```p2pinv``` modes we suggest to play around with ```skip``` in the range [0,40] and ```cfg_tar``` in the range [7,18].

**p2pinv and p2p**:
Note that you can play with the cross-and self-attention via ```--xa``` and ```--sa``` arguments. We suggest to set them to (0.6,0.2) and (0.8,0.4) for p2pinv and p2p respectively.

**ddim and p2p**:
```skip``` is overwritten to be 0.

<!-- ## Create Your Own Editing with Our Method
(1) Add your image to /example_images. <br>
(2) Run ``main_run.py --mode="our_inv"``, choose ``skip`` and ``cfg_tar``. <br>

Example:
```
python3 main_run.py --skip=20 --cfg_tar=10 --img_name=gnochi_mirror --cfg_src='a cat is sitting next to a mirror' --cfg_tar='a drawing of a cat sitting next to a mirror'
```  -->

You can edit the test.yaml file to load your image and choose the desired prompts.

<!-- ## Sources 

The DDPM code was adapted from the following [pytorch implementation of DDPM](https://github.com/lucidrains/denoising-diffusion-pytorch). 

The modified CLIP model as well as most of the code in `./text2live_util/` directory was taken from the [official Text2live repository](https://github.com/omerbt/Text2LIVE).  -->
 
## Citation
If you use this code for your research, please cite our paper:
```
@inproceedings{huberman2024edit,
  title={An edit friendly {DDPM} noise space: Inversion and manipulations},
  author={Huberman-Spiegelglas, Inbar and Kulikov, Vladimir and Michaeli, Tomer},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12469--12478},
  year={2024}
}
```

# DDPM图像融合与智能数据扩增

本项目在DDPM可编辑噪声空间中实现了强大的图像融合与基于相似度的智能数据扩增功能。

## 🔥 主要功能

### 1. DDPM噪声空间图像融合

我们实现了一种在DDPM可编辑噪声空间中进行图像融合的方法，能够生成语义连贯的结果，同时保留两张源图像的关键特征。

![图像融合示例](https://i.imgur.com/3MaQyJS.png)

**核心特点:**
- 四种融合模式:
  - **线性融合 (Linear)**: 简单的加权混合
  - **渐变融合 (Cross Fade)**: 时间步相关的渐进混合
  - **特征选择性融合 (Feature Selective)**: 保留每张图像中的主要特征
  - **层选择性融合 (Layer Selective)**: 对不同噪声层次应用不同的融合策略

**使用方法:**
```bash
python main_run.py --mode="image_fusion" --dataset_yaml="fusion_examples.yaml" --second_image="path/to/second/image.jpg" --fusion_ratio=0.5 --fusion_mode="linear"
```

### 2. 基于相似度的智能数据扩增

我们开发了一套智能数据扩增系统，能够自动检测训练集中的相似图像并融合它们，创造语义连贯的扩增样本。

**核心特点:**
- **自动相似度检测**：使用ResNet50提取图像特征并计算相似度
- **智能融合**：只融合足够相似的图像，避免生成混乱结果
- **批量处理**：高效处理整个训练集
- **多种融合策略**：支持多种融合模式，生成多样化输出

**使用方法:**
```bash
python augment_dataset_fusion.py --input_dir="training_images" --output_dir="augmented_dataset" --similarity_threshold=0.7
```

## 📚 文档

项目包含以下详细文档:

- [IMAGE_FUSION_README.md](IMAGE_FUSION_README.md) - 图像融合功能详细指南
- [FUSION_AUGMENTATION_README.md](FUSION_AUGMENTATION_README.md) - 基于相似度的数据扩增指南
- [PROJECT_CHANGES.md](PROJECT_CHANGES.md) - 项目修改完整记录

## 🧠 技术细节

我们的实现利用了DDPM可编辑噪声空间的独特特性:

- **时间步之间的统计独立性**：允许分层编辑
- **可控的随机性**：能够生成多样化输出
- **结构保留**：确保融合结果的语义连贯性

相似度检测系统使用预训练ResNet50模型提取的特征向量之间的余弦相似度，确保只有真正相似的图像才会被融合。

## 🔍 相比传统方法的优势

- **语义理解**：保留源图像的语义信息
- **自然过渡**：避免像素级混合常见的伪影
- **分层控制**：选择性地融合不同特征层次
- **智能筛选**：只融合语义相似的图像
- **多样化效果**：多种融合模式满足不同创意需求

## 📋 系统要求

- Python 3.7+
- PyTorch 1.7+
- torchvision
- scikit-learn (用于相似度计算)
- PIL (Pillow)
- numpy
- tqdm
- PyYAML

## 🚀 快速入门

1. 克隆此仓库
2. 安装依赖项
3. 运行上述示例命令
4. 查阅文档了解详细使用说明

## 📝 许可证

本项目基于MIT许可证开源 - 详见LICENSE文件