<!-- [![DDPM inversion](https://img.shields.io/badge/single%20image-generative%20model-yellow)](https://github.com/topics/single-image-generation) -->
[![Python 3.8](https://img.shields.io/badge/python-3.812+-blue)](https://www.python.org/downloads/release/python-38/)
[![torch](https://img.shields.io/badge/torch-2.0.0+-green)](https://pytorch.org/)


# DDPM Image Fusion and Data Augmentation

This repository extends the original [DDPM Inversion](https://github.com/inbarhub/DDPM_inversion) work with powerful image fusion and similarity-based data augmentation capabilities that operate in the editable DDPM noise space. 

## 🔥 New Features

### 1. Image Fusion in DDPM Noise Space

We've implemented robust image fusion functionality that allows blending two images in the editable DDPM noise space, producing semantically coherent results that preserve key features from both source images.

> **Visual Effect**: The fusion produces seamless blends where key features from both source images are preserved. Unlike pixel-level blending, our noise-space fusion maintains semantic integrity and natural transitions between different elements of the images.

**Key features:**
- Four fusion modes:
  - **Linear Fusion**: Simple weighted blending
  - **Cross Fade**: Time-dependent gradual blending
  - **Feature Selective Fusion**: Preserves dominant features from each image
  - **Layer Selective Fusion**: Applies different strategies to different noise levels

**Usage:**
```bash
python main_run.py --mode="image_fusion" --dataset_yaml="fusion_examples.yaml" --second_image="path/to/second/image.jpg" --fusion_ratio=0.5 --fusion_mode="linear"
```

### 2. Similarity-Based Data Augmentation

We've developed an intelligent data augmentation system that automatically detects similar images in a training dataset and fuses them to create semantically coherent augmented samples.

**Key features:**
- **Automatic similarity detection** using ResNet50 feature extraction
- **Smart fusion** of only sufficiently similar images
- **Batch processing** for efficient dataset augmentation
- **Multiple fusion strategies** for diverse outputs

**Usage:**
```bash
python augment_dataset_fusion.py --input_dir="training_images" --output_dir="augmented_dataset" --similarity_threshold=0.7
```

## 📚 Documentation

We've added comprehensive documentation for all new features:

- [IMAGE_FUSION_README.md](IMAGE_FUSION_README.md) - Detailed guide on image fusion
- [FUSION_AUGMENTATION_README.md](FUSION_AUGMENTATION_README.md) - Guide on similarity-based data augmentation
- [PROJECT_CHANGES.md](PROJECT_CHANGES.md) - Complete record of all project modifications

## 🧠 Technical Details

Our implementation leverages the unique properties of the DDPM editable noise space:

- **Statistical independence** between time steps allows for layered editing
- **Controllable randomness** enables diverse output generation
- **Structural preservation** ensures semantic coherence in fusion results

The similarity detection system uses cosine similarity between feature vectors extracted by a pre-trained ResNet50 model, ensuring that only meaningfully similar images are fused.

## 🔍 Advantages Over Traditional Methods

- **Semantic understanding**: Preserves semantic information from both sources
- **Natural transitions**: Avoids artifacts common in pixel-level blending
- **Layered control**: Selectively fuses different feature levels
- **Smart filtering**: Only fuses semantically similar images
- **Diverse effects**: Multiple fusion modes for different creative needs

## 🔄 Original DDPM Inversion Features

This repository builds upon [DDPM Inversion](https://github.com/inbarhub/DDPM_inversion) which provides:

- DDPM inversion technique for mapping images to noise space
- Editable representations for manipulation without text prompts
- Various noise-space editing operations (brightness, contrast, etc.)

Please refer to the [original paper](https://arxiv.org/abs/2307.10829) for more details on the foundation technology.

## 📋 Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- scikit-learn (for similarity-based data augmentation)
- PIL (Pillow)
- numpy
- tqdm
- PyYAML

## 🚀 Getting Started

1. Clone this repository
2. Install dependencies
3. Run one of the example commands above
4. Check the documentation for detailed usage instructions

## 🔗 Citation

If you use this code in your research, please cite the original paper:

```
@article{hub2023edit,
  title={An Edit Friendly DDPM Noise Space: Inversion and Manipulations},
  author={Hub, Inbar and Hertz, Amir and Fuchs, Shai},
  journal={arXiv preprint arXiv:2307.10829},
  year={2023}
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
