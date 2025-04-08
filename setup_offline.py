#!/usr/bin/env python3
"""
离线模式设置脚本
此脚本检查模型文件是否正确放置，并确保环境准备就绪
"""

import os
import sys
import shutil
from pathlib import Path

def check_model_files():
    """检查本地模型文件是否存在"""
    model_dir = Path("./stable_diff_local")
    
    required_files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/pytorch_model.bin",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "unet/diffusion_pytorch_model.bin",
        "vae/diffusion_pytorch_model.bin"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = model_dir / file_path
        if not full_path.exists():
            missing_files.append(str(full_path))
    
    if missing_files:
        print("❌ 错误: 以下必需的模型文件缺失:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n请确保您已经下载了完整的Stable Diffusion v1.4模型到'stable_diff_local'目录")
        print("可以在有网络的电脑上使用以下命令下载:")
        print("python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='CompVis/stable-diffusion-v1-4', local_dir='./stable_diff_local')\"")
        return False
    
    print("✅ 模型文件检查通过!")
    return True

def check_dependencies():
    """检查依赖项是否安装"""
    try:
        import torch
        import diffusers
        import transformers
        import PIL
        
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ Diffusers版本: {diffusers.__version__}")
        print(f"✅ Transformers版本: {transformers.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ 警告: CUDA不可用，模型将在CPU上运行，这会非常慢")
        
        return True
    except ImportError as e:
        print(f"❌ 错误: 缺少必要的依赖项: {e}")
        print("请安装所有依赖: pip install -r requirements.txt")
        return False

def setup_example_images():
    """确保示例图像目录存在"""
    example_dir = Path("./example_images")
    if not example_dir.exists():
        example_dir.mkdir()
        print("✅ 创建了example_images目录")
    else:
        print("✅ example_images目录已存在")
    
    # 检查示例图像
    images = list(example_dir.glob("*.jpg")) + list(example_dir.glob("*.png"))
    if not images:
        print("⚠️ 警告: example_images目录中没有图像")
        print("请添加一些图像到此目录以进行测试")
    else:
        print(f"✅ 发现 {len(images)} 张示例图像")

def main():
    print("===== 离线模式设置检查 =====")
    
    deps_ok = check_dependencies()
    model_ok = check_model_files()
    setup_example_images()
    
    if deps_ok and model_ok:
        print("\n✅ 所有检查通过! 您可以在离线模式下运行")
        print("\n示例运行命令:")
        print("python main_run.py --mode=image_fusion --dataset_yaml=fusion_examples.yaml --second_image=./example_images/your_second_image.jpg --fusion_ratio=0.5 --fusion_mode=linear")
    else:
        print("\n❌ 一些检查未通过，请解决上述问题")
        
    return 0 if (deps_ok and model_ok) else 1

if __name__ == "__main__":
    sys.exit(main()) 