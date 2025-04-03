import os
import argparse
import yaml
import random
import shutil
from tqdm import tqdm
import torch
import subprocess
from pathlib import Path

def create_augmentation_yaml(input_dir, output_yaml, source_prompt=""):
    """
    为输入目录中的所有图像创建一个YAML配置文件
    """
    # 获取所有图像文件
    image_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_files.extend(list(Path(input_dir).glob(f"*.{ext}")))
    
    # 创建YAML配置
    yaml_data = []
    for img_path in image_files:
        rel_path = os.path.relpath(img_path, os.getcwd())
        entry = {
            'init_img': rel_path,
            'source_prompt': source_prompt,  # 可以留空或使用通用描述
            'target_prompts': [source_prompt]  # 使用相同的提示词
        }
        yaml_data.append(entry)
    
    # 写入YAML文件
    with open(output_yaml, 'w') as f:
        yaml.dump(yaml_data, f)
    
    return len(image_files)

def run_augmentation(yaml_file, output_dir, num_variants=3, strength_range=(0.2, 0.8)):
    """
    对YAML文件中的每张图像运行数据增强
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每张图像创建多个变体
    for variant in range(num_variants):
        # 随机选择强度
        strength = random.uniform(strength_range[0], strength_range[1])
        
        # 构建命令
        cmd = [
            "python", "main_run.py",
            "--mode=noise_edit",
            f"--dataset_yaml={yaml_file}",
            "--edit_type=data_augment",
            f"--edit_strength={strength:.2f}",
            "--skip=20"  # 保持较多的原始信息
        ]
        
        # 运行命令
        print(f"运行变体 {variant+1}/{num_variants}, 强度: {strength:.2f}")
        process = subprocess.run(cmd, check=True)
        
        # 移动结果到输出目录
        # 注意: 需要找到刚刚生成的结果目录，这通常是最新创建的目录
        results_dir = sorted(Path("./results").glob("noise_edit_data_augment_*"), 
                            key=os.path.getmtime, reverse=True)[0]
        
        # 移动所有生成的图像到输出目录
        for img_file in results_dir.glob("**/*.png"):
            # 为每个变体创建唯一的文件名
            base_name = img_file.stem
            new_name = f"{base_name}_variant{variant}_strength{strength:.2f}.png"
            shutil.copy(img_file, os.path.join(output_dir, new_name))

def main():
    parser = argparse.ArgumentParser(description="训练数据集扩增工具")
    parser.add_argument("--input_dir", required=True, help="输入图像目录")
    parser.add_argument("--output_dir", required=True, help="输出图像目录")
    parser.add_argument("--variants", type=int, default=3, help="每张图像生成的变体数量")
    parser.add_argument("--prompt", default="", help="源提示词，可以为空")
    parser.add_argument("--min_strength", type=float, default=0.2, help="最小变换强度")
    parser.add_argument("--max_strength", type=float, default=0.8, help="最大变换强度")
    args = parser.parse_args()

    # 创建临时YAML文件
    temp_yaml = "temp_augment_dataset.yaml"
    num_images = create_augmentation_yaml(args.input_dir, temp_yaml, args.prompt)
    print(f"找到 {num_images} 张图像需要处理")

    # 运行增强处理
    run_augmentation(
        temp_yaml, 
        args.output_dir, 
        num_variants=args.variants,
        strength_range=(args.min_strength, args.max_strength)
    )
    
    # 清理
    os.remove(temp_yaml)
    print(f"完成! 已在 {args.output_dir} 中生成 {num_images * args.variants} 张增强图像")

if __name__ == "__main__":
    main() 