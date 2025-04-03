import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics.pairwise import cosine_similarity
import random
from tqdm import tqdm
import yaml
import shutil
from pathlib import Path

# 导入我们的DDPM模型和融合功能
import sys
sys.path.append('.')
from main_run import fuse_noise_spaces, inversion_forward_process, inversion_reverse_process

def parse_args():
    parser = argparse.ArgumentParser(description="训练集相似图像融合扩增工具")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="输入训练集目录")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="输出扩增后的训练集目录")
    parser.add_argument("--temp_yaml", type=str, default="temp_fusion.yaml",
                        help="临时配置文件路径")
    parser.add_argument("--similarity_threshold", type=float, default=0.7,
                        help="图像相似度阈值，大于该值的图像对被认为足够相似，可以融合")
    parser.add_argument("--max_pairs", type=int, default=100,
                        help="最大融合对数，防止生成过多图像")
    parser.add_argument("--fusion_modes", type=str, default="linear,cross_fade,feature_selective,layer_selective",
                        help="要使用的融合模式，逗号分隔")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="使用的设备")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="特征提取批次大小")
    parser.add_argument("--copy_original", action="store_true",
                        help="是否将原始图像复制到输出目录")
    return parser.parse_args()

def load_model(device):
    """加载预训练的ResNet50模型用于特征提取"""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # 移除最后的全连接层
    model = model.to(device)
    model.eval()
    return model

def extract_features(model, image_paths, device, batch_size=16):
    """批量提取图像特征"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    features = []
    print("提取图像特征...")
    
    # 批处理提取特征
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                batch_images.append(img_tensor)
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
                # 添加一个零张量作为占位符
                batch_images.append(torch.zeros(3, 224, 224))
        
        if batch_images:
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad():
                batch_features = model(batch_tensor)
                batch_features = batch_features.squeeze().cpu().numpy()
                
                # 处理单图像批次的情况
                if len(batch_paths) == 1:
                    batch_features = batch_features.reshape(1, -1)
                
                features.append(batch_features)
    
    # 合并所有批次的特征
    if features:
        features = np.vstack(features)
    else:
        features = np.array([])
    
    return features

def find_similar_pairs(features, image_paths, threshold=0.7, max_pairs=100):
    """找到相似的图像对"""
    n = len(features)
    similarity_matrix = cosine_similarity(features)
    
    # 创建相似对列表
    similar_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            sim = similarity_matrix[i, j]
            if sim > threshold:
                similar_pairs.append((image_paths[i], image_paths[j], sim))
    
    # 按相似度排序并限制数量
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    if max_pairs > 0 and len(similar_pairs) > max_pairs:
        similar_pairs = similar_pairs[:max_pairs]
    
    return similar_pairs

def create_temp_yaml(img1_path, img2_path, temp_yaml_path):
    """创建临时配置文件用于融合处理"""
    config = {
        "example1": {
            "init_image": img1_path,
            "source_prompt": "image1",
            "target_prompts": ["image1"]
        }
    }
    
    os.makedirs(os.path.dirname(temp_yaml_path), exist_ok=True)
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(config, f)

def fuse_images(img1_path, img2_path, temp_yaml_path, fusion_mode, fusion_ratio=0.5):
    """调用main_run中的融合功能"""
    from main_run import main
    
    # 创建临时配置文件
    create_temp_yaml(img1_path, img2_path, temp_yaml_path)
    
    # 构造命令行参数
    sys.argv = [
        "main_run.py",
        "--mode=image_fusion",
        f"--dataset_yaml={temp_yaml_path}",
        f"--second_image={img2_path}",
        f"--fusion_mode={fusion_mode}",
        f"--fusion_ratio={fusion_ratio}"
    ]
    
    # 执行融合
    result_path = main()
    return result_path

def generate_augmented_dataset(args):
    """生成融合扩增数据集"""
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取输入目录中的所有图像
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print(f"在 {args.input_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 加载模型并提取特征
    model = load_model(args.device)
    features = extract_features(model, image_paths, args.device, args.batch_size)
    
    # 寻找相似图像对
    print("寻找相似图像对...")
    similar_pairs = find_similar_pairs(features, image_paths, 
                                      args.similarity_threshold, 
                                      args.max_pairs)
    
    print(f"找到 {len(similar_pairs)} 对相似图像")
    
    # 解析融合模式
    fusion_modes = args.fusion_modes.split(',')
    
    # 对相似对进行融合
    augmented_images = []
    print("开始融合图像...")
    for img1_path, img2_path, similarity in tqdm(similar_pairs):
        # 为每对图像随机选择融合模式和融合比例
        fusion_mode = random.choice(fusion_modes)
        fusion_ratio = random.uniform(0.3, 0.7)  # 在0.3-0.7之间随机选择比例
        
        try:
            # 执行融合
            result_path = fuse_images(img1_path, img2_path, args.temp_yaml, 
                                     fusion_mode, fusion_ratio)
            
            if result_path and os.path.exists(result_path):
                # 获取图像名称用于生成输出文件名
                img1_name = os.path.splitext(os.path.basename(img1_path))[0]
                img2_name = os.path.splitext(os.path.basename(img2_path))[0]
                
                # 构造输出路径
                output_name = f"{img1_name}_{img2_name}_{fusion_mode}_{fusion_ratio:.2f}.png"
                output_path = os.path.join(args.output_dir, output_name)
                
                # 复制生成的融合图像到输出目录
                shutil.copy(result_path, output_path)
                augmented_images.append(output_path)
        except Exception as e:
            print(f"融合图像 {img1_path} 和 {img2_path} 时出错: {e}")
    
    # 如果需要，复制原始图像到输出目录
    if args.copy_original:
        print("复制原始图像到输出目录...")
        for img_path in tqdm(image_paths):
            output_path = os.path.join(args.output_dir, os.path.basename(img_path))
            shutil.copy(img_path, output_path)
    
    print(f"完成! 生成了 {len(augmented_images)} 张融合图像")
    print(f"扩增后的数据集保存在: {args.output_dir}")
    
    # 清理临时文件
    if os.path.exists(args.temp_yaml):
        os.remove(args.temp_yaml)

if __name__ == "__main__":
    args = parse_args()
    generate_augmented_dataset(args) 