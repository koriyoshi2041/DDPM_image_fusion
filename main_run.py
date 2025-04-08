import argparse
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
import os
from prompt_to_prompt.ptp_classes import AttentionStore, AttentionReplace, AttentionRefine, EmptyControl,load_512
from prompt_to_prompt.ptp_utils import register_attention_control, text2image_ldm_stable, view_images
from ddm_inversion.inversion_utils import  inversion_forward_process, inversion_reverse_process
from ddm_inversion.utils import image_grid,dataset_from_yaml

from torch import autocast, inference_mode
from ddm_inversion.ddim_inversion import ddim_inversion

import calendar
import time
import torch
import numpy as np
import torch.nn.functional as F

# 添加噪声空间编辑函数
def edit_noise_space(zs, edit_type, strength=1.0):
    """
    直接编辑噪声空间而不需要知道图像内容
    
    Args:
        zs: 噪声空间张量
        edit_type: 编辑类型 (brightness, contrast, saturation, style, etc.)
        strength: 编辑强度
    
    Returns:
        编辑后的噪声空间张量
    """
    edited_zs = zs.clone()
    
    if edit_type == "brightness":
        # 亮度调整 - 在中间层次增加噪声
        mid_layers = len(zs) // 3
        edited_zs[mid_layers:2*mid_layers] = edited_zs[mid_layers:2*mid_layers] + strength * 0.1
    
    elif edit_type == "contrast":
        # 对比度 - 放大噪声
        edited_zs = edited_zs * (1.0 + strength * 0.2)
    
    elif edit_type == "saturation":
        # 饱和度 - 调整高频噪声
        high_freq_layers = len(zs) // 4
        edited_zs[-high_freq_layers:] = edited_zs[-high_freq_layers:] * (1.0 + strength * 0.3)
    
    elif edit_type == "style":
        # 风格化 - 在多个尺度添加随机噪声
        for i in range(len(zs)):
            if i % 3 == 0:  # 每三层添加一次
                random_noise = torch.randn_like(zs[i]) * 0.1 * strength
                edited_zs[i] = edited_zs[i] + random_noise
    
    elif edit_type == "smooth":
        # 平滑 - 降低高频噪声
        high_freq_layers = len(zs) // 3
        edited_zs[-high_freq_layers:] = edited_zs[-high_freq_layers:] * (1.0 - strength * 0.2)
        
    # 添加更激进的噪声空间编辑方法
    elif edit_type == "transform":
        # 结构变换 - 在关键时间步骤应用变换矩阵
        mid_point = len(zs) // 2
        for i in range(mid_point-5, mid_point+5):
            if i >= 0 and i < len(zs):
                # 创建一个随机变换矩阵
                batch, channels, h, w = zs[i].shape
                theta = strength * np.pi / 4.0  # 最大45度旋转
                scale = 1.0 + 0.2 * strength  # 缩放因子
                
                # 应用仿射变换
                grid = F.affine_grid(
                    torch.tensor([
                        [scale * np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), scale * np.cos(theta), 0]
                    ]).unsqueeze(0).to(zs[i].device).float(),
                    zs[i].shape
                )
                edited_zs[i] = F.grid_sample(zs[i], grid, align_corners=True)
                
                # 添加一些随机噪声以增加变化
                edited_zs[i] = edited_zs[i] + torch.randn_like(zs[i]) * 0.15 * strength
    
    elif edit_type == "restructure":
        # 完全重构 - 重新组织高频与低频特征
        num_layers = len(zs)
        
        # 创建层的新排序
        # 这会彻底改变图像的结构，同时保留某些特征
        indices = list(range(num_layers))
        if strength > 0.5:  # 高强度时完全打乱
            np.random.shuffle(indices)
        else:  # 低强度时部分打乱，保留一些结构
            # 将索引分成块，并在块内打乱
            block_size = max(1, int(num_layers * (1 - strength) / 3))
            for i in range(0, num_layers, block_size):
                end = min(i + block_size, num_layers)
                if (i // block_size) % 2 == 1:  # 只打乱部分块
                    np.random.shuffle(indices[i:end])
        
        # 重新排列噪声层
        reordered_zs = edited_zs.clone()
        for i, idx in enumerate(indices):
            if idx < num_layers:  # 安全检查
                reordered_zs[i] = edited_zs[idx]
        
        edited_zs = reordered_zs
    
    elif edit_type == "abstract":
        # 抽象化 - 放大某些噪声特征，抑制其他特征
        # 类似于抽象画的效果
        for i in range(len(zs)):
            # 随机选择通道进行强化或抑制
            channels = zs[i].shape[1]
            channel_weights = torch.ones(channels, device=zs[i].device)
            
            # 随机选择一部分通道增强，一部分通道抑制
            enhance_channels = np.random.choice(channels, size=channels//3, replace=False)
            suppress_channels = np.random.choice(
                [c for c in range(channels) if c not in enhance_channels], 
                size=channels//3, 
                replace=False
            )
            
            channel_weights[enhance_channels] = 1.0 + strength * 1.5
            channel_weights[suppress_channels] = max(0.1, 1.0 - strength * 0.8)
            
            # 应用通道权重
            for c in range(channels):
                edited_zs[i][:, c, :, :] = edited_zs[i][:, c, :, :] * channel_weights[c]
            
            # 强化边缘和纹理
            if i % 3 == 0:  # 每隔几层
                # 创建高频强化滤波器
                kernel_size = 3
                laplacian_kernel = torch.tensor([
                    [0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]
                ], dtype=torch.float, device=zs[i].device).view(1, 1, kernel_size, kernel_size)
                
                # 对每个通道应用滤波器
                high_freq = torch.zeros_like(edited_zs[i])
                for c in range(channels):
                    channel_data = edited_zs[i][:, c:c+1, :, :]
                    # 应用拉普拉斯滤波器提取高频
                    high_freq[:, c:c+1, :, :] = F.conv2d(
                        F.pad(channel_data, (1, 1, 1, 1), mode='reflect'),
                        laplacian_kernel,
                        groups=1
                    )
                
                # 添加高频成分回原始数据
                edited_zs[i] = edited_zs[i] + high_freq * strength * 0.3
    
    elif edit_type == "dream":
        # 梦境效果 - 混合不同时间步的特征
        num_layers = len(zs)
        dream_zs = edited_zs.clone()
        
        # 应用递归混合和变形
        for i in range(1, num_layers-1):
            weight_prev = 0.3 * strength
            weight_current = 1.0 - 0.6 * strength
            weight_next = 0.3 * strength
            
            # 混合前后时间步的特征
            dream_zs[i] = (
                weight_prev * edited_zs[i-1] + 
                weight_current * edited_zs[i] + 
                weight_next * edited_zs[min(i+1, num_layers-1)]
            )
            
            # 添加变形扭曲
            if i % 5 == 0:
                batch, channels, h, w = dream_zs[i].shape
                # 创建扭曲网格
                grid_x, grid_y = torch.meshgrid(
                    torch.linspace(-1, 1, w, device=dream_zs[i].device),
                    torch.linspace(-1, 1, h, device=dream_zs[i].device)
                )
                grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
                
                # 添加正弦扭曲
                displacement = 0.1 * strength * torch.sin(10 * grid_y * torch.pi)
                grid[:, :, :, 0] = grid[:, :, :, 0] + displacement
                
                displacement = 0.1 * strength * torch.sin(10 * grid_x * torch.pi)
                grid[:, :, :, 1] = grid[:, :, :, 1] + displacement
                
                # 应用扭曲
                dream_zs[i] = F.grid_sample(dream_zs[i], grid, align_corners=True)
        
        edited_zs = dream_zs
    
    elif edit_type == "random":
        # 完全随机化 - 保留一些底层结构但大量重塑
        # 这会导致最彻底的变换
        random_strength = strength * 0.7  # 控制随机性程度
        structure_strength = 1.0 - random_strength  # 控制保留的结构程度
        
        # 对每个时间步长执行不同程度的随机化
        for i in range(len(zs)):
            # 生成随机噪声
            random_noise = torch.randn_like(zs[i])
            
            # 对早期层次（更接近原始图像的部分）保留更多的原始结构
            layer_weight = structure_strength + (random_strength * i / len(zs))
            random_weight = 1.0 - layer_weight
            
            # 应用加权混合
            edited_zs[i] = layer_weight * zs[i] + random_weight * random_noise
    
    elif edit_type == "data_augment":
        # 专门为训练数据扩增设计的变换
        # 目标：生成与原始图像相似但有细微变化的图像，适合用作训练数据
        
        # 保留结构特征的比例 - 较高以确保相似性
        structure_preserve = 1.0 - (strength * 0.3)  # 即使在强度=1时，仍保留70%结构
        
        # 1. 轻微的几何变形
        if np.random.random() < 0.7:  # 70%的几率应用
            mid_point = len(zs) // 2
            variation_range = max(1, int(len(zs) * 0.1))  # 只变换大约10%的时间步
            for i in range(mid_point-variation_range, mid_point+variation_range):
                if i >= 0 and i < len(zs):
                    # 轻微变形矩阵
                    batch, channels, h, w = zs[i].shape
                    theta = strength * np.pi / 20.0  # 最大9度旋转
                    scale = 1.0 + 0.05 * strength  # 轻微缩放
                    
                    # 应用仿射变换
                    grid = F.affine_grid(
                        torch.tensor([
                            [scale * np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), scale * np.cos(theta), 0]
                        ]).unsqueeze(0).to(zs[i].device).float(),
                        zs[i].shape
                    )
                    edited_zs[i] = structure_preserve * edited_zs[i] + (1-structure_preserve) * F.grid_sample(zs[i], grid, align_corners=True)
        
        # 2. 轻微的纹理变化（主要在高频部分）
        high_freq_layers = max(1, int(len(zs) * 0.2))  # 最后20%的层级（高频细节）
        for i in range(len(zs) - high_freq_layers, len(zs)):
            # 每个通道独立添加一些轻微变化
            channels = zs[i].shape[1]
            for c in range(channels):
                # 生成平滑的随机噪声（比纯随机噪声更适合作为纹理变化）
                noise = torch.randn_like(zs[i][:, c:c+1, :, :])
                # 应用高斯模糊以创建平滑噪声
                smooth_noise = F.avg_pool2d(noise, kernel_size=3, stride=1, padding=1)
                # 混合原始数据和平滑噪声
                edited_zs[i][:, c:c+1, :, :] = structure_preserve * edited_zs[i][:, c:c+1, :, :] + (1-structure_preserve) * smooth_noise * 0.1 * strength
        
        # 3. 随机通道强度变化（模拟颜色和材质变化）
        if np.random.random() < 0.5:  # 50%的几率应用
            # 选择中间层级应用（不会影响主要结构）
            mid_layers = len(zs) // 3
            for i in range(mid_layers, 2*mid_layers):
                channels = zs[i].shape[1]
                # 为每个通道创建微小的随机强度变化
                channel_weights = torch.ones(channels, device=zs[i].device)
                for c in range(channels):
                    # 轻微的随机变化，在0.9到1.1之间
                    channel_weights[c] = 1.0 + (torch.rand(1, device=zs[i].device).item() - 0.5) * 0.2 * strength
                
                # 应用通道权重，但保持结构
                for c in range(channels):
                    edited_zs[i][:, c, :, :] = structure_preserve * edited_zs[i][:, c, :, :] + (1-structure_preserve) * edited_zs[i][:, c, :, :] * channel_weights[c]
        
        # 4. 极轻微的随机噪声（主要是为了避免完全相同的像素值）
        if strength > 0.1:  # 只有当强度足够时才添加
            for i in range(len(zs)):
                # 添加极小的噪声，几乎不可见但能确保像素值的唯一性
                tiny_noise = torch.randn_like(zs[i]) * 0.01 * strength
                edited_zs[i] = edited_zs[i] + tiny_noise
        
    return edited_zs


# 添加噪声空间融合函数
def fuse_noise_spaces(zs1, zs2, fusion_ratio=0.5, fusion_mode="linear"):
    """
    融合两个噪声空间
    
    Args:
        zs1: 第一张图像的噪声向量
        zs2: 第二张图像的噪声向量
        fusion_ratio: 融合比例，0.5表示均等融合
        fusion_mode: 融合模式，可以是"linear"线性混合或"cross_fade"渐变混合等
    
    Returns:
        融合后的噪声空间向量
    """
    assert len(zs1) == len(zs2), "两张图片的噪声向量时间步数必须相同"
    
    fused_zs = []
    
    if fusion_mode == "linear":
        # 简单线性融合
        for i in range(len(zs1)):
            fused_z = zs1[i] * (1 - fusion_ratio) + zs2[i] * fusion_ratio
            fused_zs.append(fused_z)
            
    elif fusion_mode == "cross_fade":
        # 渐变混合，时间步早期偏向图像1，后期偏向图像2
        for i in range(len(zs1)):
            # 计算动态融合比例
            step_ratio = i / len(zs1)  # 0->1
            dynamic_ratio = fusion_ratio * (1 - step_ratio) + (1 - fusion_ratio) * step_ratio
            fused_z = zs1[i] * (1 - dynamic_ratio) + zs2[i] * dynamic_ratio
            fused_zs.append(fused_z)
    
    elif fusion_mode == "feature_selective":
        # 特征选择性融合，选择每个时间步中更显著的特征
        for i in range(len(zs1)):
            # 获取两个噪声向量
            z1, z2 = zs1[i], zs2[i]
            
            # 计算特征显著性（这里用简单的幅度作为衡量）
            z1_magnitude = torch.abs(z1)
            z2_magnitude = torch.abs(z2)
            
            # 创建混合掩码，选择更显著的特征
            mask = (z1_magnitude > z2_magnitude).float()
            
            # 应用融合比例进行调整
            adjusted_mask = mask * (1 - fusion_ratio) + (1 - mask) * fusion_ratio
            
            # 融合
            fused_z = z1 * adjusted_mask + z2 * (1 - adjusted_mask)
            fused_zs.append(fused_z)
    
    elif fusion_mode == "layer_selective":
        # 层选择性融合，不同层次使用不同的融合策略
        num_layers = len(zs1)
        
        # 早期层 - 结构层（主要来自图像1或图像2，由fusion_ratio决定）
        early_end = num_layers // 3
        for i in range(early_end):
            if fusion_ratio < 0.5:
                # 偏向图像1的结构
                weight1 = 1.0 - fusion_ratio * 2  # 0.5->0, 0->1
                fused_z = zs1[i] * weight1 + zs2[i] * (1 - weight1)
            else:
                # 偏向图像2的结构
                weight2 = (fusion_ratio - 0.5) * 2  # 0.5->0, 1->1
                fused_z = zs1[i] * (1 - weight2) + zs2[i] * weight2
            fused_zs.append(fused_z)
            
        # 中期层 - 混合层（线性融合）
        mid_end = 2 * num_layers // 3
        for i in range(early_end, mid_end):
            fused_z = zs1[i] * (1 - fusion_ratio) + zs2[i] * fusion_ratio
            fused_zs.append(fused_z)
            
        # 后期层 - 纹理层（特征选择性融合）
        for i in range(mid_end, num_layers):
            z1, z2 = zs1[i], zs2[i]
            z1_magnitude = torch.abs(z1)
            z2_magnitude = torch.abs(z2)
            mask = (z1_magnitude > z2_magnitude).float()
            adjusted_mask = mask * (1 - fusion_ratio) + (1 - mask) * fusion_ratio
            fused_z = z1 * adjusted_mask + z2 * (1 - adjusted_mask)
            fused_zs.append(fused_z)
            
    return torch.stack(fused_zs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--cfg_src", type=float, default=3.5)
    parser.add_argument("--cfg_tar", type=float, default=15)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--dataset_yaml",  default="test.yaml")
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--mode",  default="our_inv", help="modes: our_inv,p2pinv,p2pddim,ddim,noise_edit,image_fusion")
    parser.add_argument("--skip",  type=int, default=36)
    parser.add_argument("--xa", type=float, default=0.6)
    parser.add_argument("--sa", type=float, default=0.2)
    # 添加噪声空间编辑参数
    parser.add_argument("--edit_type", default="brightness", 
                        help="噪声空间编辑类型: brightness,contrast,saturation,style,smooth,transform,restructure,abstract,dream,random,data_augment")
    parser.add_argument("--edit_strength", type=float, default=1.0,
                        help="噪声空间编辑强度")
    # 添加图像融合参数
    parser.add_argument("--second_image", default="", 
                        help="第二张融合图像的路径，仅在image_fusion模式下使用")
    parser.add_argument("--fusion_ratio", type=float, default=0.5,
                        help="融合比例，0表示完全是第一张图，1表示完全是第二张图")
    parser.add_argument("--fusion_mode", default="linear",
                        choices=["linear", "cross_fade", "feature_selective", "layer_selective"],
                        help="融合模式: linear(线性混合), cross_fade(渐变混合), feature_selective(特征选择), layer_selective(层选择)")
    
    args = parser.parse_args()
    full_data = dataset_from_yaml(args.dataset_yaml)

    # create scheduler
    # load diffusion model
    # model_id = "CompVis/stable-diffusion-v1-4"
    model_id = "./stable_diff_local"  # 使用本地保存的模型

    device = f"cuda:{args.device_num}"

    cfg_scale_src = args.cfg_src
    cfg_scale_tar_list = [args.cfg_tar]
    eta = args.eta # = 1
    skip_zs = [args.skip]
    xa_sa_string = f'_xa_{args.xa}_sa{args.sa}_' if args.mode=='p2pinv' else '_'

    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)

    # load/reload model:
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    for i in range(len(full_data)):
        current_image_data = full_data[i]
        image_path = current_image_data['init_img']
        image_path = '.' + image_path 
        image_folder = image_path.split('/')[1] # after '.'
        prompt_src = current_image_data.get('source_prompt', "") # default empty string
        prompt_tar_list = current_image_data['target_prompts']

        if args.mode=="p2pddim" or args.mode=="ddim":
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
            ldm_stable.scheduler = scheduler
        else:
            ldm_stable.scheduler = DDIMScheduler.from_config(model_id, subfolder = "scheduler")
            
        ldm_stable.scheduler.set_timesteps(args.num_diffusion_steps)

        # load image
        offsets=(0,0,0,0)
        x0 = load_512(image_path, *offsets, device)

        # vae encode image
        with autocast("cuda"), inference_mode():
            w0 = (ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).float()

        # find Zs and wts - forward process
        if args.mode=="p2pddim" or args.mode=="ddim":
            wT = ddim_inversion(ldm_stable, w0, prompt_src, cfg_scale_src)
        else:
            wt, zs, wts = inversion_forward_process(ldm_stable, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src, prog_bar=True, num_inference_steps=args.num_diffusion_steps)

        # 为噪声编辑模式创建保存路径
        if args.mode == "noise_edit":
            save_path = os.path.join(f'./results/', args.mode+'_'+args.edit_type+'_'+str(args.edit_strength)+'_'+str(time_stamp), 
                                     image_path.split(sep='.')[0])
            os.makedirs(save_path, exist_ok=True)
            
            # 应用噪声空间编辑
            edited_zs = edit_noise_space(zs, args.edit_type, args.edit_strength)
            
            # 反向过程
            controller = AttentionStore()
            register_attention_control(ldm_stable, controller)
            # 使用原始提示词（因为我们不需要知道图像内容）
            w0, _ = inversion_reverse_process(ldm_stable, xT=wts[args.num_diffusion_steps-args.skip], 
                                             etas=eta, prompts=[prompt_src], 
                                             cfg_scales=[cfg_scale_src], prog_bar=True, 
                                             zs=edited_zs[:(args.num_diffusion_steps-args.skip)], 
                                             controller=controller)
            
            # vae decode image
            with autocast("cuda"), inference_mode():
                x0_dec = ldm_stable.vae.decode(1 / 0.18215 * w0).sample
            if x0_dec.dim()<4:
                x0_dec = x0_dec[None,:,:,:]
            img = image_grid(x0_dec)
               
            # 保存输出
            current_GMT = time.gmtime()
            time_stamp_name = calendar.timegm(current_GMT)
            image_name_png = f'edit_type_{args.edit_type}_' + f'strength_{args.edit_strength}_{time_stamp_name}' + ".png"
            save_full_path = os.path.join(save_path, image_name_png)
            img.save(save_full_path)
            continue  # 跳过下面的代码，处理下一张图片
        
        # 图像融合模式
        elif args.mode == "image_fusion":
            # 检查第二张图像参数
            if not args.second_image:
                print("错误: 图像融合模式需要指定第二张图像路径 (--second_image)")
                continue
                
            # 创建保存路径
            save_path = os.path.join(f'./results/', 
                                    f'image_fusion_{args.fusion_mode}_{args.fusion_ratio}_{time_stamp}', 
                                    f'{image_path.split(sep=".")[0]}_with_{os.path.basename(args.second_image)}')
            os.makedirs(save_path, exist_ok=True)
            
            # 加载第二张图像
            second_image_path = args.second_image
            # 确保路径正确
            if not second_image_path.startswith(".") and not second_image_path.startswith("/"):
                second_image_path = "." + second_image_path
                
            # 加载第二张图像
            x0_second = load_512(second_image_path, *offsets, device)
            
            # 使用空提示词来代替实际提示词
            empty_prompt = "a photo"  # 使用通用空提示词
            
            # VAE编码第二张图像
            with autocast("cuda"), inference_mode():
                w0_second = (ldm_stable.vae.encode(x0_second).latent_dist.mode() * 0.18215).float()
                
            # 第二张图像的前向过程 - 使用空提示词
            wt_second, zs_second, wts_second = inversion_forward_process(
                ldm_stable, w0_second, etas=eta, prompt=empty_prompt, 
                cfg_scale=cfg_scale_src, prog_bar=True, 
                num_inference_steps=args.num_diffusion_steps
            )
            
            # 融合两个噪声空间
            fused_zs = fuse_noise_spaces(
                zs, zs_second,
                fusion_ratio=args.fusion_ratio,
                fusion_mode=args.fusion_mode
            )
            
            # 确定使用哪个图像的xT
            if args.fusion_ratio <= 0.5:
                xT_base = wts[args.num_diffusion_steps-args.skip]
            else:
                xT_base = wts_second[args.num_diffusion_steps-args.skip]
            
            # 使用空提示词作为生成提示
            generation_prompt = empty_prompt
            
            # 反向过程生成融合图像
            controller = AttentionStore()
            register_attention_control(ldm_stable, controller)
            w0_fused, _ = inversion_reverse_process(
                ldm_stable,
                xT=xT_base,
                etas=eta,
                prompts=[generation_prompt],
                cfg_scales=[cfg_scale_src],
                prog_bar=True,
                zs=fused_zs[:(args.num_diffusion_steps-args.skip)],
                controller=controller
            )
            
            # VAE解码生成图像
            with autocast("cuda"), inference_mode():
                x0_fused = ldm_stable.vae.decode(1 / 0.18215 * w0_fused).sample
            if x0_fused.dim() < 4:
                x0_fused = x0_fused[None, :, :, :]
            fused_img = image_grid(x0_fused)
            
            # 保存结果
            current_GMT = time.gmtime()
            time_stamp_name = calendar.timegm(current_GMT)
            result_name = f'fusion_{args.fusion_mode}_ratio_{args.fusion_ratio}_{time_stamp_name}.png'
            save_full_path = os.path.join(save_path, result_name)
            fused_img.save(save_full_path)
            
            # 同时保存原始图像，便于比较
            img1_path = os.path.join(save_path, 'original_image1.png')
            img2_path = os.path.join(save_path, 'original_image2.png')
            image_grid(x0).save(img1_path)
            image_grid(x0_second).save(img2_path)
            
            print(f"融合图像已保存到: {save_full_path}")
            continue  # 跳过下面的代码，处理下一张图片

        # iterate over decoder prompts
        for k in range(len(prompt_tar_list)):
            prompt_tar = prompt_tar_list[k]
            save_path = os.path.join(f'./results/', args.mode+xa_sa_string+str(time_stamp), image_path.split(sep='.')[0], 'src_' + prompt_src.replace(" ", "_"), 'dec_' + prompt_tar.replace(" ", "_"))
            os.makedirs(save_path, exist_ok=True)

            # Check if number of words in encoder and decoder text are equal
            src_tar_len_eq = (len(prompt_src.split(" ")) == len(prompt_tar.split(" ")))

            for cfg_scale_tar in cfg_scale_tar_list:
                for skip in skip_zs:    
                    if args.mode=="our_inv":
                        # reverse process (via Zs and wT)
                        controller = AttentionStore()
                        register_attention_control(ldm_stable, controller)
                        w0, _ = inversion_reverse_process(ldm_stable, xT=wts[args.num_diffusion_steps-skip], etas=eta, prompts=[prompt_tar], cfg_scales=[cfg_scale_tar], prog_bar=True, zs=zs[:(args.num_diffusion_steps-skip)], controller=controller)

                    elif args.mode=="p2pinv":
                        # inversion with attention replace
                        cfg_scale_list = [cfg_scale_src, cfg_scale_tar]
                        prompts = [prompt_src, prompt_tar]
                        if src_tar_len_eq:
                            controller = AttentionReplace(prompts, args.num_diffusion_steps, cross_replace_steps=args.xa, self_replace_steps=args.sa, model=ldm_stable)
                        else:
                            # Should use Refine for target prompts with different number of tokens
                            controller = AttentionRefine(prompts, args.num_diffusion_steps, cross_replace_steps=args.xa, self_replace_steps=args.sa, model=ldm_stable)

                        register_attention_control(ldm_stable, controller)
                        w0, _ = inversion_reverse_process(ldm_stable, xT=wts[args.num_diffusion_steps-skip], etas=eta, prompts=prompts, cfg_scales=cfg_scale_list, prog_bar=True, zs=zs[:(args.num_diffusion_steps-skip)], controller=controller)
                        w0 = w0[1].unsqueeze(0)

                    elif args.mode=="p2pddim" or args.mode=="ddim":
                        # only z=0
                        if skip != 0:
                            continue
                        prompts = [prompt_src, prompt_tar]
                        if args.mode=="p2pddim":
                            if src_tar_len_eq:
                                controller = AttentionReplace(prompts, args.num_diffusion_steps, cross_replace_steps=.8, self_replace_steps=0.4, model=ldm_stable)
                            # Should use Refine for target prompts with different number of tokens
                            else:
                                controller = AttentionRefine(prompts, args.num_diffusion_steps, cross_replace_steps=.8, self_replace_steps=0.4, model=ldm_stable)
                        else:
                            controller = EmptyControl()

                        register_attention_control(ldm_stable, controller)
                        # perform ddim inversion
                        cfg_scale_list = [cfg_scale_src, cfg_scale_tar]
                        w0, latent = text2image_ldm_stable(ldm_stable, prompts, controller, args.num_diffusion_steps, cfg_scale_list, None, wT)
                        w0 = w0[1:2]
                    else:
                        raise NotImplementedError
                    
                    # vae decode image
                    with autocast("cuda"), inference_mode():
                        x0_dec = ldm_stable.vae.decode(1 / 0.18215 * w0).sample
                    if x0_dec.dim()<4:
                        x0_dec = x0_dec[None,:,:,:]
                    img = image_grid(x0_dec)
                       
                    # same output
                    current_GMT = time.gmtime()
                    time_stamp_name = calendar.timegm(current_GMT)
                    image_name_png = f'cfg_d_{cfg_scale_tar}_' + f'skip_{skip}_{time_stamp_name}' + ".png"

                    save_full_path = os.path.join(save_path, image_name_png)
                    img.save(save_full_path)