import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision import transforms
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm import tqdm
import numpy as np
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for a Single Style on SDXL")

    parser.add_argument(
        "--style_name",
        type=str,
        default="00",
        help="要训练的风格文件夹名称"
    )

    # 数据参数
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="./train_images",
        help="训练数据根目录"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_weights",
        help="LoRA权重输出根目录"
    )

    # 训练参数 - SDXL默认分辨率1024
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="图像分辨率（SDXL推荐1024）"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="训练批次大小"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=80,
        help="训练轮数"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=3,  # 新增：梯度累积步数
        help="梯度累积步数"
    )

    # 学习率 - SDXL通常需要稍小的学习率
    parser.add_argument(
        "--unet_lr",
        type=float,
        default=2e-5,
        help="UNet学习率"
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=2e-6,
        help="文本编码器学习率"
    )

    # LoRA参数
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LoRA秩的大小"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA缩放因子"
    )

    # 训练选项
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["constant", "linear", "cosine"],
        help="学习率调度器类型"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="梯度裁剪范数"
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        default=True,
        help="使用8-bit Adam优化器节省显存"
    )

    return parser.parse_args()


class StyleDataset(Dataset):
    def __init__(self, style_dir, style_name, vae, device, size=1024):
        self.style_name = style_name

        # 1. 收集图像路径
        self.image_paths = []
        for img_path in Path(style_dir).glob("*.*"):
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                self.image_paths.append(img_path)

        # 2. 预处理图像并缓存latents
        self.cached_data = []

        # 图像预处理（和之前一样）
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        for img_path in tqdm(self.image_paths):
            # 加载和预处理图像
            img_name = img_path.stem
            image = Image.open(img_path).convert('RGB')
            image = exif_transpose(image)
            original_size = (image.height, image.width)

            # 计算crop位置
            y1 = max(0, int(round((image.height - size) / 2.0)))
            x1 = max(0, int(round((image.width - size) / 2.0)))
            crop_top_left = (y1, x1)

            # 应用transform
            pixel_values = transform(image).unsqueeze(0)  # [1, 3, 1024, 1024]

            # VAE编码
            with torch.no_grad():
                pixel_values = pixel_values.to(device, dtype=torch.float32)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # 缓存所有需要的数据
            self.cached_data.append({
                'latents': latents.squeeze(0),  # [4, 128, 128]
                'prompt': f"{img_name} in {style_name} style",
                'original_size': original_size,
                'crop_top_left': crop_top_left,
            })

        logger.info(f"latents缓存完成！已缓存 {len(self.cached_data)} 个latents")

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        data = self.cached_data[idx]
        return {
            'latents': data['latents'],
            'prompt': data['prompt'],
            'original_size': data['original_size'],
            'crop_top_left': data['crop_top_left'],
        }


def collate_fn(examples):
    latents = torch.stack([example['latents'] for example in examples])
    prompts = [example['prompt'] for example in examples]
    original_sizes = [example['original_size'] for example in examples]
    crop_top_lefts = [example['crop_top_left'] for example in examples]

    return {
        'latents': latents,
        'prompts': prompts,
        'original_sizes': original_sizes,
        'crop_top_lefts': crop_top_lefts,
    }


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_model(unet, text_encoder_one, text_encoder_two, output_dir, epoch, loss, args):
    # 转换为diffusers格式
    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unet)
    )

    text_encoder_lora_layers = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(text_encoder_one)
    )

    text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(text_encoder_two)
    )

    # 使用官方保存函数
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=text_encoder_lora_layers,
        text_encoder_2_lora_layers=text_encoder_2_lora_layers,
    )

    # 保存信息
    info_path = os.path.join(output_dir, "model_info.txt")
    with open(info_path, "w") as f:
        f.write(f"epoch: {epoch + 1}\n")
        f.write(f"loss: {loss:.6f}\n")
        f.write(f"model: SDXL Base 1.0\n")
        f.write("\n训练参数:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1.初始化 Args 、Accelerator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision = "fp16" if device.type == "cuda" else "no"
    dtype = torch.float16 if mixed_precision == "fp16" else torch.float32
    variant = "fp16" if mixed_precision == "fp16" else None

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    device = accelerator.device

    # 2. 准备数据集
    style_dir = os.path.join(args.train_data_dir, args.style_name)
    logger.info(f"准备数据集: {style_dir}")

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae",
        torch_dtype=torch.float32
    )
    vae = vae.to(device)
    vae.requires_grad_(False)
    vae.eval()

    train_dataset = StyleDataset(
        style_dir=style_dir,
        style_name=args.style_name,
        vae=vae,
        device=device,
        size=args.resolution,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # 释放显存
    if device.type == "cuda":
        vae.to('cpu')
        del vae
        torch.cuda.empty_cache()
    else:
        del vae

    # 3. 加载模型组件
    logger.info("加载SDXL模型组件...")

    # 第一个tokenizer和text encoder
    tokenizer_one = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="tokenizer"
    )

    # 第二个tokenizer
    tokenizer_two = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="tokenizer_2"
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="scheduler",
    )

    # 第一个文本编码器
    text_encoder_one = CLIPTextModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="text_encoder",
        torch_dtype=dtype,
        variant=variant
    )

    # 第二个文本编码器
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="text_encoder_2",
        torch_dtype=dtype,
        variant=variant
    )

    # UNet
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet",
        torch_dtype=dtype,
        variant=variant
    )

    # 移动到设备
    text_encoder_one = text_encoder_one.to(device)
    text_encoder_two = text_encoder_two.to(device)
    unet = unet.to(device)

    # 启用梯度检查点
    unet.enable_gradient_checkpointing()
    text_encoder_one.gradient_checkpointing_enable()
    text_encoder_two.gradient_checkpointing_enable()

    # 冻结所有基础模型
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # 4. 添加LoRA
    logger.info("添加LoRA到UNet...")
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj"],
    )
    unet.add_adapter(unet_lora_config)

    logger.info("添加LoRA到第一个文本编码器...")
    text_encoder_one_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
    )
    text_encoder_one.add_adapter(text_encoder_one_lora_config)

    logger.info("添加LoRA到第二个文本编码器...")
    text_encoder_two_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
    )
    text_encoder_two.add_adapter(text_encoder_two_lora_config)

    # 将所有可训练参数强制转换为FP32（官方方案）
    logger.info("将可训练参数转换为 FP32...")
    models_to_cast = [unet]
    models_to_cast.extend([text_encoder_one, text_encoder_two])
    for model in models_to_cast:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(torch.float32)

    # 收集可训练参数
    unet_trainable_params = [p for p in unet.parameters() if p.requires_grad]
    text_encoder_one_params = [p for p in text_encoder_one.parameters() if p.requires_grad]
    text_encoder_two_params = [p for p in text_encoder_two.parameters() if p.requires_grad]

    logger.info(f"UNet可训练参数数量: {len(unet_trainable_params)}")
    logger.info(f"第一个文本编码器可训练参数数量: {len(text_encoder_one_params)}")
    logger.info(f"第二个文本编码器可训练参数数量: {len(text_encoder_two_params)}")

    # 5. 设置优化器、学习率调度器
    params_to_optimize = [
        {"params": unet_trainable_params, "lr": args.unet_lr},
        {"params": text_encoder_one_params, "lr": args.text_encoder_lr},
        {"params": text_encoder_two_params, "lr": args.text_encoder_lr},
    ]
    all_trainable_params = unet_trainable_params + text_encoder_one_params + text_encoder_two_params

    if args.use_8bit_adam:
        try:
            import bitsandbytes
            optimizer_class = bitsandbytes.optim.AdamW8bit
            logger.info("使用 8-bit Adam 优化器")
        except ImportError:
            logger.warning("bitsandbytes未安装，使用普通AdamW")
            optimizer_class = AdamW
    else:
        optimizer_class = AdamW

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.unet_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    # 学习率调度器
    total_training_steps = args.num_train_epochs * len(train_dataloader) // args.gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_training_steps),
        num_training_steps=total_training_steps,
    )

    unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
    )

    # 6. 训练循环
    logger.info("***** 训练配置 *****")
    logger.info(f"  风格 = {args.style_name}")
    logger.info(f"  样本数 = {len(train_dataset)}")
    logger.info(f"  批次大小 = {args.train_batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    logger.info(f"  有效批次大小 = {args.train_batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  训练轮数 = {args.num_train_epochs}")
    logger.info(f"  总训练步数 = {total_training_steps}")
    logger.info(f"  分辨率 = {args.resolution}")
    logger.info(f"  模型 = SDXL Base 1.0")

    best_loss = float('inf')

    # 为当前风格创建输出目录
    style_output_dir = os.path.join(args.output_dir, args.style_name)
    os.makedirs(style_output_dir, exist_ok=True)

    # 训练循环
    for epoch in range(args.num_train_epochs):
        unet.train()
        text_encoder_one.train()
        text_encoder_two.train()

        epoch_losses = []
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}")

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(unet):  # 使用accelerator管理梯度累积
                latents = batch['latents'].to(device, dtype=dtype)
                prompts = batch["prompts"]
                original_sizes = batch["original_sizes"]
                crop_top_lefts = batch["crop_top_lefts"]

                # 采样噪声和时间步
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 编码提示词
                with torch.no_grad():
                    encoder_outputs_list = []

                    for i, (text_encoder, tokenizer) in enumerate([(text_encoder_one, tokenizer_one),
                                                                   (text_encoder_two, tokenizer_two)]):
                        text_inputs = tokenizer(
                            prompts,
                            padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt"
                        )
                        text_input_ids = text_inputs.input_ids.to(device)

                        outputs = text_encoder(
                            text_input_ids,
                            output_hidden_states=True,
                            return_dict=False
                        )

                        pooled_prompt_embeds = outputs[0]
                        hidden = outputs[-1][-2]
                        encoder_outputs_list.append(hidden)

                    # 拼接两个编码器的输出
                    prompt_embeds = torch.cat(encoder_outputs_list, dim=-1)

                # 构建time_ids
                target_size = (args.resolution, args.resolution)
                add_time_ids = torch.tensor([
                    list(orig) + list(crop) + list(target_size)
                    for orig, crop in zip(original_sizes, crop_top_lefts)
                ], device=device, dtype=dtype)

                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                }

                # 预测噪声
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample

                # 计算损失
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                epoch_losses.append(loss.item())

                # 反向传播
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(all_trainable_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        # 计算epoch平均损失
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"Epoch {epoch + 1} 完成，平均损失: {avg_epoch_loss:.6f}")

        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_text_encoder_one = accelerator.unwrap_model(text_encoder_one)
            unwrapped_text_encoder_two = accelerator.unwrap_model(text_encoder_two)

            save_model(
                unet=unwrapped_unet,
                text_encoder_one=unwrapped_text_encoder_one,
                text_encoder_two=unwrapped_text_encoder_two,
                output_dir=style_output_dir,
                epoch=epoch,
                loss=best_loss,
                args=args
            )
            logger.info("保存新的最佳模型！")

    logger.info(f"训练完成！最佳损失: {best_loss:.6f}")


if __name__ == "__main__":
    main()