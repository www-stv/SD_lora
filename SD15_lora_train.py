import os
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
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for a Single Style")

    # 模型参数
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="预训练模型路径或名称"
    )

    parser.add_argument(
        "--style_name",
        type=str,
        default="00",
        help="要训练的风格文件夹名称"
    )

    # 数据参数 - 修改为单个风格目录
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

    # 训练参数
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="图像分辨率"
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

    # 学习率
    parser.add_argument(
        "--unet_lr",
        type=float,
        default=5e-4,
        help="UNet学习率"
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=2e-5,
        help="文本编码器学习率"
    )

    # LoRA参数
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
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
        "--gradient_clip_norm",
        type=float,
        default=1.0,
        help="梯度裁剪范数"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="最大梯度范数"
    )

    return parser.parse_args()


class StyleDataset(Dataset):
    def __init__(
            self,
            style_dir,
            style_name,
            size=512,
    ):
        self.size = size
        self.style_name = style_name

        self.style_dir = Path(style_dir)
        if not self.style_dir.exists():
            raise ValueError(f"风格目录不存在: {style_dir}")

        # 收集该风格的所有图像
        self.images = []
        self.image_names = []

        for img_path in self.style_dir.glob("*.*"):
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
                self.images.append(img_path)
                self.image_names.append(img_path.stem)

        logger.info(f"加载风格 '{style_name}' 的 {len(self.images)} 张图像")

        # 图像预处理
        self.image_transforms = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img_name = self.image_names[index]

        try:
            image = Image.open(img_path).convert('RGB')
            image = exif_transpose(image)
            image = self.image_transforms(image)
        except Exception as e:
            logger.error(f"加载图像失败 {img_path}: {e}")
            return self.__getitem__((index + 1) % len(self))

        # 构建提示词
        prompt = f"{img_name} in {self.style_name} style"

        return {
            "pixel_values": image,
            "prompt": prompt
        }


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    prompt = [example["prompt"] for example in examples]

    return {
        "pixel_values": pixel_values,
        "prompt": prompt,
    }


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_model(unet, text_encoder, output_dir, epoch, loss, args):
    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unet)
    )

    text_encoder_lora_layers = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(text_encoder)
    )

    StableDiffusionPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=text_encoder_lora_layers,
    )

    # 保存信息
    info_path = os.path.join(output_dir, "model_info.txt")
    with open(info_path, "w") as f:
        f.write(f"epoch: {epoch + 1}\n")
        f.write(f"loss: {loss:.6f}\n")
        f.write(f"model: {args.pretrained_model_name_or_path}\n")
        f.write("\n训练参数:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1. 初始化Accelerator
    mixed_precision = "fp16" if torch.cuda.is_available() else "no"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    device = accelerator.device
    dtype = torch.float16 if mixed_precision == "fp16" else torch.float32

    # 2. 准备数据集
    style_dir = os.path.join(args.train_data_dir, args.style_name)
    logger.info(f"准备数据集: {style_dir}")

    train_dataset = StyleDataset(
        style_dir=style_dir,
        style_name=args.style_name,
        size=args.resolution,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # 3. 加载模型组件
    logger.info("加载SD1.5模型组件...")

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer"
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    # VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=dtype
    )

    # 文本编码器
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=dtype
    )

    # UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=dtype
    )

    # 移动到设备
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)

    # 启用梯度检查点
    vae.gradient_checkpointing_enable()
    unet.enable_gradient_checkpointing()
    text_encoder.gradient_checkpointing_enable()

    # 冻结所有基础模型
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
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

    logger.info("添加LoRA到文本编码器...")
    text_encoder_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
    )
    text_encoder.add_adapter(text_encoder_lora_config)

    # 将所有可训练参数强制转换为FP32
    logger.info("将可训练参数转换为 FP32...")
    models_to_cast = [unet, text_encoder]
    for model in models_to_cast:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(torch.float32)

    # 收集可训练参数
    unet_trainable_params = [p for p in unet.parameters() if p.requires_grad]
    text_encoder_params = [p for p in text_encoder.parameters() if p.requires_grad]

    logger.info(f"UNet可训练参数数量: {len(unet_trainable_params)}")
    logger.info(f"文本编码器可训练参数数量: {len(text_encoder_params)}")

    # 5. 设置优化器、学习率调度器
    params_to_optimize = [
        {"params": unet_trainable_params, "lr": args.unet_lr},
        {"params": text_encoder_params, "lr": args.text_encoder_lr},
    ]
    all_trainable_params = unet_trainable_params + text_encoder_params

    optimizer = AdamW(
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

    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
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
    logger.info(f"  模型 = {args.pretrained_model_name_or_path}")

    best_loss = float('inf')

    # 为当前风格创建输出目录
    style_output_dir = os.path.join(args.output_dir, args.style_name)
    os.makedirs(style_output_dir, exist_ok=True)

    # 训练循环
    for epoch in range(args.num_train_epochs):
        unet.train()
        text_encoder.train()

        epoch_losses = []
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}")

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(unet):  # 使用accelerator管理梯度累积
                pixel_values = batch['pixel_values'].to(device, dtype=dtype)
                prompt = batch["prompt"]

                # 编码图像
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

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
                    text_inputs = tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    text_input_ids = text_inputs.input_ids.to(device)

                    encoder_hidden_states = text_encoder(
                        text_input_ids,
                        return_dict=False
                    )[0]

                # 预测噪声
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states
                ).sample

                # 处理方差预测
                if model_pred.shape[1] == 6:
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)

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
            # 保存前确保unwrap模型
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)

            save_model(
                unet=unwrapped_unet,
                text_encoder=unwrapped_text_encoder,
                output_dir=style_output_dir,
                epoch=epoch,
                loss=best_loss,
                args=args
            )
            logger.info("保存新的最佳模型！")

    logger.info(f"训练完成！最佳损失: {best_loss:.6f}")

if __name__ == "__main__":
    main()