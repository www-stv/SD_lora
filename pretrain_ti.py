import os
import argparse
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, DDPMScheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: Pretrain Textual Inversion")

    # 基础配置
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--style_name", type=str, required=True)
    parser.add_argument("--instance_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # TI特定参数
    parser.add_argument("--ti_lr", type=float, default=5e-3,
                        help="TI嵌入的学习率")
    parser.add_argument("--ti_epochs", type=int, default=200,
                        help="TI预训练轮数")
    parser.add_argument("--ti_reg_weight", type=float, default=1e-4,
                        help="TI正则化权重")
    parser.add_argument("--ti_token_init", type=str, default="style",
                        help="初始化TI token的词汇")

    # 其他训练参数
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


class TIDataset(Dataset):
    def __init__(self, instance_dir, style_name, tokenizer, size=512):
        self.style_name = style_name
        self.instance_images = list(Path(instance_dir).glob("*.jpg")) + \
                               list(Path(instance_dir).glob("*.png")) + \
                               list(Path(instance_dir).glob("*.jpeg"))

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(p=0.5),  # 随机翻转
            transforms.RandomRotation(degrees=10),  # 随机旋转
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        prompts_path = Path(instance_dir) / "prompts.json"
        self.prompts = {}
        if prompts_path.exists():
            with open(prompts_path, 'r') as f:
                self.prompts = json.load(f)

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.instance_images)

    def __getitem__(self, idx):
        img_path = self.instance_images[idx]
        img_name = img_path.stem

        if img_name in self.prompts:
            base_prompt = self.prompts[img_name]
        else:
            base_prompt = f"a photo of {img_name}"

        prompt = f"{base_prompt} in <{self.style_name}> style"

        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)

        return {
            'pixel_values': image_tensor,
            'prompt': prompt,
        }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载模型
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float32
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # 2. 添加新token
    style_token = f"<{args.style_name}>"
    num_added_tokens = tokenizer.add_tokens([style_token])
    print(f"添加token '{style_token}'，新增数量: {num_added_tokens}")

    style_token_id = tokenizer.convert_tokens_to_ids(style_token)

    # 3. 扩展嵌入层
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings()

    # 4. 初始化新token的嵌入
    with torch.no_grad():
        if args.ti_token_init:
            init_token_ids = tokenizer.encode(args.ti_token_init, add_special_tokens=False)
            if init_token_ids:
                init_embedding = token_embeds.weight[init_token_ids].mean(dim=0)
                print(f"从词汇 '{args.ti_token_init}' 初始化TI嵌入")
            else:
                init_embedding = torch.randn(text_encoder.config.hidden_size) * 0.02
        else:
            init_embedding = torch.randn(text_encoder.config.hidden_size) * 0.02

        token_embeds.weight[style_token_id] = init_embedding

    # 5. 创建掩码，标记哪些token需要被保护（不更新）
    index_no_updates = torch.ones(len(tokenizer), dtype=torch.bool)
    index_no_updates[style_token_id] = False

    # 保存原始嵌入（用于恢复非训练token）
    orig_embeds_params = token_embeds.weight.data.clone()

    # 6. 冻结模型
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)

    for param in text_encoder.parameters():
        param.requires_grad = False

    # 让嵌入层可训练（但训练后会恢复非目标token）
    token_embeds.weight.requires_grad = True

    # 7. 创建优化器
    optimizer = torch.optim.AdamW(
        [token_embeds.weight],
        lr=args.ti_lr,
        weight_decay=args.ti_reg_weight
    )

    # 8. 准备数据集
    dataset = TIDataset(args.instance_dir, args.style_name, tokenizer, args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)

    # 9. 训练循环
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    print(f"开始TI预训练，风格token: {style_token}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 计算总步数
    for epoch in range(args.ti_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.ti_epochs}")

        # 重置梯度累积计数器
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            pixel_values = batch['pixel_values'].to(device)
            prompts = batch['prompt']

            # 编码图片
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

            # 编码提示词
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]

            # 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 预测噪声
            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # 计算损失
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # 缩放损失（梯度累积需要）
            loss = loss / args.gradient_accumulation_steps

            # 反向传播
            loss.backward()

            total_loss += loss.item() * args.gradient_accumulation_steps

            # 每累积一定步数后更新参数
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or \
                    (batch_idx + 1) == len(dataloader):
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_([token_embeds.weight], max_norm=1.0)

                # 更新参数
                optimizer.step()

                # 关键步骤：恢复非训练token的嵌入
                with torch.no_grad():
                    token_embeds.weight.data[index_no_updates] = orig_embeds_params[index_no_updates]

                # 清零梯度
                optimizer.zero_grad()


            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{loss.item() * args.gradient_accumulation_steps:.6f}",
            })

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.6f}")

        # 保存检查点
        if (epoch + 1) % (args.ti_epochs // 10) == 0:
            save_path = output_dir / f"epoch{epoch + 1}"
            save_path.mkdir(parents=True, exist_ok=True)

            torch.save({
                'style_token': style_token,
                'style_token_id': style_token_id,
                'embedding': token_embeds.weight[style_token_id].detach().cpu(),
            }, save_path / "pretrained_ti_embedding.pt")

            info_path = os.path.join(save_path, "pretrained_ti_info.txt")
            with open(info_path, "w") as f:
                f.write(f"epoch: {epoch + 1}\n")
                f.write(f"loss: {avg_loss:.6f}\n")
                f.write(f"style_token: {style_token}\n")
                f.write("\n训练参数:\n")
                for key, value in vars(args).items():
                    f.write(f"  {key}: {value}\n")

            print(f"TI检查点已保存到 {save_path}")

    print(f"\nTI预训练完成！")


if __name__ == "__main__":
    main()