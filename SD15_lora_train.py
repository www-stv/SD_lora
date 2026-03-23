import os
import argparse
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
import json
import re
import random
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for a Single Style with DreamBooth")

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
        required=True,
        help="要训练的风格名称"
    )
    parser.add_argument(
        "--instance_dir",
        type=str,
        required=True,
        help="实例图片目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
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
        help="训练批次大小（每个子batch的大小）"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=200,
        help="训练轮数"
    )

    # DreamBooth参数
    parser.add_argument(
        "--class_dir",
        type=str,
        default=None,
        help="先验图片根目录（可选，不指定则只使用实例图片进行常规LoRA训练）"
    )
    parser.add_argument(
        "--diverse_loss_weight",
        type=float,
        default=0.5,
        help="泛化类别的先验损失权重"
    )
    parser.add_argument(
        "--matched_loss_weight",
        type=float,
        default=0.1,
        help="匹配类别的先验损失权重"
    )

    # TI参数
    parser.add_argument(
        "--pretrained_ti_path",
        type=str,
        default=None,
        help="预训练的Textual Inversion嵌入路径"
    )
    parser.add_argument(
        "--train_ti",
        action="store_true",
        help="是否同时训练TI嵌入（默认False，只训练LoRA）"
    )
    parser.add_argument(
        "--ti_lr",
        type=float,
        default=5e-4,
        help="TI嵌入的学习率（如果训练TI）"
    )
    parser.add_argument(
        "--ti_reg_weight",
        type=float,
        default=1e-4,
        help="TI正则化权重"
    )

    # LoRA参数
    parser.add_argument(
        "--unet_lr",
        type=float,
        default=5e-4,
        help="UNet学习率"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="LoRA秩的大小"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA缩放因子"
    )

    # 正交解耦参数
    parser.add_argument(
        "--use_orthogonal",
        action="store_true",
        help="是否使用正交解耦损失"
    )
    parser.add_argument(
        "--orthogonal_loss_weight",
        type=float,
        default=0.01,
        help="正交损失权重，用于解耦风格和内容"
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
        "--gradient_accumulation_steps",
        type=int,
        default=3,
        help="梯度累积步数"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="最大梯度范数"
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0.1,
        help="噪声偏移，增加训练稳定性"
    )

    return parser.parse_args()


def load_ti_embedding(text_encoder, tokenizer, ti_path, style_name, train_ti=False):
    # 创建风格token名称
    style_token = f"<{style_name}>"
    loaded_embedding = None

    # 1. 尝试加载预训练TI
    if ti_path is not None:
        print(f"尝试加载TI嵌入: {ti_path}")
        try:
            ti_data = torch.load(ti_path, map_location="cpu")
            loaded_token = ti_data.get('style_token', None)
            loaded_embedding = ti_data.get('embedding', None)

            if loaded_token is not None and loaded_embedding is not None:
                print(f"成功加载TI token: {loaded_token}")
                style_token = loaded_token  # 使用文件中的token名称
            else:
                print("警告: TI文件格式不正确，将创建新TI")
                loaded_embedding = None
        except Exception as e:
            print(f"加载TI文件失败: {e}，将创建新TI")
            loaded_embedding = None

    # 2. 如果没有加载到有效的TI，但需要训练TI，则创建新的
    if loaded_embedding is None and train_ti:
        print("创建新的TI token")
        # 初始化嵌入（使用"style"词汇）
        with torch.no_grad():
            init_token_ids = tokenizer.encode("style", add_special_tokens=False)
            if init_token_ids:
                init_embedding = text_encoder.get_input_embeddings().weight[init_token_ids].mean(dim=0)
                print(f"从词汇 'style' 初始化TI嵌入")
            else:
                init_embedding = torch.randn(text_encoder.config.hidden_size) * 0.02
                print("使用随机初始化")
        loaded_embedding = init_embedding

    # 3. 如果有有效的嵌入（无论是加载的还是新创建的），则添加到模型中
    if loaded_embedding is not None:
        # 检查token是否已存在
        if style_token not in tokenizer.get_vocab():
            # 添加新token
            num_added_tokens = tokenizer.add_tokens([style_token])
            print(f"添加token '{style_token}'，新增数量: {num_added_tokens}")

            # 调整text_encoder的嵌入层大小
            old_embeddings = text_encoder.get_input_embeddings()
            new_embeddings = torch.nn.Embedding(
                old_embeddings.num_embeddings + num_added_tokens,
                old_embeddings.embedding_dim
            )

            # 复制原有权重并设置新token
            with torch.no_grad():
                new_embeddings.weight.data[:-num_added_tokens] = old_embeddings.weight.data
                new_embeddings.weight.data[-num_added_tokens:] = loaded_embedding.to(old_embeddings.weight.dtype)

            text_encoder.set_input_embeddings(new_embeddings)
            print(f"已{'加载' if ti_path else '创建'}TI嵌入 '{style_token}'")
        else:
            # token已存在，直接更新
            token_id = tokenizer.convert_tokens_to_ids(style_token)
            with torch.no_grad():
                text_encoder.get_input_embeddings().weight[token_id] = loaded_embedding.to(
                    text_encoder.get_input_embeddings().weight.dtype
                )
            print(f"已更新token '{style_token}' 的嵌入")

        # 获取嵌入层和token ID
        token_embeds = text_encoder.get_input_embeddings()
        token_id = tokenizer.convert_tokens_to_ids(style_token)

        return style_token, token_id, token_embeds

    # 4. 不需要TI
    return None, None, None


class DreamBoothDataset(Dataset):
    def __init__(
            self,
            args,
            vae,
            device,
            size=512,
            style_token=None,
    ):
        self.size = size
        self.style_name = args.style_name
        self.style_token = style_token
        self.device = device
        self.instance_prompts = {}
        self.use_dreambooth = args.class_dir is not None
        self.args = args

        # 加载实例图片
        instance_dir_path = Path(args.instance_dir)
        if not instance_dir_path.exists():
            raise ValueError(f"实例目录不存在: {instance_dir_path}")

        instance_images = list(instance_dir_path.glob("*.*"))
        instance_images = [p for p in instance_images
                           if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']]
        print(f"找到 {len(instance_images)} 张实例图片")

        # 加载提示词
        prompts_path = instance_dir_path / "prompts.json"
        if prompts_path.exists():
            with open(prompts_path, 'r', encoding='utf-8') as f:
                self.instance_prompts = json.load(f)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # 编码实例图片
        self.instance_data = []
        for img_path in tqdm(instance_images, desc="编码实例图片"):
            img_name = img_path.stem
            try:
                clean_name = re.sub(r'[_-]', ' ', img_name)
                if img_name not in self.instance_prompts:
                    self.instance_prompts[img_name] = f"a photo of {clean_name}"

                image = Image.open(img_path).convert('RGB')
                image = exif_transpose(image)
                image_tensor = self.transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    latents = vae.encode(image_tensor).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # 构建提示词：如果使用TI则用TI token，否则用style_name
                if self.style_token:
                    style_text = self.style_token
                else:
                    style_text = f"{self.style_name}"

                self.instance_data.append({
                    'latents': latents.squeeze(0).cpu(),
                    'prompt': f"{self.instance_prompts[img_name]} in {style_text} style",
                    'class_name': img_name,
                    'base_prompt': self.instance_prompts[img_name],
                })
            except Exception as e:
                print(f"处理实例图片失败 {img_path}: {e}")

        # 如果使用DreamBooth，编码先验图片
        self.matched_class_data = {}
        self.diverse_class_data = []

        if self.use_dreambooth:
            class_dir_path = Path(args.class_dir)
            if not class_dir_path.exists():
                raise ValueError(f"先验图片目录不存在: {class_dir_path}")

            all_class_dirs = [d for d in class_dir_path.iterdir() if d.is_dir()]
            matched_dirs = []
            diverse_dirs = []

            for class_dir in all_class_dirs:
                class_name = class_dir.name
                if class_name in self.instance_prompts.keys():
                    matched_dirs.append(class_dir)
                else:
                    diverse_dirs.append(class_dir)

            print(f"匹配类别: {len(matched_dirs)} 个 ({', '.join([d.name for d in matched_dirs])})")
            print(f"多样化类别: {len(diverse_dirs)} 个")

            # 处理匹配类别的先验图片
            for class_dir in matched_dirs:
                class_name = class_dir.name
                class_images = list(class_dir.glob("*.*"))
                class_images = [p for p in class_images
                                if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']]

                if not class_images:
                    print(f"警告: 未找到类别 '{class_name}' 的先验图片")
                    continue

                print(f"为匹配类别 '{class_name}' 找到 {len(class_images)} 张先验图片")

                self.matched_class_data[class_name] = []
                for img_path in tqdm(class_images, desc=f"编码先验图片 - {class_name}"):
                    try:
                        image = Image.open(img_path).convert('RGB')
                        image = exif_transpose(image)
                        image_tensor = self.transform(image).unsqueeze(0).to(device)

                        with torch.no_grad():
                            latents = vae.encode(image_tensor).latent_dist.sample()
                            latents = latents * vae.config.scaling_factor

                        self.matched_class_data[class_name].append({
                            'latents': latents.squeeze(0).cpu(),
                            'prompt': self.instance_prompts[class_name],
                            'class_name': class_name,
                        })
                    except Exception as e:
                        print(f"处理先验图片失败 {img_path}: {e}")

            # 处理多样化类别的先验图片
            for class_dir in diverse_dirs:
                class_name = class_dir.name
                class_images = list(class_dir.glob("*.*"))
                class_images = [p for p in class_images
                                if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']]

                if not class_images:
                    continue

                for img_path in tqdm(class_images, desc=f"编码多样化先验 - {class_name}"):
                    try:
                        image = Image.open(img_path).convert('RGB')
                        image = exif_transpose(image)
                        image_tensor = self.transform(image).unsqueeze(0).to(device)

                        with torch.no_grad():
                            latents = vae.encode(image_tensor).latent_dist.sample()
                            latents = latents * vae.config.scaling_factor

                        clean_name = re.sub(r'[_-]', ' ', class_name)
                        prompt = f"a photo of {clean_name}"

                        self.diverse_class_data.append({
                            'latents': latents.squeeze(0).cpu(),
                            'prompt': prompt,
                            'class_name': class_name,
                        })
                    except Exception as e:
                        print(f"处理先验图片失败 {img_path}: {e}")

    def __len__(self):
        return len(self.instance_data)

    def __getitem__(self, index):
        data = {}
        instance_item = self.instance_data[index % len(self.instance_data)]
        class_name = instance_item['class_name']

        instance_sample = {
            'latents': instance_item['latents'],
            'prompt': instance_item['prompt'],
            'class_name': class_name,
        }
        data['instance'] = instance_sample

        if class_name in self.matched_class_data and self.matched_class_data[class_name]:
            matched_sample = random.choice(self.matched_class_data[class_name])
            data['matched_prior'] = matched_sample

        if self.diverse_class_data:
            diverse_sample = random.choice(self.diverse_class_data)
            data['diverse_prior'] = diverse_sample

        return data


def collate_fn(examples):
    batch = {}
    if 'instance' in examples[0]:
        instance_batch = {
            'latents': torch.stack([ex['instance']['latents'] for ex in examples]),
            'prompts': [ex['instance']['prompt'] for ex in examples],
            'class_names': [ex['instance']['class_name'] for ex in examples],
        }
        batch['instance'] = instance_batch

    if 'matched_prior' in examples[0]:
        matched_batch = {
            'latents': torch.stack([ex['matched_prior']['latents'] for ex in examples]),
            'prompts': [ex['matched_prior']['prompt'] for ex in examples],
            'class_names': [ex['matched_prior']['class_name'] for ex in examples],
        }
        batch['matched_prior'] = matched_batch

    if 'diverse_prior' in examples[0]:
        diverse_batch = {
            'latents': torch.stack([ex['diverse_prior']['latents'] for ex in examples]),
            'prompts': [ex['diverse_prior']['prompt'] for ex in examples],
            'class_names': [ex['diverse_prior']['class_name'] for ex in examples],
        }
        batch['diverse_prior'] = diverse_batch

    return batch


def orthogonal_loss(unet):
    orth_loss = 0.0
    count = 0
    unet_layers = {}

    for name, param in unet.named_parameters():
        if param.requires_grad and 'lora' in name.lower():
            parts = name.split('.')
            layer_name = '.'.join(parts[:3])
            if layer_name not in unet_layers:
                unet_layers[layer_name] = []
            unet_layers[layer_name].append(param.view(-1))

    all_layers = list(unet_layers.values())

    for i in range(len(all_layers)):
        for j in range(i + 1, len(all_layers)):
            w1 = all_layers[i][0]
            w2 = all_layers[j][0]
            min_dim = min(w1.size(0), w2.size(0))
            w1 = w1[:min_dim]
            w2 = w2[:min_dim]
            cos_sim = torch.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0))
            orth_loss += torch.abs(cos_sim)
            count += 1

    if count > 0:
        orth_loss = orth_loss / count
    return orth_loss


def compute_loss(batch, tokenizer, text_encoder, unet, noise_scheduler, device, dtype, args):
    latents = batch['latents'].to(device, dtype=dtype)
    prompts = batch['prompts']

    noise = torch.randn_like(latents)
    if args.noise_offset > 0:
        noise += args.noise_offset * torch.randn_like(noise)

    bsz = latents.shape[0]
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (bsz,), device=device
    ).long()

    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    with torch.no_grad():
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(device)
        encoder_hidden_states = text_encoder(text_input_ids, return_dict=False)[0]

    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    if model_pred.shape[1] == 8:
        model_pred, _ = torch.chunk(model_pred, 2, dim=1)

    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    loss = nn.functional.mse_loss(model_pred.float(), target.float())
    return loss


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def save_model(unet, output_dir, epoch, loss, args, token_embeds=None, token_id=None, style_token=None):
    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unet)
    )

    StableDiffusionPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=unet_lora_state_dict,
    )

    # 保存TI嵌入（如果训练了TI）
    if args.train_ti and token_embeds is not None and token_id is not None and style_token is not None:
        with torch.no_grad():
            ti_embedding = token_embeds.weight[token_id].detach().cpu()
            torch.save({
                'style_token': style_token,
                'style_token_id': token_id,
                'embedding': ti_embedding,
            }, Path(output_dir) / "ti_embedding.pt")

    info_path = os.path.join(output_dir, "lora_info.txt")
    with open(info_path, "w") as f:
        f.write(f"epoch: {epoch + 1}\n")
        f.write(f"loss: {loss:.6f}\n")
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

    # 2. 加载VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32
    )
    vae = vae.to(device)
    vae.requires_grad_(False)
    vae.eval()

    # 3. 加载模型组件（先加载text_encoder和tokenizer以便添加TI）
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer"
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=dtype
    )
    text_encoder = text_encoder.to(device)

    # 4. 加载TI嵌入（如果提供）
    style_token = None
    token_id = None
    token_embeds = None
    orig_embeds_params = None
    index_no_updates = None

    if args.pretrained_ti_path:
        style_token, token_id, token_embeds = load_ti_embedding(
            text_encoder, tokenizer, args.pretrained_ti_path, args.style_name, args.train_ti
        )
        if token_embeds is not None:
            token_embeds = token_embeds.to(device)
        print(f"已加载TI token: {style_token} (ID: {token_id})")

        # 创建掩码，用于保护非训练token（如果训练TI）
        if args.train_ti:
            # 创建掩码，标记哪些token需要被保护
            index_no_updates = torch.ones(len(tokenizer), dtype=torch.bool, device=device)
            index_no_updates[token_id] = False

            # 保存原始嵌入（用于恢复非训练token）
            orig_embeds_params = token_embeds.weight.data.clone()
            print(f"TI保护掩码已创建，可训练token数量: {(~index_no_updates).sum().item()}")

    # 5. 准备数据集
    train_dataset = DreamBoothDataset(
        args=args,
        vae=vae,
        device=device,
        size=args.resolution,
        style_token=style_token,
    )

    # 释放VAE显存
    if device.type == "cuda":
        vae.to('cpu')
        del vae
        torch.cuda.empty_cache()
    else:
        del vae

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # 6. 加载UNet和噪声调度器
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=dtype
    )
    unet = unet.to(device)

    # 启用梯度检查点
    unet.enable_gradient_checkpointing()
    text_encoder.gradient_checkpointing_enable()

    # 冻结所有基础模型
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 7. 添加LoRA
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
    )
    unet.add_adapter(unet_lora_config)

    # 8. 设置可训练参数
    unet_trainable_params = [p for p in unet.parameters() if p.requires_grad]
    trainable_params = [{"params": unet_trainable_params, "lr": args.unet_lr}]

    # 如果训练TI，添加TI参数
    ti_trainable_param = None
    if args.train_ti and style_token and token_embeds is not None:
        # 让整个嵌入层可训练
        token_embeds.weight.requires_grad = True

        # TI参数就是整个嵌入层（但训练后会恢复非目标token）
        ti_trainable_param = token_embeds.weight
        trainable_params.append({"params": [ti_trainable_param], "lr": args.ti_lr})
        print(f"TI嵌入将参与训练，学习率: {args.ti_lr}")
        print(f"TI嵌入形状: {ti_trainable_param.shape}")

    # 将所有可训练参数强制转换为FP32
    for model in [unet, text_encoder]:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(torch.float32)

    print(f"UNet可训练参数数量: {len(unet_trainable_params)}")
    if ti_trainable_param is not None:
        print(f"TI嵌入可训练: {ti_trainable_param.shape}")

    # 9. 设置优化器
    optimizer = AdamW(
        trainable_params,
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

    # 10. 训练循环
    print("\n训练配置:")
    print(f"  风格 = {args.style_name}")
    if style_token:
        print(f"  TI token = {style_token}")
        print(f"  训练TI = {args.train_ti}")
    print(f"  训练模式 = {'DreamBooth (混合策略)' if args.class_dir is not None else '常规LoRA'}")
    print(f"  实例图片数 = {len(train_dataset.instance_data)}")
    if args.class_dir is not None:
        print(f"  匹配类别先验图片数 = {sum(len(v) for v in train_dataset.matched_class_data.values())}")
        print(f"  泛化类别先验图片数 = {len(train_dataset.diverse_class_data)}")
        print(f"  匹配先验损失权重 = {args.matched_loss_weight}")
        print(f"  泛化先验损失权重 = {args.diverse_loss_weight}")
    if args.use_orthogonal:
        print(f"  使用正交解耦")
        print(f"  正交损失权重 = {args.orthogonal_loss_weight}")
    print(f"  批次大小 = {args.train_batch_size}")
    print(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    print(f"  训练轮数 = {args.num_train_epochs}")
    print(f"  总训练步数 = {total_training_steps}")

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(args.num_train_epochs):
        unet.train()
        text_encoder.train()

        epoch_losses = {
            'total': [],
            'instance': [],
            'matched': [],
            'diverse': [],
            'orthogonal': [],
        }

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}")

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(unet):
                instance_loss = compute_loss(
                    batch['instance'], tokenizer, text_encoder, unet,
                    noise_scheduler, device, dtype, args
                )
                epoch_losses['instance'].append(instance_loss.item())

                matched_loss = 0.0
                if 'matched_prior' in batch:
                    matched_loss = compute_loss(
                        batch['matched_prior'], tokenizer, text_encoder, unet,
                        noise_scheduler, device, dtype, args
                    )
                    epoch_losses['matched'].append(matched_loss.item())

                diverse_loss = 0.0
                if 'diverse_prior' in batch:
                    diverse_loss = compute_loss(
                        batch['diverse_prior'], tokenizer, text_encoder, unet,
                        noise_scheduler, device, dtype, args
                    )
                    epoch_losses['diverse'].append(diverse_loss.item())

                orth_loss = 0.0
                if args.use_orthogonal:
                    orth_loss = orthogonal_loss(unet)
                    epoch_losses['orthogonal'].append(orth_loss.item())

                total_loss = instance_loss + args.matched_loss_weight * matched_loss + args.diverse_loss_weight * diverse_loss + args.orthogonal_loss_weight * orth_loss
                epoch_losses['total'].append(total_loss.item())

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    all_trainable = unet_trainable_params
                    if ti_trainable_param is not None:
                        all_trainable.append(ti_trainable_param)
                    accelerator.clip_grad_norm_(all_trainable, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # 关键步骤：如果训练TI，恢复非目标token的嵌入
                if args.train_ti and style_token and token_embeds is not None and index_no_updates is not None:
                    with torch.no_grad():
                        # 恢复被保护的token
                        token_embeds.weight.data[index_no_updates] = orig_embeds_params[index_no_updates]

            postfix = {"loss": f"{total_loss.item():.6f}", "inst": f"{instance_loss.item():.4f}"}
            if 'matched_prior' in batch:
                postfix["match"] = f"{matched_loss.item():.4f}"
            if 'diverse_prior' in batch:
                postfix["div"] = f"{diverse_loss.item():.4f}"
            if args.use_orthogonal:
                postfix["orth"] = f"{orth_loss.item():.4f}"
            progress_bar.set_postfix(postfix)

        avg_epoch_loss = sum(epoch_losses['total']) / len(epoch_losses['total'])
        print(f"Epoch {epoch + 1} 完成，平均损失: {avg_epoch_loss:.6f}")

        if args.class_dir is not None:
            print(f"  实例损失: {sum(epoch_losses['instance']) / len(epoch_losses['instance']):.6f}")
            if epoch_losses['matched']:
                print(f"  匹配先验损失: {sum(epoch_losses['matched']) / len(epoch_losses['matched']):.6f}")
            if epoch_losses['diverse']:
                print(f"  多样化先验损失: {sum(epoch_losses['diverse']) / len(epoch_losses['diverse']):.6f}")

        if args.use_orthogonal and epoch_losses['orthogonal']:
            print(f"  正交损失: {sum(epoch_losses['orthogonal']) / len(epoch_losses['orthogonal']):.6f}")

        save_interval = max(1, args.num_train_epochs // 10)
        if (epoch + 1) % save_interval == 0:
            unwrapped_unet = accelerator.unwrap_model(unet)
            save_model(
                unet=unwrapped_unet,
                output_dir=Path(output_dir) / f"epoch_{epoch + 1}",
                epoch=epoch,
                loss=avg_epoch_loss,
                args=args,
                token_embeds=token_embeds if args.train_ti else None,
                token_id=token_id if args.train_ti else None,
                style_token=style_token if args.train_ti else None
            )

    print("训练完成！")


if __name__ == "__main__":
    main()