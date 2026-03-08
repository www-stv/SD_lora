import os
import argparse
import torch
from diffusers import StableDiffusionXLPipeline
from peft import LoraConfig, set_peft_model_state_dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate with single-style LoRA for SDXL")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="预训练模型路径（SDXL）"
    )

    parser.add_argument(
        "--lora_root",
        type=str,
        default="./lora_weights",
        help="LoRA权重根目录"
    )

    parser.add_argument(
        "--style",
        type=str,
        default="00",
        help="要使用的风格名称 (如 00)"
    )

    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["a cat", "a glasses", "a dog"],
        help="生成提示词列表"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_outputs",
        help="输出目录"
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
        help="推理步数"
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="引导比例"
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,  # SDXL默认1024
        help="生成图像分辨率"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )

    # LoRA参数（必须和训练时一致）
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LoRA秩的大小（必须和训练时一致）"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA缩放因子（必须和训练时一致）"
    )

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sdxl_lora_to_pipe(pipe, lora_path, rank=8, lora_alpha=16, device="cpu"):

    # 1. 加载保存的权重
    combined_dict = torch.load(os.path.join(lora_path, "pytorch_lora_weights.bin"), map_location=device)

    logger.info(f"加载的权重键: {list(combined_dict.keys())}")

    # 2. 为UNet添加LoRA适配器
    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj"],
    )
    pipe.unet.add_adapter(unet_lora_config)

    # 3. 为两个文本编码器添加LoRA适配器
    text_encoder_one_lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
    )
    pipe.text_encoder.add_adapter(text_encoder_one_lora_config)

    text_encoder_two_lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
    )
    pipe.text_encoder_2.add_adapter(text_encoder_two_lora_config)

    # 4. 加载LoRA权重
    set_peft_model_state_dict(pipe.unet, combined_dict["unet"])
    set_peft_model_state_dict(pipe.text_encoder, combined_dict["text_encoder_one"])
    set_peft_model_state_dict(pipe.text_encoder_2, combined_dict["text_encoder_two"])

    return pipe


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = os.path.join(args.output_dir, args.style)
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载基础模型（SDXL）
    logger.info("加载SDXL基础模型...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        variant="fp16" if torch.cuda.is_available() else None,
        use_safetensors=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载LoRA权重
    lora_path = os.path.join(args.lora_root, args.style)

    try:
        logger.info(f"加载LoRA从: {lora_path}")
        pipe.load_lora_weights(lora_path)
    except Exception:
        try:
            logger.error(f"LoRA官方加载方法失败，手动加载成功")
            pipe = load_sdxl_lora_to_pipe(
                pipe,
                lora_path,
                rank=args.rank,
                lora_alpha=args.lora_alpha,
                device=device
            )
        except Exception as e:
            logger.error(f"❌ LoRA加载失败: {e}")
            return

    # 3. 移动并启用模型
    if device.type == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
    else:
        pipe.enable_attention_slicing()

    pipe.unet.enable_adapters()
    pipe.text_encoder.enable_adapters()
    pipe.text_encoder_2.enable_adapters()

    # 4. 准备生成
    logger.info("开始生成图像...")

    # 记录种子用于复现
    seed_dict = {}

    for i, prompt_base in enumerate(args.prompts):
        # 为每个提示词使用不同的种子
        current_seed = args.seed + i
        set_seed(current_seed)
        seed_dict[prompt_base] = current_seed

        # 构建完整提示词
        full_prompt = f"{prompt_base} in {args.style} style"
        logger.info(f"生成 {i + 1}/{len(args.prompts)}: {full_prompt}")

        # 生成图像
        with torch.no_grad():
            image = pipe(
                prompt=full_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.resolution,
                width=args.resolution,
            ).images[0]

        # 保存图像
        output_path = os.path.join(output_dir, f"{prompt_base}.png")
        image.save(output_path)
        logger.info(f"✅ 图像已保存: {output_path}")

    logger.info(f"所有图像生成完成！")


if __name__ == "__main__":
    main()