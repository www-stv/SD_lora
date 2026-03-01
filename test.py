import os
import argparse
import torch
from diffusers import StableDiffusionPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate with single-style LoRA")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="预训练模型路径"
    )

    parser.add_argument(
        "--lora_root",
        type=str,
        default="lora_weights",
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
        default=80,
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
        default=512,
        help="生成图像分辨率"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = os.path.join(args.output_dir, args.style)
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载基础模型
    logger.info("加载基础模型...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载LoRA权重
    lora_path = os.path.join(args.lora_root, args.style)
    os.makedirs(lora_path, exist_ok=True)

    try:
        logger.info(f"加载LoRA从: {lora_path}")
        combined_dict = torch.load(os.path.join(lora_path, "pytorch_lora_weights.bin"), map_location=device)

        # 分别获取UNet和文本编码器的权重
        unet_state_dict = combined_dict["unet"]
        text_encoder_state_dict = combined_dict["text_encoder"]

        # 加载到UNet,文本编码器
        pipe.unet.load_state_dict(unet_state_dict)
        pipe.text_encoder.load_state_dict(text_encoder_state_dict)

        logger.info("LoRA权重加载成功")

    except Exception as e:
        logger.error(f"❌ LoRA加载失败: {e}")

    # 3. 移动模型到设备
    pipe = pipe.to(device)
    if device.type == "cpu":
        pipe.enable_attention_slicing()

    # 4. 准备生成
    logger.info("开始生成图像...")

    # 记录种子用于复现
    seed_dict = {}

    for i, prompt_base in enumerate(args.prompts):
        # 为每个提示词使用不同的种子（基于基础种子）
        current_seed = args.seed + i
        set_seed(current_seed)
        seed_dict[prompt_base] = current_seed

        # 构建完整提示词
        full_prompt = f"{prompt_base} in {args.style} style"
        logger.info(f"生成 {i + 1}/{len(args.prompts)}: {full_prompt}")

        # 生成图像
        with torch.no_grad():
            image = pipe(
                full_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.resolution,
                width=args.resolution,
            ).images[0]

        # 保存图像
        output_path = os.path.join(output_dir, f"{prompt_base}.png")
        image.save(output_path)

if __name__ == "__main__":
    main()