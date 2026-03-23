import os
import argparse
from pathlib import Path
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
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="生成提示词列表"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录"
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
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

    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=10,
        help="每个提示词生成的图片数量"
    )

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    # 主输出目录（prior_images）
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载基础模型
    logger.info("加载基础模型...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)
    if device.type == "cpu":
        pipe.enable_attention_slicing()

    # 2. 准备生成
    logger.info("开始生成图像...")

    # 遍历每个提示词（类别）
    for base_prompt in args.prompts:
        # 创建类别子目录（如 prior_images/cat/）
        category_dir = output_dir / base_prompt
        os.makedirs(category_dir, exist_ok=True)

        # 为每个类别生成指定数量的图片
        for img_idx in range(args.num_images_per_prompt):
            seed = args.seed + img_idx
            set_seed(seed)

            # 生成图片序号
            img_name = f"{base_prompt}_{seed}.png"
            img_path = category_dir / img_name

            prompt = f"a photo of {base_prompt}"

            # 生成图像
            with torch.no_grad():
                image = pipe(
                    prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.resolution,
                    width=args.resolution,
                ).images[0]

            image.save(img_path, quality=95)

        print(f"{base_prompt}已生成")

        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()