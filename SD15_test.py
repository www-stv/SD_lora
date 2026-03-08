import os
import argparse
import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, set_peft_model_state_dict
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
        help="要使用的风格文件夹名称"
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
        default=512,
        help="生成图像分辨率"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )

    # 添加LoRA参数（必须和训练时一致）
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
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


def load_lora_to_pipe(pipe, lora_path, rank=16, lora_alpha=16, device="cpu"):
    # 1. 加载保存的权重
    combined_dict = torch.load(os.path.join(lora_path, "pytorch_lora_weights.bin"), map_location=device)

    # 2. 为UNet添加LoRA适配器
    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj"],
    )
    pipe.unet.add_adapter(unet_lora_config)

    # 3. 为文本编码器添加LoRA适配器
    text_encoder_lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
    )
    pipe.text_encoder.add_adapter(text_encoder_lora_config)

    # 4. 加载LoRA权重
    unet_state_dict = combined_dict["unet"]
    text_encoder_state_dict = combined_dict["text_encoder"]

    # 使用PEFT的工具函数加载权重
    set_peft_model_state_dict(pipe.unet, unet_state_dict)
    set_peft_model_state_dict(pipe.text_encoder, text_encoder_state_dict)

    logger.info("✅ LoRA权重成功加载到适配器")
    return pipe


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

    try:
        pipe.load_lora_weights(lora_path)
        logger.info(f"✅ Diffusers成功加载LoRA权重: {lora_path}")

    except Exception as e:
        safetensors_path = os.path.join(lora_path, "pytorch_lora_weights.safetensors")
        bin_path = os.path.join(lora_path, "pytorch_lora_weights.bin")

        try:
            if os.path.exists(safetensors_path):
                logger.info(f"找到safetensors文件: {safetensors_path}")
                from safetensors.torch import load_file
                lora_state_dict = load_file(safetensors_path)
            elif os.path.exists(bin_path):
                logger.info(f"找到bin文件: {bin_path}")
                lora_state_dict = torch.load(bin_path, map_location="cpu")
            else:
                logger.error("未找到任何权重文件")
                return

            from diffusers.utils import convert_state_dict_to_diffusers
            pipe.unet.load_state_dict(convert_state_dict_to_diffusers(lora_state_dict), strict=False)
            logger.info("✅ 手动加载LoRA成功")
        except Exception as e2:
            logger.error(f"❌ 手动加载也失败: {e2}")
            return

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