import os
import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate with style LoRA + TI")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="预训练模型路径"
    )
    parser.add_argument(
        "--style_name",
        type=str,
        required=True,
        help="要使用的风格名称"
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help="LoRA权重根目录"
    )
    parser.add_argument(
        "--ti_path",
        type=str,
        default=None,
        help="Textual Inversion嵌入路径"
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
        "--rank",
        type=int,
        default=4,
        help="LoRA秩的大小（必须和训练时一致）"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA缩放因子（必须和训练时一致）"
    )

    return parser.parse_args()


def load_ti_embedding(pipe, ti_path):
    # 加载TI嵌入文件
    ti_data = torch.load(ti_path, map_location="cpu")

    # 获取token和嵌入向量
    style_token = ti_data.get('style_token', None)
    embedding = ti_data.get('embedding', None)

    if style_token is None or embedding is None:
        print("警告: TI文件格式不正确，将不使用TI")
        return None, None

    # 获取tokenizer和text_encoder
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # 检查token是否已存在
    if style_token not in tokenizer.get_vocab():
        # 添加新token
        num_added_tokens = tokenizer.add_tokens([style_token])
        print(f"添加新token '{style_token}'，新增数量: {num_added_tokens}")

        # 调整text_encoder的嵌入层大小
        old_embeddings = text_encoder.get_input_embeddings()
        new_embeddings = torch.nn.Embedding(
            old_embeddings.num_embeddings + num_added_tokens,
            old_embeddings.embedding_dim
        )

        # 复制原有权重
        with torch.no_grad():
            new_embeddings.weight.data[:-num_added_tokens] = old_embeddings.weight.data
            # 设置新token的嵌入
            new_embeddings.weight.data[-num_added_tokens:] = embedding.to(old_embeddings.weight.dtype)

        text_encoder.set_input_embeddings(new_embeddings)
        print(f"已加载TI嵌入 '{style_token}'")
    else:
        # 如果token已存在，则更新
        token_id = tokenizer.convert_tokens_to_ids(style_token)
        with torch.no_grad():
            text_encoder.get_input_embeddings().weight[token_id] = embedding.to(
                text_encoder.get_input_embeddings().weight.dtype
            )
        print(f"已更新token '{style_token}' 的嵌入")

    return style_token, tokenizer.convert_tokens_to_ids(style_token)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载基础模型
    print("加载基础模型...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 加载TI嵌入
    style_token = None
    if args.ti_path:
        style_token, _ = load_ti_embedding(pipe, args.ti_path)
        print(f"TI token: {style_token}")

    # 3. 加载LoRA权重
    if args.lora_dir:
        try:
            pipe.load_lora_weights(Path(args.lora_dir))
            print(f"✅ 成功加载LoRA权重")
        except Exception as e:
            print(f"❌ LoRA加载失败: {e}")

    # 4. 移动模型到设备
    pipe = pipe.to(device)
    if device.type == "cpu":
        pipe.enable_attention_slicing()

    # 5. 准备生成
    print("开始生成图像...")

    for i, prompt_base in enumerate(args.prompts):
        # 为每个提示词使用不同的种子（基于基础种子）
        current_seed = args.seed + i
        set_seed(current_seed)

        # 构建完整提示词
        if style_token:
            full_prompt = f"{prompt_base} in {style_token} style"
        else:
            full_prompt = f"{prompt_base} in {args.style_name} style"

        print(f"生成 {full_prompt}")

        # 生成图像
        with torch.no_grad():
            result = pipe(
                full_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.resolution,
                width=args.resolution,
            )
            image = result.images[0]

        # 保存图像
        safe_filename = prompt_base.replace("/", "_").replace("\\", "_").replace(" ", "_")
        output_path = output_dir / f"{safe_filename}.png"
        image.save(output_path)


if __name__ == "__main__":
    main()