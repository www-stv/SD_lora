# SD1.5 Style Transfer Training Pipeline

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Detailed Usage](#detailed-usage)
  - [Step 1: Prepare Class Images (Optional)](#step-1-prepare-class-images-optional)
  - [Step 2: Pre-train Textual Inversion (Optional)](#step-2-pre-train-textual-inversion-optional)
  - [Step 3: Train LoRA with Style Transfer](#step-3-train-lora-with-style-transfer)
  - [Step 4: Generate Styled Images](#step-4-generate-styled-images)
- [Advanced Features](#advanced-features)
  - [DreamBooth Integration](#dreambooth-integration)
  - [Orthogonal Decoupling](#orthogonal-decoupling)
  - [TI Protection Mechanism](#ti-protection-mechanism)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project provides a complete pipeline for training style transfer models on Stable Diffusion 1.5. It combines multiple state-of-the-art techniques:

- **Textual Inversion (TI)**: Learns a new token `<style_name>` that captures the target style
- **LoRA (Low-Rank Adaptation)**: Efficiently fine-tunes the UNet with low-rank matrices
- **DreamBooth**: Preserves class identity using prior preservation loss
- **Orthogonal Decoupling**: Encourages independent feature learning across LoRA layers

The pipeline is designed to be flexible, supporting both pure LoRA training and joint TI+LoRA training with various regularization strategies.

## Features

-  **Style Transfer**: Learn any visual style from a small set of example images (3 images for example)
-  **Flexible Training Modes**:
  - Pure LoRA training
  - LoRA + TI joint training
  - DreamBooth with matched/diverse class priors
-  **Advanced Regularization**:
  - Orthogonal loss for feature decoupling
  - Noise offset for training stability
  - Gradient accumulation for larger effective batch sizes

## Project Structure

```
sd15-style-transfer/
├── prepare_class_images.py    # Generate prior images for DreamBooth
├── pretrain_ti.py             # Pre-train Textual Inversion embeddings
├── SD15_lora_train.py         # Main training script (LoRA + TI)
├── SD15_test.py               # Inference script for styled generation
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── examples/                  # Example usage (create your own)
    ├── style_images/          # Target style images
    ├── prior_images/          # Generated prior images
    ├── lora_weights/          # Trained LoRa and TI weights
    └── output/                # Test generations
```

## Dataset Preparation

#### 1. Style Images
Place your style reference images in a directory (e.g., `./style_images/`):
- **Naming**: `cat.png`, `apple.png`, `glasses.png`
- **Resolution**: 512x512 (auto-resized)
- **Variety**: Different subjects, same style

#### 2. Prompts (Optional)
Create `prompts.json` in your style directory:
```json
{
    "cat": "a photo of a cute cat",
    "apple": "a photo of a red apple",
    "glasses": "a photo of stylish sunglasses"
}
```
If missing, prompts are auto-generated as: `"a photo of {img_name} in {style_name} style"`

#### 3. Prior Images for DreamBooth (Optional, recommended for small datasets)
Generate prior images when you have limited style images:

```bash
python prepare_class_images.py \
    --prompts cat apple glasses dog moon \
    --output_dir ./class_images \
    --num_images_per_prompt 10
```

**Directory structure:**
```
class_images/
├── cat/          # Matched class (matches style images)
├── apple/        # Matched class
├── glasses/      # Matched class
├── dog/          # Diverse class (generalization)
├── moon/         # Diverse class
└── ...           # More diverse classes
```

- **Matched classes**: Same subjects as style images (lower loss weight: 0.1)
- **Diverse classes**: Different subjects (higher loss weight: 0.5)


## Detailed Usage

### Step 1: Prepare Class Images (Optional)

If you want to use DreamBooth for better identity preservation, generate prior class images first.

```bash
python prepare_class_images.py \
    --prompts cat dog bird person \
    --output_dir ./prior_images \
    --num_images_per_prompt 20 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --resolution 512 \
    --seed 42
```

**Arguments:**
- `--prompts`: List of class names (e.g., cat, dog)
- `--output_dir`: Output directory for generated images
- `--num_images_per_prompt`: Number of images per class (default: 10)
- `--num_inference_steps`: Inference steps for generation (default: 50)
- `--guidance_scale`: CFG scale (default: 7.5)

### Step 2: Pre-train Textual Inversion (Optional)

Pre-train TI embeddings before LoRA training for better convergence.

```bash
python pretrain_ti.py \
    --style_name "my_style" \
    --instance_dir ./style_images \
    --output_dir ./ti_pretrain \
    --ti_epochs 200 \
    --ti_lr 5e-3 \
    --train_batch_size 1
```

**Arguments:**
- `--ti_lr`: Learning rate for TI embedding (default: 5e-3)
- `--ti_epochs`: Number of training epochs (default: 200)
- `--ti_reg_weight`: Weight decay for TI (default: 1e-4)
- `--ti_token_init`: Initialization word (default: "style")

The pretrained TI will be saved as `pretrained_ti_embedding.pt` in the output directory.

### Step 3: Train LoRA with Style Transfer

This is the main training script with multiple configuration options.

#### Basic LoRA Training

```bash
python SD15_lora_train.py \
    --style_name "my_style" \
    --instance_dir ./style_images \
    --output_dir ./lora_output \
    --num_train_epochs 200 \
    --train_batch_size 1 \
    --unet_lr 5e-4 \
    --rank 4 \
    --lora_alpha 4
```

#### Training with TI (Joint Training)

```bash
python SD15_lora_train.py \
    --style_name "my_style" \
    --instance_dir ./style_images \
    --pretrained_ti_path ./ti_pretrain/epoch200/pretrained_ti_embedding.pt \
    --train_ti \
    --ti_lr 5e-4 \
    --output_dir ./lora_output \
    --num_train_epochs 200
```

#### Training with DreamBooth

```bash
python SD15_lora_train.py \
    --style_name "my_style" \
    --instance_dir ./style_images \
    --class_dir ./prior_images \
    --diverse_loss_weight 0.5 \
    --matched_loss_weight 0.1 \
    --output_dir ./lora_output \
    --num_train_epochs 200
```

#### Full Feature Training (TI + LoRA + DreamBooth + Orthogonal)

```bash
python SD15_lora_train.py \
    --style_name "my_style" \
    --instance_dir ./style_images \
    --class_dir ./prior_images \
    --pretrained_ti_path ./ti_pretrain/epoch200/pretrained_ti_embedding.pt \
    --train_ti \
    --use_orthogonal \
    --orthogonal_loss_weight 0.01 \
    --diverse_loss_weight 0.5 \
    --matched_loss_weight 0.1 \
    --output_dir ./lora_output \
    --num_train_epochs 200 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 3 \
    --unet_lr 5e-4 \
    --ti_lr 5e-4 \
    --rank 4 \
    --noise_offset 0.1
```

### Step 4: Generate Styled Images

Use the trained LoRA and TI to generate styled images.

#### Generate with LoRA Only

```bash
python SD15_test.py \
    --style_name "my_style" \
    --lora_dir ./lora_output \
    --prompts "a cat in a garden" "a beautiful landscape" \
    --output_dir ./generated \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --seed 42
```

#### Generate with LoRA + TI

```bash
python SD15_test.py \
    --style_name "my_style" \
    --lora_dir ./lora_output \
    --ti_path ./lora_output/ti_embedding.pt \
    --prompts "a cat" "a dog" "a house by the lake" \
    --output_dir ./generated \
    --num_inference_steps 50
```

## Advanced Features

### DreamBooth Integration

The DreamBooth implementation uses a dual-prior strategy:

- **Matched Classes**: Images of the same class as your style images (e.g., if training on cat photos, matched priors are other cat photos). These help preserve class identity with higher weight (`matched_loss_weight`).

- **Diverse Classes**: Random class images from other categories. These prevent overfitting and improve generalization with lower weight (`diverse_loss_weight`).

### Orthogonal Decoupling

The orthogonal loss encourages different LoRA layers to learn orthogonal features:

```python
orth_loss = orthogonal_loss(unet)  # Computes cosine similarity between layers
total_loss += args.orthogonal_loss_weight * orth_loss
```

This helps separate style-related features from content-related features, leading to better style transfer without distorting content.

### TI Protection Mechanism

When training TI jointly with LoRA, the script implements a protection mechanism:

```python
# Only the target TI token is updated
index_no_updates[token_id] = False

# After each step, restore other tokens
token_embeds.weight.data[index_no_updates] = orig_embeds_params[index_no_updates]
```

This ensures that only the newly added token embedding is trained while preserving the base model's vocabulary.


## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [Diffusers](https://github.com/huggingface/diffusers)
- [LoRA](https://github.com/microsoft/LoRA)
- [DreamBooth](https://github.com/google/dreambooth)