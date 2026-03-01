import os
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
    )
print("下载成功")
pipe = pipe.to("cpu")

prompt = "a dog in hayao style"
print(prompt)
image = pipe(
    prompt,
    num_inference_steps=60,
    guidance_scale=9,
    height=512,
    width=512,
).images[0]

output_path = os.path.join("./test_outputs", f"dog.png")
image.save(output_path)
print("图片生成成功")