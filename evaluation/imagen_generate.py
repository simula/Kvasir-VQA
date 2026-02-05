# # !pip install diffusers
from diffusers import AutoPipelineForText2Image
import torch
import argparse
import os
import tqdm
from datasets import load_dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import numpy as np



argx = argparse.ArgumentParser()
argx.add_argument("--base_path", type=str, default="data/polyp_classification-v1")
argx.add_argument("--type", choices=["polyp", "non_polyp"], default="polyp")
argx.add_argument("--num_images", type=int, default=5000)

arg = argx.parse_args()

os.makedirs(f"{arg.base_path}/{arg.type}", exist_ok=True)

data = load_dataset("wait-whoami/ImageCLEFmed-MEDVQA-GI-2024-Dev_mod", split="train")
data= data.shuffle(seed=42)



import pandas as pd
df = pd.DataFrame([data["caption"], data['type']], index=["caption", "type"]).T
if arg.type == "polyp":
    df = df[df["type"] == 1]
elif arg.type == "non_polyp":
    df = df[df["type"] == 0]
else:
    raise ValueError("Invalid type, choose one of 'polyp' or 'non_polyp'")

all_prompts= sorted(df['caption'].unique())*10  # min 10 sampels of all_prompts
while len(all_prompts) < arg.num_images:
    all_prompts.extend(df['caption'].values.tolist())

all_prompts = all_prompts[:arg.num_images]

# if arg.type == "polyp":
#     prompt = "Generate an image containing a polyp."
# elif arg.type == "non_polyp":
#     prompt = "Generate an image with no polyps."
# else:
    # raise ValueError("Invalid type, choose one of 'polyp' or 'non_polyp'")
# negative_prompt = "low quality, blurry, unfinished"
# base save path is data/polyp_classification-v0


pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("MEDVQA_SD3_db_lora_dev", weight_name="pytorch_lora_weights.safetensors")

for i in tqdm.tqdm(range(arg.num_images)):
    # skip if image already exists
    if os.path.exists(f"{arg.base_path}/{arg.type}/image_{i}.png"):
        continue
    image = pipeline(all_prompts[i], num_inference_steps=50, guidance_scale=7, negative_prompt=None).images[0]
    image.save(f"{arg.base_path}/{arg.type}/image_{i}.png")
    # breakpoint()



## Evaluation:
# python evaluation/imagen_generate.py  --type polyp --num_images 5000