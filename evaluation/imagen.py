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

import torchvision.transforms as transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((299, 299)),          # Resize to the required input size for Inception
    transforms.ToTensor(),                  # Convert to tensor with float values in [0, 1]
    transforms.Lambda(lambda x: x * 255),   # Scale to [0, 255]
    transforms.Lambda(lambda x: x.byte())   # Convert to uint8
])

# Use ImageCLEFmed-MEDVQA-GI-2024-Development-Dataset
## Extract real images from https://drive.google.com/file/d/1bZ27-A3RLTYot65swbtu0A3p4O9Asp0-/view
# and put them in some local
real_image_dir= "MEDVQA_GI_2024/Dev-Dataset/images"
real_images = []

for image_path in os.listdir(real_image_dir):
    image = Image.open(os.path.join(real_image_dir, image_path))
    image = transform(image)
    real_images.append(image)
    print(len(real_images))

generated_image=[]

## generate some images of polyp and non_polyp questions using python evaluation/imagen_generate.py
gen_images_path= ["data/polyp_classification-v1/polyp/", "data/polyp_classification-v1/non_polyp/"]
for fold in gen_images_path:
    images = os.listdir(fold)
    # select 1000 images randomly, with seed
    np.random.seed(42)
    np.random.shuffle(images)
    images = images[:1000]
    for image_path in images:
        image = Image.open(os.path.join(fold, image_path))
        image = transform(image)
        generated_image.append(image)
        print(len(generated_image))


_real_images = torch.stack(real_images)
_generated_images = torch.stack(generated_image)
fid = FrechetInceptionDistance(feature=2048)
fid.update(_real_images, real=True)
fid.update(_generated_images, real=False)
fid_score = fid.compute()
print("fid:", fid_score)


inception = InceptionScore()
inception.update(_generated_images)
is_score = inception.compute()


inception_real = InceptionScore()
inception_real.update(_real_images)
is_score_real = inception_real.compute()

print(f"FID Score: {fid_score}, Inception Score of generated: {is_score}") #FID Score: 110.72583770751953, Inception Score: (tensor(3.0651), tensor(0.1373))
print(f"Inception Score of Real images: {is_score_real}") #  Inception Score Real: (tensor(4.0487), tensor(0.2365))
