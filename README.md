# Kvasir-VQA

Public repository for the Kvasir-VQA dataset and experiments.

- Dataset: https://datasets.simula.no/kvasir-vqa/
- Paper: https://dl.acm.org/doi/10.1145/3689096.3689458
- Advanced VQA version (Kvasir-VQA-x1): https://github.com/simula/Kvasir-VQA-x1

## 📙 Dataset
The full Kvasir-VQA dataset consists of 6,500 images with question-answer annotations and is available at https://datasets.simula.no/kvasir-vqa/.

A subset of this dataset, featuring 2,000 images and 20,241 captions, was randomly selected from five different sources. An LLM was employed to transform captions into question-answer pairs for a VQA task. The subset used in the experiments in this paper is publicly available at 🤗 https://huggingface.co/datasets/SushantGautam/ImageCLEFmed-MEDVQA-GI-2024-Dev_mod.

## 🔗 Model Checkpoints (Hugging Face)
- SD3 DreamBooth/LoRA: https://huggingface.co/SushantGautam/kvasir-vqa-sd3-lora
- Florence-2 VQA LoRA: https://huggingface.co/SushantGautam/kvasir-vqa-florence2-medvqa-lora
- Florence-2 Image Captioning LoRA: https://huggingface.co/SushantGautam/kvasir-vqa-florence2-image-captioning-lora


The code below replicates the three use-cases of the dataset as presented in the paper.
## 🖼️ Image Captioning
Finetune Florence-2 with the prefix `<DETAILED_CAPTION>` :

`python image_captioning/fine_tune_florence_2.py`

local trained model path: Florence-2-vision_tower_frozen/epoch_10

### 📝 Evaluation
`python evaluation/image_captioning.py`


## ❓Visual Question Answering (VQA)
Finetune Florence-2 model with the prefix `<medvqa>`:

`python vqa/fine_tune_florence_2_vqa.py`

 local trained model path: Florence-2-vqa/epoch_10


#### 🤖 Synthetic VQA generation with VLLM server on an A100 GPU
Serve Meta-Llama-3-8B-Instruct model on a VLLM server:

`python -m vllm.entrypoints.openai.api_server  --model meta-llama/Meta-Llama-3-8B-Instruct --tensor-parallel-size 1`

Run VLLM VQA generation with the VLLM server:

`python vqa/llm_vqa.py --vllm_url http://vllserver-hostname:8000 --num_proc 100`
 
### 📝 Evaluation
`python evaluation/vqa.py`

The generated Q/A pairs along with the generated answers for each (image, question) input form the trained VQA model (Florence-2-vqa/epoch_10) is shared as `generated_llm_vqa.csv`. The index of rows in the csv file is same as the one in the dataset (subset; see Dataset section below).

## Synthetic Medical Image Generation
Stable Diffusion 3 fine-tuning with DreamBooth/ LoRA:

`python imagen/fine_tune_dreambooth_lora_sd3.py --pretrained_model_name_or_path=stabilityai/stable-diffusion-3-medium-diffusers --dataset_name=SushantGautam/ImageCLEFmed-MEDVQA-GI-2024-Dev_mod --instance_prompt XXX --caption_column caption --report_to wandb --resolution=512 --train_batch_size=48 --gradient_accumulation_steps=1 --gradient_checkpointing --mixed_precision=fp16 --num_train_epochs=80 --optimizer=prodigy --learning_rate=1.0 --allow_tf32 --weighting_scheme=logit_normal --lr_scheduler=constant --validation_prompt="Generate an image with a polyp located in the center-right." --validation_epochs=1 --lr_warmup_steps=0 --seed=0 --rank 128 --checkpoints_total_limit=20 --output_dir=MEDVQA_SD3_db_lora_dev`

local trained model path: MEDVQA_SD3_db_lora_dev/pytorch_lora_weights.safetensors


### 📝 Evaluation

##### 🏥 Generate synthetic images
`python evaluation/imagen_generate.py --type polyp --num_images 5000`
`python evaluation/imagen_generate.py --type non_polyp --num_images 5000`

This should generate synthetic images in "data/polyp_classification-v1/polyp/" and "data/polyp_classification-v1/non_polyp/"

##### Download real images from ImageCLEFmed-MEDVQA-GI-2024-Development-Dataset: https://drive.google.com/file/d/1bZ27-A3RLTYot65swbtu0A3p4O9Asp0-/view and put all images in "MEDVQA_GI_2024/Dev-Dataset/images"

##### 📝 Evaluate the generated images
`python evaluation/imagen.py`

## 🧾 Citation
If you use this dataset or code, please cite:

```
@incollection{Gautam2024Oct,
	author = {Gautam, Sushant and Stor\aa s, Andrea M. and Midoglu, Cise and others},
	title = {{Kvasir-VQA: A Text-Image Pair GI Tract Dataset}},
	booktitle = {{ACM Conferences}},
	pages = {3--12},
	year = {2024},
	month = oct,
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	doi = {10.1145/3689096.3689458}
}
```