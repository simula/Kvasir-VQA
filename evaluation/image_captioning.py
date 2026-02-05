import requests

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

# model_path= "Florence-2/epoch_9"
model_path="Florence-2-vision_tower_frozen/epoch_10"

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", 
    trust_remote_code=True, revision='refs/pr/6')


#import load dataset
from datasets import load_dataset
prompt = "<DETAILED_CAPTION>"

data = load_dataset("wait-whoami/ImageCLEFmed-MEDVQA-GI-2024-Dev_mod", split="train").shuffle(seed=42)

from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
        

# Initialize metrics
rouge = Rouge()
vectorizer = TfidfVectorizer()

# Initialize lists to aggregate scores across the dataset
all_bleu_scores = []
all_rouge_scores = []
all_meteor_scores = []
all_cider_scores = []

import os 
import json
# Iterate over data and generate captions
for idx, i in enumerate(data):
    if len(all_bleu_scores) >= 5000:
        break  ## temporary
    inputs = processor(text=i['caption'], images=i['image'], return_tensors="pt")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=128,
        num_beams=20,
        do_sample=True,      # Enable sampling
        temperature=0.9,     # Adjust the temperature for more creativity
        top_k=100,            # Use top-k sampling to limit the pool of tokens
        top_p=0.85,          # Use top-p sampling (nucleus sampling)
        num_return_sequences=5  # Number of different sequences to generate
        )
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    # Prepare reference caption and generated captions
    reference_caption = i['caption']
    generated_captions = [sorted(processor.post_process_generation(text, task="caption", image_size=(500, 500)).values())[0] for text in generated_texts]
    print(f"Reference: {reference_caption}")
    print(f"Generated: {generated_captions}")

    # Initialize lists for storing metric scores per image
    image_bleu_scores = []
    image_rouge_scores = []
    image_meteor_scores = []
    image_cider_scores = []

    # Calculate scores for each generated caption
    for gen_caption in generated_captions:
        image_bleu_scores.append(sentence_bleu([reference_caption.split()], gen_caption.split()))
        image_rouge_scores.append(rouge.get_scores(gen_caption, reference_caption)[0]['rouge-l']['f'])
        image_meteor_scores.append(meteor_score([reference_caption.split()], gen_caption.split()))

    # Compute CIDEr scores using TF-IDF and cosine similarity
    tfidf_matrix = vectorizer.fit_transform([reference_caption] + generated_captions)
    image_cider_scores.extend(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten())

    # Aggregate scores
    all_bleu_scores.extend(image_bleu_scores)
    all_rouge_scores.extend(image_rouge_scores)
    all_meteor_scores.extend(image_meteor_scores)
    all_cider_scores.extend(image_cider_scores)
    print(f"BLEU: {np.mean(image_bleu_scores):.4f}, ROUGE: {np.mean(image_rouge_scores):.4f}, METEOR: {np.mean(image_meteor_scores):.4f}, CIDEr: {np.mean(image_cider_scores):.4f}")
    # save image and captions as json to a folder to see the results
    #maake folder if note exists outoputxx
    os.makedirs("output_captions", exist_ok=True)
    with open(f"output_captions/{idx}.json", "w") as f:
        json.dump({"reference_caption": reference_caption, "generated_captions": generated_captions}, f)
    i['image'].save(f"output_captions/{idx}.png")


# Compute overall average scores
avg_bleu = np.mean(all_bleu_scores)
avg_rouge = np.mean(all_rouge_scores)
avg_meteor = np.mean(all_meteor_scores)
avg_cider = np.mean(all_cider_scores)

print(f"Overall Average BLEU: {avg_bleu:.4f}")
print(f"Overall Average ROUGE: {avg_rouge:.4f}")
print(f"Overall Average METEOR: {avg_meteor:.4f}")
print(f"Overall Average CIDEr: {avg_cider:.4f}")