from datasets import load_dataset
import pandas as pd

data = load_dataset("wait-whoami/ImageCLEFmed-MEDVQA-GI-2024-Dev_mod")

from transformers import AutoModelForCausalLM, AutoProcessor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("Florence-2-vqa/epoch_10", trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision='refs/pr/6')

torch.cuda.empty_cache()

from torch.utils.data import Dataset

class DocVQADataset(Dataset):
    def __init__(self, data, limit_len=None):
        self.data = data
        self.df= pd.read_csv("llm_gens.csv")
        self.limit_len = limit_len

    def __len__(self):
        if self.limit_len:
            return self.limit_len
        else:
            return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        df_row = self.df.iloc[idx]
        assert df_row['caption'] == example['caption']## safe
        question = "<MedVQA>" + df_row['Q']
        first_answer = df_row['A']
        image = example['image']
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, first_answer, image


import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AdamW, AutoProcessor, get_scheduler)

def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers

test_dataset = DocVQADataset(data['train'])

# Create DataLoader
batch_size = 20
num_workers = 0

test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, shuffle=True)
all_data=[]
for batch in test_loader:
    inputs, answers = batch
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=100,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    # zip questions, answers
    for i, (answer, generated_text) in enumerate(zip(answers, generated_texts)):
        # print("\n\nQuestion: ", question)
        input_text = processor.decode(inputs["input_ids"][0], skip_special_tokens=True)
        print("\nQuestion: ", processor.decode(inputs["input_ids"][i], skip_special_tokens=True))
        print("Original Answer: ", answer)
        print("Generated Answer: ", generated_text)
        all_data.append([input_text, answer, generated_text])
df_all_data= pd.DataFrame(all_data, columns=["question", "answer", "generated_answer"])
df_all_data.to_csv("generated_llm_vqa.csv", index=False)




import pandas as pd
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

# Iterate over the data and calculate metrics
for idx, row in df_all_data.iterrows():
    reference_answer = row['answer']
    generated_answer = row['generated_answer']
    
    # Tokenize answers
    reference_tokens = reference_answer.split()
    generated_tokens = generated_answer.split()
    
    # Calculate BLEU score
    bleu_score = sentence_bleu([reference_tokens], generated_tokens)
    all_bleu_scores.append(bleu_score)
    
    # Calculate ROUGE score
    rouge_score = rouge.get_scores(generated_answer, reference_answer)[0]
    all_rouge_scores.append(rouge_score['rouge-l']['f'])
    
    # Calculate METEOR score
    meteor = meteor_score([reference_tokens], generated_tokens)
    all_meteor_scores.append(meteor)
    
    # Calculate CIDEr score using TF-IDF and cosine similarity
    tfidf_matrix = vectorizer.fit_transform([reference_answer, generated_answer])
    cider_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()[0]
    all_cider_scores.append(cider_score)

# Calculate aggregated scores
average_bleu = np.mean(all_bleu_scores)
average_rouge = np.mean(all_rouge_scores)
average_meteor = np.mean(all_meteor_scores)
average_cider = np.mean(all_cider_scores)

# Print the aggregated scores
print(f"Average BLEU score: {average_bleu}")
print(f"Average ROUGE-L score: {average_rouge}")
print(f"Average METEOR score: {average_meteor}")
print(f"Average CIDEr score: {average_cider}")