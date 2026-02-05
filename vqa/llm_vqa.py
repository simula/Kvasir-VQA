import pickle
import os
import ast
import numpy as np
import pandas as pd
import glob
from argparse import ArgumentParser

from multiprocessing import Pool
from functools import partial

import openai

from openai import OpenAI
from retrying import retry
import re
import json
import ast


from datasets import load_dataset
import pandas as pd


output_dir = f"llm_gens"
os.makedirs(output_dir, exist_ok=True)

parser = ArgumentParser()
parser.add_argument("--num_proc", type=int, default=10)
parser.add_argument("--vllm_url", default="http://g002:8000", help="URL of the vLLM server, e.g., http://g002:8000")

args = parser.parse_args()
print(args)

client = OpenAI(
    api_key= "EMPTY",
    base_url= args.vllm_url + "/v1",
)


# Use ImageCLEFmed-MEDVQA-GI-2024-Development-Dataset
## get prompt-gt.csv from https://drive.google.com/file/d/1bZ27-A3RLTYot65swbtu0A3p4O9Asp0-/view
dfx = pd.read_csv("MEDVQA_GI_2024/Dev-Dataset/prompt-gt.csv", sep=";")

@retry(stop_max_attempt_number=10, wait_fixed=0)
def call_llm(caption):
    caption = caption.replace('Generate ', "")
    message= [{"role": "system", "content": 
    '''
    You are an intelligent endoscopy procedure VQA dataset generator. 
    You are given a single caption describing one of the properties of the image that is visible.
    Your task is to generate a plain text question and answer pair what for an image with that caption. Don't refer to image.
    For quetsiosn regarding presence of text, ask if there is text in the image.
    For image from some procedure, ask about the type of procedure. 
    Ask about the polpy count, if mentioned in the caption.
    Ask about the color, size, location if the caption mentions them.
    '''
    },{"role": "user", "content":
    f'''
    Caption: {caption}
    Return only a JSON with Q and A as keys.
    Try to mention the key information given in the caption in the answer part
    '''},]
    # print(message[1]['content'])

    chat_response = client.chat.completions.create(model="meta-llama/Meta-Llama-3-8B-Instruct", messages=message)

    text= chat_response.choices[0].message.content
    # print("\n\n-------------\n\n", claim, entities, ": \n")
    # print(index, text)
    if ("{}" in text) or (len(text)<2): raise IOError("Redo, empty json is in the keys")
    datax = ast.literal_eval(re.findall(r'\{.*?\}', text, re.DOTALL)[0])
    print(caption, "\n", datax, "\n\n")
    # breakpoint()
    if len(datax['Q'])<1 or len(datax['A'])<1: raise IOError("Redo, empty json is in the keys")
    return datax


def process_row(index, row):
    save_json_as = f"{output_dir}/{index}.json"
    if os.path.exists(save_json_as):
        return
    try:
        resolved_json = call_llm(row['Prompt'],)
        resolved_json['caption'] = row['Prompt']
    except Exception as e: #openai.BadRequestError
        print(f"Error at index {index}: {e}")
        return
    print(index, resolved_json)
    with open(save_json_as, 'w') as f:
        json.dump(resolved_json, f, ensure_ascii=False)


partial_process_row = partial(process_row)
with Pool(processes=args.num_proc) as pool:
    pool.starmap(partial_process_row, dfx.iterrows())

# save as llm_gens.csv
all_files = glob.glob(f"{output_dir}/*.json")
all_data = []
for file in all_files:
    with open(file, 'r') as f:
        data = json.load(f)
        all_data.append(data)
df = pd.DataFrame(all_data, columns=["Q", "A", "caption"])
df.to_csv("llm_gens.csv", index=False)

# for index, row in dfx.iterrows():
#     process_row(index, row)

# Model served with vllm 
#python -m vllm.entrypoints.openai.api_server  --model meta-llama/Meta-Llama-3-8B-Instruct --tensor-parallel-size 2

# python llm_vqa.py --vllm_url http://g002:8000 --num_proc 100