import os
import json
from tqdm import tqdm

# Script to run on final context forest embedding output directory to check for missing embeddings

PATH_input_dir = "/home/ppathak2/Hypothesis_Generation_Active_Learning/output_trees/ICLR/ICLR_no_ctx_prompt/ICLR_drugs_final_embd"

missing_json_emb = {}
for keyword_dir in tqdm(os.listdir(PATH_input_dir)):
    with open(os.path.join(PATH_input_dir, keyword_dir, 'missing.json'), 'r') as f:
        missing_data = json.load(f)
    if len(missing_data) > 0:
        print(f"Keyword: {keyword_dir} | Missing Runs: {missing_data}")
        missing_json_emb[keyword_dir] = missing_data

print("total keywords with missing embeddings:", len(missing_json_emb))