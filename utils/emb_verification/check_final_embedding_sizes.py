import os
import json
from tqdm import tqdm

# A script that checks for any final embeddings larger than 768 dimensions in the final embedding directory

def check_final_embedding_sizes(base_dir):
    keywords = []
    for keyword in tqdm(os.listdir(base_dir)):
        keyword_dir = os.path.join(base_dir, keyword)
        if not os.path.isdir(keyword_dir):
            continue
        for fname in os.listdir(keyword_dir):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(keyword_dir, fname)
            with open(fpath, 'r') as f:
                data = json.load(f)
            for run, entry in data.items():
                emb = entry.get('final_embedding', [])
                if len(emb) > 768:
                    print(f"Keyword: {keyword} | File: {fname} | Run: {run} | Size: {len(emb)}")
                    if keyword not in keywords:
                        keywords.append(keyword)
    print(keywords)

def main():
    base_dir = "/home/ppathak2/Hypothesis_Generation_Active_Learning/output_trees/ICLR/ICLR_no_ctx_prompt/ICLR_drugs_final_embd"
    check_final_embedding_sizes(base_dir)

if __name__ == "__main__":
    main()
