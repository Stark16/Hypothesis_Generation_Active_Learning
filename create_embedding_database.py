import os
import json
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

error_keywords = {}

def polar_normalization(vec):
    min_val = np.min(vec)
    max_val = np.max(vec)
    if max_val == min_val:
        return np.zeros_like(vec)
    scaled = (vec - min_val) / (max_val - min_val)
    normalized = scaled * 2 - 1
    return normalized


def calculate_context_embedding(context_tree_embd, normalize=True):
    final_embedding = np.zeros_like(context_tree_embd[next(iter(context_tree_embd))]['w88_enc'][0])
    for keyword, values in context_tree_embd.items():
        w88_embding = np.array(values['w88_enc'][0])
        if (np.isnan(w88_embding).any() or np.isinf(w88_embding).any()):
            continue
        if normalize:
            w88_embding = polar_normalization(w88_embding)
        if (w88_embding.shape[0] > 1):
            w88_embding = w88_embding[0, :].reshape(1, w88_embding.shape[-1])
        if (len(w88_embding) == 0):
            w88_embding = np.zeros_like(final_embedding)
        final_embedding += w88_embding
    return final_embedding


def process_keyword_folder(keyword_path, save_dir):
    run_folders = [f for f in os.listdir(keyword_path)
                   if os.path.isdir(os.path.join(keyword_path, f)) and f.startswith("run_")]
    embeddings = []
    run_numbers = []

    for run_num in (run_folders):
        tree_path = os.path.join(keyword_path, run_num, "embdng_tree_v2.json")
        if not os.path.isfile(tree_path):
            continue
        with open(tree_path, "r") as f:
            context_tree_embd = json.load(f)
        try:
            emb = calculate_context_embedding(context_tree_embd, normalize=False)
        except Exception as e:
            # print("[❗] - Error while processing keyword:", keyword_path, "run:", run_num, "Error:", e)
            if keyword_path not in list(error_keywords.keys()):
                error_keywords[keyword_path]  = 1
            else:
                error_keywords[keyword_path] += 1
            continue
        emb = np.array(emb).flatten()  # Ensure emb is 1D
        # if embeddings and emb.shape[0] != embeddings[0].shape[0]:
        #     continue  # Skip inconsistent embeddings
        embeddings.append(emb)
        run_numbers.append(run_num)
    # if error_keywords:
    #     print("Error Keywords:", error_keywords)
    result = {}
    for idx, run_num in enumerate(run_numbers):
        result[str(run_num)] = {
            "final_embedding": embeddings[idx].tolist()
        }
    json_name = f"{os.path.basename(keyword_path)}_{len(run_numbers)}.json"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, json_name), "w") as f:
        json.dump(result, f, indent=2)


def main():
    PATH_output_dir = "/home/ppathak2/Hypothesis_Generation_Active_Learning/output_trees/ICLR/ICLR_no_ctx_prompt/medicinal_drug"
    save_dir = os.path.join(os.path.dirname(PATH_output_dir), "ICLR_drugs_final_embd")
    list_of_keywords = os.listdir(PATH_output_dir)

    pbar = tqdm(list_of_keywords)
    for keyword in pbar:
        pbar.set_description(f"Processing keyword {keyword}")
        keyword_path = os.path.join(PATH_output_dir, keyword)
        if os.path.isdir(keyword_path):
            process_keyword_folder(keyword_path, save_dir)
    print(error_keywords)

if __name__ == "__main__":
    main()