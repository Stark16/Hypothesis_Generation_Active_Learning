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
    
    final_embedding = np.zeros_like(context_tree_embd[next(iter(context_tree_embd))]['w88_enc'][0]).reshape(-1, 1)
    for keyword, values in context_tree_embd.items():
        w88_embding = np.array(values['w88_enc'][0]).reshape(-1, 1)

        if (np.isnan(w88_embding).any() or np.isinf(w88_embding).any()):
            continue
        if normalize:
            w88_embding = polar_normalization(w88_embding)
        # if (w88_embding.shape[0] > 1):
        #     w88_embding = w88_embding[0, :].reshape(1, w88_embding.shape[-1])
        if (len(w88_embding) == 0):
            w88_embding = np.zeros_like(final_embedding)
        final_embedding += w88_embding
    return final_embedding


def process_keyword_folder(keyword_path, save_dir):
    run_folders = [f for f in os.listdir(keyword_path)
                   if os.path.isdir(os.path.join(keyword_path, f)) and f.startswith("run_")]
    # Define all embedding strategies and their file suffixes
    strategies = [
        ("first_three", "embdng_tree_v2_first_three.json"),
        ("first_two", "embdng_tree_v2_first_two.json"),
        ("first", "embdng_tree_v2_first.json"),
        ("last_three", "embdng_tree_v2_last_three.json"),
        ("last_two", "embdng_tree_v2_last_two.json"),
        ("last", "embdng_tree_v2_last.json"),
    ]

    # For each strategy, collect embeddings for all runs
    strategy_results = {s[0]: {} for s in strategies}
    strategy_run_counts = {s[0]: 0 for s in strategies}
    missing_trees = {}
    for run_num in run_folders:
        for strat_name, strat_file in strategies:
            tree_path = os.path.join(keyword_path, run_num, strat_file)
            if not os.path.isfile(tree_path):
                missing_run = os.path.dirname(tree_path)
                if missing_run in list(missing_trees.keys()):
                    missing_trees[missing_run].append(os.path.basename(tree_path))
                else:
                    missing_trees[missing_run] = [os.path.basename(tree_path)]
                continue
            with open(tree_path, "r") as f:
                context_tree_embd = json.load(f)
            try:
                emb = calculate_context_embedding(context_tree_embd, normalize=False)
            except Exception as e:
                if keyword_path not in error_keywords:
                    error_keywords[keyword_path] = 1
                else:
                    error_keywords[keyword_path] += 1
                continue
            emb = np.array(emb).flatten()
            # Extract the root node's raw_enc (first key in context_tree_embd)
            root_key = next(iter(context_tree_embd))
            raw_root_embedding = context_tree_embd[root_key].get('raw_enc', None)
            strategy_results[strat_name][str(run_num)] = {
                "final_embedding": emb.tolist(),
                "raw_root_embedding": raw_root_embedding
            }
            strategy_run_counts[strat_name] += 1
    # Save results for each strategy in a dedicated folder for the keyword
    keyword_out_dir = os.path.join(save_dir, os.path.basename(keyword_path))
    os.makedirs(keyword_out_dir, exist_ok=True)
    with open(os.path.join(keyword_out_dir, 'missing.json'), 'w') as f:
        json.dump(missing_trees, f)
    for strat_name in strategy_results:
        result = strategy_results[strat_name]
        run_count = strategy_run_counts[strat_name]
        if run_count == 0:
            continue  # Don't save empty files
        json_name = f"final_{run_count}_{strat_name}.json"
        with open(os.path.join(keyword_out_dir, json_name), "w") as f:
            json.dump(result, f, indent=2)


def main():
    PATH_output_dir = "/home/ppathak2/Hypothesis_Generation_Active_Learning/output_trees/BATS_BERT/female-male-batch/male - female"
    save_dir = os.path.join(os.path.dirname(PATH_output_dir), "male - female_final_embd")
    list_of_keywords = os.listdir(PATH_output_dir)
    # list_of_keywords = ['pralidoxime', 'Topotecan', 'Oxytetracycline','Trimethoprim', 'Flutamide']
    # list_of_keywords = ['pralidoxime']
    # list_of_keywords = ['pralidoxime', 'perchlorate', 'Iron', 'Fructose', 'Camphor', 'coumarin', 'Capsaicin', 'Urea', 'Tetracycline', 'Aspartame', 'Isosorbide', 'Fluvastatin', 'Silver', 'betadex', 'Curcumin', 'Sulfamethoxazole', 'Lactic Acid', 'Creatine']

    pbar = tqdm(list_of_keywords)
    for keyword in pbar:
        pbar.set_description(f"Processing keyword {keyword}")
        keyword_path = os.path.join(PATH_output_dir, keyword)
        if os.path.isdir(keyword_path):
            process_keyword_folder(keyword_path, save_dir)
    print(error_keywords)

if __name__ == "__main__":
    main()