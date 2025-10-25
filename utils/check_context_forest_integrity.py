import os

## A script that performs 3 checks on the output tree directory:
# 1. Check for missing keywords compared to a source list
# 2. Check if each keyword has the expected number of runs
# 3. Check if each run contains all required tree files

def check_missing_keywords(source_list:list, path_output_tree:str):
    keywords_in_output_tree = os.listdir(path_output_tree)
    missing_keywords = []
    for keyword in source_list:
        if keyword not in keywords_in_output_tree:
            missing_keywords.append(keyword)
    return missing_keywords

def check_run_count(path_output_tree:str, target_count=int):
    mismatched_keywords = []
    for keyword in os.listdir(path_output_tree):
        path_keyword = os.path.join(path_output_tree, keyword)
        if not os.path.isdir(path_keyword):
            continue
        runs = os.listdir(path_keyword)
        try:
            runs.remove("LOG_failed_responses.json")
            if len(runs) != target_count:
                mismatched_keywords.append((keyword, len(runs)))
        except:
            continue
    return mismatched_keywords

def check_tree_integrity(path_output_tree:str):
    corrupted_keywords = {}
    trees_to_check = ["embdng_tree_v2_first_three.json", "embdng_tree_v2_first_two.json", "embdng_tree_v2_first.json",
                      "embdng_tree_v2_last_three.json", "embdng_tree_v2_last_two.json", "embdng_tree_v2_last.json"]
    for keyword in os.listdir(path_output_tree):
        path_keyword = os.path.join(path_output_tree, keyword)
        if not os.path.isdir(path_keyword):
            continue
        runs = os.listdir(path_keyword)
        try:
            runs.remove("LOG_failed_responses.json")
            for run in runs:
                path_run = os.path.join(path_keyword, run)
                run_files = os.listdir(path_run)
                for tree in trees_to_check:
                    if tree not in run_files:
                        if keyword not in corrupted_keywords:
                            corrupted_keywords[keyword] = []
                        corrupted_keywords[keyword].append((run, tree))
        except:
            continue
    return corrupted_keywords

def main():
    path_output_tree = r'/home/ppathak2/Hypothesis_Generation_Active_Learning/output_trees/ICLR/ICLR_ctx_prompt/diseases'
    source_list_path = r'/home/ppathak2/Hypothesis_Generation_Active_Learning/diseases.txt'
    
    with open(source_list_path, 'r') as f:
        source_list = [line.strip() for line in f.readlines()]
    
    target_run_count = 1
    missing_keywords = check_missing_keywords(source_list, path_output_tree)
    mismatched_keywords = check_run_count(path_output_tree, target_run_count)    
    corrupted_keywords = check_tree_integrity(path_output_tree)

    if len(missing_keywords) == 0 and len(mismatched_keywords) == 0 and len(corrupted_keywords) == 0:
        print("All checks passed successfully - ")
    if len(missing_keywords) > 0:
        print(f"Missing Keywords ({len(missing_keywords)}): {missing_keywords}")
    else:
        print("No missing keywords.")
    if len(mismatched_keywords) > 0:
        print(f"Keywords with mismatched run counts ({len(mismatched_keywords)}): {mismatched_keywords}")
    else:
        print("All keywords have the correct number of runs.")
    if len(corrupted_keywords) > 0:
        print(f"Keywords with corrupted/missing tree files ({len(corrupted_keywords)}): {corrupted_keywords}")
    else:
        print("All tree files are intact.")

if __name__ == "__main__":
    main()
