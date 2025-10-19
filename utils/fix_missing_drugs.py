import os
import shutil

PATH_context_forest = "/home/ppathak2/Hypothesis_Generation_Active_Learning/output_trees/ICLR/ICLR_no_ctx_prompt/medicinal_drugs"
PATH_missing_drugs = "/home/ppathak2/Hypothesis_Generation_Active_Learning/missing_drugs.txt"
failed_drugs = []
with open(PATH_missing_drugs, 'r') as f:
    drugs = f.readlines()
    for drug in drugs:
        try:
            all_content = os.listdir(os.path.join(PATH_context_forest, drug.strip()))
        except Exception as e:
            failed_drugs.append(drug.strip())
            continue
        run_dir = os.path.join(PATH_context_forest, drug.strip(), "run_1")
        os.makedirs(run_dir, exist_ok=True)
        for content in all_content:
            shutil.move(os.path.join(PATH_context_forest, drug.strip(), content), run_dir)
print("Failed drugs - ", failed_drugs)