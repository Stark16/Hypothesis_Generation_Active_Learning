import json
import shutil
import os

with open(r"/home/ppathak2/Hypothesis_Generation_Active_Learning/output_trees/Cycle_GMP_fix/Cyclic GMP_50.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

input_folder = r"/home/ppathak2/Hypothesis_Generation_Active_Learning/output_trees/medicinal_drugs/Cyclic GMP"
output_folder = r"/home/ppathak2/Hypothesis_Generation_Active_Learning/output_trees/temp_fix/Cyclic GMP"
all_runs = []
changed = []
for i in range(50):
    curr_run = "run_" + str(i+1)
    all_runs.append(curr_run)
    if curr_run not in (list(data.keys())):
        print(curr_run)
    
        # in_filepath = os.path.join(input_folder, curr_run, 'embdng_tree_v2.json')
        # out_filepath = os.path.join(output_folder, curr_run, 'embdng_tree_v2.json')
        # os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        # shutil.copyfile(in_filepath, out_filepath)
        # changed.append(out_filepath)

print(len(changed))
# print(all_runs)
print(list(data.keys()))