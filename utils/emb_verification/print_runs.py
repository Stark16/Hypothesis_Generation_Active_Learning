import json
import os

PATH_all_shit = "/home/ppathak2/Hypothesis_Generation_Active_Learning/output_trees/NeLaMKRR_hierarchichal3"
for keyword in os.listdir(PATH_all_shit):
    PATH_missing_json = "/home/ppathak2/Hypothesis_Generation_Active_Learning/output_trees/NeLaMKRR_hierarchichal3/<KEYWORD>/missing.json"
    with open(PATH_missing_json.replace('<KEYWORD>', keyword), 'r') as f:
        data = json.load(f)

    x = []
    for k in data.keys():
        x.append(k.split('/')[-1])
    print(f"\n\n {keyword} : {x}")
