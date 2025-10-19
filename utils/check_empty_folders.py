import os

PATH_folder_to_check = "/home/ppathak2/Hypothesis_Generation_Active_Learning/output_trees/ICLR/ICLR_no_ctx_prompt/ICLR_drugs_final_embd"
for folder in os.listdir(PATH_folder_to_check):
    path_keyword = os.path.join(PATH_folder_to_check, folder)
    if len(os.listdir(path_keyword)) < 3:
        print("Empty folder - ", path_keyword)