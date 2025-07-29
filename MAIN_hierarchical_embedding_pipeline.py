import os
from tqdm import tqdm
import torch

import context_tree_builder as ctb
import hierarchical_emb_builder as heb

class HierarchicalEmbPipeline:

    def __init__(self, keyword:str, domain:str, MODEL_gen_llm:str='microsoft/Phi-3.5-mini-instruct', MODEL_emb:str=''):
        self.PATH_self_dir = os.path.dirname(os.path.realpath(__file__))
        self.MODEL_emb = MODEL_emb
        
        # Initialize the Model objects and arguments - 
        self.OBJContextTree = ctb.ContextTree(starting_keyword=keyword, domain=domain, 
                                                model_to_load=MODEL_gen_llm)
        self.MODEL_ARGS_gen_llm = {"remember_raw_response" : False, "batch_query" : False,
                            "no_history" : False, "use_random_seed" : False}
        
        self.OBJHierarchEmb = heb.HierarchEmbdTree()

    def create_embedding(self, num_trees:int=2, depth_cap:int=2):

        for run_n in tqdm(range(num_trees), desc="Creating Context Trees"):
            NODE_root = self.OBJContextTree.bfs(self.OBJContextTree.STARTING_KEYWORD, depth_cap=depth_cap, **self.MODEL_ARGS_gen_llm)
            self.OBJContextTree.save_tree(self.OBJContextTree.STARTING_KEYWORD, NODE_root, run_n=run_n+1)

            # making sure to clear memory before each run-
            torch.cuda.empty_cache()
            self.OBJContextTree.reset_mem()
        
        # Once all the context trees are created, we calculate the hierarchical-embeddings
        PATH_context_forest = os.path.join(self.OBJContextTree.PATH_output_trees,
                                           self.OBJContextTree.DOMAIN, 
                                           self.OBJContextTree.STARTING_KEYWORD)
        self.OBJHierarchEmb.create_embeddings(PATH_context_forest)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    temprature = 40
    num_trees = 11
    depth_cap = 2

    # keyword = "heat coefficient"
    domain = "medicinal_drugs"

    with open("/home/ppathak2/Hypothesis_Generation_Active_Learning/diseases.txt", 'r') as f:
        drugs = f.readlines()
    drugs = ["Cyclic GMP"]
    for keyword in drugs:
        keyword = keyword.strip()

        print("\n\t\t", "-"*50, " ", keyword, " ", "-"*50, "\n")
        OBJ_HierarchEmbPipe = HierarchicalEmbPipeline(keyword, domain)
        # if keyword in os.listdir(os.path.join(OBJ_HierarchEmbPipe.OBJContextTree.PATH_output_trees, domain)):
        #     print(f"Tree already exists. Skipping - <{keyword}> -")
        #     continue
        
        # Setting up some configurable arguments-
        OBJ_HierarchEmbPipe.OBJContextTree.generation_args['temperature'] = temprature/100
        OBJ_HierarchEmbPipe.MODEL_ARGS_gen_llm["batch_query"] = True
        OBJ_HierarchEmbPipe.MODEL_ARGS_gen_llm["no_history"] = True

        OBJ_HierarchEmbPipe.create_embedding(num_trees, depth_cap)
       