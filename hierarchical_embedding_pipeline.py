import os
import time
from tqdm import tqdm
import torch

import context_tree_builder as ctb
import hierarchical_emb_tree_builder as heb

class HierarchicalEmbPipeline:

    def __init__(self, keyword:str, domain:str, MODEL_gen_llm:str='microsoft/Phi-3.5-mini-instruct', MODEL_emb:str='', BATCH_QUERY:bool=False):
        """This Class is the entire pipeline that created the context-forest for a given keyword in domain. It runs both the LLM and the Embedding model
        to generate the "Context-Forest" for the given set of keyword and domain.

        Args:
            keyword (str): The keyword to build the tree for
            domain (str): The domain in which we need to build this
            MODEL_gen_llm (str, optional): THe LLM of choice. Defaults to 'microsoft/Phi-3.5-mini-instruct'.
            MODEL_emb (str, optional): The Embedding model of choice. Defaults to ''.
            BATCH_QUERY (bool, optional): Boolean to enable batch-query for faster processing. Defaults to False.
        """
        self.PATH_self_dir = os.path.dirname(os.path.realpath(__file__))
        self.MODEL_emb = MODEL_emb
        
        # Initialize the Model objects and arguments - 
        self.OBJContextTree = ctb.ContextTree(starting_keyword=keyword, domain=domain, 
                                                model_to_load=MODEL_gen_llm)
        self.OBJContextTree.reset_mem_ctx()
        self.MODEL_ARGS_gen_llm = {"remember_raw_response" : False, "batch_query" : BATCH_QUERY,
                            "no_history" : False, "use_random_seed" : False}
        
        self.OBJHierarchEmb = heb.HierarchEmbdTree()

    def create_embedding(self, num_trees:int=2, depth_cap:int=2):

        TIME_context_tree_start = time.time()

        for run_n in tqdm(range(num_trees), desc="Creating Context Trees"):
            NODE_root = self.OBJContextTree.bfs(self.OBJContextTree.STARTING_KEYWORD, depth_cap=depth_cap, **self.MODEL_ARGS_gen_llm)
            self.OBJContextTree.save_tree(self.OBJContextTree.STARTING_KEYWORD, NODE_root, run_n=run_n+1)

            # making sure to clear memory before each run-
            torch.cuda.empty_cache()
            self.OBJContextTree.reset_mem_ctx()
        TIME_context_tree_total = time.time() - TIME_context_tree_start
        # Once all the context trees are created, we calculate the hierarchical-embeddings
        PATH_context_forest = os.path.join(self.OBJContextTree.PATH_output_trees,
                                           self.OBJContextTree.DOMAIN, 
                                           self.OBJContextTree.STARTING_KEYWORD)
        TIME_embedding_tree_start = time.time()
        node_per_keyword = self.OBJHierarchEmb.create_embeddings(PATH_context_forest, layer_strategy='all', topic=self.OBJContextTree.STARTING_KEYWORD)
        torch.cuda.empty_cache()
        TIME_embedding_tree_total = time.time() - TIME_embedding_tree_start

        return TIME_context_tree_total, TIME_embedding_tree_total, node_per_keyword

if __name__ == "__main__":
    temprature = 40
    num_trees = 1
    depth_cap = 3

    # keyword = "heat coefficient"
    domain = "Theoretical Physics"

    list_of_medicines = {}
    PATH_old_files = "/home/ppathak2/Hypothesis_Generation_Active_Learning/output_trees/medicinesCOVID"
    PATH_output_dir = "/home/ppathak2/Hypothesis_Generation_Active_Learning/output_trees/TESTS"
    # for med in os.listdir(PATH_old_files):
    #     list_of_medicines[med.lower().replace(' ', '_')] = med

    with open("/home/ppathak2/Hypothesis_Generation_Active_Learning/diseases.txt", 'r') as f:
        drugs = f.readlines()
        # drugs = ["COVID-19"]
    keywords = ["General relativity"]
    for keyword in keywords:
        keyword = keyword.strip()

        print("\n\t\t", "-"*50, " ", keyword, " ", "-"*50, "\n")

        OBJ_HierarchEmbPipe = HierarchicalEmbPipeline(keyword, domain)
        OBJ_HierarchEmbPipe.OBJContextTree.PATH_output_trees = PATH_output_dir
        
        # Setting up some configurable arguments-
        OBJ_HierarchEmbPipe.OBJContextTree.generation_args['temperature'] = temprature/100
        OBJ_HierarchEmbPipe.MODEL_ARGS_gen_llm["batch_query"] = True
        OBJ_HierarchEmbPipe.MODEL_ARGS_gen_llm["no_history"] = True

        _, _ = OBJ_HierarchEmbPipe.create_embedding(num_trees, depth_cap)
       