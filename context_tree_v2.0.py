import re
from collections import deque
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 
from matscibert_inf.ner.NER_inference import NER_INF
import os
import torch
import json
import random

class Node:
    def __init__(self, keyword:str, response:str, parent:object=None, depth:int=0):
        self.keyword = keyword
        self.response = response
        self.parent = parent
        self.children = []
        self.depth = depth

    def add_child(self, child_node):
        child_node.depth = self.depth + 1
        self.children.append(child_node)
    
    def add_children(self, children_nodes):
        for child_node in children_nodes:
            self.add_child(child_node)

    def is_leaf(self):
        return len(self.children) == 0

    def __repr__(self):
        return f"Node(keyword={self.keyword}), depth={self.depth}\n\nResponse='{self.response}'"
    
class ContextTree:

    def __init__(self, starting_keyword:str, domain:str,
                 model_to_load:str="microsoft/Phi-3.5-mini-instruct"):
        
        self.PATH_self_dir = os.path.dirname(os.path.realpath(__file__))
        self.PATH_output_trees = os.path.join(self.PATH_self_dir, 'output_trees')
        self.load_LLM(model_to_load)
        self.STARTING_KEYWORD = starting_keyword
        self.DOMAIN = domain
        self.messages = [
            {"role" : "system", "content" : f"You are an AI exploring the topic {self.STARTING_KEYWORD} in {self.DOMAIN} context. You're defining keywords on factual knowledge."}
        ]

        self.base_prompt = (f"Can you give a technical definition of <KEYWORD> in a few lines? "
                       f"If the word has multiple contexts, stick to a single context. "
                       f"Simply state the technical words in this definition, but don't define them. "
                       f"At the end of the definition, list out the technical words and the main context in this format - "
                       "(only mention the strongest technical keywords in order of relevance to this keyword)\n"
                       f"'tech_words=[a,b]-<<KEYWORD>>'\n"
                       f"'context=some description-<<KEYWORD>>'")
        
        self.base_keyword_prompt = (f"Can you give a technical definition of <KEYWORD> in a few lines? "
                       f"If the word has multiple contexts, stick to a single context. "
                       f"Simply state the technical words in this definition, but don't define them. "
                       f"At the end of the definition, list out the technical words and the main context in this format - "
                       "(only mention the strongest technical keywords in order of relevance to this keyword)\n"
                       f"'tech_words=[a,b]-<<KEYWORD>>'\n"
                       f"'context=some description-<<KEYWORD>>'")

        self.generation_args = { 
                            "max_new_tokens": 500, 
                            "return_full_text": False, 
                            "temperature": 0.0, 
                            "do_sample": False, 
                        }
        
        self.NER = NER_INF()
        self.NER_model = self.NER.initialize_infer()
        

    def load_LLM(self, model_to_load):
        self.LLM_tokenizer = AutoTokenizer.from_pretrained(model_to_load, trust_remote_code=False)
        self.LLM_model = AutoModelForCausalLM.from_pretrained(model_to_load, device_map="cuda:0", torch_dtype="auto", trust_remote_code=False)
        self.pipe = pipeline("text-generation", model=self.LLM_model, tokenizer=self.LLM_tokenizer)

    def _query(self, prompt:str):
        """A Method that queries a given prompt to the LLM and returns the response.

        Args:
            prompt (str): the prompt to query

        Returns:
            str: the response from LLM
        """
        self.messages.append({"role" : "user", "content" : prompt})

        output = self.pipe(self.messages, **self.generation_args)
        response = output[0]['generated_text']
        return response


    def extract_info(self, response, keyword):
        """
        Extracts both technical words and context from LLM response using regex.
        Returns a tuple (list of extracted words, extracted context).
        """
        words_match = re.search(r"tech_words=\[(.*?)\]-<" + re.escape(keyword) + r">", response)
        context_match = re.search(r"context=(.*?)-<" + re.escape(keyword) + r">", response)

        words = words_match.group(1).split(',') if words_match else []
        context = context_match.group(1).strip() if context_match else None

        filtered_keywords = []
        for word in words:
            word = word.strip()
            if word not in filtered_keywords and len(word) > 1:
                filtered_keywords.append(word)

        return filtered_keywords, context


    def linear_exploration(self, starting_keyword, depth_cap = 5):
        
        keyword = starting_keyword

        while depth_cap >=0:
            
            prompt = self.base_prompt.replace("<KEYWORD>", keyword)
            response = self._query(prompt)
            print(f"\n🔹 Depth: {depth_cap} | Keyword: {keyword}")
            print(f"📜 Prompt:\n{prompt}")
            print('-'*50)
            print("📜 Reponse: ", response)
            print("="*100)

            self.messages.append({"role" : "assistant", "content" : response})
            keyword = re.search(r"tech_words=\[(.*?)\]-<" + re.escape(keyword) + r">", response)[0].split('[')[1].split(']')[0].split()[0][:-1]
            depth_cap -= 1
        with open('tree.json', 'w', encoding='utf-8') as f:
            json.dump({"content" : self.messages}, f)

    def get_keywords(self, response:str, keyword:str, keyword_opt:str):
        new_keywords, _ = self.extract_info(response, keyword)
        ner_keywords = self.NER.infer_caption(response.split('\n\ntech_words=[')[0], self.NER_model)
        ner_keywords = self.NER.remove_o_tag(ner_keywords, {})

        ner_filtered = self.NER.infer_caption(response.split('tech_words=[')[1].split(']')[0], self.NER_model)
        ner_filtered = self.NER.remove_o_tag(ner_filtered, {})

        print(f"🔍 Keywords from the prompt: ", new_keywords)
        print()
        print(f"🔍 Keywords from the NER: ", ner_keywords)
        print()
        print(f"🔍 Keywords from the prompt filtered by the NER: ", ner_filtered)
        if keyword_opt == 'LLM':
            return new_keywords
        elif keyword_opt == 'NER':
            return ner_keywords
        elif keyword_opt == 'FILTERED':
            return ner_filtered
        
    def bfs(self, starting_keyword:str, depth_cap:int=4, keyword_opt:str='LLM', seed:int=None):
        """A method that performs BFS on the context tree for a given starting keyword.

        Args:
            starting_keyword (str): The starting keyword to eplore the tree for
            depth_cap (int, optional): the maximum depth to go for each branch. Defaults to 4.
            keyword_opt (str, optional): THe option to choose which approach to use for keyword extraction between LLM, NER, or BOTH. Defaults to 'LLM'.
            seed (int, optional): _description_. Defaults to 0.
        """
        if seed:
            torch.manual_seed(seed)
        root_response = self._query(self.base_prompt.replace("<KEYWORD>", starting_keyword))
        # Starting with the DFS tree-
        NODE_root = Node(keyword=starting_keyword, response=root_response)

        queue = deque([NODE_root])
        while queue:
            node = queue.popleft()
            if node.depth > depth_cap:
                continue
            print("\n\n", "=" * 75)
            print(f"📜 DEPTH - {node.depth} Root Response for keyword: {node.keyword} - {node.response}")
            new_keywords = self.get_keywords(node.response, node.keyword, keyword_opt)
            print("\n\n", "=" * 75)
            for keyword in new_keywords:
                if keyword.lower() == node.keyword.lower():
                    continue
                child_response = self._query(self.base_prompt.replace("<KEYWORD>", keyword))
                child_node = Node(keyword=keyword, response=child_response, depth=node.depth + 1, parent=node)
                node.add_child(child_node)
                queue.append(child_node)
        return NODE_root

    def save_tree(self, starting_keyword:str, root_node: Node):
        tree_dictionary = {}
        lookup_dictionary = {}

        def build_tree_recursive(node):
            lookup_dictionary[node.keyword] = {
                "content": str(node),
                "depth": node.depth,
                "parent": node.parent.keyword if node.parent else None
            }

            child_dict = {}
            for child in node.children:
                child_dict[child.keyword] = build_tree_recursive(child)

            return child_dict

        tree_dictionary[root_node.keyword] = build_tree_recursive(root_node)

        with open(f"{starting_keyword}_tree.json", "w") as f:
            json.dump(tree_dictionary, f, indent=4)

        with open(f"{starting_keyword}_lookup_table.json", "w") as f:
            json.dump(lookup_dictionary, f, indent=4)
            

if __name__ == "__main__":
    keywords = ["heat coefficient", "Phase Diagram", "Diffusion Coefficient"]
    # keywords = ["heat coefficient"]
    domain = "material science"

    for starting_keyword in keywords:
        OBJ_context_tree = ContextTree(starting_keyword=starting_keyword, domain=domain)
        NODE_root = OBJ_context_tree.bfs(starting_keyword, seed=0, depth_cap=3)
        OBJ_context_tree.save_tree(starting_keyword, NODE_root)
