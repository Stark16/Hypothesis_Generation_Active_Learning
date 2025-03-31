import re
from collections import deque
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 
import os
import torch
import json

class TreeNode:
    def __init__(self, keyword, depth=0, parent_context=None):
        self.keyword = keyword  # Main keyword for this node
        self.context = parent_context  # Context from parent node
        self.prompt = self.generate_prompt()  # Generates query prompt
        self.response = None  # Placeholder for LLM response
        self.children = []  # Child nodes
        self.depth = depth  # Depth level in tree

    def generate_prompt(self):
        """Generates a structured query prompt, incorporating parent context if available."""
        base_prompt = (f"Can you give a general technical definition of {self.keyword} in a few lines? "
                       f"If the word has multiple contexts, stick to a single context. "
                       f"Simply state the technical words in this definition, but don't define them. "
                       f"At the end of the definition, list out the technical words and the main context in this format - "
                       "(only mention the strongest technical keywords in order of relevance to this keyword)\n"
                       f"'tech_words=[a,b]-<{self.keyword}>'\n"
                       f"'context=some description-<{self.keyword}>'")

        if self.context:
            base_prompt = f"({self.context})\n{base_prompt}"

        return base_prompt
    
class Dfs:

    def __init__(self, keyword):
        self.keyword = keyword
        self. children = []
    
class ContextTree:

    def __init__(self, starting_keyword:str, domain:str,
                 model_to_load:str="microsoft/Phi-3.5-mini-instruct"):
        
        self.PATH_self_dir = os.path.dirname(os.path.realpath(__file__))
        self.load_LLM(model_to_load)
        self.STARTING_KEYWORD = starting_keyword
        self.DOMAIN = domain
        self.messages = [
            {"role" : "system", "content" : f"You are an AI exploring the topic {self.STARTING_KEYWORD} in {self.DOMAIN} context. You're defining keywords on factual knowledge."}
        ]

        self.base_prompt = (f"Can you give a general technical definition of KEYWORD in a few lines? "
                       f"If the word has multiple contexts, stick to a single context. "
                       f"Simply state the technical words in this definition, but don't define them. "
                       f"At the end of the definition, list out the technical words and the main context in this format - "
                       "(only mention the strongest technical keywords in order of relevance to this keyword)\n"
                       f"'tech_words=[a,b]-<KEYWORD>'\n"
                       f"'context=some description-<KEYWORD>'")

        generation_args = { 
                            "max_new_tokens": 500, 
                            "return_full_text": False, 
                            "temperature": 0.0, 
                            "do_sample": False, 
                        }
        
        self.linear_exploration(starting_keyword, generation_args)

    def load_LLM(self, model_to_load):
        self.LLM_tokenizer = AutoTokenizer.from_pretrained(model_to_load, trust_remote_code=False)
        self.LLM_model = AutoModelForCausalLM.from_pretrained(model_to_load, device_map="cuda:0", torch_dtype="auto", trust_remote_code=False)
        self.pipe = pipeline("text-generation", model=self.LLM_model, tokenizer=self.LLM_tokenizer)


    def extract_info(self, response, keyword):
        """
        Extracts both technical words and context from LLM response using regex.
        Returns a tuple (list of extracted words, extracted context).
        """
        words_match = re.search(r"tech_words=\[(.*?)\]-<" + re.escape(keyword) + r">", response)
        context_match = re.search(r"context=(.*?)-<" + re.escape(keyword) + r">", response)

        words = words_match.group(1).split(',') if words_match else []
        context = context_match.group(1).strip() if context_match else None

        words = [word.strip() for word in words if word.strip()]
        return words, context

    def linear_exploration(self, starting_keyword, generation_args, depth_cap = 5):
        
        keyword = starting_keyword

        while depth_cap >=0:
            prompt = self.base_prompt.replace("KEYWORD", keyword)
            self.messages.append({"role" : "user", "content" : prompt})

            output = self.pipe(self.messages, **generation_args)
            response = output[0]['generated_text']

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
        
    def dfs(self, node, depth, depth_cap=3):
        if not node or depth > depth_cap:
            return
        
        
        for child in node.children:
            self.dfs(child, depth + 1, depth_cap)

    def bfs(self, root):
        """
        Performs a breadth-first traversal, guiding the user through manual LLM queries.
        """
        queue = deque([root])

        while queue:
            node = queue.popleft()

            print(f"\n🔹 Depth: {node.depth} | Keyword: {node.keyword}")
            print(f"📜 Prompt:\n{node.prompt}")
            print("="*100)

            # Get user input (manual copy-paste of LLM response)
            response = self._query(node.prompt)
            node.response = response  # Store the response

            # Extract new technical words & context
            new_keywords, context = self.extract_info(response, node.keyword)
            
            if not new_keywords:
                print(f"✅ No new keywords found. This is a leaf node.\n")
                continue  # Stop if no new keywords

            print(f"🔍 Extracted keywords: {new_keywords}")
            print(f"📜 Extracted context: {context}")

            # Create child nodes and add them to the queue
            for keyword in new_keywords:
                child_node = TreeNode(keyword, node.depth + 1, context)
                node.children.append(child_node)
                queue.append(child_node)
            break


if __name__ == "__main__":
    
    OBJ_context_tree = ContextTree(starting_keyword="thermal_conductivity", domain="material science")
