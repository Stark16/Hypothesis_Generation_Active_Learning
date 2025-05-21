import re
from collections import deque
from transformers import AutoTokenizer, AutoModelForCausalLM 
from matscibert_inf.ner.NER_inference import NER_INF
import os
import torch
import json
from tqdm import tqdm
import uuid

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
            {"role" : "system", "content" : f"You are an AI assistant exploring the topic {self.STARTING_KEYWORD} in {self.DOMAIN} context. You're defining keywords on factual knowledge."}
        ]

        self.base_prompt = (f"Give a short technical definition of <KEYWORD> in a few lines. "
                            f"Stick to one specific context. Do NOT explain the technical words. "
                            f"After the definition, strictly output the following two lines — no deviation:\n\n"
                            f"tech_words=[a, b, c]-<<KEYWORD>>\n"
                            f"context=your context here-<<KEYWORD>>\n\n"
                            f"(Do NOT include any other labels, bullet points, or explanations after this.)")

        
        self.base_keyword_prompt = (f"Can you give a technical definition of <KEYWORD> in a few lines? "
                       f"If the word has multiple contexts, stick to a single context. "
                       f"Simply state the technical words in this definition, but don't define them. "
                       f"At the end of the definition, list out the technical words and the main context in this format - "
                       "(only mention the strongest technical keywords in order of relevance to this keyword)\n"
                       f"'tech_words=[a,b]-<<KEYWORD>>'\n"
                       f"'context=some description-<<KEYWORD>>'")

        self.generation_args = { 
                            "max_new_tokens": 512, 
                            "temperature": 0.05,
                            
                        }
        
        self.NER = NER_INF()
        self.NER_model = self.NER.initialize_infer()
        

    def load_LLM(self, model_to_load, LLM_device_map:str="auto"):
        self.LLM_tokenizer = AutoTokenizer.from_pretrained(model_to_load, trust_remote_code=False, padding_side='left')
        self.LLM_model = AutoModelForCausalLM.from_pretrained(model_to_load, device_map=LLM_device_map, torch_dtype="auto", trust_remote_code=False)

    def _query(self, prompt:str, remember_raw_response:bool=True, no_history:bool=False, use_random_seed:bool=False):
        """A Method that queries a given prompt with appropriate settings

        Args:
            prompt (str): the prompt to give to the generative model
            remember_raw_response (bool, optional): if set to False the model remembers the whole raw response for future context. Defaults to True.
            no_history (bool, optional): If set to True the model context is isolated for each prompt. Defaults to False.

        Returns:
            str: The response from the model as string
        """
        self.messages.append({"role" : "user", "content" : prompt})
        prompt = prompt + '<SEED=' + str(uuid.uuid4()) + '>' if use_random_seed else prompt
        if no_history:
            messages = [self.messages[0], {"role" : "user", "content" : prompt}]
            tokenized_chat = self.LLM_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        else:
            tokenized_chat = self.LLM_tokenizer.apply_chat_template(self.messages, add_generation_prompt=True, return_tensors="pt")
        output = self.LLM_model.generate(tokenized_chat, **self.generation_args)
        output = self.LLM_tokenizer.batch_decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        response = output[0].split("<|assistant|>")[-1].split("<|end|>")[0]

        if no_history:
            return response
        
        if remember_raw_response:
            self.messages.append({"role" : "assistant", "content" : response})
        else:
            self.messages.append({"role" : "assistant", "content" : response.split('\n\ntech_words=[')[0]})
        return response
    
    def _query_batch(self, keywords:list, remember_raw_response:bool=True, no_history:bool=False, use_random_seed:bool=False):
        """Method similar to _query only it batch queries a batch on len(keywords) prompts where n is number of nodes in a level of context tree

        Args:
            keywords (list): list of keywords that needs to be batch processed
            remember_raw_response (bool, optional): if set to False the model remembers the whole raw response for future context. Defaults to True.
            no_history (bool, optional): If set to True the model context is isolated for each prompt. Defaults to False.

        Returns:
            list: a list of responses from the model with string elements as individual responses.
        """

        batch_msgs = []
        base_prompt = self.base_prompt + '<SEED=' + str(uuid.uuid4()) + '>' if use_random_seed else self.base_prompt
        # first we need to create the conversation history list for each keyword in the batch-
        # depending on cofig argument no_history, the lenght of the context varies. It is usually = 2 if the flag is True.
        for keyword in keywords:
            if no_history:
                curr_messages_tree = [self.messages[0], {"role" : "user", "content" : base_prompt.replace("<KEYWORD>", keyword)}]
            else:
                curr_messages_tree = self.messages.copy()
                curr_messages_tree.append({"role" : "user", "content" : base_prompt.replace("<KEYWORD>", keyword)})
            batch_msgs.append(curr_messages_tree)

        # Then we query all the keywords in the batch together-
        batch_tokenized_chat = self.LLM_tokenizer.apply_chat_template(batch_msgs, add_generation_prompt=True, return_tensors="pt", padding=True)
        batch_output = self.LLM_model.generate(batch_tokenized_chat, **self.generation_args)
        batch_output = self.LLM_tokenizer.batch_decode(batch_output, skip_special_tokens=False, clean_up_tokenization_spaces=True)

        # Now, it is time to update the messages list to update the converation
        # This update needs to happen in the ordet the keywords are explored-
        batch_responses = []
        for i, keyword in enumerate(keywords):
            response = batch_output[i].split("<|assistant|>")[-1].split("<|end|>")[0]
            batch_responses.append(response)

            # Now, we only track the responses if no_history is False-
            if no_history == False:
                # First store the prompt of the current keyword-
                self.messages.append({"role" : "user", "content" : self.base_prompt.replace("<KEYWORD>", keyword)})
                # Then store the response for that prompt
                if remember_raw_response:
                    self.messages.append({"role" : "assistant", "content" : response})
                else:
                    self.messages.append({"role" : "assistant", "content" : response.split('\n\ntech_words=[')[0]})
            
        return batch_responses


    def extract_info(self, response, keyword):
        """
        Extracts both technical words and context from LLM response using regex.
        Returns a tuple (list of extracted words, extracted context).
        """
        words_match = re.search(r"(?i)tech_words=\[(.*?)\]", response)
        if not words_match:
            words_match = re.search(r"(?i)technical words\W*\[(.*?)\]", response)
        context_match = re.search(r"(?i)context=(.*?)-<", response)

        words = words_match.group(1).split(',') if words_match else []
        context = context_match.group(1).strip() if context_match else None

        filtered_keywords = []
        for word in words:
            word = word.strip()
            if word not in filtered_keywords and len(word) > 1 and word.lower() != keyword.lower():
                filtered_keywords.append(word)

        return filtered_keywords, context


    def linear_exploration(self, starting_keyword, depth_cap = 5, logger=False):
        
        keyword = starting_keyword

        while depth_cap >=0:
            
            prompt = self.base_prompt.replace("<KEYWORD>", keyword)
            response = self._query(prompt)
            if logger:
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
        # ner_keywords = self.NER.infer_caption(response.split('\n\ntech_words=[')[0], self.NER_model)
        # ner_keywords = self.NER.remove_o_tag(ner_keywords, {})

        # ner_filtered = self.NER.infer_caption(response.split('tech_words=[')[1].split(']')[0], self.NER_model)
        # ner_filtered = self.NER.remove_o_tag(ner_filtered, {})

        # print(f"🔍 Keywords from the prompt: ", new_keywords)
        # print()
        # print(f"🔍 Keywords from the NER: ", ner_keywords)
        # print()
        # print(f"🔍 Keywords from the prompt filtered by the NER: ", ner_filtered)
        if keyword_opt == 'LLM':
            return new_keywords
        # elif keyword_opt == 'NER':
        #     return ner_keywords
        # elif keyword_opt == 'FILTERED':
        #     return ner_filtered
        
    def bfs(self, starting_keyword:str, depth_cap:int=4, keyword_opt:str='LLM', use_random_seed:bool=False, 
            remember_raw_response:bool=True, batch_query:bool=False, no_history:bool=False):
        """A method that performs BFS on the context tree for a given starting keyword.

        Args:
            starting_keyword (str): The starting keyword to eplore the tree for
            depth_cap (int, optional): the maximum depth to go for each branch. Defaults to 4.
            keyword_opt (str, optional): THe option to choose which approach to use for keyword extraction between LLM, NER, or BOTH. Defaults to 'LLM'.
            seed (int, optional): _description_. Defaults to 0.
        """
        # Starting with the BFS tree-
        NODE_root = Node(keyword=starting_keyword, response=None)

        queue = deque([NODE_root])
        while queue:
            node = queue.popleft()
            if node.depth > depth_cap:
                continue

            # Now we wanna make sure the node we are about to explore has been prompted-
            if node.response == None:
                node_response = self._query(self.base_prompt.replace("<KEYWORD>", starting_keyword), remember_raw_response, use_random_seed=use_random_seed)
                node.response = node_response
            # print("\n\n", "=" * 75)
            # print(f"📜 DEPTH - {node.depth} Key keyword: {node.keyword}")
            new_keywords = self.get_keywords(node.response, node.keyword, keyword_opt)
            # print("\n\n", "=" * 75)

            # Prune the current keyword from the batch:
            if batch_query and len(new_keywords)>0:
                batch_responses = self._query_batch(new_keywords, remember_raw_response, no_history, use_random_seed=use_random_seed)

            for i, keyword in enumerate(new_keywords):
                if keyword.lower() == node.keyword.lower():
                    continue
                if batch_query:
                    child_response = batch_responses[i]
                else:
                    child_response = self._query(self.base_prompt.replace("<KEYWORD>", keyword), remember_raw_response, use_random_seed=use_random_seed)
                child_node = Node(keyword=keyword, response=child_response, depth=node.depth + 1, parent=node)
                node.add_child(child_node)
                queue.append(child_node)

        return NODE_root

    def save_tree(self, starting_keyword:str, root_node: Node, run_n:int=None):
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
                child_dict[child.keyword] = {"depth" : child.depth, "response" : child.response.split('tech_words')[0], "children": build_tree_recursive(child)}

            return child_dict

        # tree_dictionary[root_node.keyword] = build_tree_recursive(root_node)
        tree_dictionary[root_node.keyword] = {"depth" : root_node.depth, "response" : root_node.response.split('tech_words')[0], 
                                              "children": build_tree_recursive(root_node)}

        output_dir = os.path.join(self.PATH_output_trees, self.DOMAIN, starting_keyword)
        if run_n:
            output_dir = os.path.join(output_dir, 'run_' + str(run_n))

        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/tree.json", "w") as f:
            json.dump(tree_dictionary, f, indent=4)

        with open(f"{output_dir}/lookup_table.json", "w") as f:
            json.dump(lookup_dictionary, f, indent=4)
            
        with open(f"{output_dir}/conversation.json", "w") as f:
            conv = {'conversation' : self.messages}
            json.dump(conv, f, indent=4)


if __name__ == "__main__":
    keywords = ["heat coefficient", "Phase Diagram", "Diffusion Coefficient"]
    keywords = ["endometriosis", "covid-19", "oxygen"]
    domain = "biomedical"
    num_runs_per_tree = 200
    temprature_values = [40]
    for temprature in temprature_values:
        for starting_keyword in keywords:
            for run_n in tqdm(range(num_runs_per_tree)):
                OBJ_context_tree = ContextTree(starting_keyword=starting_keyword, domain=domain)
                OBJ_context_tree.generation_args['temperature'] = temprature/100
                NODE_root = OBJ_context_tree.bfs(starting_keyword, depth_cap=1, remember_raw_response=False, batch_query=True, no_history=True, use_random_seed=True)
                OBJ_context_tree.save_tree(starting_keyword + '_' +str(temprature_values), NODE_root, run_n=run_n+1)

                # making sure to clear memory before each run-
                torch.cuda.empty_cache()
