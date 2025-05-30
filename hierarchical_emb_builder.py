import os
import json
from tqdm import tqdm
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel

class HierarchEmbdTree:
    def __init__(self, model_control:str='allenai/scibert_scivocab_uncased', layer_strategy:str='last_two', device:str='cuda'):
        self.layer_strategy = layer_strategy
        self.device = device
        self.PATH_self_dir = os.path.dirname(os.path.realpath(__file__))
        self.MODEL_control = AutoModel.from_pretrained(model_control, output_hidden_states=True).to(self.device)
        self.TOKENIZER_control = AutoTokenizer.from_pretrained(model_control)

    def load_json_tree(self, PATH_tree:str):
        """Function to load the json tree, pretty self explainatory

        Args:
            PATH_tree (str): Path to the tree that needs to be loaded

        Returns:
            dict: dictionary object of the context tree loaded from the json file
        """
        with open (PATH_tree, 'r') as f:
            json_tree = json.load(f)
        return json_tree

    def embed_texts(self, texts):
        """A method to get batch of embeddings for given batch of texts using the loaded self.MODEL_control and tokenizer self.TOKENIZER_control.

        Args:
            texts (list): list of sentences to get the embedding for

        Raises:
            ValueError: if the self.layer_strategy is not amongst - ['static', 'first', 'first_three', 'last', 'last_two', 'last_three']

        Returns:
            tupple: (embeddings, input_ids, offsets)
        """

        encoding = self.TOKENIZER_control.batch_encode_plus(
            texts,
            max_length=512,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {k: encoding[k].to(self.device) for k in ["input_ids", "token_type_ids", "attention_mask"] if k in encoding}
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].unsqueeze(-1).unsqueeze(0)

        if self.layer_strategy == 'static':
            embeddings = self.MODEL_control.embeddings.word_embeddings(input_ids)
        else:
            with torch.no_grad():
                outputs = self.MODEL_control(**inputs)
            hidden_states = outputs.hidden_states

            if self.layer_strategy == 'first':
                embeddings = hidden_states[1]  # First transformer layer
            elif self.layer_strategy == 'first_three':
                embeddings = torch.mean(torch.stack(hidden_states[1:4]), dim=0)  # First3 transformer layers
            elif self.layer_strategy == 'last':
                embeddings = hidden_states[-1]
            elif self.layer_strategy == 'last_two':
                embeddings = torch.mean(torch.stack(hidden_states[-2:]), dim=0)
            elif self.layer_strategy == 'last_three':
                embeddings = torch.mean(torch.stack(hidden_states[-3:]), dim=0)
            else:
                raise ValueError("Invalid layer_strategy: choose from ['static', 'first', 'first_three', 'last', 'last_two', 'last_three']")

        embeddings = embeddings.masked_fill(attention_mask.logical_not(), 0)

        return embeddings, encoding["input_ids"], encoding["offset_mapping"]
    
    def tokenize_and_find(self, texts, keywords):
        if isinstance(keywords, str):
            keywords = [keywords]

        keyword_id_groups = []
        for kw in keywords:
            tokens = self.TOKENIZER_control.tokenize(kw)
            ids = self.TOKENIZER_control.convert_tokens_to_ids(tokens)
            keyword_id_groups.append(ids)

        inputs = self.TOKENIZER_control(texts,
                                max_length=512,
                                return_offsets_mapping=True,
                                add_special_tokens=False,
                                return_tensors="pt",
                                padding=True,
                                truncation=True)

        results = []
        for i in range(len(texts)):
            token_ids = inputs['input_ids'][i].tolist()
            match_positions = []

            for keyword_ids in keyword_id_groups:
                for j in range(len(token_ids) - len(keyword_ids) + 1):
                    if token_ids[j:j + len(keyword_ids)] == keyword_ids:
                        match_positions.append(tuple(range(j, j + len(keyword_ids))))

            if match_positions:
                results.append([i, match_positions])

        return results
    

    def fetch_embedding(self, keyword:str, response:str=None):
        if not response:
            return self.emb_library[keyword]
        
        # handeling a case in case the LLM response was empty-
        # We make the response = keyword itself. Reasoning is as follows-
        """
        > Since sciBERT is deterministic, if we give it a single keyword as input it's embedding will always be constant
        > We can use this to our advantage by working around the fact that a certain LLM could not 'define' a keyword, but we still know the keyword.
        > This allows us to still get an embedding in case of a missing definition with the catch that the embedding is the standalone context embedding of that keyword.
        """
        
        if (len(response.split()) == 0):
            response = keyword
        embedding, _, _ = self.embed_texts([response])
        keyword_matches = self.tokenize_and_find([response], keyword)

        embs_batch = []
        occurrence_idx = 0
        for i in range(len(embedding)):
            embs_t = []

            if occurrence_idx < len(keyword_matches):
                match_idx = keyword_matches[occurrence_idx][0]
                if match_idx == i:
                    for token_group in keyword_matches[occurrence_idx][1]:
                        embs_t.append(embedding[0, i, list(token_group), :].mean(dim=0).cpu().tolist())
                    occurrence_idx += 1
                else:
                    embs_t = []

            embs_batch.append(embs_t)
            emd = np.array(embs_batch[0])
        return emd
    
    def process_tree(self, tree_dict:dict):

        embedding_coeff = {}    # the format for data key is - [node's depth, node's num of occurances, the response from LLM]
        def process_nodes(tree_dict:dict):
            for keyword, node_dict in tree_dict.items():
                
                embdg = self.fetch_embedding(keyword, node_dict["response"])
                if keyword in list(embedding_coeff.keys()):
                    embedding_coeff[keyword]["data"][1] += 1
                else:
                    embedding_coeff[keyword] = {"data" : [node_dict['depth'], 1, node_dict['response']]}
                    embedding_coeff[keyword]["raw_enc"] = embdg.tolist()
                
                # Using formula for weight = [(1 * # of occurances) / (depth of the node)] * embedding:
                weight_coeff = (1 * embedding_coeff[keyword]["data"][1])/(embedding_coeff[keyword]["data"][0])
                embedding_coeff[keyword]["w88_enc"] = (weight_coeff * np.array(embedding_coeff[keyword]["raw_enc"])).tolist()

                # Now reccurse-
                if (len(node_dict['children']) != 0):
                    process_nodes(node_dict["children"])
        
        process_nodes(tree_dict=tree_dict)
        return embedding_coeff
    
    def create_embeddings(self, PATH_ContextForst:str):
        for subtree_dir in tqdm(os.listdir(PATH_ContextForst), desc="Calculating Embeddings"):
            if subtree_dir == 'LOG_failed_responses.json':
                continue
            PATH_tree = os.path.join(PATH_ContextForst, subtree_dir, 'tree.json')
            PATH_out_embd_tree = os.path.join(PATH_ContextForst, subtree_dir, 'embdng_tree_v2.json')
            json_tree = self.load_json_tree(PATH_tree)
            embedding_coeff = self.process_tree(json_tree)

            with open(PATH_out_embd_tree, 'w') as f:
                json.dump(embedding_coeff, f)
    

 
if __name__ == '__main__':
    
    domain = "medicine"
    PATH_self_dir = os.path.dirname(os.path.realpath(__file__))
    PATH_domain = os.path.join(PATH_self_dir, "output_trees", domain)
    # topics = os.listdir(PATH_domain)
    topics = ['didecyldimethylammonium_[40]', 'acetaminophen_[40]']

    for topic in tqdm(topics):
        PATH_current_topic = os.path.join(PATH_domain, topic)
        OBJ_EmbTree = HierarchEmbdTree()
        OBJ_EmbTree.create_embeddings(PATH_current_topic)
        
