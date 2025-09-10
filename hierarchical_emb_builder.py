import os
import json
from tqdm import tqdm
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel

class HierarchEmbdTree:
    def __init__(self, model_control:str='allenai/scibert_scivocab_uncased', device:str='cuda'):
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

    def embed_texts(self, texts, layer_strategy:str='last_two'):
        """A method to get batch of embeddings for given batch of texts using the loaded self.MODEL_control and tokenizer self.TOKENIZER_control.

        Args:
            texts (list): list of sentences to get the embedding for

        Raises:
            ValueError: if the layer_strategy is not amongst - ['static', 'first', 'first_three', 'last', 'last_two', 'last_three']

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

        if layer_strategy == 'static':
            embeddings = self.MODEL_control.embeddings.word_embeddings(input_ids)
            embeddings = embeddings.masked_fill(attention_mask.logical_not(), 0)
        elif layer_strategy == 'all':
            with torch.no_grad():
                outputs = self.MODEL_control(**inputs)
            hidden_states = outputs.hidden_states
            # Precompute all strategies in one pass
            emb_dict = {}
            # Layer indices: 1-based for transformer layers, -1 is last, etc.
            emb_dict['first'] = hidden_states[0].masked_fill(attention_mask.logical_not(), 0)
            emb_dict['first_two'] = torch.mean(torch.stack(hidden_states[0:2]), dim=0).masked_fill(attention_mask.logical_not(), 0)
            emb_dict['first_three'] = torch.mean(torch.stack(hidden_states[0:3]), dim=0).masked_fill(attention_mask.logical_not(), 0)
            emb_dict['last'] = hidden_states[-1].masked_fill(attention_mask.logical_not(), 0)
            emb_dict['last_two'] = torch.mean(torch.stack(hidden_states[-2:]), dim=0).masked_fill(attention_mask.logical_not(), 0)
            emb_dict['last_three'] = torch.mean(torch.stack(hidden_states[-3:]), dim=0).masked_fill(attention_mask.logical_not(), 0)
            # Optionally add 'first_two' for completeness
            
            # Convert each tensor to a list of numpy arrays (one per input text)
            embeddings = {k: [emb.cpu().numpy() for emb in v] for k, v in emb_dict.items()}
        else:
            with torch.no_grad():
                outputs = self.MODEL_control(**inputs)
            hidden_states = outputs.hidden_states

            if layer_strategy == 'first':
                embeddings = hidden_states[0]
            elif layer_strategy == 'first_three':
                embeddings = torch.mean(torch.stack(hidden_states[0:3]), dim=0)
            elif layer_strategy == 'last':
                embeddings = hidden_states[-1]
            elif layer_strategy == 'last_two':
                embeddings = torch.mean(torch.stack(hidden_states[-2:]), dim=0)
            elif layer_strategy == 'last_three':
                embeddings = torch.mean(torch.stack(hidden_states[-3:]), dim=0)
            else:
                raise ValueError("Invalid layer_strategy: choose from ['static', 'first', 'first_three', 'last', 'last_two', 'last_three', 'all']")

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
    

    def fetch_embedding(self, keyword:str, response:str=None, layer_strategy:str='last_three'):

            # return self.emb_library[keyword]

        # handling a case in case the LLM response was empty-
        # We make the response = keyword itself. Reasoning is as follows-
        """
        > Since sciBERT is deterministic, if we give it a single keyword as input it's embedding will always be constant
        > We can use this to our advantage by working around the fact that a certain LLM could not 'define' a keyword, but we still know the keyword.
        > This allows us to still get an embedding in case of a missing definition with the catch that the embedding is the standalone context embedding of that keyword.
        """

        embedding, _, _ = self.embed_texts([response], layer_strategy=layer_strategy)

        if layer_strategy == 'all':
            all_embeddings = {}
            for strategy, emb in embedding.items():
                keyword_matches = self.tokenize_and_find([response], keyword)
                embs_batch = []
                occurrence_idx = 0
                # emb is a list of numpy arrays, one per input text (here, just one)
                # Convert to torch tensor for processing
                emb_tensor = torch.tensor(emb)
                for i in range(len(emb_tensor)):
                    embs_t = []
                    if occurrence_idx < len(keyword_matches):
                        match_idx = keyword_matches[occurrence_idx][0]
                        if match_idx == i:
                            for token_group in keyword_matches[occurrence_idx][1]:
                                # emb_tensor shape: (batch, seq_len, hidden)
                                embs_t.append(emb_tensor[0, i, list(token_group), :].mean(dim=0).cpu().tolist())
                            occurrence_idx += 1
                        else:
                            embs_t = []
                    embs_batch.append(embs_t)
                emd = np.array(embs_batch[0])
                all_embeddings[strategy] = emd
            return all_embeddings
        else:
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
    
    def process_tree(self, tree_dict:dict, layer_strategy:str='last_three'):
        if layer_strategy == 'all':
            # Will collect a dict of dicts: {strategy: {keyword: ...}}
            strategy_dicts = {}
            def process_nodes(tree_dict:dict):
                for keyword, node_dict in tree_dict.items():
                    embdg_dict = self.fetch_embedding(keyword, node_dict["response"], layer_strategy=layer_strategy)
                    for strategy, embdg in embdg_dict.items():
                        # Clean up strategy name for dict key (optional, but can keep as is)
                        if strategy not in strategy_dicts:
                            strategy_dicts[strategy] = {}
                        strat_dict = strategy_dicts[strategy]
                        if len(embdg) > 0:
                            if keyword in strat_dict:
                                strat_dict[keyword]["data"][1] += 1
                            else:
                                strat_dict[keyword] = {"data": [node_dict['depth'], 1, node_dict['response']]}
                                strat_dict[keyword]["raw_enc"] = embdg.tolist()
                            weight_coeff = (1 * strat_dict[keyword]["data"][1])/(strat_dict[keyword]["data"][0])
                            strat_dict[keyword]["w88_enc"] = (weight_coeff * np.array(strat_dict[keyword]["raw_enc"])).tolist()
                    # Now recurse for children for all strategies
                    if (len(node_dict['children']) != 0):
                        process_nodes(node_dict["children"])
            process_nodes(tree_dict=tree_dict)
            return strategy_dicts
        else:
            embedding_coeff = {}    # the format for data key is - [node's depth, node's num of occurances, the response from LLM]
            def process_nodes(tree_dict:dict):
                for keyword, node_dict in tree_dict.items():
                    embdg = self.fetch_embedding(keyword, node_dict["response"], layer_strategy=layer_strategy)
                    if (len(embdg) > 0):
                        if keyword in list(embedding_coeff.keys()):
                            embedding_coeff[keyword]["data"][1] += 1
                        else:
                            embedding_coeff[keyword] = {"data" : [node_dict['depth'], 1, node_dict['response']]}
                            embedding_coeff[keyword]["raw_enc"] = embdg.tolist()
                        # Using formula for weight = [(1 * # of occurances) / (depth of the node)] * embedding:
                        weight_coeff = (1 * embedding_coeff[keyword]["data"][1])/(embedding_coeff[keyword]["data"][0])
                        embedding_coeff[keyword]["w88_enc"] = (weight_coeff * np.array(embedding_coeff[keyword]["raw_enc"])).tolist()
                        # Now recurse-
                        if (len(node_dict['children']) != 0):
                            process_nodes(node_dict["children"])
            process_nodes(tree_dict=tree_dict)
            return embedding_coeff
    
    def create_embeddings(self, PATH_ContextForst: str, layer_strategy: str = 'last_three', topic = 'None', topic_idx='None'):
        for subtree_dir in tqdm(os.listdir(PATH_ContextForst), desc=f"Calc Embd for - {topic} - #{topic_idx}"):
            if subtree_dir == 'LOG_failed_responses.json':
                continue
            PATH_tree = os.path.join(PATH_ContextForst, subtree_dir, 'tree.json')
            json_tree = self.load_json_tree(PATH_tree)
            tree_str = json.dumps(json_tree)
            embedding_coeff = self.process_tree(json_tree, layer_strategy=layer_strategy)

            if layer_strategy == 'all':
                # embedding_coeff is a dict: {strategy: {keyword: ...}}
                # Check that all strat_dicts have the same keys
                all_keys = [set(strat_dict.keys()) for strat_dict in embedding_coeff.values()]
                if len(all_keys) > 1:
                    first_keys = all_keys[0]
                    for idx, keys in enumerate(all_keys[1:], 1):
                        if keys != first_keys:
                            print(f"[DEBUG] Mismatched keys in strategy dicts at index {idx}:\nFirst: {sorted(list(first_keys))}\nCurrent: {sorted(list(keys))}")
                            # Place breakpoint here if needed
                for strategy, strat_dict in embedding_coeff.items():
                    PATH_out_embd_tree = os.path.join(
                        PATH_ContextForst, subtree_dir, f"embdng_tree_v2_{strategy}.json"
                    )
                    with open(PATH_out_embd_tree, 'w') as f:
                        json.dump(strat_dict, f)
            else:
                PATH_out_embd_tree = os.path.join(PATH_ContextForst, subtree_dir, 'embdng_tree_v2.json')
                with open(PATH_out_embd_tree, 'w') as f:
                    json.dump(embedding_coeff, f)
    

if __name__ == '__main__':
    
    domain = "medicinesCOVID"
    PATH_self_dir = os.path.dirname(os.path.realpath(__file__))
    PATH_domain = os.path.join(PATH_self_dir, "output_trees", domain)
    topics = os.listdir(PATH_domain)
    topics = ['pralidoxime', 'Topotecan', 'Oxytetracycline', 'Trimethoprim', 'Flutamide']
    # topics = ['pralidoxime']

    for topic_idx, topic in enumerate(topics):
        PATH_current_topic = os.path.join(PATH_domain, topic)
        OBJ_EmbTree = HierarchEmbdTree()
        # try:
        OBJ_EmbTree.create_embeddings(PATH_current_topic, layer_strategy='all', 
                                      topic=topic, topic_idx=topic_idx)
        # except Exception as e:
        #     print(e)
        
