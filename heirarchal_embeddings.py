import context_tree_v2
import os
import json
from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer, AutoModel

class EmbdTree:
    def __init__(self, model_control:str='allenai/scibert_scivocab_cased'):
        self.PATH_self_dir = os.path.dirname(os.path.realpath(__file__))
        self.MODEL_control = AutoModel.from_pretrained(model_control)
        self.TOKENIZER_control = AutoTokenizer.from_pretrained(model_control)

    def get_keyword_idx(self, tokenized_input:AutoTokenizer, keyword:str):
        """This function returns the index at which the keyword appears in the tokenized input

        Args:
            tokenized_input (AutoTokenizer): The tokenized input given to infer the LLM
            keyword (str): the keyword we need to find

        Returns:
            list: A list of token indix ranges at which the keyword appeared
        """
        keyword_token = self.TOKENIZER_control([keyword], return_tensors='pt', padding=True, truncation=True)
        keyword_token_arr = keyword_token['input_ids'].detach().cpu().numpy()[0][1:-1]      # making a sublist of tokens representing the keyword

        # TODO: OPTIMIZATION - Currently keyword idx search is happening by converiting the keyword -> token -> detokenized numpy array, make it so the search happens in string space.
        keyword_detokenized_arr = np.array(self.TOKENIZER_control.convert_ids_to_tokens(keyword_token_arr))

        # now we find the subset in the initial tokenized input-
        tokenized_input_arr = tokenized_input['input_ids'].detach().cpu().numpy()[0]
        detokenized_input_arr = np.array(self.TOKENIZER_control.convert_ids_to_tokens(tokenized_input_arr))
        n = len(keyword_detokenized_arr)
        N = len(detokenized_input_arr)
        indx = []
        ## TODO: EDGE CASE ERROR - If the keyword was in a different gramatical context than the one used in responsed. We can't find the index. E.g. keyword="quantifies", response has "quanitfy"
        for i in range(N-n+1):
            if ((keyword_detokenized_arr == detokenized_input_arr[i:i+n]).all()):
                indx.append([i, n])

        return indx

    def get_embeddings(self, input:str, keyword:str):
        
        tokenized_input = self.TOKENIZER_control([input], return_tensors='pt', padding=True, truncation=True)

        # Now infer and get the embeddings-
        output = self.MODEL_control(**tokenized_input)
        output_arr = output.last_hidden_state[0, :, :].detach().cpu().numpy()
        
        # Now to isolate our keyword embedding we need to find the index of the keyword tokens-
        indx = self.get_keyword_idx(tokenized_input, keyword)

        # TODO: EDGE CASE - Some keywords have more than 1 occurances in a LLM response, figure out what to do in that case.
        final_embedding = []
        if (len(indx) == 0):
            final_embedding = np.zeros_like(output_arr[0])

        for token_occurance in indx:
            index, keyword_token_len = token_occurance
            token_emb = np.zeros_like(output_arr[0])
            for i, k in enumerate(range(index, index+keyword_token_len)):
                token_emb +=output_arr[k]
            token_emb = (token_emb)/i+1
            final_embedding.append(token_emb)
            
        # if (len(final_embedding) == 1):
        return final_embedding[0]
        # return embeddings.detach().cpu().numpy()
    
    def load_json_tree(self, PATH_tree:str):
        
        with open (PATH_tree, 'r') as f:
            json_tree = json.load(f)
        return json_tree
    
    def process_tree(self, tree_dict:dict):

        embedding_coeff = {}    # the format for data key is - [node's depth, node's num of occurances, the response from LLM]
        def process_nodes(tree_dict:dict):
            for keyword, node_dict in tree_dict.items():
                
                embdg = self.get_embeddings(node_dict["response"], keyword)
                if keyword in list(embedding_coeff.keys()):
                    embedding_coeff[keyword]["data"][1] += 1
                else:
                    embedding_coeff[keyword] = {"data" : [node_dict['depth'] + 1, 1, node_dict['response']]}
                    embedding_coeff[keyword]["raw_enc"] = embdg.tolist()
                
                # Using formula for weight = [(1 * # of occurances) / (depth of the node)] * embedding:
                weight_coeff = (1 * embedding_coeff[keyword]["data"][1])/(embedding_coeff[keyword]["data"][0])
                embedding_coeff[keyword]["w88_enc"] = (weight_coeff * np.array(embedding_coeff[keyword]["raw_enc"])).tolist()

                # Now reccurse-
                if (len(node_dict['children']) != 0):
                    process_nodes(node_dict["children"])
        
        process_nodes(tree_dict=tree_dict)
        return embedding_coeff
    

 
if __name__ == '__main__':
    
    domain = "biomedical"
    topic = "endometriosis_[40]"
    PATH_self_dir = os.path.dirname(os.path.realpath(__file__))
    PATH_current_topic = os.path.join(PATH_self_dir, "output_trees", domain, topic)

    OBJ_EmbTree = EmbdTree()
    for subtree_dir in tqdm(os.listdir(PATH_current_topic)):

        PATH_tree = os.path.join(PATH_current_topic, subtree_dir, 'tree.json')
        PATH_out_embd_tree = os.path.join(PATH_current_topic, subtree_dir, 'embdng_tree.json')
        json_tree = OBJ_EmbTree.load_json_tree(PATH_tree)
        embedding_coeff = OBJ_EmbTree.process_tree(json_tree)

        with open(PATH_out_embd_tree, 'w') as f:
            json.dump(embedding_coeff, f)
