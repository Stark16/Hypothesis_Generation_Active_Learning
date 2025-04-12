import context_tree_v2
import os
import json

from transformers import AutoTokenizer, AutoModel

class EmbdTree:
    def __init__(self, model_control:str='allenai/scibert_scivocab_cased'):
        self.PATH_self_dir = os.path.dirname(os.path.realpath(__file__))
        self.MODEL_control = AutoModel.from_pretrained(model_control)
        self.TOKENIZER_control = AutoTokenizer.from_pretrained(model_control)

    def get_embeddings(self, input:str):
        tokenized_input = self.TOKENIZER_control([input], return_tensors='pt', padding=True, truncation=True)
        output = self.MODEL_control(**tokenized_input)
        embeddings = output.last_hidden_state[:, 0, :]
        return embeddings.detach().cpu().numpy()
    
    def process_tree(self, tree_dict:dict):

        embedding_coeff = {}
        def process_nodes(tree_dict:dict):
            for keyword, node_dict in tree_dict.items():
                if (len(node_dict['children']) == 0):
                    return
                
                embdg = self.get_embeddings(node_dict["response"])
                if keyword in list(embedding_coeff.keys()):
                    embedding_coeff[keyword]["data"][2] += 1
                else:
                    embedding_coeff[keyword]["data"] = [node_dict['depth'],  node_dict['response'], 0]
                    embedding_coeff[keyword]["raw_enc"] = embdg
                
                # Using formula for weight = [1 / (# of occurances * depth of the node)] * embedding:
                weight_coeff = 1/embedding_coeff["keyword"]["data"][2] * embedding_coeff["keyword"]["data"][1]
                embedding_coeff[keyword]["w88_enc"] = weight_coeff * embedding_coeff[keyword]["raw_enc"]

                # Now reccurse-
                process_nodes(node_dict["children"])
        
        process_nodes(tree_dict=tree_dict)
        return embedding_coeff


if __name__ == '__main__':
    OBJ_EmbTree = EmbdTree()
    OBJ_EmbTree.get_embeddings('phase diagram')