import sys
import os
sys.path.append('./MatSciBERT')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__)))
import torch
from normalize_text import normalize
from models import BERT_CRF
from transformers import AutoTokenizer, AutoConfig

class NER_INF:
    def __init__(self) -> None:
        self.PATH_self_dir = os.path.dirname(os.path.realpath(__file__))
        self.PATH_root_dir = os.path.join(self.PATH_self_dir, '../')
        self.PATH_cache_dir = os.path.join(self.PATH_root_dir, '.cache')
        self.PATH_model_dir = os.path.join(self.PATH_self_dir, 'models/matscholar')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def initialize_infer(self):
        """Intialize a model and return its object

        Returns:
            BERT_CRF: bert_ner model object
        """
        # Load the tokenizer:
        tokenizer_kwargs = {
            'cache_dir': self.PATH_cache_dir,
            'use_fast': True,
            'revision': 'main',
            'use_auth_token': None,
            'model_max_length': 512
        }
        # Tokenize inputs:
        self.tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert', **tokenizer_kwargs)

        self.label_list = ['B-APL', 'B-CMT', 'B-DSC', 'B-MAT', 'B-PRO', 'B-SMT', 'B-SPL', 'I-APL', 'I-CMT', 'I-DSC', 'I-MAT', 'I-PRO', 'I-SMT', 'I-SPL', 'O']
        id2label = {i: label for i, label in enumerate(self.label_list)}
        label2id = {label: i for i, label in enumerate(self.label_list)}
        config_kwargs = {
            'num_labels': len(self.label_list),
            'cache_dir': self.PATH_cache_dir,
            'revision': 'main',
            'use_auth_token': None,
        }

        # Model Config and Intialize_model:
        config = AutoConfig.from_pretrained('m3rg-iitd/matscibert', **config_kwargs)
        config.id2label = id2label
        config.label2id = label2id

        model = BERT_CRF('m3rg-iitd/matscibert', self.device, config, self.PATH_cache_dir)
        model = model.to(self.device)
        model.load_state_dict(torch.load(os.path.join(self.PATH_model_dir, 'pytorch_model.bin'), map_location='cpu'), strict=False)

        return model

    def infer_caption(self, sentence_caption:str, model:BERT_CRF, training_mode:bool=False):
        """Perform inference on a caption

        Args:
            sentence_caption (str): string to infer on
            model (BERT_CRF): loaded BERT_CRF model object

        Returns:
            dict: dictionary containing keys as tokens and values as lables of those token classification
        """
        sentence_caption = normalize(sentence_caption)
        inputs  = self.tokenizer(sentence_caption, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        predicted_labels = [model.encoder.config.id2label[p] for p in outputs[0]]
        results = {}
        word, label = "", None

        for token, label_id in zip(tokens, predicted_labels):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            if training_mode:
                results[token] = label_id

            else:
                if token.startswith("##"):
                    word += token[2:]
                else:
                    if word:
                        results[word] = label
                    word = token
                    label = label_id

        if word:
            results[word] = label

        return results

if __name__ == "__main__":
    test_sentence = "Roughening at low frequency was also attributed to platinum electrodissolution/electrodeposition during cycling[60]."
    OBJ_ner_inf = NER_INF()
    model = OBJ_ner_inf.initialize_infer()
    print(OBJ_ner_inf.infer_caption(test_sentence, model))
