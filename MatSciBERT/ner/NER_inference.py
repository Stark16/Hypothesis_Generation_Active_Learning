import sys
sys.path.append('./MatSciBERT')
import ner_datasets
from chemdataextractor.doc import Paragraph
import torch
from normalize_text import normalize
from models import BERT_CRF
import os
from transformers import AutoTokenizer, AutoConfig

root_dir = r'/home/ppathak/Hypothesis_Generation_Active_Learning/MatSciBERT'
cache_dir = os.path.join(root_dir, '.cache')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

scientific_caption = "Cathodic waves can be treated in terms of the equation developed by Berzins and Delahay,18"

# Trying to load the lables from the model config:
config_kwargs = {
    'num_labels': 15,
    'cache_dir': cache_dir,
    'revision': 'main',
    'use_auth_token': None,
}
label_list = ['B-APL', 'B-CMT', 'B-DSC', 'B-MAT', 'B-PRO', 'B-SMT', 'B-SPL', 'I-APL', 'I-CMT', 'I-DSC', 'I-MAT', 'I-PRO', 'I-SMT', 'I-SPL', 'O']
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}
config = AutoConfig.from_pretrained('m3rg-iitd/matscibert', **config_kwargs)
config.id2label = id2label
config.label2id = label2id
model = BERT_CRF('m3rg-iitd/matscibert', device, config, cache_dir)
ner_model_path = '/home/ppathak/Hypothesis_Generation_Active_Learning/MatSciBERT/ner/models/matscholar'
model.load_state_dict(torch.load(os.path.join(ner_model_path, 'pytorch_model.bin'), map_location='cpu'), strict=False)
model = model.to(device)
# Load the tokenizer:
tokenizer_kwargs = {
    'cache_dir': cache_dir,
    'use_fast': True,
    'revision': 'main',
    'use_auth_token': None,
    'model_max_length': 512
}
# Tokenize inputs:
tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert', **tokenizer_kwargs)

inputs  = tokenizer(scientific_caption, return_tensors="pt", truncation=True, padding=True)
inputs = {key: value.to(device) for key, value in inputs.items()}



with torch.no_grad():
    outputs = model(**inputs)

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
predicted_labels = [model.encoder.config.id2label[p] for p in outputs[0]]
for token, label in zip(tokens, predicted_labels):
    print(f"{token}: {label}")
