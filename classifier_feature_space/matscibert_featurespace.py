import torch
import os
from transformers import BertTokenizer, BertForMaskedLM
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model_checkpoint_path = "/home/ppathak/Hypothesis_Generation_Active_Learning/MatSciBERT/trained_model/trc_2005_60_32ge/checkpoint-2880"

tokenizer = BertTokenizer.from_pretrained(model_checkpoint_path)
model = BertForMaskedLM.from_pretrained(model_checkpoint_path)

model.eval()

text = "Silicon-Dioxide has a melting point of [MASK]"

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

hidden_states = outputs.hidden_states  # This is a tuple of tensors

last_hidden_state = hidden_states[-1]  # Shape: (batch_size, sequence_length, hidden_size)

sentence_embedding = last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_size)

print("Sentence Embedding Shape:", sentence_embedding.shape)
