import os
import sys
import torch
from transformers import AutoModelForMaskedLM
from matscibert_inf.ner.NER_inference import NER_INF

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        sentences = [line.strip() for line in f.readlines() if line.strip()]
    return sentences

def process_sentences(sentence):
    feature_dict = {}
    # for sentence in sentences:
    ner_results = ner_inf.infer_caption(sentence, ner_model)
    masked_sentence = " ".join([token if label == "O" else "[MASK]" for token, label in ner_results.items()])

    tokenizer = ner_inf.tokenizer
    inputs = tokenizer(masked_sentence, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states

    last_hidden_state = hidden_states[-1].mean(dim=1).squeeze(0).cpu().numpy()
    second_last_hidden_state = hidden_states[-2].mean(dim=1).squeeze(0).cpu().numpy()
    avg_last_2_hidden_states = torch.stack(hidden_states[-2:]).mean(dim=0).mean(dim=1).squeeze(0).cpu().numpy()
    avg_last_4_hidden_states = torch.stack(hidden_states[-4:]).mean(dim=0).mean(dim=1).squeeze(0).cpu().numpy()

    feature_dict[sentence] = [
        last_hidden_state.tolist(),
        second_last_hidden_state.tolist(),
        avg_last_2_hidden_states.tolist(),
        avg_last_4_hidden_states.tolist(),
        outputs
    ]

    return feature_dict


if __name__ == "__main__":
    ner_inf = NER_INF()
    ner_model = ner_inf.initialize_infer()

    QUERY_TEMPLATE = "The material [MASK] is [MASK] semiconductor"

    trained_model_path = r"D:\College\Research\Prof_jamshid\Hypothesis_Generation_Active_Learning\matscibert\mlm_models\trc_60_2005\checkpoint-2880"
    model = AutoModelForMaskedLM.from_pretrained(trained_model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for _ in range(10):
        feature_space = process_sentences(QUERY_TEMPLATE)