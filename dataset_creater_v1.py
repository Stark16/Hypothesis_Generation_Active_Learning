import os
import sys
import torch
from transformers import AutoModelForMaskedLM
from matscibert_inf.ner.NER_inference import NER_INF

class FeatureSpace:

    def __init__(self, selected_model):
        self.PATH_self_dir = os.path.dirname(os.path.realpath(__file__))
        self.PATH_all_model = os.path.join(self.PATH_self_dir, 'matscibert_inf/mlm_models', selected_model)

        # Load Model:
        self.model = AutoModelForMaskedLM.from_pretrained(self.PATH_all_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_dataset(self, file_path):
        with open(file_path, 'r') as f:
            sentences = [line.strip() for line in f.readlines() if line.strip()]
        return sentences

    def process_sentences(self, sentence):
        feature_dict = {}
        # for sentence in sentences:
        ner_results = ner_inf.infer_caption(sentence, ner_model)
        masked_sentence = " ".join([token if label == "O" else "[MASK]" for token, label in ner_results.items()])

        tokenizer = ner_inf.tokenizer
        inputs = tokenizer(masked_sentence, return_tensors="pt", truncation=True, padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        mask_token_id = tokenizer.mask_token_id
        mask_positions = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]

        logits = outputs.logits
        predicted_token_ids = [logits[0, pos].argmax(dim=-1).item() for pos in mask_positions]

        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids)

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
            predicted_tokens
        ]

        return feature_dict


if __name__ == "__main__":
    ner_inf = NER_INF()
    ner_model = ner_inf.initialize_infer()

    QUERY_TEMPLATE = "The material [MASK] is [MASK] semiconductor"

    selected_model = "trc_60_2005/checkpoint-2880"
    OBJ_FeatureSpace = FeatureSpace(selected_model)

    for _ in range(10):
        feature_space = OBJ_FeatureSpace.process_sentences(QUERY_TEMPLATE)