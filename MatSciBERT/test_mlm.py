import random
import torch
import argparse
from transformers import BertForMaskedLM, BertTokenizer
from colorama import Fore, Back, Style
from ner.NER_inference import NER_INF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OBJ_ner_inf = NER_INF()
ner_model = OBJ_ner_inf.initialize_infer()

def initialize_model_and_tokenizer(m_dir):
    tokenizer = BertTokenizer.from_pretrained(m_dir)
    model = BertForMaskedLM.from_pretrained(m_dir).to(device)
    return model, tokenizer

def get_random_sentences(file_path, n):
    with open(file_path, 'r') as file:
        sentences = file.readlines()
    return random.sample(sentences, n)

def mask_entities(sentence, entity_labels, mask_non_technical=False):
    masked_sentences = {}
    for token, label in entity_labels.items():
        if (label.startswith('I-') and not mask_non_technical) or (label == 'O' and mask_non_technical):
            masked_sentence = sentence.replace(token, '[MASK]', 1)
            masked_sentences[masked_sentence] = token
    return masked_sentences

def predict_masked_token(masked_sentence, model, tokenizer):
    inputs = tokenizer(masked_sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
    mask_logits = logits[0, mask_token_index, :]
    predicted_token_id = torch.argmax(mask_logits, dim=-1)
    predicted_token = tokenizer.decode(predicted_token_id)
    return predicted_token

def run_inference_on_sentences(sentences, model, tokenizer, mask_non_technical=False):
    correct_predictions = 0
    total_predictions = 0

    for sentence in sentences:
        sentence = sentence.strip()
        entity_labels = OBJ_ner_inf.infer_caption(sentence, ner_model)
        
        masked_sentences = mask_entities(sentence, entity_labels, mask_non_technical)
        
        for masked_sentence, original_word in masked_sentences.items():
            predicted_token = predict_masked_token(masked_sentence, model, tokenizer)

            original_word_colored = Fore.RED + original_word + Fore.WHITE
            masked_sentence_colored = masked_sentence.replace('[MASK]', original_word_colored)

            if predicted_token.strip() == original_word.strip():
                correct_predictions += 1
                predicted_token_colored = Fore.GREEN + predicted_token + Fore.WHITE
            else:
                predicted_token_colored = Fore.MAGENTA + predicted_token + Fore.WHITE
            total_predictions += 1

            print(Back.BLACK + "\tOriginal Sentence with MASK: ", Back.WHITE + masked_sentence_colored, Back.BLACK)
            print("\tPredicted Sentence: ", masked_sentence.replace('[MASK]', predicted_token_colored), Back.BLACK, '\n\n')

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions} correctly predicted)")

def main(m_dir, pth_txt, t_size, mask_non_technical):
    model, tokenizer = initialize_model_and_tokenizer(m_dir)
    random_sentences = get_random_sentences(pth_txt, t_size)
    run_inference_on_sentences(random_sentences, model, tokenizer, mask_non_technical=mask_non_technical)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on masked tokens in sentences.")
    parser.add_argument('--m_dir', type=str, default="/home/ppathak/Hypothesis_Generation_Active_Learning/MatSciBERT/trained_model/testing60_16ge/checkpoint-53", 
                        help="Directory of the trained model.")
    parser.add_argument('--pth_txt', type=str, default="/home/ppathak/Hypothesis_Generation_Active_Learning/datasets/semantic_kg/json_dataset/1990/val_norm.txt", 
                        help="Path to the text file containing sentences.")
    parser.add_argument('--t_size', type=int, default=10, 
                        help="Number of sentences to sample from the text file.")
    parser.add_argument('--mask_non_technical', action='store_true', #default=True,
                        help="If set, masks non-technical words instead of technical entities.")
    
    args = parser.parse_args()
    main(args.m_dir, args.pth_txt, args.t_size, args.mask_non_technical)
