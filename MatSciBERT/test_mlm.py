import random
import torch
import matplotlib.pyplot as plt
import argparse
import os
from transformers import BertForMaskedLM, BertTokenizer, AutoConfig, AutoTokenizer
from colorama import Fore, Back, Style
from tqdm import tqdm
from ner.NER_inference import NER_INF
import re
import pandas as pd

class TestingMLM:

    def __init__(self) -> None:
        self.PATH_self_dir = os.path.dirname(os.path.realpath(__file__))
        self.PATH_bert_cache = os.path.join(self.PATH_self_dir, 'bert_temp_cache')
        self.PATH_results_dir = os.path.join(self.PATH_self_dir, 'mlm_tests/2005_model')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.OBJ_ner_inf = NER_INF()
        self.MODEL_ner = self.OBJ_ner_inf.initialize_infer()
        self.ENTITYCLS = self.OBJ_ner_inf.label_list
        self.NUM_NONTECHPERSENT = 4

    # def load_bert(self, model_name):
    #     model_revision = 'main'

    #     tokenizer_kwargs = {
    #         'cache_dir': self.PATH_bert_cache,
    #         'use_fast': True,
    #         'revision': model_revision,
    #         'use_auth_token': None,
    #     }
    #     bert_tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    #     bert_config_kwargs = {
    #         'cache_dir': self.PATH_bert_cache,
    #         'revision': model_revision,
    #         'use_auth_token': None,
    #     }
    #     bert_config = AutoConfig.from_pretrained(model_name, **bert_config_kwargs)
    #     model = BertForMaskedLM.from_pretrained(model_name, from_tf=False, config=bert_config, cache_dir=self.PATH_bert_cache, 
    #                                             revision=model_revision, use_auth_token=None,)
    #     model.resize_token_embeddings(len(bert_tokenizer))
    #     model.to(torch.device(self.device))  

    def load_bert(self, m_dir):
        tokenizer = BertTokenizer.from_pretrained(m_dir)
        model = BertForMaskedLM.from_pretrained(m_dir).to(self.device)
        return model, tokenizer

    def get_random_sentences(self, file_path, n):
        with open(file_path, 'r') as file:
            sentences = file.readlines()
        return random.sample(sentences, n)

    def mask_entities(self, sentence, entity_labels, target_entity_class=None, mask_non_technical=False):
        masked_sentences = {}
        lower_sentence = sentence.lower()

        # Gather non-technical tokens to limit them later if needed
        non_technical_tokens = []

        for token, label in entity_labels.items():
            lower_token = token.lower()
            if target_entity_class:
                if label == target_entity_class and lower_token in lower_sentence:
                    masked_sentence = lower_sentence.replace(lower_token, '[MASK]', 1)
                    masked_sentences[masked_sentence] = token
            else:
                if label.startswith('I-') and not mask_non_technical and lower_token in lower_sentence:
                    masked_sentence = lower_sentence.replace(lower_token, '[MASK]', 1)
                    masked_sentences[masked_sentence] = token
                elif label == 'O' and mask_non_technical and lower_token in lower_sentence:
                    # Only add non-technical tokens that are words (remove punctuation and special characters)
                    if re.match(r"^[a-zA-Z0-9]+$", lower_token):  
                        non_technical_tokens.append(lower_token)

        # Limit non-technical tokens to a reasonable subset, e.g., 3 random selections
        if mask_non_technical and non_technical_tokens:
            sample_tokens = random.sample(non_technical_tokens, min(len(non_technical_tokens), 3))
            for token in sample_tokens:
                masked_sentence = lower_sentence.replace(token, '[MASK]', 1)
                masked_sentences[masked_sentence] = token

        return masked_sentences

    def predict_masked_token(self, masked_sentence, model, tokenizer):
        inputs = tokenizer(masked_sentence, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
        mask_logits = logits[0, mask_token_index, :]
        predicted_token_id = torch.argmax(mask_logits, dim=-1)
        predicted_token = tokenizer.decode(predicted_token_id)
        return predicted_token

    def run_inference_on_sentences(self, sentences, model, tokenizer, mask_non_technical=False):
        correct_predictions = 0
        total_predictions = 0
        results = []

        for sentence in sentences:
            sentence = sentence.strip()
            entity_labels = self.OBJ_ner_inf.infer_caption(sentence, self.MODEL_ner)
            
            masked_sentences = self.mask_entities(sentence, entity_labels, mask_non_technical=mask_non_technical)
            for masked_sentence, original_word in masked_sentences.items():
                predicted_token = self.predict_masked_token(masked_sentence, model, tokenizer)

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
                
                results.append([sentence, masked_sentence.replace('[MASK]', predicted_token), original_word, predicted_token, 
                                predicted_token.strip() == original_word.strip(), not mask_non_technical, entity_labels[original_word]])

        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions} correctly predicted)")
        return results
    
    def plot_results(self, DF_results, test_configs):
        plt.figure(figsize=(12, 8))

        correct_counts = []
        wrong_counts = []
        accuracies = []

        for model_name in [config[3] for config in test_configs]:
            model_results = DF_results[DF_results['ModelName'] == model_name]
            correct_count = model_results[model_results['IsCorrect'] == True].shape[0]
            wrong_count = model_results[model_results['IsCorrect'] == False].shape[0]
            
            correct_counts.append(correct_count)
            wrong_counts.append(wrong_count)
            
            accuracy = (correct_count / (correct_count + wrong_count)) * 100 if (correct_count + wrong_count) > 0 else 0
            accuracies.append(accuracy)

        bar_width = 0.6
        labels = ['Base BERT - Tech', 'Base BERT - Non-Tech', 'Trained BERT - Tech', 'Trained BERT - Non-Tech']
        bar_positions = range(len(labels))

        plt.bar(bar_positions, correct_counts, color='green', label='Correct Predictions', width=bar_width)
        plt.bar(bar_positions, wrong_counts, bottom=correct_counts, color='red', label='Wrong Predictions', width=bar_width)

        for i, (correct, wrong, accuracy) in enumerate(zip(correct_counts, wrong_counts, accuracies)):
            plt.text(i, correct / 2, f'{correct}', ha='center', va='center', color='white', fontsize=10)
            plt.text(i, correct + wrong / 2, f'{wrong}', ha='center', va='center', color='white', fontsize=10)
            plt.text(i, correct + wrong + 1, f'Acc: {accuracy:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.xticks(bar_positions, labels)
        plt.ylabel('Count of Predictions')
        plt.title('Correct vs Incorrect Predictions by Model and Masking Mode')
        plt.legend()

        plot_path = os.path.join(self.PATH_results_dir, '500ex_2005_tr_2005_80_32ge_prediction_accuracy_comparison.png')
        plt.savefig(plot_path)
        plt.close()


    def test(self, trained_bert_mlm, pth_txt, t_size):
        if not os.path.exists(self.PATH_results_dir):
            os.makedirs(self.PATH_results_dir)
        df_columns=['UnMaksed', 'Predicted', 'GT_Word', 'P_Word', 'IsCorrect', 'IsTechEntity', 'EntityCls', 'ModelName']
        DF_results = pd.DataFrame(columns=df_columns)

        random_sentences = self.get_random_sentences(pth_txt, t_size)
        MODEL_bert, TOK_bert = self.load_bert('bert-base-uncased')
        MODEL_mlm_bert, TOK_mlm_bert = self.load_bert(trained_bert_mlm)

        test_configs = [(MODEL_bert, TOK_bert, False, 'base_bert_tech'), (MODEL_bert, TOK_bert, True, 'base_bert_non_tech'),
                        (MODEL_mlm_bert, TOK_mlm_bert, False, 'mlm_bert_tech'), (MODEL_mlm_bert, TOK_mlm_bert, True, 'mlm_bert_non_tech')]

        for model, tokenizer, mask_non_technical, model_name in tqdm(test_configs):
            results = self.run_inference_on_sentences(random_sentences, model, tokenizer, mask_non_technical)
            for result in results:
                result.append(model_name) 
                DF_results.loc[len(DF_results)] = result
        csv_path = os.path.join(self.PATH_results_dir, '500ex_2005_tr_2005_80_32ge_test_results.csv')
        DF_results.to_csv(csv_path, index=False, sep='\t')

        self.plot_results(DF_results, test_configs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on masked tokens in sentences.")
    parser.add_argument('--m_dir', type=str, default="/home/ppathak/Hypothesis_Generation_Active_Learning/MatSciBERT/trained_model/tr_2005_80_32ge/checkpoint-1699", 
                        help="Directory of the trained model.")
    parser.add_argument('--pth_txt', type=str, default="/home/ppathak/Hypothesis_Generation_Active_Learning/datasets/semantic_kg/json_dataset/2005/val_norm.txt", 
                        help="Path to the text file containing sentences.")
    parser.add_argument('--t_size', type=int, default=500, 
                        help="Number of sentences to sample from the text file.")
    parser.add_argument('--mask_non_technical', action='store_true', #default=True,
                        help="If set, masks non-technical words instead of technical entities.")
    
    args = parser.parse_args()
    PATH_model, PATH_val_txt, test_size, _ = args.m_dir, args.pth_txt, args.t_size, args.mask_non_technical
    OBJ_MlmTest = TestingMLM()
    OBJ_MlmTest.test(PATH_model, PATH_val_txt, test_size)
