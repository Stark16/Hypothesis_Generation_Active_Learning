import random
from tqdm import tqdm
from argparse import ArgumentParser
from tokenizers.normalizers import BertNormalizer
import os


parser = ArgumentParser()
parser.add_argument('--train_file', required=True, type=str)
parser.add_argument('--val_file', required=True, type=str)
parser.add_argument('--op_train_file', required=False, type=str)
parser.add_argument('--op_val_file', required=False, type=str)
args = parser.parse_args()


f = open(r'./vocab_mappings.txt', 'r', encoding='utf-8')
mappings = f.read().strip().split('\n')
f.close()

mappings = {m[0]: m[2:] for m in mappings}

norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)

def normalize_and_save(file_path, save_file_path):
    f = open(file_path)
    corpus = f.read().strip().split('\n')
    f.close()
    
    random.seed(42)
    corpus = [norm.normalize_str(sent) for sent in tqdm(corpus)]
    corpus_norm = []
    for sent in tqdm(corpus):
        norm_sent = ""
        for c in sent:
            if c in mappings:
                norm_sent += mappings[c]
            elif random.uniform(0, 1) < 0.3:
                norm_sent += c
            else:
                norm_sent += ' '
        corpus_norm.append(norm_sent)
    
    f = open(save_file_path, 'w')
    f.write('\n'.join(corpus_norm))
    f.close()
output_train = os.path.dirname(args.train_file) + '/train_norm.txt'
output_val = os.path.dirname(args.train_file) + '/val_norm.txt'

normalize_and_save(args.train_file, output_train)
normalize_and_save(args.val_file, output_val)
