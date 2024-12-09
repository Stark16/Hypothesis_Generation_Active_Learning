import os
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
training_device = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = training_device
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
import sys
from datetime import datetime, timedelta
import random
import re
sys.path.append('./MatSciBERT')
import test_mlm
ner_path = os.path.join(os.path.dirname(__file__), '../ner')
sys.path.append(ner_path)
import NER_inference


from argparse import ArgumentParser

from transformers import (
    AutoConfig,
    BertForMaskedLM,
    AutoTokenizer,
    DataCollatorForWholeWordMask,
    Trainer,
    TrainingArguments,
    set_seed,
)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from transformers import TrainerCallback

class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if 'loss' in logs:
            trainer.state.log_history.append({'step': state.global_step, 'loss': logs['loss']})

torch.cuda.set_device(0)
torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
print('using device:', device)


def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path

HYPERPARAMETERS = {
    "per_device_train_batch_size" : 64,
    "gradient_accumulation_steps" : 32,
    "learning_rate" : 1e-4,
    "num_train_epochs" : 150,
    "logging_steps" : 8,
    'mlm_probability' : 0.15,
    "training_device" : training_device
}

parser = ArgumentParser()
parser.add_argument('--train_file', default=r"/home/ppathak/Hypothesis_Generation_Active_Learning/datasets/semantic_kg/json_dataset/1970/train_norm.txt", type=str)
parser.add_argument('--val_file', default=r"/home/ppathak/Hypothesis_Generation_Active_Learning/datasets/semantic_kg/json_dataset/1970/val_norm.txt", type=str)
parser.add_argument('--model_save_dir', default=r"/home/ppathak/Hypothesis_Generation_Active_Learning/MatSciBERT/trained_model/all_mask_non_tech", type=str)
parser.add_argument('--cache_dir', default=None, type=str)
parser.add_argument('--tech', default=False, type=bool)
args = parser.parse_args()

model_revision = 'main'
# model_name = 'allenai/scibert_scivocab_uncased'
model_name = "bert-base-uncased"

output_dir = ensure_dir(args.model_save_dir + '_{}_{}_{}ge'.format(os.path.basename(os.path.dirname(args.train_file)), 
                                                            HYPERPARAMETERS['num_train_epochs'], 
                                                            HYPERPARAMETERS['gradient_accumulation_steps']))

cache_dir = ensure_dir(args.cache_dir) if args.cache_dir else ensure_dir(output_dir + '/cache')
logging_dir = ensure_dir(output_dir + '/logs')
output_dir = ensure_dir(output_dir + r'/checkpoints')
technical_words_only = args.tech

assert os.path.exists(args.train_file)
assert os.path.exists(args.val_file)

SEED = 42
set_seed(SEED)

config_kwargs = {
    'cache_dir': cache_dir,
    'revision': model_revision,
    'use_auth_token': None,
}
config = AutoConfig.from_pretrained(model_name, **config_kwargs)

tokenizer_kwargs = {
    'cache_dir': cache_dir,
    'use_fast': True,
    'revision': model_revision,
    'use_auth_token': None,
}
tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

max_seq_length = 512

start_tok = tokenizer.convert_tokens_to_ids('[CLS]')
sep_tok = tokenizer.convert_tokens_to_ids('[SEP]')
pad_tok = tokenizer.convert_tokens_to_ids('[PAD]')

def mask_entities(sentence, entity_labels, target_entity_class=None, mask_technical=False):
    masked_sentences = {}
    lower_sentence = sentence.lower()

    non_technical_tokens = []

    for token, label in entity_labels.items():
        lower_token = token.lower()

        if (lower_token in stop_words or len(lower_token) == 1 or re.match(r"^\d+$", lower_token) or lower_token not in lower_sentence):
            continue  # Skip this token if it's a stop word, single character, or number

        if target_entity_class:         # For masking any needed label
            if label == target_entity_class:
                masked_sentence = lower_sentence.replace(lower_token, '[MASK]', 1)
                masked_sentences[masked_sentence] = token
        else:
            if label.startswith('I-') and mask_technical:     # Mask techincal words (currently only masks entities that have the 'inside' (I-) label)
                masked_sentence = lower_sentence.replace(lower_token, '[MASK]', 1)
                masked_sentences[masked_sentence] = token
            elif label == 'O' and not mask_technical:
                if re.match(r"^[a-zA-Z0-9]+$", lower_token):  
                    non_technical_tokens.append(lower_token)

    if not mask_technical and non_technical_tokens:
        sample_tokens = random.sample(non_technical_tokens, min(len(non_technical_tokens), 3))
        for token in sample_tokens:
            masked_sentence = lower_sentence.replace(token, '[MASK]', 1)
            masked_sentences[masked_sentence] = token

    return masked_sentences

def full_sent_tokenize(file_name, OBJ_ner:NER_inference, model_ner:NER_inference.NER_INF, mask_technical=False):
    f = open(file_name, 'r')
    sents = f.read().strip().split('\n')
    f.close()

    tok_sents = []
    masked_sentences = []

    for s in tqdm(sents):
        entity_labels = OBJ_ner.infer_caption(s, model_ner)
        masked_sents = mask_entities(s, entity_labels, mask_technical=mask_technical)

        for masked_sentence in masked_sents.keys():
            tokenized_sent = tokenizer(masked_sentence, return_attention_mask=True, truncation=True, padding=True)['input_ids']
            tok_sents.append(tokenized_sent)
    
    # tok_sents = [tokenizer(s, return_attention_mask=True, truncation=True, padding=True)['input_ids'] for s in tqdm(sents)]
    for s in tok_sents:
        s.pop(0)
    
    res = [[]]
    l_curr = 0
    
    for s in tok_sents:
        l_s = len(s)
        idx = 0
        while idx < l_s - 1:
            if l_curr == 0:
                res[-1].append(start_tok)
                l_curr = 1
            s_end = min(l_s, idx + max_seq_length - l_curr) - 1
            res[-1].extend(s[idx:s_end] + [sep_tok])
            idx = s_end
            if len(res[-1]) == max_seq_length:
                res.append([])
            l_curr = len(res[-1])
    
    for s in res[:-1]:
        assert s[0] == start_tok and s[-1] == sep_tok
        assert len(s) == max_seq_length
        
    attention_mask = []
    for s in res:
        attention_mask.append([1] * len(s) + [0] * (max_seq_length - len(s)))
    
    return {'input_ids': res, 'attention_mask': attention_mask}

def get_min_and_last(values, label):
    min_val = min(values)
    min_epoch = np.argmin(values) + 1
    last_val = values[-1]
    last_epoch = len(values)
    print(f"{label} -> Min: {min_val} at epoch {min_epoch}, Last: {last_val} at epoch {last_epoch}")
    return min_val, min_epoch, last_val, last_epoch

class MSC_Dataset(torch.utils.data.Dataset):
    def __init__(self, inp):
        self.inp = inp

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.inp.items()}
        return item

    def __len__(self):
        return len(self.inp['input_ids'])
    
OBJ_ner = NER_inference.NER_INF()
model_ner = OBJ_ner.initialize_infer()

training_config = {"train_dir" : args.train_file.replace('\\', '/'),
                   "val_dir" : args.val_file.replace('\\', '/'),
                   "mask_only_technical_words" : technical_words_only,
                   "output_directories" : {"Model" : output_dir.replace('\\', '/'),
                                          "Cache" : cache_dir.replace('\\', '/'),
                                          "Logs" : logging_dir.replace('\\', '/')},
                    "training_resumed" : None,
                    "parallel_preprocessing" : None,
                    "training example_count" : 0,
                    "validation example_count" : 0,
                    "timeline" : {}}
                    
# Save the training config for future reference
with open(logging_dir + '/training_config.json', 'w') as f:
    json.dump(training_config, f)

dataset_loading_start_time = time.time()
train_dataset = MSC_Dataset(full_sent_tokenize(args.train_file, OBJ_ner, model_ner, mask_technical=technical_words_only))
eval_dataset = MSC_Dataset(full_sent_tokenize(args.val_file, OBJ_ner, model_ner, mask_technical=technical_words_only))

t_val = datetime(1,1,1) + timedelta(seconds=int(time.time() - dataset_loading_start_time))
training_config['timeline']['Dataset Loading'] = f"%d Days : %d Hours : %d Mins : %d Seconds" % (t_val.day-1, t_val.hour, t_val.minute, t_val.second)
training_config['training example_count'] = len(train_dataset)
training_config['validation example_count'] = len(eval_dataset)
with open(logging_dir + '/training_config.json', 'w') as f:
    json.dump(training_config, f)

print(len(train_dataset), len(eval_dataset))


model = BertForMaskedLM.from_pretrained(
    model_name,
    from_tf=False,
    config=config,
    cache_dir=cache_dir,
    revision=model_revision,
    use_auth_token=None,
)

NUM_FORZEN_LAYERS = 10
for i in range(NUM_FORZEN_LAYERS):
    for param in model.bert.encoder.layer[i].parameters():
        param.requires_grad = False

model.resize_token_embeddings(len(tokenizer))
model.to(torch.device(device))
# model = torch.nn.DataParallel(model)

data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=HYPERPARAMETERS['mlm_probability'])

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=HYPERPARAMETERS['per_device_train_batch_size'],
    per_device_eval_batch_size=HYPERPARAMETERS['per_device_train_batch_size'],
    gradient_accumulation_steps=HYPERPARAMETERS['gradient_accumulation_steps'],
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=4,
    load_best_model_at_end=True,
    warmup_ratio=0.048,
    learning_rate=HYPERPARAMETERS['learning_rate'],
    weight_decay=1e-2,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    max_grad_norm=0.0,
    num_train_epochs=HYPERPARAMETERS['num_train_epochs'],
    seed=SEED,
    # logging_strategy='no'
    logging_first_step=True,
    logging_strategy='steps',
    logging_steps=HYPERPARAMETERS['logging_steps']
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[CustomLoggingCallback()]
)

torch.cuda.empty_cache()
print("Device Used: ", trainer.args.device)
print(torch.cuda.current_device())
resume = None if len(os.listdir(output_dir)) == 0 else True
training_config['training_resumed'] = resume
with open(logging_dir + '/training_config.json', 'w') as f:
    json.dump(training_config, f)

train_start = time.time()
train_res = trainer.train(resume_from_checkpoint=resume)
print(train_res)
t_val = datetime(1,1,1) + timedelta(seconds=int(time.time() - train_start))
training_config['timeline']['Training'] = f"%d Days : %d Hours : %d Mins : %d Seconds" % (t_val.day-1, t_val.hour, t_val.minute, t_val.second)
with open(logging_dir + '/training_config.json', 'w') as f:
    json.dump(training_config, f)

torch.cuda.empty_cache()

eval_start_time = time.time()
train_output = trainer.evaluate(train_dataset)
eval_output = trainer.evaluate()
t_val = datetime(1,1,1) + timedelta(seconds=int(time.time() - eval_start_time))
training_config['timeline']['Evaluation'] = f"%d Days : %d Hours : %d Mins : %d Seconds" % (t_val.day-1, t_val.hour, t_val.minute, t_val.second)
with open(logging_dir + '/training_config.json', 'w') as f:
    json.dump(training_config, f)



# Evaluating:
log_history = trainer.state.log_history

train_loss = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
train_step = [log['step'] for log in trainer.state.log_history if 'loss' in log]
eval_loss = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
eval_steps = [log['step'] for log in trainer.state.log_history if 'eval_loss' in log]
epochs = range(1, training_args.num_train_epochs + 1)

train_loss = np.array(train_loss)
eval_loss = np.array(eval_loss)
train_perplexity = np.exp(train_loss)
eval_perplexity = np.exp(eval_loss)

steps = range(8, len(train_loss) * 8 + 1, 8)
steps_per_epoch = 17
epochs = [step // steps_per_epoch + 1 for step in steps]

train_loss_min, train_loss_min_epoch, train_loss_last, train_loss_last_epoch = get_min_and_last(train_loss, "Training Loss")
training_config['Training Loss'] = f"Training Loss -> Min: {train_loss_min} at epoch {train_loss_min_epoch}, Last: {train_loss_last} at epoch {train_loss_last_epoch}"

eval_loss_min, eval_loss_min_epoch, eval_loss_last, eval_loss_last_epoch = get_min_and_last(eval_loss, "Evaluation Loss")
training_config['Evaluation Loss'] = f"Evaluation Loss -> Min: {eval_loss_min} at epoch {eval_loss_min_epoch}, Last: {eval_loss_last} at epoch {eval_loss_last_epoch}"

train_perplexity_min, train_perplexity_min_epoch, train_perplexity_last, train_perplexity_last_epoch = get_min_and_last(train_perplexity, "Training Perplexity")
training_config['Training Perplexity'] = f"Training Perplexity -> Min: {train_perplexity_min} at epoch {train_perplexity_min_epoch}, Last: {train_perplexity_last} at epoch {train_perplexity_last_epoch}"

eval_perplexity_min, eval_perplexity_min_epoch, eval_perplexity_last, eval_perplexity_last_epoch = get_min_and_last(eval_perplexity, "Evaluation Perplexity")
training_config['Evaluation Perplexity'] = f"Evaluation Perplexity -> Min: {eval_perplexity_min} at epoch {eval_perplexity_min_epoch}, Last: {eval_perplexity_last} at epoch {eval_perplexity_last_epoch}"
with open(logging_dir + '/training_config.json', 'w') as f:
    json.dump(training_config, f)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(train_step, train_loss, label="Training Loss", marker='o', color='blue')
ax1.plot(eval_steps, eval_loss, label="Evaluation Loss", marker='x', color='orange')

ax1.scatter(train_loss_min_epoch, train_loss_min, color='blue', s=100)
ax1.text(train_loss_min_epoch, train_loss_min, f"Min: {train_loss_min:.2f}", color='blue', ha='right')
ax1.scatter(train_loss_last_epoch, train_loss_last, color='blue', s=100)
ax1.text(train_loss_last_epoch, train_loss_last, f"Last: {train_loss_last:.2f}", color='blue', ha='right')

ax1.scatter(eval_loss_min_epoch, eval_loss_min, color='orange', s=100)
ax1.text(eval_loss_min_epoch, eval_loss_min, f"Min: {eval_loss_min:.2f}", color='orange', ha='right')
ax1.scatter(eval_loss_last_epoch, eval_loss_last, color='orange', s=100)
ax1.text(eval_loss_last_epoch, eval_loss_last, f"Last: {eval_loss_last:.2f}", color='orange', ha='right')

ax1.set_xlabel("Steps")
ax1.set_ylabel("Loss")
ax1.set_title("Training and Evaluation Loss per Step")
ax1.legend()

ax2.plot(train_step, train_perplexity, label="Training Perplexity", marker='o', color='blue')
ax2.plot(eval_steps, eval_perplexity, label="Evaluation Perplexity", marker='x', color='orange')

ax2.scatter(train_perplexity_min_epoch, train_perplexity_min, color='blue', s=100)
ax2.text(train_perplexity_min_epoch, train_perplexity_min, f"Min: {train_perplexity_min:.2f}", color='blue', ha='right')
ax2.scatter(train_perplexity_last_epoch, train_perplexity_last, color='blue', s=100)
ax2.text(train_perplexity_last_epoch, train_perplexity_last, f"Last: {train_perplexity_last:.2f}", color='blue', ha='right')

ax2.scatter(eval_perplexity_min_epoch, eval_perplexity_min, color='orange', s=100)
ax2.text(eval_perplexity_min_epoch, eval_perplexity_min, f"Min: {eval_perplexity_min:.2f}", color='orange', ha='right')
ax2.scatter(eval_perplexity_last_epoch, eval_perplexity_last, color='orange', s=100)
ax2.text(eval_perplexity_last_epoch, eval_perplexity_last, f"Last: {eval_perplexity_last:.2f}", color='orange', ha='right')

ax2.set_xlabel("Steps")
ax2.set_ylabel("Perplexity")
ax2.set_title("Training and Evaluation Perplexity per Step")
ax2.legend()

plt.tight_layout()
plt.show()
plt.savefig(logging_dir + '/training_plots.png', format='png', dpi=300)

print(train_output)
print(eval_output)

torch.cuda.empty_cache()

testing_start_time = time.time()

print("[INFO] Training finished. Performing Checkpoint Evaluations...")
OBJ_MlmTest = test_mlm.TestingMLM()
for model_checkpoint in tqdm(os.listdir(output_dir)):
    print('================================ [INFO] Testing Checkpoint - ', model_checkpoint, '[INFO] ================================')
    PATH_model_checkpoint = os.path.join(output_dir, model_checkpoint)
    PATH_output = os.path.join(logging_dir + '_' + model_checkpoint, 'simple')

    OBJ_MlmTest.test(PATH_trained_bert_mlm=PATH_model_checkpoint, pth_txt=args.val_file, PATH_output=PATH_output, t_size=500, use_pipeline=True, ignore_stop_words=True)

# Save the training config for future reference
t_val = datetime(1,1,1) + timedelta(seconds=int(time.time() - testing_start_time))
training_config['timeline']['Custom Testing'] = f"%d Days : %d Hours : %d Mins : %d Seconds" % (t_val.day-1, t_val.hour, t_val.minute, t_val.second)
with open(logging_dir + '/training_config.json', 'w') as f:
    json.dump(training_config, f)