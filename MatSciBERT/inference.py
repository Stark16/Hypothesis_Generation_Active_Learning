import torch
from transformers import BertForMaskedLM, BertTokenizer
from colorama import Fore, Back, Style

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = r"D:\College\Research\Prof_jamshid\Hypothesis_Generation_Active_Learning\MatSciBERT\trained_models\checkpoint-500"


def predict_masked_token(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]

    mask_logits = logits[0, mask_token_index, :]

    predicted_token_id = torch.argmax(mask_logits, dim=-1)
    predicted_token = tokenizer.decode(predicted_token_id)

    return predicted_token

list_of_examples = {
    "The cathodic waves are irreversible." : "The cathodic [MASK] are irreversible.",
    "The wave vector dependence of the paramagnetic scattering at 300 K." : "The wave vector dependence of the paramagnetic [MASK] at 300 K.",
    "Highly puri&d DMSO was used as solvent." : "Highly [MASK] DMSO was used as solvent.",
    "For the solutions with added iodine, a platinum electrode dipped into the same solution worked as a reference electrode." : "For the solutions with added iodine, a [MASK] electrode dipped into the same solution worked as a reference electrode.",
    "Both electrodes were horizontally placed, facing each other 5 cm apart." : "Both [MASK] were horizontally placed, facing each other 5 cm apart.",
    "Between the electrodes a sintered glass disk divided the cell into a cathodic and an anodic section." : "Between the [MASK] a sintered glass disk divided the cell into a cathodic and an anodic section.",
    "Sodium perchlorate was employed as the supporting electrolyte for all the solutions" : "Sodium perchlorate was employed as the supporting [MASK] for all the solutions"

}

for example_sentence_real, example_sentence_maksed in list_of_examples.items():
    model = BertForMaskedLM.from_pretrained(model_dir).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    predicted_token = Fore.LIGHTBLUE_EX + predict_masked_token(example_sentence_maksed) + Fore.WHITE

    mask_idx = example_sentence_maksed.index('[MASK]')
    
    maksed_word = Fore.RED + example_sentence_real[mask_idx:].split(' ')[0] + Fore.WHITE
    example_sentence_real = example_sentence_maksed.replace('[MASK]', maksed_word)

    print(Back.BLACK +"\tS1: ", Back.WHITE + example_sentence_real, Back.BLACK)
    # print(f"Original sentence MASKED: ", Back.WHITE + example_sentence_maksed, Back.BLACK)
    print("\tS2: ", example_sentence_maksed.replace('[MASK]', predicted_token), Back.BLACK, '\n\n')
