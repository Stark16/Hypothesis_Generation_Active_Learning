# ⚡ Parallel Training Optimization for MatSciBERT ⚡  

## 🚀 Overview  
This branch focuses on optimizing the **MatSciBERT** training pipeline to **significantly reduce training time**. By restructuring how masked language modeling (MLM) is handled, we achieved up to a **5x speed boost**! 🎯  

## 🎯 The Problem  
In our original training setup, the **Named Entity Recognition (NER)** model was used to generate **masking candidates** dynamically **for every training batch**. This led to **unnecessary re-computation**, significantly slowing down the training process. The solution? **Move the NER inferencing from training time to preprocessing!**  

## 🔧 The Solution  
Here’s the step-by-step breakdown of what we did:  

✅ **Identified the bottleneck:** The initial pipeline was dynamically generating mask words for every batch, leading to redundant **NER inferencing** during training.  
✅ **Precomputed masking candidates:** We split the dataset into chunks and generated all **masking candidates in advance** before training.  
✅ **Used GPU parallelization:** Since PyTorch has some limitations in handling **multi-GPU parallel processing**, we had to work around a known bug to efficiently distribute the preprocessing workload.  
✅ **Modified the custom masking callback:** Instead of dynamically inferring masks during training, the pipeline now directly **retrieves precomputed masking candidates**, eliminating the need for repeated NER inferencing.  

## 🛠 **Training**  
----------------  

Here are some key details about the training process:  

- Prepare the dataset for training
  - ```
    cd ./MatSciBERT
    python pretraining/normalize_corpus.py --train_file <PATH to training.txt> --val_file <PATH to val.txt> --op_train_file <PATH to output train.txt> --op_val_file <PATH to output val.txt>
    ```
- Train the MLM model-
  - ```
    python pretraining/matscibert_train.py --train_file <PATH to the normlized train.txt> --val_file <PATH to the normlized val.txt> --model_save_dir <model_save_dir> --cache_dir <PATH where cache should be saved>
    ```
  - Optional Argument- `tech` in `matscibert_train.py` if set to `True` the masking candidates are limited to only technical words.
