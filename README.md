# 📚 **MatSciBERT Fine-Tuning**

## 🧠 **Overview**

This branch adapts **MatSciBERT**, a domain-specific language model, to generate **embeddings** for scientific statements in the field of material science. Two separate models are trained:

1. **2015 Model**: Fine-tuned on a dataset of material science papers published until **2015**, capturing knowledge and trends up to that point.
  
2. **2024 Model**: Fine-tuned on a combined dataset from **Semantic Scholar** and **MatKG**, which includes papers published between **2016 and 2024**, representing more recent developments in material science.

We extract embeddings from specific layers of these models to create **feature spaces** for material science knowledge. These spaces allow us to:

- **Model-Generated Predictions of Potential Discoveries**: The **2015 model** generates points within the 2015 feature space that it believes represent **potential discoveries**. These are areas of material science knowledge that may not have been fully explored or validated at the time.
  
- **Intermediate Unconfirmed Space**: The points identified by the 2015 model are treated as an **intermediate, unconfirmed space**, where the model suggests possible discoveries that require verification.

- **Cross-Validation with 2024 Feature Space**: To confirm whether these potential discoveries are actually valid, we compare the points from the 2015 space with the **2024 feature space**. If the same points from the 2015 model are found to be present or validated in the 2024 model, we confirm them as **actual discoveries** in material science.

By comparing these two feature spaces, we aim to identify:

- **Emerging Discoveries**: Discoveries predicted by the 2015 model and validated in the 2024 model, showing that the field of material science has indeed advanced in these areas.
- **Knowledge Gaps**: Areas where the 2015 model’s predictions do not align with the 2024 feature space, indicating potential areas of scientific progress.
- **Confirmed Patterns**: Hypotheses or trends that are verified by both feature spaces, highlighting the solid evolution of material science knowledge over time (Active Learning).

---

## 🧩 **Key Features**

- **Custom Model**: **MatSciBERT** is fine-tuned specifically for material science, enabling the model to better understand domain-specific terminology and relationships.
- **Two-Time-Point Feature Spaces**: The project creates feature spaces for material science knowledge at **two different time points**—2015 and 2024. By comparing these spaces, we aim to identify new discoveries, validate them, and trace the progression of material science over time.
- **NER-Based Masking**: Using a **Named Entity Recognition (NER)** technique, the model generates **masking candidates** from scientific statements. These candidates help the model focus on key material science terms (e.g., chemical compounds, materials, or properties) during training.

---

## ⚙️ **Data Pipeline**

1. **Dataset Preparation**:
   - We downloaded **all available material science datasets** using the **SemanticScholar API**.
   - The **MatKG dataset** (provided via ArangoDB) contains **entity relations**. We used these relations to filter the research papers from **Semantic Scholar**.
   - The papers were filtered by **year**, and we split the dataset into two sets: one containing papers published **before 2015**, and the other containing papers published between **2016 and 2024**.

2. **Text Extraction**:
   - For each paper, we extracted the **full body text** and used **NLTK** to break the text down into **sentences**.
   - From these sentences, we used the **MatSciBERT NER** to generate **masking word candidates** (key material science terms such as chemical elements, properties, or materials).

3. **Masking and Training**:
   - The sentences were then masked by randomly selecting a word from the generated **masking candidates** and training **BERT-MLM** using the **MatSciBERT pipeline**.
   - This resulted in a model, which we call **Semantic-KG-BERT**, that generates embeddings from the sentences for use in hypothesis testing and feature space analysis.

---

## 🔄 **Training Process**

1. **Fine-Tuning the Models**:
   - The **2015 model** was trained using the dataset of material science research papers published **before 2015**. This model captures knowledge up until that point.
   - The **2024 model** was fine-tuned on a combined dataset from **Semantic Scholar** and **MatKG** for papers published between **2016 and 2024**, reflecting more recent developments in material science.

2. **Data Normalization and Collator**:
   - We use the script **normalize_corpus.txt** to **normalize** the words used in the dataset. Normalization here refers to standardizing the text by converting words into a consistent form (e.g., handling different variations of words, correcting spelling errors, and ensuring uniform use of terminology) so that the model can better understand and process them.
   - A **custom data collator** was developed to handle the task of **pitching masking candidates**. These candidates are selected based on the material science domain and include important entities, such as chemical compounds, properties, or materials. The collator identifies these key terms and proposes them as candidates for **masking**—where certain words in the sentence are randomly replaced with a **[MASK]** token. This process helps the model focus on predicting important terms, improving its ability to learn the relationships between different material science concepts.

3. **Model Training**:
   - The goal of training **MatSciBERT** (and its resulting model, **Semantic-KG-BERT**) is to **generate embeddings** for scientific statements in the material science domain. These embeddings are used to explore relationships and patterns between different material science concepts. 
   - Once trained, we input **scientific statements**—both with and without masked words—into **Semantic-KG-BERT**. The model then generates embeddings from the last three layers, and these embeddings are **weight-averaged** to produce a robust representation of the statement’s meaning within the material science context.

4. **Evaluation**:
   - We use **perplexity** and **accuracy** as evaluation metrics for **label prediction**. Perplexity measures how well the model predicts the masked words, while accuracy evaluates the model’s ability to correctly identify the predicted words from the masked positions.

---

## 🛠️ **Technologies Used**

- **ArangoDB**
- **Hugging Face Transformers**
- **Python**
- **PyTorch**
- **NLTK**
- **Semantic Scholar API**
- **MatSciBERT (Semantic-KG-BERT)**

---

## 🚀 **Future Work**

- **Hypothesis Generation**: Future work will focus on using the embeddings generated by **Semantic-KG-BERT** to predict hypotheses in material science and other scientific fields. This will involve leveraging the embeddings to propose potential discoveries and validate them.
- **Embedding Space Analysis**: We plan to analyze the feature spaces created by the **2015 model** and **2024 model**. This analysis will focus on identifying how scientific knowledge has evolved over time and help refine the models for better predictions in the future.

---

## 📚 **References**

- **Semantic Scholar Dataset**: [Semantic Scholar](https://www.semanticscholar.org/)
- **MatKG Dataset**: [MatKG](https://matkg.org/)

---

Let me know if this version is clearer!
