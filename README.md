# 🚀 Towards Hypothesis Prediction using Active Learning [[DePaul Research](https://drive.google.com/file/d/17-rAkKbAvTCTzBNMniW6QPe66BzO36h-/view?usp=sharing)]

## 🧠 **The Vision**  

Scientific progress relies on **hypothesis generation**—asking the right questions to explore the unknown. Our research aims to build an **AI-driven system** that can **predict, refine, and validate scientific hypotheses**, starting with **Material Science**. We use **active learning** to bridge the gap between **existing knowledge** and **emerging discoveries**.

The research follows two main directions:

1. **🧩 Structuring Knowledge** – Creating **hierarchical embeddings** to represent scientific concepts in a more structured way. This approach will allow AI to **reason** like a researcher and reduce the **black-box** nature of current systems.
  
2. **🎭 Hypothesis Prediction** – Training AI to **anticipate knowledge gaps** and suggest **scientific hypotheses**. This involves **fine-tuning language models** to understand research papers, detect gaps, and improve predictions over time.

> To get a detailed synopsis of the research, check out [this link](https://drive.google.com/file/d/17-rAkKbAvTCTzBNMniW6QPe66BzO36h-/view?usp=sharing).

#### [LLM Knowledge Tree Demo](https://drive.google.com/file/d/1SbmQ9ulhwpSOeBLk7izHLnAQu3MRwkcL/view?usp=sharing) - 
![Desktop 2025 09 23 - 13 16 11 01](https://github.com/user-attachments/assets/77e9278e-996d-4990-9f2e-5cd561ec2e92)

![image](https://github.com/user-attachments/assets/b650a66d-18e1-48fd-97dd-43fe8e650bb9)


---

## 🔀 **Project Structure & Branches**  

This repository is divided into multiple branches, each contributing to the broader research goal:

### 📚 **[MatSciBERT Fine-Tuning](https://github.com/Stark16/Hypothesis_Generation_Active_Learning/tree/matscibert)**  
This branch adapts **MatSciBERT**, a domain-specific language model, to generate **embeddings** for scientific statements. Two models are trained—one on papers till 2015, and the other till 2024. We then extract embeddings from specific layers of these models and use them to create a **feature space** for material science knowledge. This space is used to predict new patterns and validate them by comparing with the 2024 feature space. Our custom version of MatSciBERT is called **Semantic-KG-BERT**, as it is trained on a combination of the **[Semantic Scholar](https://www.semanticscholar.org/about)** dataset and **[MatKG](https://openreview.net/pdf?id=cR1iE6MQ1y)**.

### ⚡ **[Parallel Training Optimization](https://github.com/Stark16/Hypothesis_Generation_Active_Learning/tree/parallel_training)**  
Efficient training is key to large-scale hypothesis generation. This branch optimizes the **MatSciBERT pipeline** to achieve a **5× boost in training speed**. By restructuring the training pipeline, this enables faster and more effective learning, making the models more practical for real-world scientific research.

### 🌳 **[Feature Space (Hierarchical Embeddings)](https://github.com/Stark16/Hypothesis_Generation_Active_Learning/tree/feature_space)**  
This branch explores **hierarchical embeddings**, a novel approach where knowledge is structured in a **context-aware, tree-like** structure rather than high-dimensional space. This makes the AI’s reasoning more structured, allowing for **meaningful hypothesis prediction** by organizing scientific concepts based on their relationships and depth.

---

💡 **Together, these branches aim to build a system that can**:  
✔️ Understand **scientific concepts in context**  
✔️ Predict **missing links in knowledge**  

🚀 **This is just the beginning.** Future work will refine these models and extend them to other scientific fields. Feel free to explore any of the branches.

---

# Hierarchical Embedding Pipeline Guide

This section explains the overarching process and architecture of the codebase used for hypothesis generation via context-aware hierarchical embeddings. The central concept involves defining scientific keywords by building a tree of related concepts (Context Tree), extracting deep contextual embeddings for these concepts (Context Forest / Embedding Trees), and evaluating the output against scientific or analogy datasets (e.g., BATS).

## 1. High-Level Process Overview

The research flow consists of four primary stages.

### Stage 1: Context-Tree Creation
A purely text-based knowledge tree is generated for a given starting keyword. An LLM (Language Model) is prompted to define the target word and list related sub-keywords. These sub-keywords become branches. The model recursively queries these new branches up to a maximum depth, creating a hierarchy of related terms and definitions centered on the root keyword.

### Stage 2: Embedding-Tree (Context Forest) Creation
Once the text-based Context Tree is built, it needs to be mapped to a continuous vector space so that semantic relationships can be analyzed mathematically. A scientific BERT model (e.g., SciBERT, MatSciBERT) ingests each node’s text and definition. The text is passed through the model, and embeddings from specific layers (typically the last few hidden layers) are extracted to form the corresponding Embedding Tree (or Context Forest).

### Stage 3: Final Embedding Aggregation
Due to the LLM's non-deterministic nature and varying contexts, multiple Context Trees are generated for the same starting keyword across different "runs". This stage aggregates the resulting embedding trees into a single, cohesive representation (the "Final Embedding") for the root keyword, establishing a robust vector that captures its multidimensional scientific context.

### Stage 4: Evaluation (BATS Test)
The Bigger Analogy Test Set (BATS) is used to evaluate the semantic properties of the final embeddings. It tests whether the vector arithmetic of the hierarchical embeddings appropriately models semantic analogies (e.g., inflectional morphology, lexicographic semantics).


## 2. Project Scripts and their Dependencies

The repository utilizes a modular structure. Here are the core scripts, their purposes, and their interdependencies:

*   **`MAIN_context_tree_emb_pipeline.py`**
    *   **Role:** The entry point for running the embedding creation process. It handles batch processing of multiple keywords from a given domain.
    *   **Dependencies:** Imports `HierarchicalEmbPipeline` from `hierarchical_embedding_pipeline.py`.
    *   **Output:** Generates execution time analysis plots and coordinates the output directories.

*   **`hierarchical_embedding_pipeline.py`**
    *   **Role:** The orchestrator class. It coordinates the creation of both the initial text tree and the corresponding embedding tree. It heavily relies on the builder scripts.
    *   **Dependencies:** Imports `context_tree_builder.py` and `hierarchical_emb_tree_builder.py`.

*   **`context_tree_builder.py`**
    *   **Role:** Manages interactions with the generation LLM (e.g., Phi-3.5-mini-instruct). Provides the `ContextTree` and `Node` classes. Uses a breadth-first search (BFS) approach to construct the text-based tree.
    *   **Dependencies:** Runs largely independently but relies on `transformers` and HuggingFace models.

*   **`hierarchical_emb_tree_builder.py`**
    *   **Role:** Houses the `HierarchEmbdTree` class. It loads the output JSON from the `context_tree_builder` and processes the text through a specified embedding model.
    *   **Dependencies:** Dependent on the JSON output structure defined by `context_tree_builder.py`.

*   **`create_embedding_database.py`** (and utilities like `check_final_embedding_sizes.py`)
    *   **Role:** Reads the raw embedding JSONs (representing individual runs) and aggregates them into the final structured embeddings, performing any necessary normalization (e.g., polar normalization).
    *   **Dependencies:** Relies on the JSON outputs from `hierarchical_emb_tree_builder.py`.

*   **`tests/run_bats_test.py`** (and associated notebooks)
    *   **Role:** Evaluates the generated final embeddings against the BATS analogy datasets.
    *   **Dependencies:** Requires the aggregated final embedding outputs from the database creation stage.


## 3. Detailed Execution Steps within Scripts

### Step 1: `context_tree_builder.py` (Text Generation)
1.  **Initialization:** The `ContextTree` class is initialized with a starting keyword, a domain, and an LLM identifier.
2.  **Breadth-First Search (BFS):** The `bfs` method starts with the root keyword. It passes a customized prompt to the LLM requesting a definition and related sub-keywords.
3.  **Parsing:** `extract_info` isolates the definition and the list of related terms from the LLM's response.
4.  **Tree Expansion:** New `Node` objects are created for the related terms and added as children to the current node. The process repeats until the `depth_cap` is reached.
5.  **Output:** The tree structure is serialized and saved as `tree.json`.

**Example JSON Data (`tree.json` structure):**
```json
{
    "keyword": "General relativity",
    "response": "General relativity is a fundamental theory... tech_words=[gravity, spacetime, mass, energy]",
    "depth": 1,
    "children": [
        {
            "keyword": "gravity",
            "response": "Gravity is the phenomenon... tech_words=[force, mass, attraction]",
            "depth": 2,
            "children": [...]
        }
    ]
}
```

### Step 2: `hierarchical_emb_tree_builder.py` (Embedding Extraction)
1.  **Loading:** The `HierarchEmbdTree` loads the `tree.json` outputted by the context tree builder.
2.  **Tokenization & Model Inference:** The text in the tree is tokenized. A scientific BERT model processes the text to map semantic features to hidden layer weights.
3.  **Layer Selection:** The `embed_texts` method applies a specific layer strategy (e.g., extracting embeddings from the `last_three` layers, or `all` layers) to derive vectors.
4.  **Vector Mapping:** Word tokens within the text matching the node's keyword are located, and their specific token embeddings are pulled to represent the concept.
5.  **Output:** An output JSON mapping nodes and their depths to their corresponding multi-dimensional vectors (`embdng_tree_v2_*.json`).

**Example JSON Data (`embdng_tree_v2_*.json` structure):**
```json
{
    "General relativity": {
        "depth": 1,
        "occurrences": 1,
        "embedding": [0.12, -0.45, 0.89, ...]
    },
    ...
}
```

### Step 3: `create_embedding_database.py` (Aggregation)
1.  **Folder Scanning:** Searches output directories for all `embdng_tree_v2_*.json` files belonging to the multiple runs of a specific keyword.
2.  **Vector Stacking:** Consolidates embeddings utilizing mathematical models (e.g. averaging vector coordinates across runs).
3.  **Output:** Produces combined files typically prefixed as `final_<runs>_<strategy>.json` ensuring a consistent matrix form utilized for evaluation.

### Step 4: `tests/run_bats_test.py` (Evaluation)
1.  **Setup:** Loads the BATS dataset containing pairs of words defining specific relationships (e.g., male-female pairs).
2.  **Analogy Processing:** Loads the aggregated final embeddings for these specific words.
3.  **Arithmetic:** Uses vector arithmetic logic: `A - B + C = D` (e.g., `King - Man + Woman`) and checks the proximity of the resultant vector to the target embedding `Queen`.
4.  **Reporting:** Outputs accuracy scores determining the contextual mapping integrity engineered by the hierarchical embeddings.

