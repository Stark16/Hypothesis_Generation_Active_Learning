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
