# 🚀 Towards Hypothesis Prediction using Active Learning [DePaul Research]

## 🧠 **The Vision**  

Scientific progress relies on **hypothesis generation**—asking the right questions to explore the unknown. Our research aims to build an **AI-driven system** that can **predict, refine, and validate scientific hypotheses**, starting with **Material Science**. We use **active learning** to bridge the gap between **existing knowledge** and **emerging discoveries**.

The research follows two main directions:

1. **🧩 Structuring Knowledge** – Creating **hierarchical embeddings** to represent scientific concepts in a more structured way. This approach will allow AI to **reason** like a researcher and reduce the **black-box** nature of current systems.
  
2. **🎭 Hypothesis Prediction** – Training AI to **anticipate knowledge gaps** and suggest **scientific hypotheses**. This involves **fine-tuning language models** to understand research papers, detect gaps, and improve predictions over time.

---

## 🔀 **Project Structure & Branches**  

This repository is divided into multiple branches, each contributing to the broader research goal:

### 📚 **MatSciBERT Fine-Tuning**  
This branch adapts **MatSciBERT**, a domain-specific language model, to generate **embeddings** for scientific statements. Two models are trained—one on papers till 2015, and the other till 2024. We then extract embeddings from specific layers of these models and use them to create a **feature space** for material science knowledge. This space is used to predict new patterns and validate them by comparing with the 2024 feature space.

### ⚡ **Parallel Training Optimization**  
Efficient training is key to large-scale hypothesis generation. This branch optimizes the **MatSciBERT pipeline** to achieve a **5× boost in training speed**. By restructuring the workflow, this enables faster and more effective learning, making the models more practical for real-world scientific research.

### 🌳 **Feature Space (Hierarchical Embeddings)**  
This branch explores **hierarchical embeddings**, where knowledge is structured in a **context-aware, tree-like** structure rather than high-dimensional space. This makes the AI’s reasoning more structured, allowing for **meaningful hypothesis prediction** by organizing scientific concepts based on their relationships and depth.

---

💡 **Together, these branches aim to build a system that can**:  
✔️ Understand **scientific concepts in context**  
✔️ Predict **missing links in knowledge**  

🚀 **This is just the beginning.** Future work will refine these models and extend them to other scientific fields. Feel free to explore any of the branches.
