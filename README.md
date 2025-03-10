# 🚀 Towards Hypothesis Prediction using Active Learning  

## 🧠 **The Vision**  

Scientific progress has always depended on **hypothesis generation**—the ability to ask the right questions and explore the unknown. This research aims to build an **AI-driven system** that can **predict, refine, and validate hypotheses**. W are starting in the domain of Material Science. The use of **active learning** is to bridge the gap between **existing knowledge** and **emerging discoveries**.

The research follows **two key directions**:  

1. **🧩 Understanding & Structuring Knowledge** – Developing a structured way to **represent scientific concepts** so that AI can **reason about them like a researcher** if needed. This involves creating **hierarchical embeddings** that encode relationships between scientific terms more effectively than traditional vector spaces. This also makes the whole idea less of a **black-box**. It is possible through a new approach we are working on called as heirarchal statements and heirarchal embeddings.  

2. **🎭 Predicting & Evaluating Hypotheses** – This is the larger picture. Training an AI system to **anticipate missing knowledge or potential connections between scientific observations/theories**. This leads to suggesting of **scientific hypotheses**. This involves **fine-tuning language models** to understand research papers, detect knowledge gaps, and iteratively improve its predictions. The overview of our research is showcased in this presentation.  

---

## 🔀 **Project Structure & Branches**  

Each branch of this repository contributes to the broader goal of the research:  

### 🌳 **Feature Space (Hierarchical Embeddings)**  
To make AI reasoning more structured, this branch explores **hierarchical embeddings**, where knowledge is organized in a **context-aware, tree-like structure** rather than just floating in high-dimensional space. This helps the AI **understand definitions, relationships, and conceptual depth**, whiel also making the embeddings we will later generate less abstract (Mkaing it less Black-Box). This helps in forming the foundation for **meaningful hypothesis prediction**.  

### 📚 **MatSciBERT Fine-Tuning**  
This branch focuses on adapting **MatSciBERT**, a domain-specific language model, to predict missing scientific knowledge. Instead of passively reading research papers, the model is **trained to identify key terms, mask them, and predict their relevance**.   The idea

### ⚡ **Parallel Training Optimization**  
Efficient training is critical for large-scale hypothesis generation. This branch improves the **MatSciBERT pipeline**, optimizing how data is processed to achieve a **5× boost in training speed**. By restructuring the workflow, AI can focus on learning **faster and more effectively**, making it more practical for real-world scientific research.  

---

💡 **Together, these branches form a pipeline where AI can**:  
✔️ Understand **scientific concepts in context**  
✔️ Predict **missing links in knowledge**  
✔️ Efficiently **iterate on hypotheses**  

🚀 **This is just the beginning.** Future work will refine these models and extend them to broader scientific fields, making AI a true research collaborator.
