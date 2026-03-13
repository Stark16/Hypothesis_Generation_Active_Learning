# Hierarchical Embedding Pipeline - AI Agent Instructions

## Project Overview
This is a scientific AI research project focused on **hypothesis generation using active learning**. The system creates hierarchical, context-aware embeddings for scientific concepts to predict knowledge gaps and generate scientific hypotheses. The main pipeline generates **context trees** from scientific keywords and converts them into **hierarchical embeddings** using domain-specific language models.

## Architecture & Core Components

### Main Pipeline Flow
1. **Context Tree Generation** (`context_tree_builder.py`) - Uses LLMs to build hierarchical knowledge trees from scientific keywords
2. **Hierarchical Embedding Creation** (`hierarchical_emb_builder.py`) - Converts context trees into multi-layer embeddings using SciBERT/MatSciBERT
3. **Embedding Database** (`create_embedding_database.py`) - Aggregates and normalizes embeddings across multiple runs
4. **Analysis Notebooks** - Jupyter notebooks for analyzing embedding patterns and hypothesis validation

### Key Classes & Patterns
- **Node class**: Tree structure with `keyword`, `response`, `parent`, `children`, and `depth` attributes
- **HierarchicalEmbPipeline**: Main orchestrator that coordinates tree building and embedding generation
- **ContextTree**: Handles LLM interactions and BFS tree traversal (uses `microsoft/Phi-3.5-mini-instruct` by default)
- **HierarchEmbdTree**: Manages embedding extraction using multiple layer strategies (`first_three`, `last_two`, etc.)

## Critical Development Patterns

### File Structure Convention
```
output_trees/
└── <domain>/
    └── <keyword>/
        ├── run_1/
        │   ├── tree_*.json (context trees)
        │   └── embdng_tree_v2_*.json (embeddings by layer strategy)
        └── final_<runs>_<strategy>.json (aggregated results)
```

### Model & Device Management
- Always call `torch.cuda.empty_cache()` after each tree generation run
- Reset memory context with `OBJContextTree.reset_mem_ctx()` between runs
- Use `device='cuda'` by default but handle CPU fallback
- Multiple embedding strategies: `['first', 'first_two', 'first_three', 'last', 'last_two', 'last_three', 'all']`

### Configuration Patterns
- Temperature controls creativity: `generation_args['temperature'] = 0.05` (low) to `0.4` (high)
- Batch processing: Set `MODEL_ARGS_gen_llm["batch_query"] = True` for efficiency
- Memory management: Use `no_history = True` to prevent context overflow
- Depth control: `depth_cap` parameter limits tree depth (typically 2-4)

## Development Workflows

### Running the Main Pipeline
```python
# Standard execution pattern from MAIN_hierarchical_embedding_pipeline.py
OBJ_HierarchEmbPipe = HierarchicalEmbPipeline(keyword, domain)
OBJ_HierarchEmbPipe.create_embedding(num_trees=2, depth_cap=2)
```

### Data Integrity Checking
Use utilities in `utils/` directory:
- `check_context_forest_integrity.py` - Validates output completeness
- `check_for_missing_embd.py` - Identifies missing embeddings
- `fix_missing_drugs.py` - Repairs incomplete datasets

### Embedding Analysis
- Load aggregated results from `final_<N>_<strategy>.json` files
- Each contains `final_embedding` (tree-aggregated) and `raw_root_embedding` per run
- Use polar normalization: `(vec - min) / (max - min) * 2 - 1`

## Domain-Specific Considerations

### Scientific Domains
- **Material Science**: Uses MatSciBERT models trained on scientific literature
- **Medical/Drug Research**: Processes drug-disease relationships from text files
- **Physics**: Handles complex scientific concepts and laws

### Model Selection
- **Generation LLM**: `microsoft/Phi-3.5-mini-instruct` for context tree creation
- **Embedding Model**: `allenai/scibert_scivocab_uncased` or custom MatSciBERT variants
- **Custom Models**: Check `MatSciBERT/trained_model/` for domain-specific models

### Error Handling
- Track failed responses in `LOG_failed_responses.json`
- Handle NaN/infinite values in embeddings with proper filtering
- Use try-catch blocks around model loading and GPU operations

## Branch-Specific Context
- **Current branch**: `hierarchical_tree_embedding` - Main development branch
- **matscibert branch**: Contains fine-tuned MatSciBERT models
- **parallel_training branch**: Optimized training pipeline (5x speed boost)
- **feature_space branch**: Hierarchical embedding experiments

When modifying code, ensure GPU memory management, proper error logging, and maintain the hierarchical output structure for downstream analysis.