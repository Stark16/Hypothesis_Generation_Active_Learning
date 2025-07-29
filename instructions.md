# Hierarchical Embedding Output Structure & Instructions

## Overview

For each keyword (e.g., a drug or disease), we generate contextual embeddings using multiple transformer layer strategies. The output is organized for easy access, comparison, and downstream analysis.

---

## Directory Structure

```
output_trees/
└── NeLaMKRR_hierarchichal/
    ├── <keyword_1>/
    │   ├── final_50_first_three.json
    │   ├── final_50_first_two.json
    │   ├── final_50_first.json
    │   ├── final_50_last_three.json
    │   ├── final_50_last_two.json
    │   └── final_50_last.json
    ├── <keyword_2>/
    │   └── ...
    └── ...
```

- Each `<keyword>` (e.g., abemaciclib, aspirin, etc.) has its own folder.
- Inside each folder, there are 6 JSON files, one for each embedding strategy:
  - `final_50_first_three.json`
  - `final_50_first_two.json`
  - `final_50_first.json`
  - `final_50_last_three.json`
  - `final_50_last_two.json`
  - `final_50_last.json`
- The number `50` indicates the number of runs per keyword.

---

## File Content Structure

Each JSON file (e.g., `final_50_first_three.json`) contains the final embeddings for all 50 runs for that keyword and strategy:

```json
{
  "run_1": {
    "final_embedding": [ ... ],        // Aggregated embedding for the whole tree (numpy array, flattened)
    "raw_root_embedding": [ ... ]      // The raw embedding vector for the root node (from 'raw_enc')
  },
  "run_2": {
    "final_embedding": [ ... ],
    "raw_root_embedding": [ ... ]
  },
  ...
}
```

- `final_embedding`: The aggregated embedding for the entire context tree for that run and strategy.
- `raw_root_embedding`: The direct embedding of the root node (i.e., the main keyword/definition for that run and strategy).

---

## How to Use

- To access all embeddings for a given keyword and strategy, open the corresponding JSON file in that keyword’s folder.
- Each run is indexed by its run number (e.g., `"run_1"`, `"run_2"`, ...).
- Both the overall tree embedding and the root node embedding are available for each run.

---

## Notes

- All embeddings are stored as lists of floats (flattened numpy arrays).
- The structure is consistent across all keywords and strategies.
- If you need to process or analyze these embeddings, simply iterate over the runs in the relevant JSON file.

---

If you have any questions about the structure or need example code to load these embeddings, let me know!
