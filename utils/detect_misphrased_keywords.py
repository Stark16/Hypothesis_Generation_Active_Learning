"""
Misphrased Keyword Detection Script
====================================
Scans all context trees under output_trees/BATS/ and detects nodes where the
keyword's exact token-ID sequence is NOT found in the LLM-generated response.

This replicates the same `tokenize_and_find` logic used in
hierarchical_emb_tree_builder.py — when the keyword tokens aren't found,
`fetch_embedding` produces an empty array and the node is silently skipped
during embedding creation. This script quantifies how many nodes are affected.

Usage:
    python utils/detect_misphrased_keywords.py
    python utils/detect_misphrased_keywords.py --bats-dir output_trees/BATS --output utils/misphrased_keywords_report.txt
    python utils/detect_misphrased_keywords.py --tokenizer allenai/scibert_scivocab_uncased
"""

import os
import json
import argparse
import glob
from collections import defaultdict
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Core detection logic — mirrors tokenize_and_find from
# hierarchical_emb_tree_builder.py (lines 105-137)
# ---------------------------------------------------------------------------

def tokenize_and_find(tokenizer, text: str, keyword: str) -> bool:
    """Return True if the keyword's token-ID subsequence is found in text.

    This is a faithful replica of HierarchEmbdTree.tokenize_and_find applied
    to a single (text, keyword) pair.  We only need the boolean — no need to
    track positions.

    Uses plain Python lists (no tensors) for speed since we only need the
    boolean match result, not embeddings.
    """
    # Tokenize the keyword into IDs
    kw_tokens = tokenizer.tokenize(keyword)
    kw_ids = tokenizer.convert_tokens_to_ids(kw_tokens)

    if not kw_ids:
        # Edge case: keyword tokenizes to nothing
        return False

    # Tokenize the full response text (same settings as production, but
    # without return_tensors to avoid slow torch.tensor conversion)
    token_ids = tokenizer.encode(
        text,
        max_length=512,
        add_special_tokens=False,
        truncation=True,
    )

    # Sliding-window exact subsequence match (same as production)
    kw_len = len(kw_ids)
    for j in range(len(token_ids) - kw_len + 1):
        if token_ids[j : j + kw_len] == kw_ids:
            return True

    return False


# ---------------------------------------------------------------------------
# Tree traversal
# ---------------------------------------------------------------------------

def walk_tree(tokenizer, tree_dict: dict, results: list, tree_path: str, category: str):
    """Recursively visit every node in a context tree and check keyword match.

    Each entry appended to *results* is a dict:
        keyword, depth, matched (bool), response_snippet, tree_path, category,
        empty_response (bool)
    """
    for keyword, node in tree_dict.items():
        depth = node.get("depth", -1)
        response = node.get("response", "")
        children = node.get("children", {})

        entry = {
            "keyword": keyword,
            "depth": depth,
            "tree_path": tree_path,
            "category": category,
            "empty_response": False,
            "matched": False,
            "response_snippet": "",
        }

        if not response or not response.strip():
            entry["empty_response"] = True
            entry["response_snippet"] = "<empty>"
        else:
            matched = tokenize_and_find(tokenizer, response, keyword)
            entry["matched"] = matched
            # Keep a truncated snippet for the report
            snippet = response.strip().replace("\n", " ")
            entry["response_snippet"] = snippet[:150] + ("..." if len(snippet) > 150 else "")

        results.append(entry)

        # Recurse into children
        if children:
            walk_tree(tokenizer, children, results, tree_path, category)


# ---------------------------------------------------------------------------
# Discovery — find all tree.json files
# ---------------------------------------------------------------------------

def discover_trees(bats_dir: str):
    """Yield (tree_json_path, category) for every tree.json under bats_dir."""
    pattern = os.path.join(bats_dir, "**", "tree.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        # Derive category from the relative path
        # Typical: BATS/{category}/{subcategory}/{keyword}/run_N/tree.json
        # or:      BATS/{category}/{keyword}/run_N/tree.json
        rel = os.path.relpath(path, bats_dir)
        parts = rel.split(os.sep)
        category = parts[0] if parts else "unknown"
        yield path, category


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: list, output_path: str, bats_dir: str):
    """Write a human-readable text report summarising misphrased keywords."""

    # --- Aggregate stats ---
    total = len(results)
    empty_resp = sum(1 for r in results if r["empty_response"])
    matched = sum(1 for r in results if r["matched"] and not r["empty_response"])
    mismatched = sum(1 for r in results if not r["matched"] and not r["empty_response"])

    # Per-category breakdown
    cat_stats = defaultdict(lambda: {"total": 0, "matched": 0, "mismatched": 0, "empty": 0})
    for r in results:
        c = cat_stats[r["category"]]
        c["total"] += 1
        if r["empty_response"]:
            c["empty"] += 1
        elif r["matched"]:
            c["matched"] += 1
        else:
            c["mismatched"] += 1

    # Per-depth breakdown
    depth_stats = defaultdict(lambda: {"total": 0, "matched": 0, "mismatched": 0, "empty": 0})
    for r in results:
        d = depth_stats[r["depth"]]
        d["total"] += 1
        if r["empty_response"]:
            d["empty"] += 1
        elif r["matched"]:
            d["matched"] += 1
        else:
            d["mismatched"] += 1

    # All misphrased entries (sorted by category, then keyword)
    misphrased = sorted(
        [r for r in results if not r["matched"] and not r["empty_response"]],
        key=lambda r: (r["category"], r["keyword"]),
    )

    # --- Build report text ---
    lines = []
    sep = "=" * 80

    lines.append(sep)
    lines.append("  MISPHRASED KEYWORD DETECTION REPORT")
    lines.append(f"  Scanned directory: {bats_dir}")
    lines.append(sep)
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Total nodes scanned      : {total}")
    lines.append(f"  Keyword found (matched)   : {matched}")
    lines.append(f"  Keyword NOT found (miss)  : {mismatched}")
    lines.append(f"  Empty responses           : {empty_resp}")
    if total - empty_resp > 0:
        rate = mismatched / (total - empty_resp) * 100
        lines.append(f"  Mismatch rate             : {rate:.2f}%  ({mismatched}/{total - empty_resp} non-empty nodes)")
    lines.append("")

    lines.append("PER-CATEGORY BREAKDOWN")
    lines.append("-" * 80)
    lines.append(f"  {'Category':<35} {'Total':>6} {'Match':>6} {'Miss':>6} {'Empty':>6} {'Miss%':>7}")
    lines.append(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        denom = s["total"] - s["empty"]
        pct = f"{s['mismatched']/denom*100:.1f}%" if denom > 0 else "N/A"
        lines.append(f"  {cat:<35} {s['total']:>6} {s['matched']:>6} {s['mismatched']:>6} {s['empty']:>6} {pct:>7}")
    lines.append("")

    lines.append("PER-DEPTH BREAKDOWN")
    lines.append("-" * 80)
    lines.append(f"  {'Depth':>6} {'Total':>6} {'Match':>6} {'Miss':>6} {'Empty':>6} {'Miss%':>7}")
    lines.append(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for depth in sorted(depth_stats.keys()):
        s = depth_stats[depth]
        denom = s["total"] - s["empty"]
        pct = f"{s['mismatched']/denom*100:.1f}%" if denom > 0 else "N/A"
        lines.append(f"  {depth:>6} {s['total']:>6} {s['matched']:>6} {s['mismatched']:>6} {s['empty']:>6} {pct:>7}")
    lines.append("")

    lines.append(sep)
    lines.append(f"  DETAILED MISPHRASED ENTRIES  ({len(misphrased)} total)")
    lines.append(sep)
    lines.append("")

    current_cat = None
    for r in misphrased:
        if r["category"] != current_cat:
            current_cat = r["category"]
            lines.append(f"--- [{current_cat}] ---")
            lines.append("")

        rel_tree = os.path.relpath(r["tree_path"], bats_dir)
        lines.append(f"  Keyword  : \"{r['keyword']}\"")
        lines.append(f"  Depth    : {r['depth']}")
        lines.append(f"  Tree     : {rel_tree}")
        lines.append(f"  Response : {r['response_snippet']}")
        lines.append("")

    lines.append(sep)
    lines.append("  END OF REPORT")
    lines.append(sep)

    report_text = "\n".join(lines)

    # Write to file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_text)

    return report_text, mismatched, total, empty_resp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect misphrased keywords in BATS context trees"
    )
    default_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--bats-dir",
        default=os.path.join(default_root, "output_trees", "BATS"),
        help="Root directory containing BATS context trees",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(default_root, "utils", "misphrased_keywords_report.txt"),
        help="Path to write the output report",
    )
    parser.add_argument(
        "--tokenizer",
        default="bert-base-uncased",
        help="HuggingFace tokenizer to use (must match embedding pipeline)",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Scanning trees in: {args.bats_dir} ...")
    results = []
    tree_count = 0
    for tree_path, category in discover_trees(args.bats_dir):
        tree_count += 1
        try:
            with open(tree_path, "r") as f:
                tree_dict = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [WARN] Skipping {tree_path}: {e}")
            continue

        walk_tree(tokenizer, tree_dict, results, tree_path, category)

    print(f"Scanned {tree_count} tree files, {len(results)} total nodes.")

    report_text, mismatched, total, empty = generate_report(results, args.output, args.bats_dir)
    print(f"\nResults: {mismatched} misphrased out of {total - empty} non-empty nodes")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
