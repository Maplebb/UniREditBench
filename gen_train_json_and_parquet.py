import json
import re
import random
from pathlib import Path
from typing import Union, List, Dict, Any
import pandas as pd
import pyarrow.parquet as pq
import argparse
import os

# ---------- Basic Utilities ----------
def load_json_list(path: Union[str, Path]) -> List[dict]:
    """Loads data from a JSON or JSONL file."""
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    else:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list), "JSON root object should be a list"
        return data

_CODEFENCE_RE = re.compile(
    r'^\s*```(?:json|js|javascript)?\s*\n(.*)\n```\s*$', re.DOTALL | re.IGNORECASE
)

def _strip_code_fence(s: str) -> str:
    """Strips markdown code fences from a string."""
    m = _CODEFENCE_RE.match(s)
    return m.group(1) if m else s

# ---------- Main Process ----------
def export_json_and_parquet(
    src_json: Union[str, Path],
    dataset_dir: Union[str, Path],
    out_json: Union[str, Path],
    out_parquet_dir: Union[str, Path],
    *,
    n_chunks: int = 8,
    seed: int = 42,
    ensure_exists: bool = True,
) -> None:
    """
    Filters source JSON, checks for file existence, and exports to
    a consolidated JSON file and sharded Parquet files.
    """
    out_json = Path(out_json)
    out_parquet_dir = Path(out_parquet_dir)

    # Ensure output directories exist
    out_parquet_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    items = load_json_list(src_json)

    rows: List[Dict[str, Any]] = []
    missing_images = 0
    missing_fields = 0
    parse_failures = 0 # Note: This is initialized but never incremented.

    for it in items:
        instruction = it.get("instruction")
        reasoning = it.get("reasoning")
        rules = it.get("rules", "")
        rel_original_image_path = it.get("original_image_path")
        rel_edit_image_path = it.get("edited_image_path")

        if not instruction or not reasoning:
            missing_fields += 1
            continue
        
        # Construct full paths using os.path.join, as requested
        bpath = os.path.join(dataset_dir, rel_original_image_path)
        apath = os.path.join(dataset_dir, rel_edit_image_path)

        # Check for existence using os.path.exists, as requested
        if ensure_exists and (not os.path.exists(apath) or not os.path.exists(bpath)):
            if not os.path.exists(bpath):
                print(f"[WARN] Skipped due to missing images: {bpath}")
            else:
                print(f"[WARN] Skipped due to missing images: {apath}")
            missing_images += 1
            continue

        rows.append({
            "before_image_path": bpath,
            "after_image_path":  apath,
            "instruction":       instruction.strip(),
            "reasoning":         reasoning.strip(),
            "rules":             rules.strip(),
        })

    # 1) Write the filtered JSON (preserving the original order, not shuffled)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # 2) Shuffle randomly and write N parquet chunks
    rng = random.Random(seed)
    shuffled = rows[:]  # Create a copy to shuffle
    rng.shuffle(shuffled)

    N = len(shuffled)
    base, extra = divmod(N, n_chunks)
    start = 0
    info_dict = {}
    
    # Define column order
    column_order = ["before_image_path", "after_image_path", "instruction", "reasoning", "rules"]

    for i in range(n_chunks):
        take = base + (1 if i < extra else 0)
        part = shuffled[start:start + take]
        start += take

        df = pd.DataFrame(part, columns=column_order)
        parquet_path = out_parquet_dir / f"chunk_{i}.parquet"
        df.to_parquet(parquet_path, index=False)

        # Read parquet metadata
        table = pq.ParquetFile(parquet_path)
        info_dict[str(parquet_path)] = {
            "num_row_groups": table.num_row_groups,
            "num_rows": table.metadata.num_rows,
        }

        print(f"[OK] {len(df):4d} rows -> {parquet_path}")

    # Write parquet_info.json
    parquet_info_json = out_parquet_dir / "parquet_info.json"
    with open(parquet_info_json, "w", encoding="utf-8") as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Valid samples: {N} | Output JSON: {out_json}")
    print(f"[INFO] Generated parquet_info.json: {parquet_info_json}")
    if missing_images:
        print(f"[WARN] Skipped due to missing images: {missing_images} items")
    if missing_fields:
        print(f"[WARN] Skipped due to missing fields: {missing_fields} items")
    if parse_failures:
        print(f"[WARN] Skipped due to parsing errors: {parse_failures} items")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export filtered JSON and Parquet files.")
    parser.add_argument("--src_json", type=Path, required=True, help="Path to source JSON/JSONL")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Path to base images directory")
    parser.add_argument("--out_json", type=Path, required=True, help="Path to output filtered JSON")
    parser.add_argument("--out_parquet_dir", type=Path, required=True, help="Directory for output Parquet shards")
    
    args = parser.parse_args()

    # --------- Example Call (uncomment and modify paths/templates as needed) ---------
    export_json_and_parquet(
        src_json=args.src_json,                # Your source JSON/JSONL
        dataset_dir=args.dataset_dir,          # Base directory
        out_json=args.out_json,                # Filtered output JSON
        out_parquet_dir=args.out_parquet_dir,  # Sharded parquet output directory
        n_chunks=8,
        seed=42,
    )