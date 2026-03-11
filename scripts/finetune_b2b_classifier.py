"""
Fine-tune gpt-4o-mini as a B2B article classifier.

Data sources:
  1. data/dynamic_dataset.jsonl  — 2,383 pipeline-labeled examples (NLI-validated)
  2. ANLI R3 (HuggingFace)       — adversarial hard negatives from news premises
  3. financial_phrasebank        — falls back to dynamic_dataset positives if unavailable

Output: fine-tuning job ID (monitor at platform.openai.com/finetune)
Usage: venv/Scripts/python.exe scripts/finetune_b2b_classifier.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


SYSTEM_PROMPT = (
    "You are a B2B sales intelligence classifier. "
    "Given a news article title and text, reply with exactly YES if it describes "
    "a business-to-business event (company launching product, funding, acquisition, "
    "enterprise deal, SaaS, industrial tech, B2B partnership, regulatory action on a "
    "named company) or NO if it is consumer news, sports, politics, crime, entertainment, "
    "or general market commentary with no specific company as primary actor."
)


def _make_message(text: str, label: int) -> dict:
    answer = "YES" if label == 1 else "NO"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text[:800]},  # cap at 800 chars
            {"role": "assistant", "content": answer},
        ]
    }


def load_dynamic_dataset(path: Path) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                text = ex.get("text", "").strip()
                label = ex.get("label", -1)
                if text and label in (0, 1):
                    examples.append({"text": text, "label": label, "source": ex.get("source", "dynamic")})
            except Exception:
                pass
    return examples


def load_anli_negatives(n: int = 600) -> list[dict]:
    """Pull hard negatives from ANLI R3 premises (news/wiki sentences, non-B2B)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("anli", trust_remote_code=False)
        r3 = ds["train_r3"]
        # Use premises: they are real news/wiki sentences → many clearly non-B2B
        # Filter for short-to-medium length, avoid pure business news
        candidates = []
        for ex in r3:
            premise = ex.get("premise", "").strip()
            if 50 < len(premise) < 600:
                candidates.append(premise)
        random.shuffle(candidates)
        selected = candidates[:n]
        print(f"[anli] Selected {len(selected)} hard negatives from ANLI R3")
        return [{"text": t, "label": 0, "source": "anli_r3"} for t in selected]
    except Exception as e:
        print(f"[anli] Failed to load ANLI R3: {e}")
        return []


def load_financial_phrasebank() -> list[dict]:
    """Load financial_phrasebank positive examples via direct Parquet download."""
    try:
        import requests
        import io
        import pandas as pd
        # Use the Parquet version on HuggingFace datasets
        url = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/train-00000-of-00001.parquet"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            df = pd.read_parquet(io.BytesIO(resp.content))
            print(f"[phrasebank] Loaded {len(df)} rows. Columns: {list(df.columns)}")
            # label: 0=negative, 1=neutral, 2=positive → use positive (2) and neutral (1)
            positives = df[df["label"] == 2]["sentence"].tolist()
            print(f"[phrasebank] {len(positives)} positive (label=2) sentences")
            return [{"text": t, "label": 1, "source": "financial_phrasebank"} for t in positives[:500]]
        else:
            print(f"[phrasebank] HTTP {resp.status_code}")
    except Exception as e:
        print(f"[phrasebank] Failed: {e}")
    return []


def build_training_file(examples: list[dict], path: Path, val_frac: float = 0.15) -> tuple[Path, Path]:
    """Convert examples to OpenAI JSONL and split train/val."""
    random.shuffle(examples)
    n_val = max(10, int(len(examples) * val_frac))
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    train_path = path / "finetune_train.jsonl"
    val_path = path / "finetune_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(_make_message(ex["text"], ex["label"])) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(_make_message(ex["text"], ex["label"])) + "\n")

    pos_train = sum(1 for e in train_examples if e["label"] == 1)
    neg_train = len(train_examples) - pos_train
    print(f"Train: {len(train_examples)} ({pos_train} pos, {neg_train} neg)")
    print(f"Val:   {len(val_examples)}")
    return train_path, val_path


def upload_and_finetune(train_path: Path, val_path: Path) -> str:
    """Upload files to OpenAI and start fine-tuning job."""
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("\nUploading training file...")
    with open(train_path, "rb") as f:
        train_file = client.files.create(file=f, purpose="fine-tune")
    print(f"Train file ID: {train_file.id}")

    print("Uploading validation file...")
    with open(val_path, "rb") as f:
        val_file = client.files.create(file=f, purpose="fine-tune")
    print(f"Val file ID: {val_file.id}")

    print("\nStarting fine-tuning job (gpt-4o-mini)...")
    job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=val_file.id,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": 3},
        suffix="b2b-classifier",
    )
    print(f"\n✓ Fine-tuning job started!")
    print(f"  Job ID:  {job.id}")
    print(f"  Status:  {job.status}")
    print(f"  Monitor: https://platform.openai.com/finetune/{job.id}")
    print(f"\nOnce complete, update OPENAI_FINETUNE_MODEL in .env:")
    print(f"  OPENAI_FINETUNE_MODEL=<model_id_from_job>")
    return job.id


def main():
    random.seed(42)
    out_dir = ROOT / "data" / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load pipeline-labeled dataset
    dynamic_path = ROOT / "data" / "dynamic_dataset.jsonl"
    examples = load_dynamic_dataset(dynamic_path)
    print(f"[dynamic] Loaded {len(examples)} examples")

    # 2. Add ANLI R3 hard negatives
    anli_negs = load_anli_negatives(n=600)
    examples.extend(anli_negs)

    # 3. Add financial_phrasebank positives
    phrasebank = load_financial_phrasebank()
    examples.extend(phrasebank)

    pos = sum(1 for e in examples if e["label"] == 1)
    neg = len(examples) - pos
    print(f"\nTotal: {len(examples)} ({pos} pos, {neg} neg)")

    # 4. Build JSONL files
    train_path, val_path = build_training_file(examples, out_dir)

    # 5. Upload and start fine-tuning
    job_id = upload_and_finetune(train_path, val_path)

    # 6. Save job ID for later wiring
    (out_dir / "finetune_job.json").write_text(
        json.dumps({"job_id": job_id, "status": "pending"}, indent=2)
    )
    print(f"\nJob ID saved to data/finetune/finetune_job.json")


if __name__ == "__main__":
    main()
