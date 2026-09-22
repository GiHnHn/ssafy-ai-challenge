"""Evaluate leakage-safe answer transfer from byte-identical reference images.

Validation queries are the final 10% of train.csv.  References are restricted to
the first 90% of train.csv plus dev consensus labels, so validation labels never
enter retrieval.  An answer is transferred only when the reference answer text
is present verbatim among the query choices.
"""

import hashlib
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from tqdm import tqdm


import os
ROOT = Path(os.environ.get("SSAFY_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
LETTERS = list("abcd")
VALID_SIZE = 508


def normalize(text):
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", str(text)).lower()


def sha1_file(path):
    digest = hashlib.sha1()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_hash(frame):
    result = frame.copy()
    result["sha1"] = [
        sha1_file(ROOT / path)
        for path in tqdm(result["path"], desc="SHA1", leave=False)
    ]
    return result


def dev_consensus(row):
    votes = [
        str(row[f"answer{i}"]).strip().lower()
        for i in range(1, 6)
        if pd.notna(row[f"answer{i}"])
        and str(row[f"answer{i}"]).strip().lower() in LETTERS
    ]
    if not votes:
        return None, 0
    answer, count = Counter(votes).most_common(1)[0]
    return answer, count


def make_references(train_reference, dev, minimum_dev_votes):
    records = []
    for _, row in train_reference.iterrows():
        records.append(
            {
                "source": "train",
                "id": row["id"],
                "sha1": row["sha1"],
                "dhash": int(row["dhash"]) if "dhash" in row.index else None,
                "question": row["question"],
                "question_norm": normalize(row["question"]),
                "answer_text": str(row[row["answer"]]),
                "votes": 5,
            }
        )
    for _, row in dev.iterrows():
        answer, votes = dev_consensus(row)
        if answer is not None and votes >= minimum_dev_votes:
            records.append(
                {
                    "source": "dev",
                    "id": row["id"],
                    "sha1": row["sha1"],
                    "dhash": int(row["dhash"]) if "dhash" in row.index else None,
                    "question": row["question"],
                    "question_norm": normalize(row["question"]),
                    "answer_text": str(row[answer]),
                    "votes": votes,
                }
            )
    return pd.DataFrame(records)


def transfer_candidates(queries, references):
    by_hash = defaultdict(list)
    for ref in references.to_dict("records"):
        by_hash[ref["sha1"]].append(ref)

    output = []
    for _, query in queries.iterrows():
        query_question = normalize(query["question"])
        option_to_letter = {
            normalize(query[letter]): letter for letter in LETTERS
        }
        candidates = []
        for ref in by_hash.get(query["sha1"], []):
            answer_norm = normalize(ref["answer_text"])
            if answer_norm not in option_to_letter:
                continue
            similarity = SequenceMatcher(
                None, query_question, ref["question_norm"]
            ).ratio()
            candidates.append(
                {
                    **ref,
                    "similarity": similarity,
                    "pred": option_to_letter[answer_norm],
                }
            )
        candidates.sort(
            key=lambda item: (
                item["similarity"],
                item["votes"],
                item["source"] == "train",
            ),
            reverse=True,
        )
        best = candidates[0] if candidates else None
        output.append(
            {
                "id": query["id"],
                "retrieval_pred": None if best is None else best["pred"],
                "similarity": None if best is None else best["similarity"],
                "reference_source": None if best is None else best["source"],
                "reference_id": None if best is None else best["id"],
                "reference_question": None if best is None else best["question"],
                "reference_answer_text": None if best is None else best["answer_text"],
                "candidate_count": len(candidates),
            }
        )
    return pd.DataFrame(output)


def evaluate(valid, baseline, candidates, label):
    joined = valid.merge(candidates, on="id", how="left").merge(
        baseline[["id", "pred", "category"]].rename(
            columns={"pred": "baseline_pred"}
        ),
        on="id",
        how="left",
    )
    assert joined["baseline_pred"].notna().all()
    base_correct = joined["baseline_pred"] == joined["answer"]
    print(f"\n=== {label} ===")
    print("baseline:", int(base_correct.sum()), "/", len(joined))
    for threshold in [1.0, 0.98, 0.95, 0.90, 0.80, 0.70, 0.0]:
        use = joined["similarity"].fillna(-1) >= threshold
        predictions = joined["baseline_pred"].where(
            ~use, joined["retrieval_pred"]
        )
        correct = predictions == joined["answer"]
        retrieval_correct = (
            joined.loc[use, "retrieval_pred"] == joined.loc[use, "answer"]
        )
        fixes = int((use & ~base_correct & correct).sum())
        breaks = int((use & base_correct & ~correct).sum())
        print(
            f"sim>={threshold:>4.2f}: coverage={int(use.sum()):>3}, "
            f"retrieval_acc={int(retrieval_correct.sum())}/{int(use.sum())}, "
            f"hybrid={int(correct.sum())}/{len(joined)}, fixes={fixes}, breaks={breaks}"
        )
    available = joined[joined["retrieval_pred"].notna()].copy()
    if len(available):
        available["retrieval_correct"] = (
            available["retrieval_pred"] == available["answer"]
        )
        available["baseline_correct"] = (
            available["baseline_pred"] == available["answer"]
        )
        print("\nvalidation retrieval candidates:")
        print(
            available[
                [
                    "id", "category", "answer", "baseline_pred",
                    "retrieval_pred", "similarity", "reference_source",
                    "reference_id", "baseline_correct", "retrieval_correct",
                    "question", "reference_question",
                ]
            ].to_string(index=False)
        )
    return joined


def main():
    train = add_hash(pd.read_csv(ROOT / "train.csv"))
    dev = add_hash(pd.read_csv(ROOT / "dev.csv"))
    test = add_hash(pd.read_csv(ROOT / "test.csv"))
    baseline = pd.read_csv(ROOT / "성능파일" / "qwen3_vl_8b_valid.csv")

    train_reference = train.iloc[:-VALID_SIZE].copy()
    valid = train.iloc[-VALID_SIZE:].copy()

    for minimum_votes in (5, 4, 3):
        references = make_references(train_reference, dev, minimum_votes)
        valid_candidates = transfer_candidates(valid, references)
        evaluate(
            valid,
            baseline,
            valid_candidates,
            f"exact-image transfer; dev votes >= {minimum_votes}",
        )

        test_candidates = transfer_candidates(test, references)
        print("test candidate coverage by similarity:")
        for threshold in (1.0, 0.98, 0.95, 0.90, 0.80, 0.70, 0.0):
            coverage = int(
                (test_candidates["similarity"].fillna(-1) >= threshold).sum()
            )
            print(f"  sim>={threshold:>4.2f}: {coverage}")

        valid_candidates.to_csv(
            ROOT / f"duplicate_transfer_valid_votes{minimum_votes}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        test_candidates.to_csv(
            ROOT / f"duplicate_transfer_test_votes{minimum_votes}.csv",
            index=False,
            encoding="utf-8-sig",
        )


if __name__ == "__main__":
    main()
