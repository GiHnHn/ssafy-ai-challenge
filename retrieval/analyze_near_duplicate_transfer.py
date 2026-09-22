"""Evaluate question-aware answer transfer from perceptually similar images."""

from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from analyze_duplicate_transfer import (
    LETTERS,
    VALID_SIZE,
    evaluate,
    make_references,
    normalize,
)


import os
ROOT = Path(os.environ.get("SSAFY_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
BYTE_POPCOUNT = np.asarray(
    [bin(value).count("1") for value in range(256)], dtype=np.uint8
)


def attach_hashes(frame, split, cache):
    hashes = cache.loc[cache["split"] == split, ["id", "sha1", "dhash"]].copy()
    hashes["dhash"] = hashes["dhash"].map(int)
    result = frame.merge(hashes, on="id", how="left", validate="one_to_one")
    assert result[["sha1", "dhash"]].notna().all().all()
    return result


def hamming_distances(query, references):
    values = np.asarray(references, dtype=np.uint64)
    xor_values = np.bitwise_xor(values, np.uint64(query))
    xor_bytes = xor_values.view(np.uint8).reshape(-1, 8)
    return BYTE_POPCOUNT[xor_bytes].sum(axis=1)


def near_transfer_candidates(queries, references, maximum_distance):
    reference_records = references.to_dict("records")
    answer_index = defaultdict(list)
    for index, ref in enumerate(reference_records):
        answer_index[normalize(ref["answer_text"])].append(index)

    output = []
    for _, query in tqdm(
        queries.iterrows(), total=len(queries), desc=f"near transfer d<={maximum_distance}"
    ):
        query_question = normalize(query["question"])
        option_to_letter = {normalize(query[letter]): letter for letter in LETTERS}
        candidate_indices = sorted(
            {
                index
                for answer_text in option_to_letter
                for index in answer_index.get(answer_text, [])
            }
        )
        best = None
        close_count = 0
        if candidate_indices:
            candidate_hashes = [
                reference_records[index]["dhash"] for index in candidate_indices
            ]
            distances = hamming_distances(int(query["dhash"]), candidate_hashes)
            close_positions = np.flatnonzero(distances <= maximum_distance)
            close_count = len(close_positions)
            ranked = []
            for position in close_positions:
                ref = reference_records[candidate_indices[int(position)]]
                similarity = SequenceMatcher(
                    None, query_question, ref["question_norm"]
                ).ratio()
                ranked.append(
                    {
                        **ref,
                        "image_distance": int(distances[int(position)]),
                        "similarity": similarity,
                        "pred": option_to_letter[normalize(ref["answer_text"])],
                    }
                )
            ranked.sort(
                key=lambda item: (
                    item["similarity"],
                    -item["image_distance"],
                    item["votes"],
                    item["source"] == "train",
                ),
                reverse=True,
            )
            best = ranked[0] if ranked else None
        output.append(
            {
                "id": query["id"],
                "retrieval_pred": None if best is None else best["pred"],
                "similarity": None if best is None else best["similarity"],
                "image_distance": None if best is None else best["image_distance"],
                "reference_source": None if best is None else best["source"],
                "reference_id": None if best is None else best["id"],
                "reference_question": None if best is None else best["question"],
                "reference_answer_text": None if best is None else best["answer_text"],
                "candidate_count": close_count,
            }
        )
    return pd.DataFrame(output)


def compact_evaluation(valid, baseline, candidates, maximum_distance):
    joined = valid.merge(candidates, on="id", how="left").merge(
        baseline[["id", "pred"]].rename(columns={"pred": "baseline_pred"}),
        on="id",
        how="left",
    )
    base_correct = joined["baseline_pred"] == joined["answer"]
    rows = []
    for threshold in [1.0, 0.98, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
        use = joined["similarity"].fillna(-1) >= threshold
        predictions = joined["baseline_pred"].where(~use, joined["retrieval_pred"])
        correct = predictions == joined["answer"]
        rows.append(
            {
                "max_dhash_distance": maximum_distance,
                "min_question_similarity": threshold,
                "coverage": int(use.sum()),
                "retrieval_correct": int(
                    (joined.loc[use, "retrieval_pred"] == joined.loc[use, "answer"]).sum()
                ),
                "hybrid_correct": int(correct.sum()),
                "fixes": int((use & ~base_correct & correct).sum()),
                "breaks": int((use & base_correct & ~correct).sum()),
            }
        )
    return pd.DataFrame(rows), joined


def main():
    cache = pd.read_csv(ROOT / "image_hash_cache.csv", dtype={"dhash": str})
    train = attach_hashes(pd.read_csv(ROOT / "train.csv"), "train", cache)
    dev = attach_hashes(pd.read_csv(ROOT / "dev.csv"), "dev", cache)
    test = attach_hashes(pd.read_csv(ROOT / "test.csv"), "test", cache)
    baseline_valid = pd.read_csv(ROOT / "성능파일" / "qwen3_vl_8b_valid.csv")
    baseline_test = pd.read_csv(ROOT / "성능파일" / "qwen3_vl_8b_test.csv")

    train_reference = train.iloc[:-VALID_SIZE].copy()
    valid = train.iloc[-VALID_SIZE:].copy()
    valid_references = make_references(train_reference, dev, minimum_dev_votes=3)

    result_frames = []
    valid_candidate_frames = {}
    for maximum_distance in (0, 1, 2, 4, 6, 8):
        candidates = near_transfer_candidates(
            valid, valid_references, maximum_distance
        )
        metrics, joined = compact_evaluation(
            valid, baseline_valid, candidates, maximum_distance
        )
        result_frames.append(metrics)
        valid_candidate_frames[maximum_distance] = candidates

    results = pd.concat(result_frames, ignore_index=True)
    print("\n=== perceptual duplicate transfer validation grid ===")
    print(results.to_string(index=False))
    print("\nBest rows by hybrid score, then fewer interventions:")
    print(
        results.sort_values(
            ["hybrid_correct", "coverage", "breaks"],
            ascending=[False, True, True],
        ).head(12).to_string(index=False)
    )
    results.to_csv(
        ROOT / "near_duplicate_transfer_grid.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # For test, all labeled train rows are legitimate references.
    test_references = make_references(train, dev, minimum_dev_votes=3)
    for maximum_distance in (0, 2, 4, 6, 8):
        candidates = near_transfer_candidates(test, test_references, maximum_distance)
        joined = baseline_test[["id", "pred", "category"]].merge(
            candidates, on="id", how="left"
        )
        print(f"\ntest dHash distance <= {maximum_distance}:")
        for threshold in (1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70):
            use = joined["similarity"].fillna(-1) >= threshold
            changed = use & (joined["pred"] != joined["retrieval_pred"])
            print(
                f"  sim>={threshold:.2f}: coverage={int(use.sum())}, "
                f"changed={int(changed.sum())}"
            )
        candidates.to_csv(
            ROOT / f"near_duplicate_transfer_test_d{maximum_distance}.csv",
            index=False,
            encoding="utf-8-sig",
        )


if __name__ == "__main__":
    main()
