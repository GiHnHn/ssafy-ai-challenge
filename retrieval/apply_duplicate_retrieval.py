"""Apply the validated duplicate-image retrieval gate to test predictions."""

import argparse
from pathlib import Path

import pandas as pd


import os
ROOT = Path(os.environ.get("SSAFY_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "성능파일" / "qwen3_vl_8b_test.csv",
        help="CSV containing id and either pred or answer",
    )
    parser.add_argument(
        "--retrieval",
        type=Path,
        default=ROOT / "near_duplicate_transfer_test_d0.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "submission_8b_duplicate_retrieval.csv",
    )
    parser.add_argument("--minimum-similarity", type=float, default=0.70)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    prediction_column = "answer" if "answer" in predictions.columns else "pred"
    assert prediction_column in predictions.columns
    retrieval = pd.read_csv(args.retrieval)
    required = {"id", "retrieval_pred", "similarity", "image_distance"}
    assert required.issubset(retrieval.columns)

    joined = predictions[["id", prediction_column]].rename(
        columns={prediction_column: "original_pred"}
    ).merge(retrieval[list(required)], on="id", how="left", validate="one_to_one")
    use = (
        joined["retrieval_pred"].notna()
        & (joined["similarity"] >= args.minimum_similarity)
        & (joined["image_distance"] == 0)
    )
    joined["answer"] = joined["original_pred"].where(
        ~use, joined["retrieval_pred"]
    )
    assert len(joined) == len(predictions)
    assert joined["id"].tolist() == predictions["id"].tolist()
    assert joined["answer"].isin(list("abcd")).all()

    changed = joined.loc[
        use & (joined["original_pred"] != joined["answer"]),
        [
            "id", "original_pred", "answer", "similarity", "image_distance"
        ],
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joined[["id", "answer"]].to_csv(
        args.output, index=False, encoding="utf-8-sig"
    )
    print(f"retrieval gate coverage: {int(use.sum())}/{len(joined)}")
    print(f"changed answers: {len(changed)}")
    print(changed.to_string(index=False))
    print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
