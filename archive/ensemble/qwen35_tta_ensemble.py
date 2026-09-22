from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

import qwen35_640_ensemble as base


ROOT = Path(__file__).resolve().parent
TTA = ROOT / "TTA_file"
OUT = ROOT / "qwen35_tta_ensemble"
OUT.mkdir(parents=True, exist_ok=True)

TTA_NAMES = [
    "flip640",
    "original768",
    "flip768",
    "tta640",
    "multiscale_original",
    "tta768",
    "full_tta",
]


def align_probability(path: Path, reference: pd.DataFrame) -> np.ndarray:
    frame = base.load(path)
    aligned = base.align(reference, frame, base.PROB_COLUMNS)
    return base.probabilities(aligned)


tta_valid = {
    name: align_probability(TTA / f"qwen35_tta_{name}_valid.csv", base.valid8)
    for name in TTA_NAMES
}
tta_test = {
    name: align_probability(TTA / f"qwen35_tta_{name}_test.csv", base.test8)
    for name in TTA_NAMES
}

# The public-LB 0.94599 anchor is exactly 50% of the old/512 equal blend and
# 50% of the fresh 640 epoch-2 model (geometric/log-probability blend).
pwinv = base.blend([base.pequalv, base.p640v[2]], np.asarray([0.50, 0.50]))
pwint = base.blend([base.pequalt, base.p640t[2]], np.asarray([0.50, 0.50]))
win_valid_prediction = pwinv.argmax(axis=1)
win_test_prediction = pwint.argmax(axis=1)

known_winner = base.load(
    base.OUT / "submission_lb93732_plus_640e2_w0.500_duplicate.csv"
)
reconstructed = base.LETTERS[base.duplicate_override(win_test_prediction)]
assert reconstructed.tolist() == known_winner["answer"].tolist()

# The TTA notebook reused this exact source file for original640.
tta_original640_valid = align_probability(
    TTA / "qwen35_tta_original640_valid.csv", base.valid8
)
tta_original640_test = align_probability(
    TTA / "qwen35_tta_original640_test.csv", base.test8
)
assert np.allclose(tta_original640_valid, base.p640v[2], atol=1e-7)
assert np.allclose(tta_original640_test, base.p640t[2], atol=1e-7)


def score(probability: np.ndarray) -> dict[str, int]:
    prediction = probability.argmax(axis=1)
    return {
        "overall": int((prediction == base.gold).sum()),
        **{
            category: int(
                (
                    prediction[base.categories == category]
                    == base.gold[base.categories == category]
                ).sum()
            )
            for category in sorted(set(base.categories))
        },
    }


def oracle_score(probabilities: list[np.ndarray]) -> dict[str, int]:
    correct = np.zeros(len(base.gold), dtype=bool)
    for probability in probabilities:
        correct |= probability.argmax(axis=1) == base.gold
    return {
        "overall": int(correct.sum()),
        **{
            category: int(correct[base.categories == category].sum())
            for category in sorted(set(base.categories))
        },
    }


def final_answers(probability: np.ndarray) -> np.ndarray:
    return base.LETTERS[base.duplicate_override(probability.argmax(axis=1))]


def save_submission(name: str, probability: np.ndarray) -> str:
    answers = final_answers(probability)
    frame = pd.DataFrame({"id": base.test8["id"], "answer": answers})
    assert len(frame) == 5074 and frame["id"].is_unique
    path = OUT / f"submission_{name}.csv"
    frame.to_csv(path, index=False)
    return str(path)


def changes_from_winner(probability: np.ndarray) -> int:
    return int((final_answers(probability) != known_winner["answer"].to_numpy()).sum())


# Diagnostics and two-model anchor/TTA line searches.
diagnostics = [{"model": "lb94599_anchor", **score(pwinv)}]
for name in TTA_NAMES:
    diagnostics.append({"model": name, **score(tta_valid[name])})

weight_grid = np.round(np.arange(0.0, 1.0001, 0.025), 3)
splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=768)
splits = list(splitter.split(np.zeros(len(base.gold)), base.categories))
two_model_rows = []
two_model_cv_rows = []
two_model_results = {}

for name in TTA_NAMES:
    valid_candidates = [
        base.blend([pwinv, tta_valid[name]], np.asarray([1.0 - w, w]))
        for w in weight_grid
    ]
    test_candidates = [
        base.blend([pwint, tta_test[name]], np.asarray([1.0 - w, w]))
        for w in weight_grid
    ]
    predictions = np.stack([value.argmax(axis=1) for value in valid_candidates])
    full_scores = (predictions == base.gold[None, :]).sum(axis=1)
    best_indices = np.flatnonzero(full_scores == full_scores.max())
    # Among validation ties, keep the smallest intervention from the confirmed winner.
    best_index = int(best_indices[0])

    selected_indices = []
    fold_rows = []
    for fold_number, (train_index, validation_index) in enumerate(splits):
        train_scores = (
            predictions[:, train_index] == base.gold[train_index][None, :]
        ).sum(axis=1)
        fold_best = np.flatnonzero(train_scores == train_scores.max())
        selected = int(fold_best[0])
        selected_indices.append(selected)
        selected_prediction = predictions[selected, validation_index]
        fold_rows.append({
            "repeat": fold_number // 5,
            "fold": fold_number % 5,
            "baseline_correct": int(
                (win_valid_prediction[validation_index] == base.gold[validation_index]).sum()
            ),
            "blend_correct": int(
                (selected_prediction == base.gold[validation_index]).sum()
            ),
            "selected_w_tta": float(weight_grid[selected]),
        })

    folds = pd.DataFrame(fold_rows)
    repeats = folds.groupby("repeat")[["baseline_correct", "blend_correct"]].sum()
    gains = repeats["blend_correct"] - repeats["baseline_correct"]
    mean_weight = float(weight_grid[np.asarray(selected_indices)].mean())
    consensus_index = int(np.argmin(np.abs(weight_grid - mean_weight)))
    consensus_valid = valid_candidates[consensus_index]
    consensus_test = test_candidates[consensus_index]
    two_model_results[name] = {
        "best_valid": valid_candidates[best_index],
        "best_test": test_candidates[best_index],
        "consensus_valid": consensus_valid,
        "consensus_test": consensus_test,
        "best_weight": float(weight_grid[best_index]),
        "consensus_weight": float(weight_grid[consensus_index]),
    }
    two_model_rows.append({
        "tta": name,
        "oracle_with_anchor": oracle_score([pwinv, tta_valid[name]])["overall"],
        "best_valid_score": int(full_scores[best_index]),
        "best_w_tta": float(weight_grid[best_index]),
        "cv_gain_mean": float(gains.mean()),
        "cv_gain_min": int(gains.min()),
        "cv_gain_max": int(gains.max()),
        "cv_positive_repeats": int((gains > 0).sum()),
        "cv_zero_repeats": int((gains == 0).sum()),
        "cv_negative_repeats": int((gains < 0).sum()),
        "cv_mean_selected_w_tta": mean_weight,
        "cv_consensus_w_tta": float(weight_grid[consensus_index]),
        "cv_consensus_valid_score": score(consensus_valid)["overall"],
        "consensus_test_changes_vs_lb94599": changes_from_winner(consensus_test),
    })
    folds.assign(tta=name).to_csv(OUT / f"cv_folds_{name}.csv", index=False)

two_model_table = pd.DataFrame(two_model_rows).sort_values(
    ["cv_gain_mean", "cv_consensus_valid_score", "best_valid_score"],
    ascending=False,
).reset_index(drop=True)

# Let the confirmed components be optimized directly: equal(old,512), original
# 640, and full TTA. This can replace the 640 branch instead of double-counting it.
three_weights = base.simplex_weights(3, 0.025)
three_valid_predictions = base.candidate_predictions(
    [base.pequalv, base.p640v[2], tta_valid["full_tta"]], three_weights
)
target_weights = np.asarray([0.50, 0.50, 0.00])
full_scores = (three_valid_predictions == base.gold[None, :]).sum(axis=1)
best = np.flatnonzero(full_scores == full_scores.max())
best_index = int(
    best[np.argmin(((three_weights[best] - target_weights[None, :]) ** 2).sum(axis=1))]
)

three_cv_rows = []
three_selected = []
for fold_number, (train_index, validation_index) in enumerate(splits):
    train_scores = (
        three_valid_predictions[:, train_index]
        == base.gold[train_index][None, :]
    ).sum(axis=1)
    fold_best = np.flatnonzero(train_scores == train_scores.max())
    selected = int(
        fold_best[
            np.argmin(
                ((three_weights[fold_best] - target_weights[None, :]) ** 2).sum(axis=1)
            )
        ]
    )
    three_selected.append(selected)
    prediction = three_valid_predictions[selected, validation_index]
    three_cv_rows.append({
        "repeat": fold_number // 5,
        "fold": fold_number % 5,
        "baseline_correct": int(
            (win_valid_prediction[validation_index] == base.gold[validation_index]).sum()
        ),
        "blend_correct": int((prediction == base.gold[validation_index]).sum()),
        "w_equal": float(three_weights[selected, 0]),
        "w_640": float(three_weights[selected, 1]),
        "w_full_tta": float(three_weights[selected, 2]),
    })

three_cv_folds = pd.DataFrame(three_cv_rows)
three_cv_repeats = three_cv_folds.groupby("repeat")[[
    "baseline_correct", "blend_correct"
]].sum()
three_gains = three_cv_repeats["blend_correct"] - three_cv_repeats["baseline_correct"]
mean_three_weights = three_weights[np.asarray(three_selected)].mean(axis=0)
consensus_index = int(
    np.argmin(((three_weights - mean_three_weights[None, :]) ** 2).sum(axis=1))
)
consensus_three_weights = three_weights[consensus_index]
best_three_weights = three_weights[best_index]
best_three_valid = base.blend(
    [base.pequalv, base.p640v[2], tta_valid["full_tta"]], best_three_weights
)
best_three_test = base.blend(
    [base.pequalt, base.p640t[2], tta_test["full_tta"]], best_three_weights
)
consensus_three_valid = base.blend(
    [base.pequalv, base.p640v[2], tta_valid["full_tta"]], consensus_three_weights
)
consensus_three_test = base.blend(
    [base.pequalt, base.p640t[2], tta_test["full_tta"]], consensus_three_weights
)

# Category-conditioned soft blending with the only globally positive-CV view.
# Each fold learns a separate original768 weight from the other four folds;
# the held-out rows are never used to choose their weight.
original768_valid_candidates = [
    base.blend([pwinv, tta_valid["original768"]], np.asarray([1.0 - w, w]))
    for w in weight_grid
]
original768_test_candidates = [
    base.blend([pwint, tta_test["original768"]], np.asarray([1.0 - w, w]))
    for w in weight_grid
]
original768_predictions = np.stack(
    [value.argmax(axis=1) for value in original768_valid_candidates]
)
category_names = sorted(set(base.categories))
category_selected_weights = {category: [] for category in category_names}
category_cv_rows = []
for fold_number, (train_index, validation_index) in enumerate(splits):
    fold_prediction = win_valid_prediction[validation_index].copy()
    row = {"repeat": fold_number // 5, "fold": fold_number % 5}
    for category in category_names:
        category_train = train_index[base.categories[train_index] == category]
        train_scores = (
            original768_predictions[:, category_train]
            == base.gold[category_train][None, :]
        ).sum(axis=1)
        best = np.flatnonzero(train_scores == train_scores.max())
        selected = int(best[0])
        selected_weight = float(weight_grid[selected])
        category_selected_weights[category].append(selected_weight)
        row[f"w_{category}"] = selected_weight
        local = base.categories[validation_index] == category
        fold_prediction[local] = original768_predictions[selected, validation_index[local]]
    row["baseline_correct"] = int(
        (win_valid_prediction[validation_index] == base.gold[validation_index]).sum()
    )
    row["blend_correct"] = int(
        (fold_prediction == base.gold[validation_index]).sum()
    )
    category_cv_rows.append(row)

category_cv_folds = pd.DataFrame(category_cv_rows)
category_cv_repeats = category_cv_folds.groupby("repeat")[[
    "baseline_correct", "blend_correct"
]].sum()
category_gains = (
    category_cv_repeats["blend_correct"] - category_cv_repeats["baseline_correct"]
)
category_consensus_weights = {
    category: float(
        weight_grid[
            np.argmin(
                np.abs(
                    weight_grid
                    - np.mean(category_selected_weights[category])
                )
            )
        ]
    )
    for category in category_names
}
category_soft_valid = pwinv.copy()
category_soft_test = pwint.copy()
for category, weight in category_consensus_weights.items():
    valid_mask = base.categories == category
    test_mask = base.test_categories == category
    category_soft_valid[valid_mask] = base.blend(
        [pwinv[valid_mask], tta_valid["original768"][valid_mask]],
        np.asarray([1.0 - weight, weight]),
    )
    category_soft_test[test_mask] = base.blend(
        [pwint[test_mask], tta_test["original768"][test_mask]],
        np.asarray([1.0 - weight, weight]),
    )

best_tta_name = str(two_model_table.iloc[0]["tta"])
best_tta_result = two_model_results[best_tta_name]

# The 0.200 validation optimum improves over the 0.175 CV consensus on exactly
# one material validation row. On test, using 0.200 globally also flips four
# counting rows, so keep the proven 0.175 blend everywhere except material.
material_bump_valid = best_tta_result["consensus_valid"].copy()
material_bump_test = best_tta_result["consensus_test"].copy()
material_valid_mask = base.categories == "material"
material_test_mask = base.test_categories == "material"
material_bump_valid[material_valid_mask] = best_tta_result["best_valid"][
    material_valid_mask
]
material_bump_test[material_test_mask] = best_tta_result["best_test"][
    material_test_mask
]

paths = {
    "recommended_two_model_cv": save_submission(
        f"lb94599_plus_{best_tta_name}_cv_w{best_tta_result['consensus_weight']:.3f}_duplicate",
        best_tta_result["consensus_test"],
    ),
    "two_model_validation_optimal": save_submission(
        f"lb94599_plus_{best_tta_name}_validbest_w{best_tta_result['best_weight']:.3f}_duplicate",
        best_tta_result["best_test"],
    ),
    "recommended_material_bump": save_submission(
        "lb94836_original768_w0.175_material_w0.200_duplicate",
        material_bump_test,
    ),
    "three_source_cv": save_submission(
        "equal_640_fulltta_cv_"
        f"w{consensus_three_weights[0]:.3f}_{consensus_three_weights[1]:.3f}_"
        f"{consensus_three_weights[2]:.3f}_duplicate",
        consensus_three_test,
    ),
    "three_source_validation_optimal": save_submission(
        "equal_640_fulltta_validbest_"
        f"w{best_three_weights[0]:.3f}_{best_three_weights[1]:.3f}_"
        f"{best_three_weights[2]:.3f}_duplicate",
        best_three_test,
    ),
    "category_soft_original768_cv": save_submission(
        "lb94599_original768_category_soft_cv_duplicate", category_soft_test
    ),
    "full_tta_standalone": save_submission(
        "q35_full_tta_standalone_duplicate", tta_test["full_tta"]
    ),
}

summary = {
    "leaderboard_anchor": 0.94599,
    "anchor_score": score(pwinv),
    "diagnostics": diagnostics,
    "two_model_ranked": two_model_table.to_dict(orient="records"),
    "recommended_two_model": {
        "tta": best_tta_name,
        "consensus_weight": best_tta_result["consensus_weight"],
        "score": score(best_tta_result["consensus_valid"]),
        "test_changes_vs_anchor": changes_from_winner(best_tta_result["consensus_test"]),
    },
    "material_bump_from_lb94836": {
        "weights": {"default": 0.175, "material": 0.200},
        "score": score(material_bump_valid),
        "test_changes_vs_lb94599": changes_from_winner(material_bump_test),
        "test_changes_vs_w0.175": int(
            (
                final_answers(material_bump_test)
                != final_answers(best_tta_result["consensus_test"])
            ).sum()
        ),
    },
    "three_source": {
        "model_order": ["equal_old_512", "640e2", "full_tta"],
        "validation_best_weights": best_three_weights.tolist(),
        "validation_best_score": score(best_three_valid),
        "cv_mean_selected_weights": mean_three_weights.tolist(),
        "cv_consensus_weights": consensus_three_weights.tolist(),
        "cv_consensus_score": score(consensus_three_valid),
        "cv_gain_mean": float(three_gains.mean()),
        "cv_gain_min": int(three_gains.min()),
        "cv_gain_max": int(three_gains.max()),
        "cv_positive_repeats": int((three_gains > 0).sum()),
        "cv_zero_repeats": int((three_gains == 0).sum()),
        "cv_negative_repeats": int((three_gains < 0).sum()),
        "consensus_test_changes_vs_anchor": changes_from_winner(consensus_three_test),
    },
    "category_soft_original768": {
        "cv_consensus_weights": category_consensus_weights,
        "score": score(category_soft_valid),
        "cv_gain_mean": float(category_gains.mean()),
        "cv_gain_min": int(category_gains.min()),
        "cv_gain_max": int(category_gains.max()),
        "cv_positive_repeats": int((category_gains > 0).sum()),
        "cv_zero_repeats": int((category_gains == 0).sum()),
        "cv_negative_repeats": int((category_gains < 0).sum()),
        "test_changes_vs_anchor": changes_from_winner(category_soft_test),
    },
    "submission_paths": paths,
}

pd.DataFrame(diagnostics).to_csv(OUT / "diagnostics.csv", index=False)
two_model_table.to_csv(OUT / "two_model_cv_summary.csv", index=False)
three_cv_folds.to_csv(OUT / "three_source_cv_folds.csv", index=False)
three_cv_repeats.to_csv(OUT / "three_source_cv_repeats.csv")
category_cv_folds.to_csv(OUT / "category_soft_cv_folds.csv", index=False)
category_cv_repeats.to_csv(OUT / "category_soft_cv_repeats.csv")
with (OUT / "summary.json").open("w", encoding="utf-8") as file:
    json.dump(summary, file, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
