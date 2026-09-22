from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold


ROOT = Path(__file__).resolve().parent
PERF = ROOT / "성능파일"
HIGH = ROOT / "640_file"
OUT = ROOT / "qwen35_640_ensemble"
OUT.mkdir(parents=True, exist_ok=True)

LETTERS = np.asarray(list("abcd"))
LETTER_TO_INDEX = {letter: index for index, letter in enumerate(LETTERS)}
PROB_COLUMNS = [f"prob_{letter}" for letter in LETTERS]
EPS = 1e-12


def load(path: Path) -> pd.DataFrame:
    assert path.exists(), path
    frame = pd.read_csv(path)
    assert frame["id"].is_unique, path
    return frame


def align(reference: pd.DataFrame, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = reference[["id"]].merge(
        frame[["id", *columns]], on="id", how="left", validate="one_to_one"
    )
    assert not result[columns].isna().any().any()
    return result


def probabilities(frame: pd.DataFrame) -> np.ndarray:
    values = frame[PROB_COLUMNS].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all() and (values >= 0).all()
    return values / values.sum(axis=1, keepdims=True)


def blend(probability_list: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    score = np.zeros_like(probability_list[0], dtype=np.float64)
    for probability, weight in zip(probability_list, weights):
        score += float(weight) * np.log(np.clip(probability, EPS, 1.0))
    score -= score.max(axis=1, keepdims=True)
    values = np.exp(score)
    return values / values.sum(axis=1, keepdims=True)


def simplex_weights(n_models: int, step: float) -> np.ndarray:
    units = int(round(1.0 / step))
    rows = []
    for cuts in itertools.combinations(range(units + n_models - 1), n_models - 1):
        augmented = (-1, *cuts, units + n_models - 1)
        counts = [augmented[i + 1] - augmented[i] - 1 for i in range(n_models)]
        rows.append(np.asarray(counts, dtype=np.float64) / units)
    result = np.stack(rows)
    assert np.allclose(result.sum(axis=1), 1.0)
    return result


def candidate_predictions(probability_list: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    logs = np.stack(
        [np.log(np.clip(probability, EPS, 1.0)) for probability in probability_list], axis=0
    )
    scores = np.einsum("cm,mnk->cnk", weights, logs, optimize=True)
    return scores.argmax(axis=2).astype(np.int8)


valid8 = load(PERF / "qwen3_vl_8b_valid.csv")
test8 = load(PERF / "qwen3_vl_8b_test.csv")
paired27 = load(PERF / "paired_counting_valid_analysis.csv")
test27 = load(PERF / "qwen38_adapter_counting_test.csv")
valid512 = load(PERF / "qwen35_tuned_valid.csv")
test512 = load(PERF / "qwen35_tuned_test.csv")
valid640 = {
    epoch: load(HIGH / f"qwen35_640_valid_epoch{epoch}.csv") for epoch in (1, 2, 3)
}
test640 = {
    epoch: load(HIGH / f"qwen35_640_epoch{epoch}_test.csv") for epoch in (1, 2, 3)
}

for frame in [valid512, *valid640.values()]:
    assert frame["id"].tolist() == valid8["id"].tolist()
    assert frame["answer"].tolist() == valid8["answer"].tolist()
for frame in [test512, *test640.values()]:
    assert frame["id"].tolist() == test8["id"].tolist()

gold = valid8["answer"].map(LETTER_TO_INDEX).to_numpy(dtype=np.int8)
categories = valid8["category"].astype(str).to_numpy()
test_categories = test8["category"].astype(str).to_numpy()
is_count = categories == "counting"
test_is_count = test_categories == "counting"

p8v, p8t = probabilities(valid8), probabilities(test8)
p512v, p512t = probabilities(valid512), probabilities(test512)
p640v = {epoch: probabilities(frame) for epoch, frame in valid640.items()}
p640t = {epoch: probabilities(frame) for epoch, frame in test640.items()}

aligned27v = align(
    valid8.loc[is_count],
    paired27.rename(columns={f"prob_{letter}_27b": f"prob_{letter}" for letter in LETTERS}),
    PROB_COLUMNS,
)
aligned27t = align(test8.loc[test_is_count], test27, PROB_COLUMNS)
p27v, p27t = probabilities(aligned27v), probabilities(aligned27t)

# Old 0.92629 ensemble, the 0.93732 equal blend, and the actual public-LB
# 0.93969 anchor. The winning anchor uses 70/30 old-current/Q35-512 for
# counting and 50/50 for every non-counting category.
pcurrentv = p8v.copy()
pcurrentt = p8t.copy()
pcurrentv[is_count] = blend([p8v[is_count], p27v], np.asarray([0.51, 0.49]))
pcurrentt[test_is_count] = blend([p8t[test_is_count], p27t], np.asarray([0.51, 0.49]))
pequalv = blend([pcurrentv, p512v], np.asarray([0.50, 0.50]))
pequalt = blend([pcurrentt, p512t], np.asarray([0.50, 0.50]))
panchorv = pequalv.copy()
panchort = pequalt.copy()
panchorv[is_count] = blend(
    [pcurrentv[is_count], p512v[is_count]], np.asarray([0.70, 0.30])
)
panchort[test_is_count] = blend(
    [pcurrentt[test_is_count], p512t[test_is_count]], np.asarray([0.70, 0.30])
)

model_valid = {
    "lb93969_anchor": panchorv,
    "lb93732_equal": pequalv,
    "old_current": pcurrentv,
    "q35_512_e3": p512v,
    "q35_640_e1": p640v[1],
    "q35_640_e2": p640v[2],
    "q35_640_e3": p640v[3],
}
model_test = {
    "lb93969_anchor": panchort,
    "lb93732_equal": pequalt,
    "old_current": pcurrentt,
    "q35_512_e3": p512t,
    "q35_640_e1": p640t[1],
    "q35_640_e2": p640t[2],
    "q35_640_e3": p640t[3],
}
prediction_valid = {name: value.argmax(axis=1) for name, value in model_valid.items()}
prediction_test = {name: value.argmax(axis=1) for name, value in model_test.items()}

diagnostics = []
for name, prediction in prediction_valid.items():
    diagnostics.append({
        "model": name,
        "overall": int((prediction == gold).sum()),
        **{
            category: int((prediction[categories == category] == gold[categories == category]).sum())
            for category in sorted(set(categories))
        },
    })

oracle_sets = {
    "anchor+640e2": ["lb93969_anchor", "q35_640_e2"],
    "anchor+640all": ["lb93969_anchor", "q35_640_e1", "q35_640_e2", "q35_640_e3"],
    "512+640all": ["q35_512_e3", "q35_640_e1", "q35_640_e2", "q35_640_e3"],
    "all": list(model_valid),
}
oracle_rows = []
for label, names in oracle_sets.items():
    correct = np.zeros(len(gold), dtype=bool)
    for name in names:
        correct |= prediction_valid[name] == gold
    oracle_rows.append({
        "models": label,
        "overall": int(correct.sum()),
        **{
            category: int(correct[categories == category].sum())
            for category in sorted(set(categories))
        },
    })

# Two-model line search around the confirmed leaderboard winner.
two_model_rows = []
two_model_probabilities = {}
for w640 in np.arange(0.0, 1.0001, 0.025):
    valid_probability = blend([panchorv, p640v[2]], np.asarray([1.0 - w640, w640]))
    test_probability = blend([panchort, p640t[2]], np.asarray([1.0 - w640, w640]))
    prediction = valid_probability.argmax(axis=1)
    test_prediction = test_probability.argmax(axis=1)
    key = round(float(w640), 3)
    two_model_probabilities[key] = (valid_probability, test_probability)
    two_model_rows.append({
        "w640": key,
        "wbase": round(1.0 - float(w640), 3),
        "overall": int((prediction == gold).sum()),
        **{
            category: int((prediction[categories == category] == gold[categories == category]).sum())
            for category in sorted(set(categories))
        },
        "test_changes_vs_anchor": int((test_prediction != prediction_test["lb93969_anchor"]).sum()),
    })
two_model_table = pd.DataFrame(two_model_rows)
best_two_score = int(two_model_table["overall"].max())
best_two_weight = float(
    two_model_table[two_model_table["overall"] == best_two_score]
    .assign(distance=lambda x: (x["w640"] - 0.5).abs())
    .sort_values(["distance", "w640"]).iloc[0]["w640"]
)
best_two_valid, best_two_test = two_model_probabilities[best_two_weight]

# Nested-style repeated CV for the two-model weight. Each fold chooses its
# weight using only the other four folds; ties stay closer to the LB anchor.
two_weights = two_model_table["w640"].to_numpy(dtype=np.float64)
two_candidate_predictions = np.stack([
    two_model_probabilities[float(weight)][0].argmax(axis=1) for weight in two_weights
])
two_cv_rows = []
two_selected_weights = []
two_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=642)
for fold_number, (train_index, validation_index) in enumerate(
    two_splitter.split(np.zeros(len(gold)), categories)
):
    correct_train = (
        two_candidate_predictions[:, train_index] == gold[train_index][None, :]
    ).sum(axis=1)
    best = np.flatnonzero(correct_train == correct_train.max())
    selected = int(best[np.argmin(two_weights[best])])
    two_selected_weights.append(float(two_weights[selected]))
    fold_prediction = two_candidate_predictions[selected, validation_index]
    two_cv_rows.append({
        "repeat": fold_number // 5,
        "fold": fold_number % 5,
        "baseline_correct": int(
            (prediction_valid["lb93969_anchor"][validation_index] == gold[validation_index]).sum()
        ),
        "blend_correct": int((fold_prediction == gold[validation_index]).sum()),
        "selected_w640": float(two_weights[selected]),
    })
two_cv_folds = pd.DataFrame(two_cv_rows)
two_cv_repeats = two_cv_folds.groupby("repeat")[["baseline_correct", "blend_correct"]].sum()
two_cv_mean_weight = float(np.mean(two_selected_weights))
two_cv_consensus_weight = float(two_weights[np.argmin(np.abs(two_weights - two_cv_mean_weight))])
two_cv_consensus_valid, two_cv_consensus_test = two_model_probabilities[two_cv_consensus_weight]

# Repeated CV over base + all three 640 epochs.
cv_model_names = ["lb93969_anchor", "q35_640_e1", "q35_640_e2", "q35_640_e3"]
weights = simplex_weights(4, 0.05)
candidate_valid_predictions = candidate_predictions(
    [model_valid[name] for name in cv_model_names], weights
)
target_weight = np.asarray([1.0, 0.0, 0.0, 0.0])
splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=640)
cv_rows = []
selected_weight_indices = []
for fold_number, (train_index, validation_index) in enumerate(
    splitter.split(np.zeros(len(gold)), categories)
):
    correct_train = (
        candidate_valid_predictions[:, train_index] == gold[train_index][None, :]
    ).sum(axis=1)
    best = np.flatnonzero(correct_train == correct_train.max())
    penalty = ((weights[best] - target_weight[None, :]) ** 2).sum(axis=1)
    selected = int(best[np.argmin(penalty)])
    selected_weight_indices.append(selected)
    fold_prediction = candidate_valid_predictions[selected, validation_index]
    cv_rows.append({
        "repeat": fold_number // 5,
        "fold": fold_number % 5,
        "baseline_correct": int(
            (prediction_valid["lb93969_anchor"][validation_index] == gold[validation_index]).sum()
        ),
        "blend_correct": int((fold_prediction == gold[validation_index]).sum()),
        **{f"w_{name}": weights[selected, index] for index, name in enumerate(cv_model_names)},
    })
cv_folds = pd.DataFrame(cv_rows)
cv_repeats = cv_folds.groupby("repeat")[["baseline_correct", "blend_correct"]].sum()
mean_weights = weights[np.asarray(selected_weight_indices)].mean(axis=0)
consensus_index = int(np.argmin(((weights - mean_weights[None, :]) ** 2).sum(axis=1)))
consensus_weights = weights[consensus_index]
consensus_valid = blend([model_valid[name] for name in cv_model_names], consensus_weights)
consensus_test = blend([model_test[name] for name in cv_model_names], consensus_weights)

# Category router and leakage-controlled category-router CV.
router_names = ["lb93969_anchor", "q35_640_e1", "q35_640_e2", "q35_640_e3"]
router = {}
router_rows = []
for category in sorted(set(categories)):
    mask = categories == category
    scores = {
        name: int((prediction_valid[name][mask] == gold[mask]).sum()) for name in router_names
    }
    best_score = max(scores.values())
    selected = next(name for name in router_names if scores[name] == best_score)
    router[category] = selected
    router_rows.append({"category": category, **scores, "selected": selected})

router_valid_prediction = np.asarray([
    prediction_valid[router[category]][index] for index, category in enumerate(categories)
])
router_test_prediction = np.asarray([
    prediction_test[router[category]][index] for index, category in enumerate(test_categories)
])

router_cv_rows = []
router_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=641)
for fold_number, (train_index, validation_index) in enumerate(
    router_splitter.split(np.zeros(len(gold)), categories)
):
    fold_prediction = prediction_valid["lb93969_anchor"][validation_index].copy()
    selections = {}
    for category in sorted(set(categories)):
        category_train = train_index[categories[train_index] == category]
        scores = {
            name: int((prediction_valid[name][category_train] == gold[category_train]).sum())
            for name in router_names
        }
        best_score = max(scores.values())
        selected = next(name for name in router_names if scores[name] == best_score)
        selections[category] = selected
        local_mask = categories[validation_index] == category
        fold_prediction[local_mask] = prediction_valid[selected][validation_index[local_mask]]
    router_cv_rows.append({
        "repeat": fold_number // 5,
        "fold": fold_number % 5,
        "baseline_correct": int(
            (prediction_valid["lb93969_anchor"][validation_index] == gold[validation_index]).sum()
        ),
        "router_correct": int((fold_prediction == gold[validation_index]).sum()),
        **{f"selected_{category}": selections[category] for category in sorted(set(categories))},
    })
router_cv_folds = pd.DataFrame(router_cv_rows)
router_cv_repeats = router_cv_folds.groupby("repeat")[[
    "baseline_correct", "router_correct"
]].sum()


def duplicate_override(prediction: np.ndarray) -> np.ndarray:
    override = load(ROOT / "submission_8b_duplicate_retrieval.csv")
    aligned = test8[["id"]].merge(override, on="id", how="left", validate="one_to_one")
    override_prediction = aligned["answer"].map(LETTER_TO_INDEX).to_numpy()
    # Retrieval artifact was defined against the plain 8B submission.
    intervention = override_prediction != p8t.argmax(axis=1)
    result = prediction.copy()
    result[intervention] = override_prediction[intervention]
    assert intervention.sum() == 3
    return result


# Ensure the reconstructed probability anchor exactly matches the submitted
# 0.93969 file after the same three duplicate-retrieval overrides.
known_anchor = load(
    ROOT / "qwen35_final_ensemble" / "submission_q35_countw30_noncountw50_duplicate.csv"
)
reconstructed_anchor = LETTERS[duplicate_override(prediction_test["lb93969_anchor"])]
assert reconstructed_anchor.tolist() == known_anchor["answer"].tolist()


def save_submission(name: str, prediction: np.ndarray) -> str:
    prediction = duplicate_override(prediction)
    frame = pd.DataFrame({"id": test8["id"], "answer": LETTERS[prediction]})
    assert len(frame) == len(test8) and frame["id"].tolist() == test8["id"].tolist()
    assert frame["answer"].isin(LETTERS).all()
    path = OUT / f"submission_{name}.csv"
    frame.to_csv(path, index=False)
    return str(path)


submission_paths = {
    "640_epoch2": save_submission("q35_640_epoch2_duplicate", prediction_test["q35_640_e2"]),
    "base_plus_640_best_two": save_submission(
        f"lb93969_plus_640e2_w{best_two_weight:.3f}_duplicate", best_two_test.argmax(axis=1)
    ),
    "base_plus_640_two_cv_consensus": save_submission(
        f"lb93969_plus_640e2_cv_w{two_cv_consensus_weight:.3f}_duplicate",
        two_cv_consensus_test.argmax(axis=1),
    ),
    "cv_consensus": save_submission("base_640epochs_cv_consensus_duplicate", consensus_test.argmax(axis=1)),
    "category_router": save_submission("base_640epochs_category_router_duplicate", router_test_prediction),
}

summary = {
    "leaderboard_anchor": 0.93969,
    "diagnostics": diagnostics,
    "oracles": oracle_rows,
    "two_model": {
        "best_valid_score": best_two_score,
        "best_w640": best_two_weight,
        "test_changes_vs_anchor": int(
            (best_two_test.argmax(axis=1) != prediction_test["lb93969_anchor"]).sum()
        ),
        "cv_baseline_mean": float(two_cv_repeats["baseline_correct"].mean()),
        "cv_blend_mean": float(two_cv_repeats["blend_correct"].mean()),
        "cv_gain_mean": float(
            (two_cv_repeats["blend_correct"] - two_cv_repeats["baseline_correct"]).mean()
        ),
        "cv_gain_min": int(
            (two_cv_repeats["blend_correct"] - two_cv_repeats["baseline_correct"]).min()
        ),
        "cv_gain_max": int(
            (two_cv_repeats["blend_correct"] - two_cv_repeats["baseline_correct"]).max()
        ),
        "cv_mean_selected_w640": two_cv_mean_weight,
        "cv_consensus_w640": two_cv_consensus_weight,
        "cv_consensus_valid_score": int(
            (two_cv_consensus_valid.argmax(axis=1) == gold).sum()
        ),
    },
    "cv": {
        "baseline_mean": float(cv_repeats["baseline_correct"].mean()),
        "blend_mean": float(cv_repeats["blend_correct"].mean()),
        "gain_mean": float((cv_repeats["blend_correct"] - cv_repeats["baseline_correct"]).mean()),
        "gain_min": int((cv_repeats["blend_correct"] - cv_repeats["baseline_correct"]).min()),
        "gain_max": int((cv_repeats["blend_correct"] - cv_repeats["baseline_correct"]).max()),
        "model_names": cv_model_names,
        "mean_selected_weights": mean_weights.tolist(),
        "consensus_weights": consensus_weights.tolist(),
        "consensus_valid_score": int((consensus_valid.argmax(axis=1) == gold).sum()),
        "consensus_test_changes_vs_anchor": int(
            (consensus_test.argmax(axis=1) != prediction_test["lb93969_anchor"]).sum()
        ),
    },
    "category_router": {
        "selection": router,
        "valid_score": int((router_valid_prediction == gold).sum()),
        "test_changes_vs_anchor": int(
            (router_test_prediction != prediction_test["lb93969_anchor"]).sum()
        ),
        "cv_baseline_mean": float(router_cv_repeats["baseline_correct"].mean()),
        "cv_router_mean": float(router_cv_repeats["router_correct"].mean()),
        "cv_gain_mean": float(
            (router_cv_repeats["router_correct"] - router_cv_repeats["baseline_correct"]).mean()
        ),
        "cv_gain_min": int(
            (router_cv_repeats["router_correct"] - router_cv_repeats["baseline_correct"]).min()
        ),
        "cv_gain_max": int(
            (router_cv_repeats["router_correct"] - router_cv_repeats["baseline_correct"]).max()
        ),
        "cv_selection_counts": {
            category: router_cv_folds[f"selected_{category}"].value_counts().to_dict()
            for category in sorted(set(categories))
        },
    },
    "submission_paths": submission_paths,
}

pd.DataFrame(diagnostics).to_csv(OUT / "diagnostics.csv", index=False)
pd.DataFrame(oracle_rows).to_csv(OUT / "oracles.csv", index=False)
two_model_table.to_csv(OUT / "base_640e2_weight_grid.csv", index=False)
two_cv_folds.to_csv(OUT / "base_640e2_cv_folds.csv", index=False)
two_cv_repeats.to_csv(OUT / "base_640e2_cv_repeats.csv")
cv_folds.to_csv(OUT / "repeated_cv_folds.csv", index=False)
cv_repeats.to_csv(OUT / "repeated_cv_repeats.csv")
pd.DataFrame(router_rows).to_csv(OUT / "category_router.csv", index=False)
router_cv_folds.to_csv(OUT / "category_router_cv_folds.csv", index=False)
router_cv_repeats.to_csv(OUT / "category_router_cv_repeats.csv")
with (OUT / "summary.json").open("w", encoding="utf-8") as file:
    json.dump(summary, file, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
