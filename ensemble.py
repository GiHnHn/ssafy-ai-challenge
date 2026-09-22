"""Reproduce the recorded final probability blend; no model training is performed."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

LETTERS = np.array(list('abcd'))
PROBS = [f'prob_{x}' for x in LETTERS]

def load(path):
    frame = pd.read_csv(path)
    if 'id' not in frame or frame['id'].isna().any() or not frame['id'].is_unique:
        raise ValueError(f'Unique non-null id required: {path}')
    return frame

def aligned(reference, frame, columns):
    if not set(reference['id']).issubset(set(frame['id'])):
        raise ValueError('Missing prediction IDs')
    result = reference[['id']].merge(frame[['id', *columns]], on='id', how='left', validate='one_to_one')
    if result[columns].isna().any().any():
        raise ValueError('Missing values after ID alignment')
    return result

def probability(frame):
    values = frame[PROBS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values.sum(axis=1) <= 0).any():
        raise ValueError('Invalid probability row')
    return values / values.sum(axis=1, keepdims=True)

def blend(left, right, right_weight):
    score = (1-right_weight)*np.log(np.clip(left, 1e-12, 1)) + right_weight*np.log(np.clip(right, 1e-12, 1))
    score -= score.max(axis=1, keepdims=True)
    result = np.exp(score)
    return result / result.sum(axis=1, keepdims=True)

def reproduce(root):
    base = load(root/'성능파일/qwen3_vl_8b_test.csv')
    if 'category' not in base:
        raise ValueError('8B predictions must include category')
    p8 = probability(base)
    def read(relative):
        return probability(aligned(base, load(root/relative), PROBS))
    current = p8.copy()
    counting = base['category'].eq('counting').to_numpy()
    p27 = probability(aligned(base.loc[counting], load(root/'성능파일/qwen38_adapter_counting_test.csv'), PROBS))
    current[counting] = blend(p8[counting], p27, 0.49)
    equal = blend(current, read('성능파일/qwen35_tuned_test.csv'), 0.50)
    p94599 = blend(equal, read('640_file/qwen35_640_epoch2_test.csv'), 0.50)
    p94836 = blend(p94599, read('TTA_file/qwen35_tta_original768_test.csv'), 0.175)
    refit = blend(read('full_refit_file/qwen35_full_refit_640_test.csv'), read('full_refit_file/qwen35_full_refit_768_test.csv'), 0.175)
    final = blend(p94836, refit, 0.123)
    answers = LETTERS[final.argmax(axis=1)]
    # Preserve the historical gate: only interventions relative to plain 8B
    # are carried to the final ensemble, rather than all eligible references.
    retrieval = load(root/'near_duplicate_transfer_test_d0.csv')
    fields = ['retrieval_pred','similarity','image_distance']
    merged = base[['id']].merge(retrieval[['id', *fields]], on='id', how='left', validate='one_to_one')
    eligible = merged['retrieval_pred'].notna() & merged['similarity'].ge(0.70) & merged['image_distance'].eq(0)
    if not merged.loc[eligible, 'retrieval_pred'].isin(LETTERS).all():
        raise ValueError('Invalid retrieval answer')
    intervention = eligible.to_numpy() & (merged['retrieval_pred'].to_numpy() != LETTERS[p8.argmax(axis=1)])
    answers[intervention] = merged.loc[intervention, 'retrieval_pred'].to_numpy()
    return pd.DataFrame({'id':base['id'], 'answer':answers}), int(intervention.sum())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/submission.csv'))
    args = parser.parse_args()
    submission, interventions = reproduce(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f'rows={len(submission)}, retrieval_interventions={interventions}, output={args.output}')

if __name__ == '__main__':
    main()
