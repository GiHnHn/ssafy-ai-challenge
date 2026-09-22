"""Find exact and low-distance image duplicates across train/dev/test."""

import hashlib
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


import os
ROOT = Path(os.environ.get("SSAFY_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
RESAMPLE_LANCZOS = (
    Image.Resampling.LANCZOS
    if hasattr(Image, "Resampling")
    else Image.LANCZOS
)


def resolve_path(relative_path):
    path = Path(str(relative_path))
    return path if path.is_absolute() else ROOT / path


def sha1_file(path):
    digest = hashlib.sha1()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(path):
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("L").resize(
            (9, 8), RESAMPLE_LANCZOS
        )
    pixels = np.asarray(image, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def cross_split_groups(frame, hash_column):
    groups = []
    for _, group in frame.groupby(hash_column):
        if group["split"].nunique() >= 2:
            groups.append(group[["split", "id", "path", hash_column]])
    return groups


def nearest_distances(query_hashes, reference_hashes):
    reference_values = np.asarray(reference_hashes, dtype=np.uint64)
    byte_popcount = np.asarray(
        [bin(value).count("1") for value in range(256)], dtype=np.uint8
    )
    distances = []
    nearest_indices = []
    for query in tqdm(query_hashes, desc="nearest dHash"):
        xor_values = np.bitwise_xor(reference_values, np.uint64(query))
        xor_bytes = xor_values.view(np.uint8).reshape(-1, 8)
        row_distances = byte_popcount[xor_bytes].sum(axis=1)
        best_index = int(row_distances.argmin())
        best_distance = int(row_distances[best_index])
        distances.append(best_distance)
        nearest_indices.append(best_index)
    return np.asarray(distances), np.asarray(nearest_indices)


def main():
    frames = []
    for split in ("train", "dev", "test"):
        frame = pd.read_csv(ROOT / f"{split}.csv")[["id", "path", "question"]]
        frame["split"] = split
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["absolute_path"] = all_rows["path"].map(resolve_path)
    missing = all_rows.loc[
        ~all_rows["absolute_path"].map(Path.exists), "absolute_path"
    ]
    assert len(missing) == 0, f"Missing images: {missing.head().tolist()}"

    cache_path = ROOT / "image_hash_cache.csv"
    cache = None
    if cache_path.exists():
        candidate = pd.read_csv(cache_path)
        expected_keys = all_rows[["split", "id", "path"]].astype(str)
        candidate_keys = candidate[["split", "id", "path"]].astype(str)
        if len(candidate) == len(all_rows) and candidate_keys.equals(expected_keys):
            cache = candidate

    if cache is None:
        print(f"Hashing {len(all_rows)} images ...")
        all_rows["sha1"] = [
            sha1_file(path)
            for path in tqdm(all_rows["absolute_path"], desc="SHA1")
        ]
    else:
        print(f"Using hash cache: {cache_path}")
        all_rows["sha1"] = cache["sha1"].astype(str).values
    sha_groups = cross_split_groups(all_rows, "sha1")
    print("exact byte-identical cross-split groups:", len(sha_groups))
    for group in sha_groups[:10]:
        print(group.to_string(index=False))

    if cache is None or "dhash" not in cache.columns:
        workers = min(8, os.cpu_count() or 1)
        print(f"Computing dHash with {workers} threads ...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            all_rows["dhash"] = list(
                tqdm(
                    executor.map(dhash, all_rows["absolute_path"]),
                    total=len(all_rows),
                    desc="dHash",
                )
            )
        all_rows[["split", "id", "path", "sha1", "dhash"]].to_csv(
            cache_path, index=False
        )
        print(f"Saved hash cache: {cache_path}")
    else:
        all_rows["dhash"] = cache["dhash"].astype(np.uint64).values
    dhash_groups = cross_split_groups(all_rows, "dhash")
    print("exact dHash cross-split groups:", len(dhash_groups))
    for group in dhash_groups[:10]:
        print(group.to_string(index=False))

    reference = all_rows[all_rows["split"].isin(["train", "dev"])].reset_index(drop=True)
    test = all_rows[all_rows["split"] == "test"].reset_index(drop=True)
    distances, nearest = nearest_distances(
        test["dhash"].tolist(), reference["dhash"].tolist()
    )
    print("test -> train/dev nearest dHash distance:")
    print(pd.Series(distances).describe(percentiles=[0.01, 0.05, 0.1, 0.25]).to_dict())
    print(
        "counts",
        {threshold: int((distances <= threshold).sum()) for threshold in [0, 1, 2, 4, 6, 8]},
    )
    close_indices = np.flatnonzero(distances <= 4)
    for index in close_indices[:30]:
        ref = reference.iloc[nearest[index]]
        query = test.iloc[index]
        print(
            f"distance={distances[index]} | test={query['id']} | "
            f"reference={ref['split']}:{ref['id']}"
        )


if __name__ == "__main__":
    main()
