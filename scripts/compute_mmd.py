"""Deterministic RBF-MMD permutation test on author-supplied embeddings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _load_features(path: Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "features" not in loaded.files:
            raise ValueError(f"{path} must contain an array named 'features'")
        array = loaded["features"]
        loaded.close()
    else:
        array = loaded
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("feature arrays must have shape (n>=2, d>=1)")
    if not np.isfinite(array).all():
        raise ValueError("feature arrays contain non-finite values")
    return array


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairwise_sqdist(values: np.ndarray) -> np.ndarray:
    norms = np.sum(values * values, axis=1, keepdims=True)
    return np.maximum(norms + norms.T - 2.0 * values @ values.T, 0.0)


def median_bandwidth(pooled: np.ndarray) -> float:
    distances = _pairwise_sqdist(pooled)
    upper = distances[np.triu_indices(distances.shape[0], k=1)]
    positive = upper[upper > 0]
    if positive.size == 0:
        raise ValueError("median heuristic is undefined for identical pooled embeddings")
    return float(np.sqrt(np.median(positive)))


def _mmd_from_kernel(kernel: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    return float(
        kernel[np.ix_(left, left)].mean()
        + kernel[np.ix_(right, right)].mean()
        - 2.0 * kernel[np.ix_(left, right)].mean()
    )


def permutation_mmd(reference: np.ndarray, variant: np.ndarray,
                    permutations: int, seed: int) -> dict:
    if reference.shape[1] != variant.shape[1]:
        raise ValueError("reference and variant feature dimensions differ")
    if int(permutations) <= 0:
        raise ValueError("permutations must be positive")
    pooled = np.vstack([reference, variant])
    sigma = median_bandwidth(pooled)
    kernel = np.exp(-_pairwise_sqdist(pooled) / (2.0 * sigma * sigma))
    n_reference = reference.shape[0]
    left = np.arange(n_reference)
    right = np.arange(n_reference, pooled.shape[0])
    observed = _mmd_from_kernel(kernel, left, right)
    rng = np.random.default_rng(int(seed))
    null = np.empty(int(permutations), dtype=np.float64)
    for index in range(int(permutations)):
        order = rng.permutation(pooled.shape[0])
        null[index] = _mmd_from_kernel(kernel, order[:n_reference], order[n_reference:])
    return {
        "mmd2": observed,
        "p_value": float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1)),
        "bandwidth_sigma": sigma,
        "permutations": int(permutations),
        "seed": int(seed),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=0)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-features", required=True, type=Path)
    parser.add_argument("--variant-features", required=True, type=Path)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    reference = _load_features(args.reference_features)
    variant = _load_features(args.variant_features)
    result = permutation_mmd(reference, variant, args.permutations, args.seed)
    result.update({
        "reference_features": str(args.reference_features.resolve()),
        "variant_features": str(args.variant_features.resolve()),
        "reference_features_sha256": _sha256(args.reference_features),
        "variant_features_sha256": _sha256(args.variant_features),
        "reference_shape": list(reference.shape),
        "variant_shape": list(variant.shape),
        "kernel": "gaussian_rbf",
        "bandwidth": "pooled_median_heuristic",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
