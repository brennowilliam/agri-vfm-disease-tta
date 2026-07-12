"""Label-shift target-prior estimators, for the OTTER baseline and honest comparison.

Reviewer point M1: our Sinkhorn-with-uniform-prior is, mechanically, OTTER (Shin et al.,
NeurIPS 2024) with a uniform instead of an estimated prior. To position honestly we must
(a) compare against a principled prior estimate, and (b) show that for the worst-group /
tail objective the uniform prior beats the accuracy-oriented estimated prior.

This module provides the classical source-free-compatible estimators:
  - source_confusion : soft C×C confusion from a held-out source split (a source *statistic*,
                       shipped once; no source images at adaptation — same footing as prototypes)
  - bbse_prior       : Black-Box Shift Estimation (Lipton et al., 2018)
  - mlls_prior       : Maximum-Likelihood Label Shift / Saerens EM (Alexandari et al., 2020)

All operate on softmaxed scorer logits (probabilities). OTTER is then
`sinkhorn(logits, r=bbse_prior(...))` using tta.sinkhorn.
"""
from __future__ import annotations

import numpy as np


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def source_confusion(src_logits: np.ndarray, src_labels: np.ndarray, C: int) -> tuple[np.ndarray, np.ndarray]:
    """Soft confusion matrix Ĉ[k,j] = E_source[ p̂_j(x) | y=k ] and source prior.

    Built from a labelled source (held-out) split — a compact C×C statistic, not source data.
    Returns (confusion (C×C), source_prior (C,))."""
    probs = _softmax_rows(src_logits)
    conf = np.zeros((C, C))
    src_prior = np.zeros(C)
    for k in range(C):
        m = src_labels == k
        src_prior[k] = m.mean()
        if m.any():
            conf[k] = probs[m].mean(axis=0)     # mean predicted distribution for true class k
    return conf, src_prior


def bbse_prior(confusion: np.ndarray, target_logits: np.ndarray, source_prior: np.ndarray) -> np.ndarray:
    """BBSE: solve Ĉᵀ w = q̂ for importance weights w=p_t(y)/p_s(y); prior = w·source_prior.

    q̂ = mean softmax prediction on the (unlabelled) target. Clip negatives, renormalise."""
    q = _softmax_rows(target_logits).mean(axis=0)          # (C,)
    # Ĉ is E[p̂|y=k] in row k, so predicted marginal = Ĉᵀ @ p_t(y); solve for p_t(y) directly.
    w, *_ = np.linalg.lstsq(confusion.T, q, rcond=None)
    prior = np.clip(w, 0, None)
    if prior.sum() < 1e-8:
        return np.ones_like(prior) / len(prior)
    return prior / prior.sum()


def mlls_prior(target_logits: np.ndarray, source_prior: np.ndarray,
               n_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """MLLS / Saerens EM: iteratively reweight target posteriors to a self-consistent prior.

    Assumes the scorer probabilities are calibrated to the SOURCE prior (we pass source_prior)."""
    probs = _softmax_rows(target_logits)                   # p̂(y|x) under source prior
    src = source_prior + 1e-8
    r = np.ones(len(src)) / len(src)                        # start uniform
    for _ in range(n_iter):
        w = r / src
        adj = probs * w                                    # unnormalised reweighted posteriors
        adj /= (adj.sum(axis=1, keepdims=True) + 1e-12)
        new = adj.mean(axis=0)
        if np.abs(new - r).sum() < tol:
            r = new
            break
        r = new
    return r / r.sum()
