#!/usr/bin/env python3
"""G3 -- null controls, both of them BINDING.

v2.0 (audit V-01):
  * label shuffle ran on a single seed with no interval;
  * the second control compared two genuinely different negative classes, returned 1.0000, and
    was explicitly excluded from `passed` with a comment calling it "informational".
    That is not the control the rebuild plan specified.

v2.1:
  1. label shuffle over N seeds -> the mean AUC and its 95% interval must straddle 0.5.
  2. same-generator / different-label: take ONE class, split its SOCKETS at random into two
     pseudo-classes, and try to tell them apart. Same generator, same TLS stack, same server,
     same budgets -- so any separation is the harness fingerprinting capture conditions rather
     than protocols. Run per class, over N seeds. BINDING.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

BAND = (0.45, 0.55)
N_SEEDS = 10
#: a control whose 95% interval is wider than this cannot answer the question either way
MAX_INFORMATIVE_CI_WIDTH = 0.20


def _fit_auc(X, y, groups=None, seed=0) -> float:
    """Group-aware holdout AUC. Groups keep a socket out of both halves."""
    rng = np.random.RandomState(seed)
    if groups is None:
        idx = rng.permutation(len(y))
        cut = int(len(y) * 0.75)
        tr, te = idx[:cut], idx[cut:]
    else:
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        cut = int(len(uniq) * 0.75)
        train_groups = set(uniq[:cut])
        mask = np.asarray([g in train_groups for g in groups])
        tr, te = np.where(mask)[0], np.where(~mask)[0]
    if len(tr) < 20 or len(te) < 20:
        return float("nan")
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return float("nan")
    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_depth=5,
                                       random_state=seed).fit(X[tr], y[tr])
    return float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))


def _ci(vals: List[float]):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(v.mean())
    half = 1.96 * float(v.std(ddof=1)) / np.sqrt(v.size) if v.size > 1 else 0.0
    return mean, mean - half, mean + half


def label_shuffle(X, y, groups=None, n_seeds: int = N_SEEDS) -> List[float]:
    out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(1000 + s)
        out.append(_fit_auc(X, rng.permutation(y), groups=groups, seed=s))
    return out


def same_generator_split(X, labels, groups, class_name: str, order=None,
                         n_seeds: int = N_SEEDS) -> List[float]:
    """Same generator, different label -- the TEMPORAL version.

    Within ONE class (one generator, one TLS stack, one server, one budget distribution), label
    the earlier half of the captures 0 and the later half 1, then try to tell them apart with a
    socket-grouped holdout. Nothing about the PROTOCOL differs between the halves, so any
    separation is the corpus carrying wall-clock drift: host load, Tor circuit state, server
    warm-up, thermal throttling. That is finding F-08, and it is a control that can genuinely
    fail -- which a randomly-relabelled control cannot, because a random label assigned to a
    held-out group is unlearnable by construction.
    """
    mask = np.asarray(labels) == class_name
    if mask.sum() < 60:
        return []
    Xc, gc = X[mask], np.asarray(groups)[mask]
    if order is None:
        return []
    oc = np.asarray(order, dtype=float)[mask]
    if not np.isfinite(oc).all() or len(np.unique(oc)) < 4:
        return []
    y = (oc > np.median(oc)).astype(int)
    if len(np.unique(y)) < 2 or len(np.unique(gc)) < 8:
        return []
    return [_fit_auc(Xc, y, groups=gc, seed=s) for s in range(n_seeds)]


def run(X, y, y_multi=None, labels=None, groups=None, order=None,
        verbose=True, n_seeds: int = N_SEEDS) -> dict:
    res: Dict[str, object] = {}

    sh = label_shuffle(X, y, groups=groups, n_seeds=n_seeds)
    m, lo, hi = _ci(sh)
    res["label_shuffle"] = {"mean": m, "ci95": [lo, hi], "n_seeds": len(sh)}
    # A control passes when chance CANNOT BE REJECTED, i.e. the 95% interval contains 0.5.
    # Requiring the point estimate to sit inside a fixed band is both too strict for a noisy
    # estimate and too lax for a tight one.
    shuffle_ok = bool(np.isfinite(m) and lo <= 0.5 <= hi)
    shuffle_weak = bool(np.isfinite(hi - lo) and (hi - lo) > MAX_INFORMATIVE_CI_WIDTH)

    per_class = {}
    underpowered: List[str] = []
    same_gen_ok = True
    if labels is not None and groups is not None:
        for c in sorted(set(labels)):
            vals = same_generator_split(X, labels, groups, c, order=order, n_seeds=n_seeds)
            if not vals:
                continue
            cm, clo, chi = _ci(vals)
            ok = bool(np.isfinite(cm) and clo <= 0.5 <= chi)
            weak = bool(np.isfinite(chi - clo) and (chi - clo) > MAX_INFORMATIVE_CI_WIDTH)
            per_class[c] = {"mean": cm, "ci95": [clo, chi], "passed": ok, "underpowered": weak}
            same_gen_ok &= ok
            underpowered.extend([c] if weak else [])
    res["same_generator"] = per_class

    if verbose:
        print(f"  1. label shuffle ({len(sh)} seeds)      AUC = {m:.4f}  "
              f"95% CI [{lo:.4f}, {hi:.4f}]   must contain 0.5   "
              f"{'PASS' if shuffle_ok else 'FAIL'}"
              f"{'  (UNDERPOWERED)' if shuffle_weak else ''}")
        print(f"  2. same-generator / EARLY-vs-LATE ({n_seeds} seeds, socket-grouped):")
        if not per_class:
            print("       (skipped: needs per-row labels, socket groups and capture timestamps)")
        for c, v in per_class.items():
            print(f"       {c:<24} AUC = {v['mean']:.4f}  95% CI "
                  f"[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]   "
                  f"{'PASS' if v['passed'] else 'FAIL'}"
                  f"{'  (UNDERPOWERED -- too few independent sockets, see G5)' if v['underpowered'] else ''}")
        if per_class and not same_gen_ok:
            print("       FAIL -- two arbitrary halves of ONE generator are separable. The model "
                  "is reading capture conditions, not protocol.")

    res["passed"] = bool(shuffle_ok and (same_gen_ok if per_class else True))
    res["underpowered"] = underpowered
    res["binding_controls"] = ["label_shuffle", "same_generator"]
    return res
