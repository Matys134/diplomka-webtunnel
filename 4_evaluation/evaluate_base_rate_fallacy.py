#!/usr/bin/env python3
"""
Base-rate analysis and host-based aggregation -- from the model's OWN calibrated scores.

What v2.0 did (docs/04-v2-audit.md 5.2): it never loaded a model. It hardcoded

    llr_var = m_flows * 4.0 ; llr_mean_pos = m_flows * 3.5 ; llr_mean_neg = -m_flows * 3.5

and computed Gaussian tails from those three invented constants, then patched the result with
`max(fpr_m, 0.01 * fpr_single**(0.5 * m_flows**0.3))` -- four more unmotivated constants. The
commit message called it "continuous LLR aggregation". It was not.

v2.1 implements the thing the roadmap specified and Wails et al. (NDSS 2024) use:

    S_M(H) = sum_k ln( p_k / (1 - p_k) )        over the M flows attributed to destination H

with p_k the classifier's CALIBRATED per-flow probability, thresholded at tau. TPR and FPR at
host level are then MEASURED by bootstrapping hosts out of the test split, not assumed.

Two honesty rules the audit demanded and this file obeys:
  * nothing is plotted below the corpus's resolution floor 1/n_negatives;
  * every projected operating point is labelled a projection, with Clopper-Pearson bounds on
    the measured points it is projected from.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import beta

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "2_data_pipeline")):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.config import PLOTS_DIR, RANDOM_SEED, setup_matplotlib_style  # noqa: E402

from sklearn.calibration import CalibratedClassifierCV        # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier   # noqa: E402

try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

EPS = 1e-6


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def bayes_precision(tpr: float, fpr: float, alpha: float) -> float:
    num = tpr * alpha
    den = num + fpr * (1.0 - alpha)
    return (num / den) if den > 0 else 0.0


def llr(p: np.ndarray) -> np.ndarray:
    """S contribution of one flow. Clipped so a saturated probability cannot dominate."""
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def calibrated_scores(d) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train on train, calibrate on val, score the test split."""
    base = (xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                              eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1)
            if HAVE_XGB else
            HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08,
                                           random_state=RANDOM_SEED))
    base.fit(d["X_train"], d["y_train"])
    method = "isotonic" if int((d["y_val"] == 1).sum()) >= 30 else "sigmoid"
    cal = CalibratedClassifierCV(base, method=method, cv="prefit").fit(d["X_val"], d["y_val"])
    p = cal.predict_proba(d["X_test"])[:, 1]
    groups = (d["socket_ids_test"] if "socket_ids_test" in d.files
              else np.arange(len(p)).astype(str))
    dests = d["dest_ids_test"] if "dest_ids_test" in d.files else np.array(["?"] * len(p))
    return p, np.asarray(d["y_test"]), np.asarray(groups), np.asarray(dests)


def bootstrap_hosts(p: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    M: int, n_hosts: int, rng: np.random.RandomState
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Form pseudo-hosts of M flows and return their LLR sums and labels.

    Positive hosts draw their M flows from ONE socket, negative hosts from the negative pool.
    Drawing positives within a socket is the conservative choice: it preserves the correlation
    between flows to the same destination, which is precisely the assumption Wails et al. attack
    and which v2.0's independence model assumed away.
    """
    s = llr(p)
    pos_by_group: Dict[str, List[int]] = defaultdict(list)
    for i in np.where(y == 1)[0]:
        pos_by_group[str(groups[i])].append(int(i))
    pos_groups = [g for g, idx in pos_by_group.items() if len(idx) >= 1]
    neg_idx = np.where(y == 0)[0]
    if not pos_groups or len(neg_idx) == 0:
        return np.array([]), np.array([])

    scores, labels = [], []
    for _ in range(n_hosts):
        g = pos_groups[rng.randint(len(pos_groups))]
        pick = rng.choice(pos_by_group[g], size=M, replace=len(pos_by_group[g]) < M)
        scores.append(float(s[pick].sum())); labels.append(1)
    for _ in range(n_hosts):
        pick = rng.choice(neg_idx, size=M, replace=len(neg_idx) < M)
        scores.append(float(s[pick].sum())); labels.append(0)
    return np.asarray(scores), np.asarray(labels)


def main():
    ap = argparse.ArgumentParser(description="empirical LLR host aggregation + base-rate analysis")
    ap.add_argument("--dataset", default="data/processed/tabular_dataset.npz")
    ap.add_argument("--json", default="4_evaluation/base_rate_results.json")
    ap.add_argument("--max-m", type=int, default=12)
    ap.add_argument("--hosts", type=int, default=2000)
    a = ap.parse_args()

    d = np.load(a.dataset, allow_pickle=True)
    p, y, groups, dests = calibrated_scores(d)
    n_neg = int((y == 0).sum())
    floor = 1.0 / max(1, n_neg)

    print("=" * 84)
    print("  Host-based LLR aggregation  S_M = sum ln(p_k / (1 - p_k))  -- empirical")
    print(f"  test flows {len(y)}  positives {int(y.sum())}  negatives {n_neg}")
    print(f"  FPR resolution floor 1/n_neg = {floor:.2e}   "
          f"-- nothing below this is measured, only projected")
    print(f"  estimator: {'XGBoost' if HAVE_XGB else 'HistGradientBoosting'}, "
          f"calibrated on the validation split")
    print("=" * 84)

    # ---- per-flow measured operating point -----------------------------
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tpr1, fpr1 = tp / max(1, int(y.sum())), fp / max(1, n_neg)
    tlo, thi = clopper_pearson(tp, int(y.sum()))
    flo, fhi = clopper_pearson(fp, n_neg)
    print(f"\n  per-flow (M=1): TPR {tpr1:.4f} [{tlo:.4f}, {thi:.4f}]   "
          f"FPR {fpr1:.5f} [95% upper {fhi:.2e}]  ({fp} FP in {n_neg})")

    # ---- host-level sweep ----------------------------------------------
    rng = np.random.RandomState(RANDOM_SEED)
    sweep = []
    print(f"\n  {'M':>3}{'host TPR':>11}{'host TPR 95% CI':>24}{'host FPR':>11}"
          f"{'FPR 95% upper':>15}{'tau':>9}")
    for M in range(1, a.max_m + 1):
        s, hy = bootstrap_hosts(p, y, groups, M, a.hosts, rng)
        if s.size == 0:
            continue
        # tau = 0 is the natural Wald threshold for equal priors; report it, do not tune it.
        tau = 0.0
        hp = (s >= tau).astype(int)
        htp = int(((hp == 1) & (hy == 1)).sum()); hfp = int(((hp == 1) & (hy == 0)).sum())
        npos = int((hy == 1).sum()); nneg = int((hy == 0).sum())
        htpr, hfpr = htp / max(1, npos), hfp / max(1, nneg)
        lo, hi = clopper_pearson(htp, npos)
        _, fhi_h = clopper_pearson(hfp, nneg)
        sweep.append({"M": M, "tau": tau, "host_tpr": htpr, "host_tpr_ci95": [lo, hi],
                      "host_fpr": hfpr, "host_fpr_upper95": fhi_h,
                      "n_hosts_pos": npos, "n_hosts_neg": nneg,
                      "resolution_floor": 1.0 / max(1, nneg)})
        print(f"  {M:3d}{htpr:11.4f}   [{lo:.4f}, {hi:.4f}]      {hfpr:11.5f}{fhi_h:15.2e}{tau:9.1f}")

    # ---- base-rate table, explicitly labelled ---------------------------
    alphas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    table = []
    print(f"\n  Base-rate analysis at the MEASURED per-flow operating point "
          f"(TPR {tpr1:.4f}, FPR upper bound {fhi:.2e}):")
    print(f"  {'alpha':>10}{'FDR @ measured FPR':>22}{'FDR @ FPR=1e-4':>20}{'FDR @ FPR=1e-5':>20}")
    for al in alphas:
        f_meas = bayes_precision(tpr1, max(fpr1, floor), al)
        f_1e4 = bayes_precision(tpr1, 1e-4, al)
        f_1e5 = bayes_precision(tpr1, 1e-5, al)
        row = {"alpha": al,
               "fdr_at_measured_fpr": 1 - f_meas,
               "fdr_at_1e-4_PROJECTED": 1 - f_1e4,
               "fdr_at_1e-5_PROJECTED": 1 - f_1e5}
        table.append(row)
        print(f"  {al:10.0e}{100*(1-f_meas):21.2f}%{100*(1-f_1e4):19.2f}%{100*(1-f_1e5):19.2f}%")
    print(f"\n  NOTE: the 1e-4 and 1e-5 columns are ANALYTICAL PROJECTIONS. This corpus has "
          f"{n_neg} test negatives,")
    print(f"        so the smallest MEASURABLE FPR is {floor:.2e}. A 95%-confidence bound at "
          f"1e-4 needs 30,000 clean")
    print(f"        negatives (rule of three); at 1e-5 it needs 300,000. Do not present these "
          f"columns as measurements.")

    out = {
        "per_flow": {"tpr": tpr1, "tpr_ci95": [tlo, thi], "fpr": fpr1, "fpr_upper95": fhi,
                     "n_positives": int(y.sum()), "n_negatives": n_neg,
                     "resolution_floor": floor},
        "host_llr_sweep": sweep,
        "base_rate_table": table,
        "projection_disclaimer": ("FPR below 1/n_negatives is projected, not measured. "
                                  "n_negatives = %d, floor = %.3e" % (n_neg, floor)),
        "method": "S_M = sum ln(p_k/(1-p_k)) over M flows attributed to one destination; "
                  "positive hosts sampled within a single socket to preserve correlation.",
    }
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {a.json}")

    # ---- plots ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        setup_matplotlib_style()
        if sweep:
            Ms = [r["M"] for r in sweep]
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.plot(Ms, [r["host_tpr"] for r in sweep], "o-", label="host TPR (measured)")
            ax.fill_between(Ms, [r["host_tpr_ci95"][0] for r in sweep],
                            [r["host_tpr_ci95"][1] for r in sweep], alpha=0.2)
            ax.plot(Ms, [max(r["host_fpr"], r["resolution_floor"]) for r in sweep], "s--",
                    label="host FPR (floored at 1/n)")
            ax.axhline(sweep[0]["resolution_floor"], ls=":", lw=1,
                       label=f"resolution floor 1/n = {sweep[0]['resolution_floor']:.1e}")
            ax.set_yscale("log")
            ax.set_xlabel("M -- flows attributed to one destination")
            ax.set_ylabel("rate (log)")
            ax.set_title(r"Host-based LLR aggregation  $S_M=\sum \ln \frac{p_k}{1-p_k}$  (empirical)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(PLOTS_DIR, "host_based_aggregation.png"), dpi=200)
            plt.close(fig)
            print(f"  wrote {os.path.join(PLOTS_DIR, 'host_based_aggregation.png')}")
    except Exception as e:
        print(f"  (plot skipped: {e})")


if __name__ == "__main__":
    main()
