#!/usr/bin/env python3
"""Self-test for the build gates: each gate must PASS on clean input and FAIL on the specific
defect it exists to catch.

This file is the answer to the v2.1 audit's most important finding (V-01): four of six gates
reported PASS on assertions that could not fail. A gate suite is only evidence if it has been
shown to discriminate, so every gate below is exercised in BOTH directions against synthetic
fixtures that reproduce the exact historical defect.

    python3 checks/test_gates.py
"""
from __future__ import annotations

import os
import sys
import hashlib
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import budget_parity, corpus_gates, null_controls, split_integrity, tripwire  # noqa: E402
from common.contracts import LATTICE_OFFSET, TOR_CELL_BYTES  # noqa: E402

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def mkflow(label, ja4, ch_len, alpn, socket_id="s", lattice_up=0.0, n=40):
    """Synthetic FlowRecord-like object."""
    recs, ct = [], []
    for i in range(n):
        on = i < int(n * lattice_up)
        ln = LATTICE_OFFSET + TOR_CELL_BYTES * (1 + i % 3) if on else 40 + (i * 7) % 300
        recs.append((i * 0.01, 1 if i % 2 == 0 else -1, ln))
        ct.append(23)
    return SimpleNamespace(label=label, ja4=ja4, clienthello_len=ch_len, alpn_offered=alpn,
                           socket_id=socket_id, capture_id=f"{label}_{socket_id}",
                           records=recs, content_types=ct, hs_end_idx=0)


# ---------------------------------------------------------------------------
print("\nG1 -- stack parity")
CLEAN_JA4 = "t13d1514h2_8daaf6152771_02713d6af862"
clean = ([mkflow("webtunnel", CLEAN_JA4, 570, ["h2", "http/1.1"], f"w{i}") for i in range(30)] +
         [mkflow("websocket_chat", CLEAN_JA4, 570, ["h2", "http/1.1"], f"c{i}") for i in range(30)])
check("G1 passes when JA4, ALPN and ClientHello lengths agree",
      corpus_gates.stack_parity(clean, verbose=False)["passed"])

# the v2.0 defect: webtunnel is a stock-Go stack, 267 B, no ALPN
broken = ([mkflow("webtunnel", "t13d190900_9dc949149365_e7c285222651", 267, [], f"w{i}") for i in range(30)] +
          [mkflow("websocket_chat", CLEAN_JA4, 570, ["h2", "http/1.1"], f"c{i}") for i in range(30)])
r = corpus_gates.stack_parity(broken, verbose=False)
check("G1 fails on a different JA4 / ALPN / disjoint ClientHello length (F-03)",
      not r["passed"] and len(r["distinct_ja4"]) == 3 - 1 and "webtunnel" in r["length_disjoint_classes"])

# the v2.0 blind spot: a class with NO handshake at all must not be silently skipped
silent = ([mkflow("webtunnel", None, None, None, f"w{i}") for i in range(30)] +
          [mkflow("websocket_chat", CLEAN_JA4, 570, ["h2", "http/1.1"], f"c{i}") for i in range(30)])
r = corpus_gates.stack_parity(silent, verbose=False)
check("G1 fails when a class contributes no ClientHello at all (F-02 blind spot)",
      not r["passed"] and "webtunnel" in r["silent_classes"])

# ---------------------------------------------------------------------------
print("\nG2 -- leakage tripwire")
rng = np.random.RandomState(0)
names = ["up_lattice_frac", "harmless_a", "harmless_b"]
n = 400
y = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)
X_ok = np.c_[np.r_[rng.uniform(.7, .9, n // 2), rng.uniform(.0, .01, n // 2)],
             rng.normal(size=n), rng.normal(size=n)]
r = tripwire.run(X_ok, y, X_ok, y, names, verbose=False)
check("G2 passes when the only strong feature is the REGISTERED lattice invariant",
      r["passed"] and "up_lattice_frac" not in r["unexplained"])

names_bad = ["up_len_max", "harmless_a", "harmless_b"]
r = tripwire.run(X_ok, y, X_ok, y, names_bad, verbose=False)
check("G2 fails on an unregistered perfect separator (the up_len_max ceiling, F-09)",
      not r["passed"] and "up_len_max" in r["unexplained"])

# ---------------------------------------------------------------------------
print("\nG3 -- null controls")
N3 = 1200
X_noise = rng.normal(size=(N3, 6))
labels_one = np.array(["direct_web_browsing"] * N3)
groups_one = np.array([f"s{i//4}" for i in range(N3)])
order_one = np.arange(N3, dtype=float)          # capture wall-clock order
y_noise = rng.randint(0, 2, N3)
r = null_controls.run(X_noise, y_noise, labels=labels_one, groups=groups_one,
                      order=order_one, verbose=False, n_seeds=8)
check("G3 passes on pure noise (label shuffle AND early-vs-late at chance)", r["passed"],
      f"shuffle={r['label_shuffle']['mean']:.3f} "
      f"temporal={r['same_generator']['direct_web_browsing']['mean']:.3f}")

# the control must FAIL when the corpus drifts with wall-clock time (F-08)
drift = (order_one / N3) * 6.0 + rng.normal(scale=0.3, size=N3)
X_drift = np.c_[drift, rng.normal(size=(N3, 5))]
r = null_controls.run(X_drift, y_noise, labels=labels_one, groups=groups_one,
                      order=order_one, verbose=False, n_seeds=8)
sg = r["same_generator"].get("direct_web_browsing", {})
check("G3 fails when early and late captures of ONE generator are separable (F-08 drift)",
      not r["passed"] and not sg.get("passed", True),
      f"early-vs-late AUC={sg.get('mean', float('nan')):.3f}")

# ---------------------------------------------------------------------------
print("\nG4 -- budget parity (paired)")
fn = ["total_bytes", "total_pkts", "iat_max"]
m = 150
budgets = np.array([f"b{i%m}" for i in range(2 * m)])
ym = np.r_[np.zeros(m), np.ones(m)].astype(int)      # 0 = webtunnel (positive_index=0)
base = rng.uniform(20000, 60000, m)
X_par = np.zeros((2 * m, 3))
X_par[:m, 0] = base * rng.uniform(0.95, 1.05, m)
X_par[m:, 0] = base * rng.uniform(0.95, 1.05, m)
X_par[:, 1] = rng.uniform(40, 60, 2 * m)
X_par[:, 2] = rng.uniform(0.1, 0.4, 2 * m)
r = budget_parity.run(X_par, ym, fn, ["webtunnel", "neg"], budget_ids=budgets, verbose=False)
check("G4 passes when both classes honour the same paired budget", r["passed"],
      f"worst p={r['worst_p']:.2g} ratio={r['worst_ratio']:.2f}")

X_bad = X_par.copy()
X_bad[m:, 0] *= 8.0                                   # negatives move 8x the bytes
r = budget_parity.run(X_bad, ym, fn, ["webtunnel", "neg"], budget_ids=budgets, verbose=False)
check("G4 fails when a generator ignores its budget (F-06)",
      not r["passed"], f"worst ratio={r['worst_ratio']:.2f}")

# ---------------------------------------------------------------------------
print("\nG5 -- split integrity")
def fake_npz(socket_train, socket_val, socket_test, y_all=None):
    socks = np.r_[socket_train, socket_val, socket_test]
    y = y_all if y_all is not None else (np.arange(len(socks)) % 2)
    files = {"socket_ids_train": socket_train, "socket_ids_val": socket_val,
             "socket_ids_test": socket_test, "socket_ids_all": socks,
             "conn_ids_train": np.array([f"c{x}" for x in socket_train]),
             "conn_ids_val": np.array([f"c{x}" for x in socket_val]),
             "conn_ids_test": np.array([f"c{x}" for x in socket_test]),
             "y_train": y[:len(socket_train)],
             "y_val": y[len(socket_train):len(socket_train) + len(socket_val)],
             "y_test": y[len(socket_train) + len(socket_val):]}
    ns = SimpleNamespace(files=list(files), **{})
    ns.__getitem__ = files.__getitem__          # type: ignore[attr-defined]
    class NPZ(dict):
        @property
        def files(self): return list(self.keys())
    return NPZ(files)

tr = np.array([f"s{i}" for i in range(600)])
va = np.array([f"s{i}" for i in range(600, 750)])
te = np.array([f"s{i}" for i in range(750, 900)])
y_all = np.array([1 if i % 2 == 0 else 0 for i in range(900)])
r = split_integrity.run(fake_npz(tr, va, te, y_all), verbose=False)
check("G5 passes on a socket-disjoint split with enough independent positive sockets",
      r["passed"], f"pos_sockets={r['positive_sockets']}")

# the historical defect: ONE socket carries most positives and spans all three splits
tr2 = np.array(["s_hot"] * 200 + [f"s{i}" for i in range(200)])
va2 = np.array(["s_hot"] * 40 + [f"s{i}" for i in range(200, 240)])
te2 = np.array(["s_hot"] * 40 + [f"s{i}" for i in range(240, 280)])
y2 = np.r_[np.ones(200), np.zeros(200), np.ones(40), np.zeros(40), np.ones(40), np.zeros(40)].astype(int)
r = split_integrity.run(fake_npz(tr2, va2, te2, y2), verbose=False)
check("G5 fails when one socket spans train/val/test (F-01, port 56446)",
      not r["passed"] and not r["socket_disjoint"]["passed"],
      f"{r['socket_disjoint']['n_clashing_keys']} socket(s) in >1 split")

# ---------------------------------------------------------------------------
print("\nG6 -- provenance")
class M:
    def __init__(self, cid, prov, ft):
        self.capture_id, self.provenance, self.target_5tuple = cid, prov, ft
        self.label = self.profile = self.dest_id = "x"
    @property
    def is_authoritative(self): return self.provenance == "collector"

ft = ("172.20.0.30", 4242, "172.20.0.10", 443, "tcp")
sid = "172.20.0.30:4242->172.20.0.10:443/tcp"
fl = [SimpleNamespace(capture_id="c1", socket_id=sid, label="webtunnel")]
r = corpus_gates.provenance(fl, {"c1": M("c1", "collector", ft)}, n_captures_on_disk=1,
                            attrition={}, verbose=False)
check("G6 passes on collector-written ground truth matching the wire", r["passed"])

r = corpus_gates.provenance(fl, {"c1": M("c1", "repaired-legacy", ft)}, n_captures_on_disk=1,
                            attrition={}, verbose=False)
check("G6 fails on reconstructed (non-authoritative) provenance (V-02)",
      not r["passed"] and r["non_authoritative"] == 1)

wrong = ("172.20.0.3", 0, "172.20.0.10", 443, "tcp")   # the exact v2.0 hardcoded tuple
r = corpus_gates.provenance(fl, {"c1": M("c1", "collector", wrong)}, n_captures_on_disk=1,
                            attrition={}, verbose=False)
check("G6 fails when the observed socket disagrees with the manifest 5-tuple",
      not r["passed"] and r["tuple_mismatch"] == 1)

r = corpus_gates.provenance(fl, {"c1": M("c1", "collector", ft)}, n_captures_on_disk=5,
                            attrition={}, verbose=False)
check("G6 fails when captures on disk are neither flows nor logged drops",
      not r["passed"] and r["unaccounted"] == 4)

# ---------------------------------------------------------------------------
n_pass = sum(1 for _, ok, _ in RESULTS if ok)
print("\n" + "=" * 78)
print(f"  {n_pass}/{len(RESULTS)} gate self-tests passed")
print("  Every gate has now been shown to discriminate in BOTH directions.")
print("=" * 78)
sys.exit(0 if n_pass == len(RESULTS) else 1)
