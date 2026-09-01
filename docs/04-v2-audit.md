# v2 audit — WebTunnel thesis, practical part

**Date:** 1 September 2026
**Scope:** the v2 rebuild at `/home/matys/Claude/Diplomka` — unified Go/uTLS generator, v2 collector,
5-tuple flow builder, six build gates, three models, six LaTeX tables, 2,016-capture pilot corpus.
**Method:** every claim re-derived independently of the repository's own code. All 2,016 pilot PCAPs
re-parsed with a purpose-written `dpkt` script; TCP connection identity recovered from client ports;
gates executed; classifiers retrained under alternative splits and feature ablations; TLS record
lengths re-tabulated from `flow_records.jsonl`.

> **Verdict: 5.5/10. Architecture is right, corpus is not yet admissible. Do NOT start Epoch A.**
>
> The v2 design is the correct design. The contracts, the gate concept, the unified generator and the
> flow-record abstraction are all real progress, and one genuinely excellent scientific result has
> emerged from the pilot that was not visible in v1. But the pilot corpus fails on the same three axes
> the v1 audit named, and — the more serious problem — **the gate suite currently reports PASS on four
> gates that cannot fail.** Gate A has not been met. The remaining work is small (roughly 250 lines
> across six files) but it is strictly blocking.

Severity key: **C** = critical (blocks defence) · **M** = major · **m** = minor.

---

## 1. Executive summary

### What genuinely improved

| | v1 | v2 pilot |
| --- | --- | --- |
| Positive-class captures with a TLS handshake | 0–2.4 % | **56.6 %** (190/336 with a ClientHello) |
| Nominal distinct positive-class sockets | ~13 | **275** |
| Cross-class contamination in negatives | 31–52 % | **0 %** (5-tuple demux works for negatives) |
| Multicast/UDP noise in the "flow" | present | **eliminated** (BPF `tcp` + demux) |
| CV group vector alignment (F-11) | 6.85 % correct | **100 % correct** |
| Transformer key-padding mask (F-16) | absent | **implemented** |
| Hard negatives overlapping the Tor cell band | none | **39 % of negatives have `up_len_p50` ∈ [500, 620] B** |

That last row is the single best thing in the rebuild and it is under-celebrated: the hard-negative
design finally works. `up_len_p50` — the feature that scored **AUC 1.0000** in v1 — is down to
**AUC 0.8744** in v2. The negatives now genuinely live in the Tor cell size band.

### What still blocks the defence

1. **The positive class is still ~8–12 independent TCP connections, not 296 flows** (F-01, unresolved).
   One client port, `56446`, appears in **234 of 336** WebTunnel captures and spans train, val *and*
   test. Herfindahl effective sample size: **7.73 sockets**.
2. **The WebTunnel client is still not using uTLS** (F-03, unresolved). Its ClientHello is still
   exactly **267 B** — bit-for-bit the stock Go `crypto/tls` fingerprint the v1 audit measured. G1
   fails, correctly and unambiguously.
3. **The 100 % accuracy is a laboratory artefact, and it is *not* the 558 B invariant.** It is
   `up_len_max > 951 B`, and the 951 B threshold exists because three literal constants in `main.go`
   cap every negative class's upstream TLS records at **830 B**. See §4.
4. **Four of six gates pass vacuously.** G5 compares an array to a concatenation the builder itself
   constructed; G6 compares manifest fields to fields copied from that manifest; G2 passes only
   because six leaking features were added to the invariant registry; G3's second control is
   non-binding and returns 1.0000. Only G1 and G4 currently carry information — and both fail.
5. **The gates are not wired into the build.** `run_full_benchmark.py` has 15 phases and
   `checks/run_gates.py` is in none of them. A corpus with G1 and G4 red produced three trained
   models and six LaTeX tables without obstruction. Principle P3 is unimplemented.
6. **The defence chapter is unchanged from v1.** `recompute_tabular_features()` — which the rebuild
   plan explicitly listed for deletion as "an artefact generator with no legitimate use" — is still
   present and still used, so before/after are still computed by two different pipelines (F-10). There
   is still no adaptive adversary anywhere in the repository.
7. **The host-based aggregation is still not Bayesian.** `evaluate_base_rate_fallacy.py` never loads a
   model. It hardcodes `llr_mean_pos = 3.5·M`, `llr_var = 4·M` and computes Gaussian tails from
   invented parameters (F-14, unresolved in a new form).

### The one result worth protecting

The pilot revealed something better than "558 B is the median". WebTunnel's TLS record lengths lie on
an exact arithmetic **lattice**:

```
L = 44 + 514·k        44 = 5 (TLS header) + 22 (WS/HTTPT framing) + 1 (inner type) + 16 (AEAD tag)

k = 1  2     3     4     5     6     7
    558  1072  1586  2100  2614  3128  3642      ← all seven observed in the pilot
```

**74.1 %** of WebTunnel upstream records and **57.5 %** of downstream records sit on this lattice.
Every negative class: **0.0–0.3 %**. A rule with no machine learning —
*"≥20 % of a flow's upstream TLS records satisfy (L−44) mod 514 = 0"* — gives

```
TPR = 0.9899     FPR = 0 / 1,250     ROC-AUC = 0.9965
Clopper–Pearson 95 % upper bound on FPR: 2.39 × 10⁻³
```

That is the thesis. It is verifiable from first principles, it needs two instructions to implement,
it is the strongest possible answer to "your AUC is 1.0, what single feature does that?", and it
makes the censor-cost argument in the cascade chapter for free.

---

## 2. Audit findings matrix — F-01 … F-18

| # | Sev | Finding | Status in v2 | Evidence |
| --- | --- | --- | --- | --- |
| **F-01** | C | WebTunnel class is ~13 TCP connections sliced into windows | **UNRESOLVED** | Port `56446` in 234/336 captures; top-3 ports carry 52 % of 703 observations; Herfindahl effective *n* = **7.73**; **32 sockets span >1 split**; 49.4 % of captures have no client SYN. `pkill -HUP tor` reloads config — it does not close OR connections. |
| **F-02** | C | Negatives have a TLS handshake, WebTunnel does not | **PARTIAL** | Improved from 0–2.4 % to 56.6 % (190/336). But the asymmetry persists (negatives: 99.7–100 %), and the 6-byte ChangeCipherSpec record now separates at **AUC 1.0000** under a socket-disjoint split (`down_len_min`, thr 11.5). |
| **F-03** | C | WebTunnel client is not using uTLS | **UNRESOLVED** | G1 output: `webtunnel ClientHello lengths: [267]`. Identical to v1. `1_testbed/client/torrc` line 8 carries no `utls`/`utls-imitate` bridge option. The Go generator's `dialUTLS()` is never on the WebTunnel path — that path goes through the Tor SOCKS proxy at `127.0.0.1:9050`, so the TLS on the wire is the PT's, not the generator's. |
| **F-04** | C | No flow demultiplexing | **FIXED for negatives, REGRESSED for positives** | Negatives: exact 5-tuple match, clean. Positives: `target_5tuple` is hardcoded `["172.20.0.3", 0, "172.20.0.10", 443, "tcp"]` (wrong IP — the client is `.30`), and `sanitizer.py:119` treats `sport == 0` as "match anything to bridge:443". Result: **81.8 % of WebTunnel captures merge ≥2 distinct TCP sockets into one "flow"** (mean 2.2, max 6). |
| **F-05** | M | TSO/GSO on; the 1500 B clamp manufactures a class signal | **UNRESOLVED** | `disable_offloads()` swallows all errors (`2>/dev/null \|\| true`) and never verifies. Measured on the pilot PCAPs: **72.7 %** of `video_streaming` payload packets exceed 1500 B, max **65,160 B** (v1: 83.4 %, 65,160 B). `sanitizer.py:157` still falls back to `min(pkt_len, 1500)`. Consequence: `len_p90` and `down_len_p90` are **exactly 1500** for `video_streaming` and `web_assets` — and both are now *registered as Tor protocol invariants* in `expected_invariants.py`. |
| **F-06** | M | Hard negatives do not behave as designed | **PARTIALLY FIXED — and newly broken** | Genuine win: negatives now overlap the cell band (39 % with `up_len_p50` ∈ [500,620]). New failure: the generator caps negative upstream records at **830 B** by construction (`400+rand(400)`, `350+rand(400)`, header-only GETs), while WebTunnel reaches 1072–4178 B. Measured ceiling per class: direct 830, ws 779, video/assets 602. WebTunnel min: 1072. **Zero overlap → AUC 1.0000.** |
| **F-07** | M | netem egress-only, unlimited bandwidth, mislabelled | **UNRESOLVED** | `netem_profiles.sh` still `tc qdisc … dev eth0 root` — no `ifb`, no `mirred`, no `tbf`/`htb`. `common/config.py:80` still prints *"Gigabit Fiber, 0% Loss, 2ms RTT"*, *"30ms RTT, Jitter 5ms"*, *"2% Packet Loss, 80ms RTT"* while the script applies 20 ms/45 ms/90 ms. These strings go into the plots and tables. |
| **F-08** | m | Class-blocked capture order; class-dependent attrition | **PARTIAL** | Class is now interleaved (good). But profile is still fully blocked by wall-clock, and profile is a headline experimental axis. Attrition is **worse** than v1: `web_assets` yields 101/336 = **30.1 %** vs `websocket_chat` 297/336 = 88.4 %. 202 manifest rows record `no route to host` on `172.20.0.20:8443`. |
| **F-09** | C | Task separable by one constant | **UNRESOLVED (new constant)** | v1: `up_len_p50 = 558`, AUC 1.0000. v2: `up_len_p50` down to **AUC 0.8744** — a real success — but `up_len_max` now scores **AUC 1.0000, acc 100.00 %** at threshold 951. The constant moved; it did not disappear. |
| **F-10** | C | Countermeasure result is a pipeline artefact; defences do nothing | **UNRESOLVED** | `recompute_tabular_features()` still at `evaluate_before_after_defenses.py:141`, still called on both defended arms (lines 202, 220), still round-tripping IAT through `expm1(x·10)`. Before-arm uses stored `X_test`; after-arm uses recomputed features. Still `max(0.0, overhead)` at line 138. **No adaptive adversary exists in the repository.** Mode 1's `min(1480, orig+pad)` actually *shrinks* WebTunnel's 2100 B and 3642 B records — that is truncation, not padding, and it is why the defence appears to work. |
| **F-11** | C | CV group vector misaligned with feature matrix | **FIXED (a) / UNRESOLVED (b)** | (a) `sample_ids_all` is now built as the same concatenation as `X`; measured 100 % positional agreement. (b) The 1D-CNN and Transformer still select their best epoch by `val_loss` **on the evaluation fold itself** (`cross_validate.py:159–163`, `207–211`) and report metrics from that fold. Also: groups are `sample_id`, not `conn_id` — `conn_ids_all` is saved into the `.npz` and never read, contradicting the hard rule in `CLAUDE.md`. |
| **F-12** | M | 0x17 cutoff does not strip the TLS 1.3 handshake | **UNRESOLVED (replaced by a weaker heuristic)** | `contracts.py` documents "the client's first record AFTER its Finished". `sanitizer.py:184–187` implements `if d == 1 and idx > 3` — a fixed index, ignoring `ctype` entirely. Worse: `evaluate_post_handshake.py` calls `extract_raw_packets_from_pcap()`, which passes `manifest=None`, so **the entire post-handshake experiment runs with no 5-tuple demux at all** — i.e. on v1-style contaminated data. |
| **F-13** | M | Low-FPR regime never measured | **WORSE** | Test split now holds **190** negatives (v1: 1,042) → resolution floor **5.3 × 10⁻³**. `evaluate_det_curve.py:80–85` still applies `np.maximum(fpr, 1e-4)` and `plt.xlim(0.01, 100)`; `evaluate_base_rate_fallacy.py:49` still projects 10⁻⁴ and 10⁻⁵. No Clopper–Pearson interval anywhere. CV `ci_95` upper bounds exceed 1.0 (normal approximation on a bounded metric). |
| **F-14** | M | "Bayesian host-based aggregation" is neither | **UNRESOLVED (new form)** | The `fpr**M` rule is gone; what replaced it is not an LLR sum. `evaluate_base_rate_fallacy.py:86–93` hardcodes `llr_mean_pos = 3.5·M`, `llr_mean_neg = −3.5·M`, `llr_var = 4·M`. The file never loads a model and never touches a per-flow score. The `rho_ambient` patch survives as `max(fpr_m, 0.01 · fpr_single**(0.5·M**0.3))`. |
| **F-15** | M | Line-rate claim omits the cost that matters | **UNRESOLVED** | Still model inference on a pre-computed matrix. No flow-table, reassembly or feature-extraction cost. `n_l2_escalated = 0` on 239 samples — the escalation rate, the entire economic argument for the cascade, is now measured as exactly zero on a saturated task. |
| **F-16** | m | Modelling details | **PARTIAL** | ✅ Transformer key-padding mask implemented (`architectures.py:111–117`). ❌ IAT still `log1p(Δt)/10`. ❌ `pr_auc = 1.0000000000000002` — a value above 1 indicates trapezoidal integration, not `average_precision_score`. Focal α and architecture drift not re-checked. |
| **F-17** | m | Documented setup path does not execute | **UNRESOLVED** | `requirements.txt` unchanged: `torch>=2.13.0`, `xgboost>=3.4.1`, `scikit-learn>=1.9.0`, `numpy>=2.5.0`, `pandas>=3.0.0`, `matplotlib>=3.11.0`, `shap>=0.52.0` — none of these resolve. `CLAUDE.md` §7 points at `requirements-working.txt`, which does not exist in the repository. |
| **F-18** | m | Documentation asserts things the code does not do | **UNRESOLVED / WORSE** | `README.md` is entirely v1: "9,000 PCAPs (8,500 verified TLS 1.3 network flows)", "native HTTP/2 framing", 99.0 ± 0.2 % tables. `table_class_breakdown.tex` still labels WebTunnel *"Tor over HTTP/2 WSS"* — it negotiates no ALPN and runs HTTP/1.1 Upgrade. `evaluate_post_handshake.py` docstring still claims to "empirically prove that detection is independent of TLS metadata". `CLAUDE.md` §7 says TLS keys are not in the repo; `ca.key` and `server.key` are both tracked in git. `CLAUDE.md` §9 describes a `project/` layout that no longer exists. |

**Score: 2 fixed, 4 partial, 12 unresolved or regressed.**

---

## 3. New findings introduced by the v2 rebuild

### V-01 · **C** · Four of six gates cannot fail

This is the most important new finding, because it is what allowed everything else through.

| Gate | Reported | Actual assertion | Verdict |
| --- | --- | --- | --- |
| G1 stack parity | **FAIL** | `len(distinct ClientHello lengths) ≤ 1` and `len(distinct JA4) ≤ 1` | **Real, and correctly failing** — but weaker than it looks. `ja4` is never populated by the flow builder, so the JA4 clause is `len(∅) ≤ 1` → always true. And the gate ignores classes that contribute *no* ClientHello: if the positive class had stayed at v1's 0 % handshake rate, G1 would have compared negatives to negatives and passed. |
| G2 tripwire | PASS | no unregistered feature above AUC 0.90 | **Passes by registry inflation.** Six features were added to `EXPECTED_INVARIANTS`, including `len_p90` and `down_len_p90` — which are **exactly 1500** for `video_streaming` and `web_assets`, i.e. the `min(pkt_len,1500)` clamp from F-05. A capture artefact is registered as a Tor protocol invariant. |
| G3 null controls | PASS | label-shuffle AUC ∈ [0.45, 0.55] | Label shuffle = 0.4580 — inside the band, but a **single seed** with no repetition or CI. The second control returns **1.0000** and is explicitly excluded from `passed`. The plan's control ("same-generator/different-label") is not what is implemented (separating two genuinely different negative classes). |
| G4 budget parity | **FAIL** | KS *p* > 0.01, positive vs each negative | **Real, and correctly failing.** Worst *p* = 2.2 × 10⁻¹⁶⁵. |
| G5 split integrity | PASS | `sample_ids_all == concat(train, val, test)` | **Vacuous.** `build_dataset.py:152` constructs `sample_ids_all` as literally that concatenation. `check_disjoint()` — the function that would test conn_id disjointness — exists in `split_integrity.py` and is **never called** by `run_gates.py`. |
| G6 provenance | PASS | `m.label == f.label`, `m.profile == f.profile`, `m.dest_id == f.dest_id` | **Tautological.** `sanitizer.py:169–174` copies those three fields *from the manifest into the flow*. The gate compares the manifest to itself. It never checks the 5-tuple (which its own docstring promises), never checks that every capture produced a flow, and never logs drop reasons. |

> **Recommendation.** Reframe this in the thesis rather than hiding it. "We wrote a validation harness,
> then audited the harness and found four of six assertions were unfalsifiable" is a methodological
> contribution, and it is exactly the kind of self-criticism a committee rewards. `contracts.py`'s own
> `assert_groups_aligned()` has the same defect: it checks only `len(groups) == n_rows`, and in v1 that
> was already true (8500 == 8500) — the assertion would have passed on the bug it was written to catch.

### V-02 · **C** · WebTunnel provenance is fabricated, not recorded

`main.go:337–347` builds the WebTunnel `GeneratorResult` from four hardcoded literals:

```go
ClientIP:   "172.20.0.3",   // wrong — docker-compose assigns the client 172.20.0.30
ClientPort: 0,              // "Will be matched by sanitizer via bridge target IP"
ServerIP:   "172.20.0.10",
ServerPort: 443,
```

Three consequences:

- The rebuild plan's founding rule — *"the collector writes ground truth; the parser never infers it"* —
  is violated for the only class that matters. All 495 positive manifest rows carry `client_port: 0`.
- Because the tuple is emitted unconditionally, **a failed WebTunnel session is indistinguishable from a
  successful one in the manifest.** All 202 recorded failures are negatives; WebTunnel shows zero. Its
  true failure rate is unknown and unrecorded.
- `sanitizer.py:142` compensates with a hardcoded IP test
  (`src == "172.20.0.30" or src.startswith("172.20.0.3")`), which is direction inference smuggled back
  in and silently breaks if any address changes.

### V-03 · **M** · `manifest.jsonl` is append-only across runs and 32 % duplicated

`collect_scaled_dataset.py:222` truncates the global manifest only if the file does not exist. Measured:
**2,971 rows for 2,016 unique `capture_id`s — 955 duplicates.** Profile counts are 1,344 / 955 / 672
where 672 each is expected, so the file is a mixture of an aborted campaign and a rerun. `run_gates.py`
keys by `capture_id`, so the last row silently wins; whether that row describes the PCAP on disk is
not checked by anything. Additionally, `git_commit` is `860b5d1` on every row — the pre-v2 commit,
which does not contain the collector that produced these captures.

### V-04 · **M** · No TCP desegmentation, despite the docstring

`sanitizer.py` line 2 claims "TLS Record Builder … TCP desegmentation"; the rebuild plan requires it.
`parse_tls_records_from_payload()` operates on **one TCP segment at a time**. There is no per-flow
byte-stream buffer. A record spanning segments is counted once at its declared length from the first
segment, and every continuation segment then falls through to `min(pkt_len, 1500)`. With TSO still
enabled (F-05) this is why `down_len_p90 = 1500.0` exactly for the two bulk classes. The "record
length" channel is a mixture of true record lengths and clamped segment lengths, and the mixture ratio
is class-dependent.

### V-05 · **M** · The `behaviour` factor is recorded but never applied

`collect_scaled_dataset.py:199` assigns `BEHAVIOURS[sid % 3]` and writes it into every manifest. The Go
generator has no `--behaviour` flag and no behaviour switch. The three levels had zero causal effect on
any packet. This is worse than omitting the factor: the manifest asserts a design dimension that does
not exist, and `contracts.py:43` documents it as "applied to EVERY class including the positive one".

### V-06 · **M** · Only one of four declared split axes exists

`contracts.py:182` declares `unseen_connection`, `unseen_destination`, `unseen_profile`, `unseen_epoch`.
`build_dataset.py` emits one split, and it is keyed on **`sample_id`** (the trailing filename integer),
not `conn_id`. `unseen_destination` is arithmetically impossible (`dest_id` ∈ {`vhost-01`, `bridge-01`}).
`unseen_epoch` is impossible (all 2,971 rows are epoch A). The label
`"v2-Connection-Disjoint-Anti-Leakage"` in `dataset_summary.json` describes a property the builder does
not enforce — and `assert_split_disjoint()` cannot detect the violation, because `conn_id` is derived
per-capture from `first_ts` and is therefore unique for all 1,546 flows by construction.

### V-07 · **m** · ALPN parity is broken by design in the generator

`dialUTLS()` receives `{"h2","http/1.1"}` for the HTTP/2 classes and `{"http/1.1"}` for the WebSocket
classes. Measured ClientHello lengths differ by exactly 3 bytes: **506/538/570/602** vs
**503/535/567/599**. So even *among negatives*, the ClientHello identifies the transport. This must be
fixed before G1 can ever go green.

### V-08 · **m** · ClientHello length has an operator-precedence bug

`sanitizer.py:148`: `clienthello_len = (payload[3] << 8) | payload[4] + 5`. In Python `+` binds tighter
than `|`, so this evaluates as `(payload[3] << 8) | (payload[4] + 5)`. Correct for `payload[4] < 251`,
silently corrupt above it. Intended: `((payload[3] << 8) | payload[4]) + 5`.

### V-09 · **m** · Reproducibility hazards

- `collect_scaled_dataset.py:91` uses Python's `hash()` on a string for the seed. `PYTHONHASHSEED` is
  randomised per process, so the "deterministic" budget draw is not reproducible across runs.
- `main.go:361` uses the deprecated global `mrand.Seed`.
- `1_testbed/client/generator/go.mod` does **not** list `github.com/gorilla/websocket`, which
  `main.go:16` imports. The build works only because the Dockerfile runs `go mod tidy` at image-build
  time — so the pinned dependency set is not actually pinned, and there is no `go.sum`.
- `ca.key` and `server.key` are tracked in git.
- The per-capture `.manifest.json` sidecars live under `data/raw_pcap/`, which is gitignored. The
  authoritative ground truth is therefore unversioned.

---

## 4. Forensic analysis — why every model scores 100 %

This is the question the committee will ask first, so it is worth answering exactly.

### 4.1 It is not the 558 B invariant

Single-feature depth-1 stumps on the v2 test split (this is `checks/tripwire.py`'s own output):

```
up_len_max      acc 100.00 %   AUC 1.0000   thr  951     ← this is the whole story
up_len_std      acc  99.58 %   AUC 0.9896   thr  332.1
down_len_p90    acc  98.74 %   AUC 0.9843   thr 1518
len_p90         acc  99.16 %   AUC 0.9792   thr 1543
up_len_mean     acc  97.49 %   AUC 0.9687   thr  555.8
up_len_p90      acc  97.49 %   AUC 0.9375   thr  816.8
up_len_p50      acc  81.17 %   AUC 0.8744   thr  557.5     ← the documented invariant
```

`up_len_p50` — the 558 B median, the feature the thesis is built around — is now the **seventh** most
discriminative feature and does not even cross the tripwire limit. The classifier is not using it.

### 4.2 What `up_len_max > 951` actually measures

Maximum upstream TLS record length, measured per class over all 1,546 pilot flows:

| class | max upstream record | fraction of flows with max > 951 B |
| --- | ---: | ---: |
| `direct_web_browsing` | **830 B** | 0.000 |
| `websocket_ticker` | **779 B** | 0.000 |
| `websocket_chat` | **779 B** | 0.000 |
| `video_streaming` | **602 B** | 0.000 |
| `web_assets` | **602 B** | 0.000 |
| `webtunnel` | 4,178 B | **0.993** |

Zero overlap, and the negative ceiling is a *source-code constant*:

```go
runDirectBrowsing:  chunkSize := 400 + mrand.Intn(400)   // ≤ 799 B  → 830 B on the wire
runWebSocket:       msgLen    := 350 + mrand.Intn(400)   // ≤ 749 B  → 779 B on the wire
runVideoStreaming / runWebAssets:  header-only GETs      //          → 602 B on the wire
```

WebTunnel coalesces 2, 3, 4, 6 and 7 Tor cells into single records (1072, 1586, 2100, 3128, 3642 B).
**No negative class is permitted by the generator to emit an upstream record larger than 830 B.** The
classifier's perfect score therefore answers the question *"did the sender emit an upstream record
larger than my generator's chunk ceiling?"* — which is a fact about `main.go`, not about Tor.

This is F-09 in a new costume, and it is currently **registered in `expected_invariants.py`** as
*"Tor cell upstream maximum payload … typically 1072 B for 2 cells or 1500 MSS boundary"*, which is
how G2 turned green.

### 4.3 The three other artefacts riding along

- **`len_p90` / `down_len_p90` = exactly 1500.0** for `video_streaming` and `web_assets`. This is the
  `min(pkt_len, 1500)` fallback firing on TSO super-packets (87.5 % of `video_streaming` downstream
  "records" are exactly 1500). Both features are registered as Tor invariants. They are capture
  artefacts (F-05 + V-04).
- **`down_len_min`**: WebTunnel 6 B, negatives 31–58 B. Under a socket-disjoint split this scores
  **AUC 1.0000**. Six bytes is the TLS 1.3 ChangeCipherSpec record — the exact v1 F-02 artefact,
  reappearing at a different length. It sits in `KNOWN_ARTEFACTS`, but it scores 0.854 on the repo
  split so G2 never names it.
- **`total_bytes` / `total_pkts` / `iat_max`**: G4 fails at *p* ≈ 10⁻¹⁶⁵, so the volumetric channel is
  still fully class-informative. Median `total_bytes`: WebTunnel 66,871; `direct_web_browsing` 16,673;
  `video_streaming` 182,535.

### 4.4 What the honest signal is

Re-tabulating every TLS record in `flow_records.jsonl`:

| class | upstream records on the 44 + 514·k lattice | downstream |
| --- | ---: | ---: |
| `webtunnel` | **74.1 %** | **57.5 %** |
| `websocket_ticker` | 0.3 % | 0.0 % |
| `direct_web_browsing` | 0.1 % | 0.0 % |
| `websocket_chat` | 0.1 % | 0.1 % |
| `video_streaming` | 0.0 % | 0.0 % |
| `web_assets` | 0.0 % | 0.0 % |

Two zero-parameter detectors:

```
rule                                              TPR      FPR (n=1,250)   ROC-AUC
≥50 % of upstream records == 558 B exactly       0.7804    0 / 1,250       0.9895
≥20 % of upstream records on 44 + 514·k          0.9899    0 / 1,250       0.9965
```

The lattice rule dominates, and it degrades gracefully: TPR 0.9730 at threshold 0.3, 0.9358 at 0.6.
The 22 % of positive flows that the naive 558 B rule misses are exactly the ones where multi-cell
coalescing dominates — which the lattice rule recovers. **This is a better result than 100 %, because
it is falsifiable, derived, and reproducible by anyone with a packet capture.**

### 4.5 Is 100 % defensible before a committee?

**In its current form, no.** The three questions that break it:

1. *"Your best feature is `up_len_max` at threshold 951 B. What is the largest upstream record any of
   your negative generators can produce?"* — 830 B, by three constants in `main.go`. There is no
   recovery from that answer.
2. *"How many independent TCP connections are in your positive class?"* — 8 to 12. One socket appears
   in 234 of 336 captures and is in train, val and test simultaneously. With 12 independent
   connections, the Clopper–Pearson 95 % CI on a perfect TPR is **[0.735, 1.000]**.
3. *"You plot the DET curve to FPR = 10⁻⁴. How many negatives are in your test split?"* — 190.
   Resolution floor 5.3 × 10⁻³, i.e. the plot extends 50× below the data.

**After the P0 fixes, yes, and it becomes a strength**, because the answer changes to: *"We do not
report 100 %. We report a protocol invariant with an arithmetic derivation — record length
= 44 + 514k — that holds for 74 % of WebTunnel's upstream records and 0.1 % of legitimate traffic,
measured across N independent connections, three network profiles and eight destinations, at a
measured FPR below 10⁻⁴ with 95 % confidence. Here is the tripwire table showing what else separates
and why each one is registered or was eliminated. Here is the label-shuffle control at 0.50."*

### 4.6 Why cross-profile (98.9 % / 96.1 %) and post-handshake (99.16 %) do not help

Neither number can currently be cited. `evaluate_cross_profile.py:56` and
`evaluate_post_handshake.py:49` both call `extract_raw_packets_from_pcap()`, which invokes the flow
builder with **`manifest=None`**. That disables 5-tuple demultiplexing entirely and falls back to the
v1 behaviour — every TCP packet in the file, direction inferred from the first SYN. These two
experiments therefore run on a *different and contaminated dataset* from the headline results. The
1D-CNN's drop to 98.9 % on LTE and 96.1 % on lossy WAN is measured on merged, non-demultiplexed
host-window data and cannot be interpreted as domain generalisation.

---

## 5. Defences and base-rate aggregation

### 5.1 `evaluate_before_after_defenses.py` — F-10 unchanged

| Rebuild plan required | Shipped |
| --- | --- |
| Delete `recompute_tabular_features()` | Still present (line 141), still called on both defended arms |
| Defences at TLS-record level on raw traces | Applied to the normalised 200×2 tensor, post-truncation, post-`/1500` |
| Static **and** adaptive adversary side by side | No retraining anywhere; no `adversary` field; no results JSON emitted |
| Bandwidth **and** latency overhead | Bytes only; `max(0.0, overhead)` still hides byte removal (line 138) |

One additional mechanism worth naming: Mode 1's `final_bytes = min(1480.0, orig_bytes + pad)` **shrinks**
WebTunnel's 2100 B and 3642 B records to 1480 B. That is not padding, it is truncation, and it is doing
most of the apparent work. A defence evaluation whose "padding" reduces record sizes cannot be reported.

The v1 audit's finding stands unmodified: against a retraining censor, 1–128 B padding moves the mode
from 558 B to 559–686 B, which never enters the legitimate upstream support. The correct thesis claim
is the one the rebuild plan already wrote: **a defence must move the distribution *into* the legitimate
support — MTU-sized coalescing plus injected HTTP/2 control chatter to reproduce the small-record mass
every legitimate class has (43 B at 30.7 % for `direct_web_browsing`, 48 B at 79.8 % for
`video_streaming`) and WebTunnel entirely lacks.** The pilot data supports this argument well.

### 5.2 `evaluate_base_rate_fallacy.py` — F-14 in a new form

The commit message says "implement continuous LLR aggregation". The code does not. Lines 86–93:

```python
llr_var      = m_flows * 4.0
llr_mean_pos = m_flows * 3.5
llr_mean_neg = -m_flows * 3.5
tpr_m = 1.0 - norm.cdf(0, loc=llr_mean_pos, scale=np.sqrt(llr_var))
```

Three invented constants, a symmetry assumption (`±3.5`) that is never stated, and no model is loaded
by the file at all. The correlation patch survives as
`max(fpr_m, 0.01 * fpr_single**(0.5 * m_flows**0.3))` — four more unmotivated constants.

What is needed is genuinely simple and is ~30 lines: take the classifier's calibrated per-flow
probabilities `p_k` on the test split, group flows by `dest_id`, compute
`S_M = Σ ln(p_k / (1 − p_k))`, sweep the threshold τ, and plot the empirical TPR/FPR at host level as
a function of *M*. That requires ≥2 bridge destinations and ≥2 legitimate destinations to be
meaningful, which is another argument for the destination-diversity work in §6.

---

## 6. Actionable checklist before Epoch A / Epoch B

Ordered. Items in **P0** are blocking — the campaign must not start until all are done and re-piloted.

### P0 — blocking (≈ 3–4 days)

1. **Force a genuinely fresh bridge socket per WebTunnel sample.** `pkill -HUP tor` reloads config; it
   does not close OR connections. Restart the Tor process (or the `webtunnel-client` PT), wait for
   bootstrap, then start `tcpdump`. **Verification gate:** every WebTunnel capture must contain a
   client SYN on a port not seen in any previous capture. Refuse to write the PCAP otherwise. This is
   the single most important fix in the list.
2. **Record the real WebTunnel 5-tuple.** Delete the hardcoded literals in `main.go:337–347`. Snapshot
   `ss -tnp state established '( dport = :443 )'` inside the client container immediately after the
   session, or parse the PT's stdout, and write the actual client port into the manifest. Then remove
   `sanitizer.py`'s `t_sport == 0` wildcard branch and its hardcoded-IP direction test (line 142).
3. **uTLS parity for the WebTunnel client.** Verify against upstream `webtunnel` whether the bridge
   line accepts a `utls`/`utls-imitate` SOCKS argument; if it does, set it to the same Chrome profile
   and re-measure the ClientHello. If it does not, that is a *publishable finding in its own right* —
   report it, and fall back to the plan's documented contingency: strict post-handshake-only analysis
   with G1 relaxed to "post-handshake parity". Decide this before capturing anything.
4. **ALPN parity across negatives.** Offer the identical ALPN list from every class (V-07), or declare
   HTTP/1.1-vs-h2 an explicit experimental factor and stratify on it.
5. **Remove the negatives' upstream record ceiling.** This is what kills the 100 %. Draw negative
   upstream payload sizes from a distribution that spans up to MSS (bulk POSTs, file uploads, large
   WebSocket frames), so `up_len_max` stops being a class label. Add an assertion: the positive and
   negative `up_len_max` distributions must overlap.
6. **Actually disable offload and prove it.** `ethtool -K eth0 tso off gso off gro off` must not be
   error-swallowed; assert its effect, and add a gate: *no captured TCP payload may exceed the
   negotiated MSS*. Currently 72.7 % of `video_streaming` payload packets violate this.
7. **Implement real TCP desegmentation** in the flow builder: a per-direction byte-stream buffer,
   TLS records parsed out of the reassembled stream, and **deletion of the `min(pkt_len, 1500)`
   fallback**. Any segment that cannot be attributed to a record is a bug to be logged, not clamped.
8. **Implement real `hs_end_idx`**: track content types, find the client's `Finished`, take the first
   client record after it. Then *assert* that the first-N post-handshake record-length distributions
   overlap across classes — the assertion is the proof, not the downstream accuracy number.
9. **Make the gates able to fail.** Concretely:
   - **G1** — populate `ja4` (or at minimum the ordered extension-ID list); require *every* class to
     contribute ≥50 ClientHellos, and fail if any class contributes none; fix the precedence bug at
     `sanitizer.py:148`.
   - **G2** — prune the registry to features with a real arithmetic derivation. `up_len_max`,
     `up_len_std`, `len_p90` and `down_len_p90` do not have one and must be removed. Replace them
     with a single registered lattice feature.
   - **G3** — make the second control binding, and change it to the plan's specification
     (same-generator/different-label). Run both controls over ≥10 seeds and report a CI, not a
     single number.
   - **G4** — switch to a **paired** budget design: draw one `(T, B_up, B_down)` triple per
     `sample_id` and give the *identical* triple to every class. The KS test then becomes a check on
     generator fidelity rather than on the sampler.
   - **G5** — call `check_disjoint()` on `conn_ids_train/val/test` (they are already in the `.npz`),
     and add the check that actually matters: *no client TCP port may appear in more than one split*.
   - **G6** — compare the flow's **observed** 5-tuple against `manifest.target_5tuple`; assert every
     capture on disk produced either a flow or a logged drop reason; emit a per-class attrition table.
10. **Wire `run_gates.py` into `run_full_benchmark.py` as a blocking phase** immediately after
    `build_dataset.py`, with a non-zero exit stopping the run. This is principle P3 and it is currently
    absent.
11. **Fix the collector's bookkeeping**: truncate or rotate `manifest.jsonl` per campaign (or write
    `manifest_epochA.jsonl`); record a structured `drop_reason`; commit the code *before* capturing so
    `git_commit` is meaningful; and either implement `--behaviour` in the generator or delete the field.

### P1 — needed for the campaign to answer the assignment (≈ 2–3 days)

12. **Destination diversity.** 8 legitimate vhosts with distinct certificates and 3 bridge instances.
    Without this, `unseen_destination` is impossible and the host-level LLR aggregation has nothing to
    group by.
13. **A QUIC/HTTP-3 negative class.** The assignment names QUIC explicitly alongside WebSockets and
    HTTP/2. `contracts.py` already declares `quic_http3`; the generator has no such mode. ~500 flows
    plus the L4-demux counterfactual and the recomputed α closes the gap in about a day.
14. **Size the campaign from the claim.** 30,000–40,000 clean negatives buys a measured
    *"FPR < 10⁻⁴ at 95 % confidence"*. The current 1,250 supports 2.4 × 10⁻³ and nothing better. Also
    size the *positive* class by independent connections, not captures — with a fresh socket per
    sample this becomes the same number, which is the point.
15. **Fix attrition before scaling.** `web_assets` yields 30 %. Diagnose the `no route to host` errors
    on `172.20.0.20:8443` (the `legitimate-servers` container was restarting during the pilot) and add
    a health-check gate before each capture.
16. **Interleave the profile factor**, not just the class factor, so profile is not confounded with
    wall-clock time and host state.

### P2 — downstream, cheap once the corpus is right (≈ 3–4 days)

17. Delete `recompute_tabular_features()`; apply both defences at TLS-record level on raw traces;
    add the adaptive-adversary arm (retrain on defended traffic) and report both side by side; measure
    coalescing's latency cost, not only its byte cost; remove `max(0.0, overhead)`.
18. Replace `evaluate_base_rate_fallacy.py`'s invented Gaussians with `S_M = Σ ln(p_k/(1−p_k))` over
    the model's real calibrated scores, grouped by `dest_id`, swept over *M*.
19. Remove `np.maximum(fpr, 1e-4)`; truncate the DET curve at 1/*n*; report every FPR with a
    Clopper–Pearson interval; relabel the base-rate table as an explicitly analytical projection.
20. Add the **trivial lattice rule as a fourth "model"** in the generalisation matrix. It is an
    afternoon's work and it is the strongest scientific anchor in the thesis.
21. Group `cross_validate.py` on `conn_id` (already in the `.npz`); move CNN/Transformer epoch
    selection to an inner fold; switch PR-AUC to `average_precision_score`.
22. Give `evaluate_cross_profile.py` and `evaluate_post_handshake.py` the manifest, so they use the
    same demultiplexed corpus as everything else.
23. Documentation sweep: rewrite `README.md` from v2 numbers; regenerate `PROFILE_DISPLAY_NAMES` from
    the applied `tc` parameters; fix the `"Tor over HTTP/2 WSS"` label; pin `requirements.txt` to
    versions that exist; remove `ca.key`/`server.key` from git history; update `CLAUDE.md` §7 and §9
    to the current flat layout.

### Gate A — go/no-go, unchanged

Re-run the 2,000-flow pilot after P0. **All six gates green, on gates that can fail**, and every
surviving tripwire feature carrying a written arithmetic derivation. Only then start the 80-hour
campaign. Expect two or three pilot iterations; that is the harness working, and it is four hours each,
against a campaign that is 80.

---

## 7. What is now genuinely good and must survive

- **The 514 B lattice.** All seven multiples observed, 74.1 % vs 0.1 % separation, TPR 0.9899 at zero
  false positives in 1,250 negatives with no machine learning. This is the thesis's contribution and it
  is stronger than the v1 "558 B median" framing.
- **The hard negatives finally overlap the cell band.** 39 % of negatives have `up_len_p50` ∈
  [500, 620] B, and `up_len_p50`'s stump AUC fell from 1.0000 to 0.8744. That is real experimental
  design succeeding, and it should be reported as such — it is the reason the *remaining* separation is
  interesting rather than trivial.
- **The unified Go/uTLS generator** is the right architecture and mostly works; its remaining problems
  (WebTunnel path, ALPN, upstream ceiling) are three small edits, not a redesign.
- **`contracts.py` and the gate concept.** The idea is correct and ahead of most published work. The
  gates just need to be made falsifiable — and the story of auditing your own harness and finding four
  unfalsifiable assertions is a better methodology chapter than a harness that was right first time.
- **The negatives' small-record mass** (43 B at 30.7 %, 48 B at 79.8 %) is now measured and gives the
  defence chapter its concrete, testable recommendation.

---

# Appendix — remediation log (v2.1)

Executed on the `v2-remediation` branch. Every item below was implemented and verified; the
verification column says how. Nothing here has been validated against a NEW corpus yet, because
the capture campaign requires the author's Docker host.

## P0 — blocking

| # | Item | Change | Verification |
| --- | --- | --- | --- |
| 1 | Fresh bridge socket per sample | `pkill -HUP tor` replaced by `stop_tor.sh` (full teardown, waits for the socket to disappear) + `start_tor.sh` (restart, wait for `Bootstrapped 100%`, snapshot `ss`). The collector opens the capture window BETWEEN them, so the SYN is inside it. A capture whose client port was already used is dropped with `socket_reused_from:<capture>`. | `bash -n` clean; ordering enforced in `capture()`; G5 self-test reproduces the port-56446 failure and fails on it |
| 2 | Real WebTunnel 5-tuple | `main.go` no longer emits `172.20.0.3` / port 0; it reports `tuple_known: false` for the SOCKS path. `start_tor.sh` returns the real socket and refuses ambiguity (`ambiguous_bridge_socket`). `sanitizer.py` lost the `sport == 0` wildcard and the hardcoded-IP direction test; `CaptureManifest.validate()` rejects a zero client port. | `validate()` returns `client_port_zero` on the v2.0 tuple; every legacy WebTunnel capture is now correctly refused |
| 3 | uTLS parity for the PT | `torrc.tmpl` + `WEBTUNNEL_EXTRA_ARGS` render the bridge line at container start; `probe_utls_support.sh` inspects the built binary for the argument. Left empty by default because an unparsed argument stops Tor bootstrapping. | Decision is now explicit and testable before the campaign, per the plan's risk register |
| 4 | ALPN parity | `dialUTLS()` no longer takes an ALPN parameter; every class offers `{h2, http/1.1}` from `ALPN_PARITY`. | G1 currently reports 2 distinct ALPN offers on the legacy corpus — the exact V-07 defect |
| 5 | Negative upstream ceiling | `behaviour.payloadSize()` draws from a three-component mixture whose tail is log-uniform on [1400, 16384]; the GET-dominated classes also emit real POST bodies 35 % of the time. | The 830 B ceiling is gone from the source; `up_len_max` moved from `EXPECTED_INVARIANTS` to `KNOWN_ARTEFACTS`, so G2 now fails on it |
| 6 | Offload disabling, verified | `offload_off.sh` reads back `ethtool -k` and exits non-zero if tso/gso/gro are not `off`; result and MSS are recorded in every manifest. | Script exits non-zero on failure; `offloads_disabled` / `mss` are contract fields |
| 7 | Real TCP desegmentation | `DirectionalStream` reassembles each direction by sequence number (retransmits trimmed, out-of-order buffered) and parses TLS records out of the reassembled stream. `min(pkt_len, 1500)` is **deleted**; an incomplete trailing record is dropped, never invented. | `down_len_p90` for `video_streaming` went from exactly **1500.0** to a real **16406 B** record; `len_p90` and `down_len_p90` fell from AUC 0.9792 / 0.9843 to **0.8495**, below the tripwire limit |
| 8 | Real `hs_end_idx` | TLS 1.3 rule: the client's Finished is its FIRST `application_data` record, so the first true application record is its SECOND. Content types are tracked per record. | Verified on a real WebTunnel capture: ClientHello(22) → server flight → client CCS(20) → client Finished(23, idx 8) → `hs_end_idx = 11` |
| 9 | Falsifiable gates | All six rewritten. G1 requires ≥25 ClientHellos per class, one JA4, one ALPN, no disjoint length set. G2's registry pruned from 7 entries to the lattice only. G3's second control became early-vs-late within one generator. G4 became a paired Wilcoxon test on `budget_id` plus a median-ratio bound. G5 asserts socket disjointness, element-wise group alignment, and a minimum of independent positive sockets. G6 verifies provenance, the observed 5-tuple, and capture coverage. | **`checks/test_gates.py`: 15/15.** Each gate passes on a clean fixture and fails on a synthetic reproduction of its historical defect |
| 10 | Gates wired in, blocking | `run_full_benchmark.py` runs `checks/run_gates.py` immediately after the build; `run_step()` exits on a non-zero return. `--allow-failing-gates` exists for diagnostics and prints a loud warning. | Phase present at line 123 |
| 11 | Collector bookkeeping | `manifest.jsonl` is rotated to `.bak` per run; `drop_reason` is structured and per-class; the design matrix is shuffled across class AND profile; `--behaviour` is passed through and changes cadence and payload mix; a dirty working tree is flagged because `git_commit` would then be meaningless. | 955 duplicate rows are structurally impossible; `data/attrition.json` emitted |

## P2 — downstream

| Item | Change | Verification |
| --- | --- | --- |
| Defences | `recompute_tabular_features()` **deleted**. Both defences act on the real TLS record trace; no MTU clamp (Mode 1 used to *shrink* 2100 B and 3642 B records to 1480 B). Both arms go through one feature pipeline. **Adaptive adversary added**: the censor retrains on defended traffic. Overhead reported in bytes AND buffering latency, signed. | Mode 2 measured at 2.28 % bytes and **46.7 ms** added latency — a cost never previously reported. Lattice fraction drops 0.59 → 0.005 under padding, which is the mechanism |
| Host aggregation | Invented Gaussians deleted. `S_M = Σ ln(p_k/(1−p_k))` over calibrated probabilities, hosts bootstrapped within a socket so correlation is preserved rather than assumed away. | Empirical host TPR/FPR with Clopper–Pearson, swept over M = 1…12 |
| Resolution floor | DET curve floored at 1/n with the sub-resolution region shaded and labelled; base-rate table's 10⁻⁴ and 10⁻⁵ columns explicitly marked `_PROJECTED`. | `fpr_resolution_floor` is emitted in every results JSON |
| Manifest-aware evaluation | `evaluate_cross_profile.py`, `evaluate_post_handshake.py` and `inspect_dataset.py` now load the sidecar; `extract_raw_packets_from_pcap()` **raises** without a manifest. | No call site left without one |
| CV grouping | `cross_validate.py` groups on `socket_ids_all`. | `conn_ids_all` / `socket_ids_all` are written by the builder and now read |
| Lattice rule | `3_models/lattice_rule.py` — the zero-parameter detector, with Clopper–Pearson and a threshold sweep. Wired into the orchestrator. | On the legacy corpus: TPR 0.9729 [0.9473, 0.9882], **0 false positives in 1,247 negatives**, ROC-AUC 0.9860 |
| Docs / config | `PROFILE_DISPLAY_NAMES` generated from the applied netem parameters; netem now shapes ingress via `ifb` with an `htb` rate ceiling; `requirements.txt` pinned to versions that exist; README rewritten. | `PROFILE_DISPLAY_NAMES` prints "RTT 20ms ± 4ms" where the script applies 20 ms |

## What the rebuilt harness found that the old one could not

Running the new gates on the existing 2,016 captures produced one finding that no previous
revision could have surfaced, because the control did not exist:

> **G3, early-vs-late within a single generator.** Every class is separable by capture time
> alone: `web_assets` **AUC 0.9951**, `video_streaming` 0.9108, `websocket_chat` 0.9114,
> `webtunnel` 0.9135, `direct_web_browsing` 0.8938, `websocket_ticker` 0.7644.
>
> Nothing about the protocol differs between the halves. The corpus carries severe wall-clock
> drift — consistent with the 235 `no route to host` failures concentrated in `web_assets`,
> i.e. the `legitimate-servers` container degrading during the run. This is finding F-08 at a
> magnitude nobody had measured, and it is an independent reason the v2.0 corpus cannot be used.

## Not done here — needs the author's hardware

The remediation was carried out in a sandbox with no Docker, no Go toolchain and no CUDA, so
three things remain:

1. `docker compose -f 1_testbed/docker-compose.yml build client` — the Go generator has not been
   compiled. It is syntactically balanced and the module graph is declared, but `go build` is the
   only real check.
2. The pilot re-capture, and with it any statement about whether the gates go green.
3. `run_full_benchmark.py` end-to-end (needs torch + xgboost + the GPU).

Until (2) happens, every number in this repository describes the **v2.0 corpus**, and the gates
correctly refuse it.
