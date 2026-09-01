# Audit findings — WebTunnel thesis, practical part

**Date:** September 2026
**Scope:** `project/` (v1), 9,000 PCAPs, 18 code modules, 6 LaTeX tables
**Method:** every quantitative claim re-derived independently of the repository's own code.
8,174 of 9,000 captures re-parsed with a purpose-written `dpkt` pipeline into four
extraction variants; classifiers retrained from scratch on each; TLS ClientHellos parsed
by hand from raw records; defence results reproduced with the repo's own saved model.

**Verdict: 4/10 for the practical part as it stands.**
Engineering and infrastructure: ~8/10. Scientific validity: ~3/10.
Recoverable — the failures are concentrated in data collection, the cheapest layer to rebuild.

Severity key: **C** = critical (blocks defence) · **M** = major · **m** = minor.

---

## Summary: the three that matter

1. **The positive class is ~13 TCP connections, not 1,500 flows** (F-01).
2. **A single feature reaches 100.00 % accuracy, AUC 1.0000** (F-09).
3. **The countermeasure result is a pipeline bug, and the defences do nothing** (F-10).

---

# Step 1 — Assignment vs. roadmap

## Roadmap evolution

Draft A (*Resilience Research Roadmap*) is theory-first: Bayesian derivation of the base-rate
problem, threat-model decomposition, leakage taxonomy. Draft B (*Traffic Analysis Roadmap*) is
engineering-first: concrete four-namespace topology, exact `tc-netem` invocations, PyTorch layer
specs. The synthesis in `ULTIMATNI_ROADMAPA_WEBTUNNEL_DIPLOMKA.md` is genuinely strong work.
Its §3 identifies three traps that much published work walks into:

- **Trap 1 — TrafficFormer is not downloadable.** Correct, and correctly resolved.
- **Trap 2 — the uTLS/JA4 stack asymmetry.** Correctly diagnosed. Two remedies specified
  (Mode A: strip the handshake; Mode B: same client stack for negatives). **Neither implemented.**
- **Trap 3 — flow-level vs. host-based aggregation.** Correctly diagnosed, right equation given.
  **The equation was never implemented** (see F-14).

## Commitments made and silently dropped

| Roadmap commitment | What shipped | Consequence |
| --- | --- | --- |
| §5.2 Destination-Split — test destinations never in training | One bridge IP, one mock-server IP. Arithmetically impossible. | No evidence the models learn *protocol* rather than *server*. |
| §3.2 Mode A or Mode B for the TLS-stack asymmetry | WebTunnel = Go `crypto/tls`; negatives = Python OpenSSL. Handshake stripping is an ablation only, and mis-implemented. | Headline "full flow" numbers contaminated by a 267 B vs 517 B ClientHello. |
| §4.1 Three client behaviour profiles over WebTunnel | One function, `run_webtunnel_request()`. The three "profiles" in code are *network* profiles. | Positive class has near-zero behavioural variance; negatives have five behaviours. |

## Would I have structured it differently from the assignment alone?

Yes — the difference is one of stance. The assignment asks for *"analýza a zhodnocení
detekovatelnosti"*: an assessment of **how detectable** WebTunnel is. The implementation is
built to **maximise classifier accuracy**. The second has a trivial optimum — make the classes
different and report 99 %.

A detectability study inverts the design pressure: spend the engineering budget making the
classes *as similar as you can honestly make them* (same client TLS stack, same destination host,
same session-length distribution, same behavioural mixture, same capture-window policy), then
report how much signal survives. Under that design 99 % is a finding; under the current design
it is a null result, because the design guarantees it.

Four ordering changes I would have made:

- **A leakage tripwire in week one, before any deep learning.** A depth-1 stump per feature,
  run after every dataset build; fail if any exceeds ~90 %. Eleven lines of code, would have
  caught the entire problem eight months earlier.
- **Negative controls as first-class experiments.** Label-shuffle and
  same-generator/different-label. Neither exists.
- **Dataset budget over model budget.** Three model families on 8,500 correlated samples is
  worse science than one family on 100,000 independent ones. The assignment asks for two.
- **Defences designed against an adaptive adversary from the start.**

## Over- and under-engineering

**Over-engineered:** fifteen orchestrator phases. The cascaded line-rate pipeline, the FDR
projection curves, the DET curve to 10⁻⁴, the SHAP/saliency layer and the two-tier defence
simulation are sophisticated instruments pointed at a dataset that cannot support any of them.
Three model families where two would do; a Transformer 150× slower than XGBoost for 0.5 points
less accuracy on a saturated task.

**Under-engineered:** everything upstream of the models — destination diversity (2 endpoints),
positive-class independence (~13 connections), behavioural diversity of the positive class
(1 script), negative-set scale for the low-FPR regime (1,042 test negatives against a claimed
10⁻⁴–10⁻⁵ operating point), and open-world evaluation.

> **Scale reference.** Wails, Sullivan, Sherr & Jansen (NDSS 2024) — the paper the thesis builds
> its central argument on — evaluate on 60,000,000 real-world flows to 600,000 destinations
> across six orders of magnitude of base rate. This thesis: 8,500 lab captures, 2 destinations,
> ~13 independent positive connections. That gap is fine for an MSc **provided the claims are
> scaled to it**. Claiming a 10⁻⁵ FPR operating point from 1,042 negatives is not.

## The QUIC question

The rationale under review: *WebTunnel is an L4 TCP transport, and an in-path censor
demultiplexes L4 before L7 inspection, therefore QUIC/UDP negatives cannot produce false
positives and their absence is harmless.*

**The mechanism is sound. The argument as stated is not sufficient**, for four reasons:

1. **It is a scope decision presented as a proof.** Restricting the negative universe to TCP/TLS
   is legitimate — but must be stated as a scope boundary with a measured cost. The assignment
   names QUIC alongside WebSockets and HTTP/2; silent omission reads as non-fulfilment.
2. **It changes the base rate, the thesis's own central quantity.** Excluding QUIC removes
   25–35 % of modern egress HTTPS from the denominator, *raising* α by roughly 1.4–1.5× and
   making the censor's problem easier. Compute and report it — it strengthens the chapter.
3. **Host-based aggregation does not demultiplex.** Wails et al. score *destinations*, not flows.
   A bridge IP that also answers QUIC contributes host-level evidence a per-flow L4 filter never
   sees. The argument protects the flow classifier, not the host classifier — and the thesis
   uses both.
4. **It is a moving target.** MASQUE/HTTP-3 proxying and QUIC-obfuscated VPN transports are live
   deployment paths. An opponent will ask "and when WebTunnel moves to HTTP/3?"

**Cheap fix (≈1 day)** that converts this into a defended decision: add ~500 QUIC/HTTP-3 negative
flows; demonstrate the L4 demultiplexer discards them at zero cost; then run the interesting
counterfactual — feed QUIC flows to the TCP-trained classifiers *with the demux disabled* and
report what a sloppy real-world DPI parser would do. Add the α recomputation.

One awkward detail regardless: the captures already **contain** UDP — mDNS and SSDP multicast
from the Docker host — and `sanitizer.py` merges it into the "flow". UDP is in the dataset as
uncontrolled noise while QUIC is absent as a class.

---

# Step 2 — Testbed and anti-leakage pipeline

## F-01 · **C** · The WebTunnel class is ~13 TCP connections sliced into windows
`1_testbed/client/torrc`, `capture/collect_scaled_dataset.py`

Tor maintains a single long-lived TLS connection to its bridge. The collector starts `tcpdump`,
fires one short request through the existing tunnel, and stops. Consecutive "sessions" are
consecutive *windows* of the same socket.

```
profile     TRAIN (id<=350)                          TEST (id>425)     SHARED
broadband   34626, 54478, 54490, 54500, 54502        54502             54502
lte         36122, 51040                             51040             51040
lossy       15000, 34636, 35498, 49660, 50860,53980  34636, 50860      34636, 50860
```

Three consequences, each fatal alone:
- Effective sample size of the positive class is ~13, not 1,500 — every CI, σ and 5-fold
  interval is understated by roughly an order of magnitude.
- The "Session-Stratified-Anti-Leakage" split leaks at exactly the level it claims to protect:
  the identical TCP connection appears in training and test.
- The window almost never contains connection setup → the WebTunnel class has no TLS
  handshake at all (F-02).

## F-02 · **C** · Every negative capture has a TLS handshake; virtually no WebTunnel capture does
`2_data_pipeline/sanitizer.py`

Fraction of captures containing a ClientHello (42 sampled per class/profile):

```
class                  broadband   lte     lossy    mean SYNs (bb)
webtunnel                 0.0%     2.4%    2.4%        0.00
direct_web_browsing       100%     100%     98%        2.00
websocket_ticker          100%     100%     43%        2.07
websocket_chat            100%     100%     67%        2.00
video_streaming           100%     100%     74%        2.00
web_assets                100%     100%     69%        2.00
```

Zero-payload packets are filtered, so SYNs never reach the model — but the handshake's *record
lengths* do, and they are deterministic constants: `517` (ClientHello), `1677`/`1686`
(ServerHello flight), `80` (ChangeCipherSpec), `510`. A depth-1 stump on `down_len_min` at
threshold 79 — "did I see the 80-byte CCS record?" — reaches **98.58 %** on the repo's own test
set. This is why `xgboost_results.json` reports ROC-AUC and PR-AUC of exactly 1.0.

## F-03 · **C** · The WebTunnel client is not using uTLS
`1_testbed/client/Dockerfile`

```
                        record   ciphers  ALPN            GREASE  n_ext
webtunnel (Go client)    267 B     19     (absent)          no     10
direct_web_browsing      517 B     43     http/1.1, h2      no     12
websocket_ticker         517 B     31     (absent)          no     11
video_streaming          517 B     43     http/1.1, h2      no     12

webtunnel ext ids: 0, 11, 65281, 23, 18, 5, 10, 13, 43, 51
  → no session_ticket(35), no padding(21), no ALPN(16), no GREASE
  → Go crypto/tls default, NOT a uTLS Chrome profile
```

Two consequences. WebTunnel here is JA3/JA4-detectable outright and a 267-vs-517-byte
ClientHello is a perfect discriminator whenever setup is captured — so the "post-handshake
analysis is necessary" premise is never exercised. And **the absent ALPN means nginx cannot
negotiate h2**: this connection is HTTP/1.1 Upgrade → WebSocket, while `direct_web_browsing`,
`video_streaming` and `web_assets` run HTTP/2. The README's "native HTTP/2 framing" claim and
the LaTeX label "Tor over HTTP/2 WSS" are both wrong, and the mismatch is itself a confound.

Verify against upstream whether the bridge line needs an explicit `utls` / `utls-imitate`
option. If stock Go TLS is genuinely the default deployment behaviour, that is a reportable
finding — but it must be established deliberately, not discovered by an opponent.

## F-04 · **C** · There is no flow demultiplexing
`sanitizer.py::extract_raw_packets_from_pcap`

`tcpdump -i eth0` runs with no BPF filter and the extractor iterates every packet regardless of
5-tuple.

- **Cross-class contamination.** The Tor client keeps running during negative captures:
  **31 %** of `direct_web_browsing` and **52 %** of `websocket_ticker` broadband captures contain
  WebTunnel packets.
- **Host-window, not flow, classification.** The unit is "everything one container did in ~5 s",
  including mDNS/SSDP multicast. No in-path censor has that view, and the cascade chapter assumes
  per-flow scoring throughout.
- **Direction inversion.** `client_ip` is taken from the *first packet in the file*. When that is
  inbound — 12 % of broadband and up to 38 % of lossy WebTunnel captures — the direction sign
  flips for the whole sample.

## F-05 · **M** · TSO/GSO enabled; the 1500 B clamp manufactures a class signal

Share of captured payload packets exceeding 1500 B (broadband):

```
video_streaming   83.4%   max 65,160 B      web_assets       74.1%   max 36,200 B
webtunnel         23.7%   max 20,272 B      direct_browsing   8.0%   max  2,286 B
websocket_ticker   0.9%   max  2,896 B      websocket_chat    4.9%   max  1,677 B
```

Segmentation offload means `tcpdump` records pre-segmentation super-packets, not wire packets;
`min(pkt_len, 1500)` then collapses them onto one value. High-throughput classes get an
artificial spike at exactly 1500 while WebTunnel's small records pass through untouched.
Fix: `ethtool -K eth0 tso off gso off gro off` on the capture interface, or re-segment offline.

## F-06 · **M** · The hard negatives do not behave as designed
`client/traffic_generator.py`, `legitimate_servers/server.py`

Credit where due: the generator deliberately places WebSocket and GraphQL payloads in the
320–750 B band to overlap the Tor cell. It did not survive contact with the wire.

```
webtunnel         558×1904   1072×253   1448×106   138×17   652×13
                  → 81.4% of all upstream records are exactly 558 B
websocket_ticker  46×201  47×190  45×145  517×60  80×60  304×60  48×58
websocket_chat    52×74   53×74   517×60  302×60  80×60  51×48   30×47
```

The 44–55 byte records are HTTP/2 `WINDOW_UPDATE`/`PING`/`SETTINGS` and WebSocket ping/pong.
WebTunnel — a single long-lived stream with no flow-control dynamics — emits none. That
asymmetry *is* a real protocol property (roadmap vulnerability #3; Huma NDSS 2026 names the
same weakness), but its magnitude here is inflated by sessions so short that control chatter
dominates the payload.

Two behavioural artefacts: capture-window duration is a per-class constant driven by the script
(`websocket_ticker` 20.6 ± 6.2 s vs `web_assets` 1.18 ± 0.31 s), and `total_pkts`/`total_bytes`
are features #46/#47. The ticker's 20 s is itself a bug — the generator asks for 2–4 s, but the
server never reads from `/ws/ticker`, so the close handshake stalls.

## F-07 · **M** · Network profiles are egress-only, bandwidth-unlimited, and mislabelled
`router/netem_profiles.sh`, `common/config.py`

- `tc qdisc … dev eth0 root` shapes **egress only**. Downstream loss — the direction carrying
  video and asset traffic — is never emulated. Use an `ifb` device with `tc mirred`.
- No `tbf`/`htb` rate limit, so "4G/LTE" has unlimited bandwidth and no bufferbloat.
- `PROFILE_DISPLAY_NAMES` says *"Broadband (Gigabit Fiber, 0% Loss, 2ms RTT)"*, *"4G/LTE (30ms
  RTT, Jitter 5ms)"*, *"Lossy WAN (2% Packet Loss, 80ms RTT, Jitter 15ms)"* — none match the
  shell script (20 ms ± 4 ms / 45 ms ± 15 ms / 90 ms ± 25 ms Gilbert–Elliot). These strings are
  what the plots and tables print, so the thesis would ship parameters that were never applied.

## F-08 · **m** · Class-blocked capture order and class-dependent attrition

All 500 samples of a class are captured consecutively, so class is perfectly confounded with
wall-clock time, host load and Tor circuit state. Attrition is class-dependent:
`websocket_ticker` retains 1,268/1,500 (84.5 %) vs `direct_web_browsing` 1,494 (99.6 %), because
failed short sessions fall below the `len(packets) < 3` floor. The discarded samples are
systematically the hardest ones — survivorship bias that flatters every metric.

---

# Step 3 — Models and evaluation

## F-09 · **C** · The task is separable by one constant

Feature matrices rebuilt from raw PCAPs four ways to test whether sanitization is at fault.
It is not — the problem is upstream of it.

```
EXTRACTION VARIANT                              acc      ROC-AUC  avg-prec  FPR@TPR95
V0  all packets merged (repo pipeline)        100.00%   1.0000   1.0000    0.0000
V1  single target TCP flow only               100.00%   1.0000   1.0000    0.0000
V2  single flow, first 10 packets dropped     100.00%   1.0000   1.0000    0.0000
V3  V2 + segmentation offload undone          100.00%   1.0000   1.0000    0.0000

FEATURE SUBSET ON V3
sizes only  (28 feats)                        100.00%   1.0000   1.0000    0.0000
timing only (11 feats)                         99.92%   1.0000   1.0000    0.0000
volume only (4 feats: up-ratios, totals)      100.00%   1.0000   1.0000    0.0000
no volume, no duration features               100.00%   1.0000   1.0000    0.0000

SINGLE-FEATURE DEPTH-1 STUMPS ON V3
up_len_mean / p10 / p25 / p50 / p75           100.00%   1.0000   (thresholds 316–531 B)
up_len_min                                     99.92%   0.9953
down_len_min                                   99.09%   0.9919   (the 80 B ChangeCipherSpec)
```

Class-conditional medians on V3:

```
class                 ratio_up_bytes  up_len_p50  total_bytes  total_pkts
webtunnel                 0.141         558.0        78,486        71
direct_web_browsing       0.323          65.5         3,131        16
websocket_ticker          0.017          46.0        43,036       188
websocket_chat            0.403          52.0           618        11
video_streaming           0.000          66.0     1,624,312      1143
web_assets                0.001          74.0     1,254,747       919

webtunnel up_len_p50: p1 = p50 = p99 = 558.0
negatives up_len_p50: p1 = 40.5, p50 = 65.5, p99 = 81.0   → zero overlap
```

Four features — two directional ratios and two totals — reach a perfect score. So does *one*.
Stripping handshakes, isolating a single flow and undoing offload change nothing: the separation
is created by the experimental design, not by the packet representation.

The project's own SHAP figure already shows this — `up_len_p25` carries ~3× the mean absolute
attribution of the runner-up, with `down_len_min` in fourth place. The XAI layer diagnosed the
problem correctly; it just was not read as a warning.

## F-10 · **C** · The countermeasure result is a pipeline artefact; the defences do nothing
`evaluate_before_after_defenses.py`

The "before" condition uses the stored `X_test`. The "after" condition rebuilds features from
the 200×2 tensor via `recompute_tabular_features()`, which truncates to 200 packets and
round-trips the IAT channel through `expm1(x·10)` on a clipped value. Before and after are
computed by two different pipelines.

Reproduced with the repository's own saved model on its own test split:

```
condition                                                  accuracy   recall
(1) original features, saved model  — repo "before"          98.90%   100.00%
(2) CONTROL: recomputed from the UNMODIFIED tensor           86.66%    24.89%   ← no defence
(3) Mode 1 adaptive padding, 4.1% overhead — repo "after"    88.63%    36.00%
(4) Mode 2 cell coalescing, 4.4% overhead — repo "after"     88.87%    37.33%

ADAPTIVE ADVERSARY (retrain the censor on defended traffic)
Mode 1 padding, retrained XGBoost                           100.00%   100.00%   AUC 1.0000
Mode 2 coalescing, retrained XGBoost                        100.00%   100.00%   AUC 1.0000
reference: retrained on UNdefended recomputed features       98.90%   100.00%   AUC 1.0000
```

Undefended traffic through the same recomputation path collapses to 24.89 % — *lower* than the
36.0 % reported for defended traffic. Read literally, the repository's own numbers say the
defence makes WebTunnel **easier** to detect. Both effects are the artefact.

Against a retraining censor both defences give exactly zero protection. Mechanism: the model
memorised a delta function at 558 B, so *any* perturbation breaks the frozen model, while
padding of 1–128 B moves the mode to 559–686 B — still nowhere near the negatives' 40–81 B
support.

The tell was already in the LaTeX table: 1D-CNN recall stays at 100.0 % and Transformer at
99.6 % under both defences. Only XGBoost moved. A defence affecting exactly one of three
classifiers is a bug, not a defence — yet the README promotes the XGBoost number as the headline.

Two further problems: `recompute_tabular_features()` is applied to the negatives too, so the
accuracy column is contaminated in both arms; and coalescing is scored purely in bytes while its
real cost — buffering latency and throughput on an interactive Tor circuit — is never measured.
`max(0.0, overhead)` also hides that coalescing removes bytes before padding adds them back.

## F-11 · **C** · The CV group vector is misaligned with the feature matrix
`3_models/cross_validate.py`

`X_tab = concat(X_train, X_val, X_test)` is a permutation of file order; `groups =
sample_ids_all` is in file order. They are never re-indexed.

```
len(sample_ids_all)                                            8500
len(concat(train, val, test) ids)                              8500
arrays identical?                                              False
positions where the CV group label is the correct session id   6.85%
```

`StratifiedGroupKFold` therefore partitions on a label vector wrong for 93 % of rows; the claim
that session groups are "strictly respected" is false, and the ±0.2 % fold spreads are
optimistic. Also in the same file: the docstring says groups are 1–100 when they are 1–500; and
for the CNN/Transformer the best epoch is selected by loss *on the evaluation fold itself*, then
metrics reported from that same fold — model selection on test data.

## F-12 · **M** · The 0x17 cutoff does not strip the TLS 1.3 handshake
`evaluate_post_handshake.py`, `sanitizer.py`

**Protocol-level:** in TLS 1.3 the server's `EncryptedExtensions`, `Certificate`,
`CertificateVerify` and `Finished` travel inside records whose *outer* ContentType is `0x17`.
Cutting at the first `0x17` removes only ClientHello, ServerHello and ChangeCipherSpec — the
certificate flight and session tickets survive. Empirically the negatives still begin with a
deterministic 510 B or 304 B record in ~100 % of captures; WebTunnel never does. The heuristic
is also brittle: it only fires when a record header happens to start a TCP segment.

**Logical:** the ablation is uninformative by construction. Per F-02 the WebTunnel class has no
handshake to strip, so the operation only modifies negatives. "Accuracy stayed at 98 %" says
nothing about Tor cell quantization — it says the classes remained different after one class was
edited.

Fix: cut at the *client's* first application record after its `Finished`, then **assert** that
the first-N record-length distributions overlap across classes before training. The assertion is
the proof; the accuracy number is not.

## F-13 · **M** · The low-FPR regime is never measured
`evaluate_det_curve.py`, `evaluate_base_rate_fallacy.py`

The test split holds 1,042 negatives → smallest observable non-zero FPR is 1/1042 ≈ 9.6 × 10⁻⁴.
Yet the DET curve is plotted to 10⁻⁴ with `np.maximum(fpr, 1e-4)` flooring it, and the base-rate
chapter projects operating points at 10⁻⁴ and 10⁻⁵.

By the rule of three, a 10⁻⁵ upper bound at 95 % confidence needs ~3 × 10⁵ negatives with zero
false positives — short by two and a half orders of magnitude. This is the one requirement the
assignment names explicitly (*"s důrazem na režim s nízkou mírou falešných poplachů"*) and it is
currently the least supported claim in the work.

Two acceptable resolutions, pick one deliberately: scale the negative set to ≥10⁵ flows, or keep
the scale and report every FPR with Clopper–Pearson intervals, truncate the DET curve at 1/N,
and relabel the base-rate table as an explicitly *analytical projection*. The second is entirely
defensible for an MSc; presenting projections as measurements is not.

## F-14 · **M** · The "Bayesian host-based aggregation" is neither
`evaluate_base_rate_fallacy.py`

The roadmap gives the correct formulation — `Score(H) = Σ log(pₖ/(1−pₖ))` thresholded at τ.
The code implements `tpr_m = tpr**M` and `fpr_m = fpr**M`: an AND-rule over *M* independent hard
decisions, assuming independence between flows to the same host — precisely the assumption the
NDSS paper attacks. The "correlated" curve then patches this with `max(fpr**M, ρ·√fpr)` where
ρ = 0.01 is unmotivated. Nothing touches the model's actual per-flow scores.

## F-15 · **M** · The line-rate claim omits the cost that matters
`evaluate_cascaded_pipeline.py`

2,204,567 flows/s is model inference on a pre-computed 48-column matrix. A real in-path censor
must first maintain a flow table, reassemble TCP, parse TLS record boundaries and compute 48
statistics per flow — one to three orders of magnitude larger, and not benchmarked anywhere.

- The table places 62.23 µs single-flow latency next to 2.7 M flows/s batch throughput as if they
  compose. The 62 µs is Python call overhead for a batch of one.
- The 2.1 % escalation rate is measured on a saturated task where nothing is ambiguous. On a
  realistic dataset that fraction — the entire economic argument for the cascade — will be larger.
- The model calibrates a threshold of 0.928 on validation, then the cascade evaluates at 0.5 with
  escalation band [0.05, 0.95].

Reframe: a per-flow rule "≥50 % of upstream records are exactly 558 B" needs no ML and runs in a
few instructions. If the thesis argues about censor cost, that rule is its strongest argument.

## F-16 · **m** · Modelling details that would draw questions

- **PR-AUC via trapezoidal `auc(recall, precision)`** is optimistically biased; use
  `average_precision_score`.
- **No key-padding mask in the Transformer.** Flows are zero-padded to 200 and `[CLS]` attends
  over padding, so the model can read flow length from the attention pattern — and flow length is
  class-correlated (F-06).
- **IAT channel squashed.** `log1p(Δt)/10` maps 0–5 s onto [0, 0.18] while the size channel spans
  [−0.97, 0.97] — a tenth of the dynamic range.
- **Focal loss α = 0.75** in code vs α = 0.25 in the roadmap and in Lin et al.
- **Architecture drift** from the roadmap spec (kernels 5/5/3 vs 7/5/3, FC 256→64 vs 256→128,
  dropout 0.3 vs 0.4). Harmless, but the thesis must describe what runs.

## F-17 · **m** · The documented setup path does not execute

`pip install -r requirements.txt` fails: it pins `xgboost>=3.4.1` and `scikit-learn>=1.9.0`
while the newest published releases resolve to 3.2.0 and 1.7.2; `numpy>=2.5.0`, `scipy>=1.18.0`,
`pandas>=3.0.0`, `matplotlib>=3.11.0`, `shap>=0.52.0` are in the same category. Also
`data/raw_pcap/` and `*.pcap` are gitignored, so the 4.1 GB of evidence is not archived anywhere
reproducible.

## F-18 · **m** · Documentation asserts things the code does not do

- "9,000 PCAPs (8,500 verified TLS 1.3 network flows)" — they are captures, not flows (F-04),
  the WebTunnel ones are not independent (F-01), and nginx permits TLS 1.2.
- "under native HTTP/2 framing" — WebTunnel negotiates no ALPN; the WebSocket classes use
  HTTP/1.1 (F-03).
- "empirically prove that model detection is NOT dependent on TLS metadata" — the ablation
  cannot support this (F-12).
- Mode 2's docstring says "~11–14 % overhead"; the README and table say 4.5 %.
- Every class-breakdown accuracy is 97–100 %, which should have read as a warning.

---

# The five defence questions

Each is answerable well *after* the P0 work. Answering them from the current results is not
possible — which is the clearest way to see why P0 is blocking.

### 1. "Your ROC-AUC is exactly 1.0. What single feature achieves that, and why is it not an artefact?"

Currently answerable only by conceding: `up_len_p50`, constant at 558 B across p1/p50/p99 of the
positive class against 40.5–81 B for negatives, with handshake asymmetry and the persistent
connection both contributing.

After P0 the answer becomes a strength: *"It is 558 B, and that is a genuine protocol invariant —
here is the arithmetic (514 B Tor cell + 22 B framing + 22 B TLS 1.3 AEAD overhead), here is the
tripwire table showing no other single feature exceeds chance, here is the label-shuffle control
at 50 %, and here is the same result across independent connections, destinations and client
stacks. The remaining discriminative power after handshake removal and stack equalisation is X %."*

Prepare the tripwire table as a slide. Volunteering it converts your worst question into your
methodology contribution.

### 2. "You report 8,500 flows. How many *independent* WebTunnel connections are in the corpus?"

Today: ~13, and the same connection is in both training and test. There is no way to argue around
it — the client ports are in the PCAPs. After P0: *"Each capture is one TCP connection,
established fresh; connection identity is the CV grouping key; no connection appears in more than
one fold. Independent positive connections: N."* Quote the effective sample size explicitly and
give CIs at the connection level. If a reviewer computes your effective *n* before you do, you
have lost the room.

### 3. "Your countermeasures reduce recall to 36 %. What happens when the censor retrains?"

Measured: **100 % recall, AUC 1.0000, AP 1.0000** for both modes. And the 36 % is itself an
artefact — undefended traffic through the same path gives 24.89 %.

The honest version is a better thesis. Report static and adaptive side by side and explain *why*
padding fails: 1–128 B shifts 558 B to 559–686 B, which never enters the legitimate upstream
support of 40–81 B. Then derive the requirement that follows — a defence must move the
distribution *into* that support, meaning MTU-sized coalescing **and** injected HTTP/2 control
chatter (`WINDOW_UPDATE`, `PING`, `SETTINGS`) to reproduce the small-record mass every legitimate
class has and WebTunnel entirely lacks. Concrete, testable, and independently corroborated by
Huma (NDSS 2026).

### 4. "The assignment names QUIC. Where is it?"

State the L4-demux mechanism as a scope boundary with a measured cost, not as a reason the
requirement does not apply. Then add: a ~500-flow QUIC negative class demonstrating the demux
discards it at zero cost; the counterfactual with the demux absent; and the recomputed base rate
(excluding 25–35 % of egress HTTPS raises α ~1.4× and makes the censor's task *easier*). Also
acknowledge the limit: host-based aggregation scores destinations, so a bridge that also answers
QUIC contributes host-level evidence the L4 filter never sees.

### 5. "You quote FPR = 10⁻⁴ and 10⁻⁵ from 1,042 negative flows. Justify that."

It cannot be justified as a measurement — the resolution floor is 9.6 × 10⁻⁴, and a 10⁻⁵ upper
bound at 95 % confidence needs ~3 × 10⁵ clean negatives. Two defensible positions: scale and
measure, or keep the scale and report Clopper–Pearson intervals, truncate the DET curve at 1/N,
and present the base-rate table as a *sensitivity analysis under assumed FPR* — close to how
Wails et al. frame their own projections. What is not defensible is a DET curve plotted an order
of magnitude below the data's resolution.

---

# What is already good and must survive the rewrite

- **The roadmap's critical deconstruction (§3).** Better threat modelling than many published
  papers. Keep it, and add a fourth trap describing what went wrong here.
- **The 558 B arithmetic.** A clean, verifiable protocol result and the real contribution.
- **Reproducible orchestration.** Fifteen-phase master runner, seeded config, auto-generated
  LaTeX and 300 DPI figures. Rare at MSc level.
- **The XAI layer.** SHAP + gradient saliency already surfaced the leak. Reframe from
  "interpretation" to "leakage diagnosis" and it becomes a method contribution.
- **Hard-negative design intent.** Placing legitimate payloads in the 320–750 B band is exactly
  right — it just needs verification on the wire.
- **The economic framing.** Base-rate fallacy, cascaded inspection, censor cost. Correct framing;
  only the numbers underneath need replacing.
