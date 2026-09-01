# Raw evidence — measurements behind the audit

Every number here was produced by re-parsing the raw PCAPs independently of `project/`'s own
code, or by re-running `project/`'s own saved model with added controls. Reproduce any of it with
the scripts in `audit/`.

- **Corpus:** 9,000 PCAPs, 4.1 GB, at `…/webtunnel_pcaps_9000/raw_pcap/`
- **Re-parsed:** 8,174 captures survived the independent extraction (single target flow, ≥5
  payload packets)
- **Split used throughout:** the repo's own — `sample_id ≤ 350` train, `351–425` val, `> 425` test

---

## 1. Corpus composition

```
9,000 files = 6 classes × 3 profiles × 500 samples

per-class mean PCAP size (bytes)
                        broadband      lte      lossy
webtunnel                  87,634    83,697    72,076
direct_web_browsing        11,260    11,501    11,706
websocket_ticker           65,022    68,646    29,371
websocket_chat              8,304     8,172     5,456
video_streaming         1,769,457 1,716,100 1,223,535
web_assets              1,334,236 1,318,546   885,693
```

Repo's own `dataset_summary.json` after sanitization (8,500 of 9,000 retained):

```
webtunnel            1,480 / 1,500   (98.7%)
direct_web_browsing  1,494 / 1,500   (99.6%)
websocket_ticker     1,268 / 1,500   (84.5%)   ← class-dependent attrition
websocket_chat       1,361 / 1,500   (90.7%)
video_streaming      1,442 / 1,500   (96.1%)
web_assets           1,455 / 1,500   (97.0%)

splits: train 5,951 · val 1,282 · test 1,267 (225 positive, 1,042 negative)
```

---

## 2. Connection independence (F-01)

Client-side TCP source ports on the bridge connection, sampling every 7th capture in the training
range and every 2nd in the test range:

```
profile     TRAIN (id<=350)                              TEST (id>425)      SHARED
broadband   34626, 54478, 54490, 54500, 54502            54502              54502
lte         36122, 51040                                 51040              51040
lossy       15000, 34636, 35498, 49660, 50860, 53980     34636, 50860       34636, 50860
```

Across 42 sampled captures per profile the WebTunnel class uses **2 distinct client ports in
broadband, 2 in lte, 5 in lossy**. Effective independent positive connections ≈ 13 total.

Reproduce: `audit/pcap_forensics.py --mode ports`

---

## 3. Handshake asymmetry (F-02)

42 sampled captures per class/profile:

```
class                  %captures with ClientHello        mean SYN count (broadband)
                       broadband   lte     lossy
webtunnel                 0.0%     2.4%    2.4%                 0.00
direct_web_browsing       100%     100%     98%                 2.00
websocket_ticker          100%     100%     43%                 2.07
websocket_chat            100%     100%     67%                 2.00
video_streaming           100%     100%     74%                 2.00
web_assets                100%     100%     69%                 2.00
```

Deterministic negative-class record sequence at flow start:
`0x16` len 517 (ClientHello) → `0x16` len 1677/1686 (ServerHello flight) → `0x14` len 80 (CCS) →
`0x17` len 510 → …

WebTunnel captures begin directly with `0x17` len 558.

Single-feature stump on the repo's own test set:
`down_len_min > 79` → **98.58 %** accuracy, AUC 0.9635.

Reproduce: `audit/pcap_forensics.py --mode handshake`

---

## 4. TLS fingerprints (F-03)

Parsed by hand from the raw ClientHello records:

```
                        record   n_ciphers  ALPN              GREASE  n_ext
webtunnel (Go client)    267 B      19      (absent)            no     10
direct_web_browsing      517 B      43      http/1.1, h2        no     12
websocket_ticker         517 B      31      (absent)            no     11
video_streaming          517 B      43      http/1.1, h2        no     12

webtunnel extension ids: 0(SNI), 11(ec_point_formats), 65281(renegotiation_info),
                         23(extended_master_secret), 18(SCT), 5(status_request),
                         10(supported_groups), 13(signature_algorithms),
                         43(supported_versions), 51(key_share)

absent vs a real uTLS Chrome profile: GREASE, ALPN(16), session_ticket(35), padding(21),
                                      compress_certificate, ALPS
→ this is the Go crypto/tls default fingerprint
```

Consequence: no ALPN → nginx cannot negotiate h2 → WebTunnel runs **HTTP/1.1 Upgrade → WebSocket**
while `direct_web_browsing`, `video_streaming`, `web_assets` run **HTTP/2**.

Server certificate is shared by both server containers: `CN = webtunnel.local`,
issuer `CN = WebTunnel-Testbed-Root-CA`, SAN `DNS:webtunnel.local, DNS:localhost,
IP:127.0.0.1, IP:172.20.0.10` — note `legitimate-servers` is not in the SAN, which is why the
generator uses `verify=False`.

Reproduce: `audit/pcap_forensics.py --mode clienthello`

---

## 5. Cross-class contamination and capture hygiene (F-04, F-05, F-08)

```
%captures containing WebTunnel packets (broadband)
  direct_web_browsing   31.0%
  websocket_ticker      52.4%
  video_streaming        9.5%
  websocket_chat         7.1%
  web_assets             0.0%

first packet is inbound (→ whole-capture direction sign flipped)
  webtunnel broadband   12%      webtunnel lossy      38%
  ws_ticker lossy       29%      web_assets lossy     33%

captures with no TCP/UDP at all (dropped by len<3 floor)
  websocket_ticker lossy  33%    websocket_chat lossy  26%

%captured payload packets > 1500 B  (TSO/GSO artefact)
  video_streaming   83.4%   max 65,160 B
  web_assets        74.1%   max 36,200 B
  webtunnel         23.7%   max 20,272 B
  direct_browsing    8.0%   max  2,286 B
  websocket_chat     4.9%   max  1,677 B
  websocket_ticker   0.9%   max  2,896 B

capture-window duration, mean ± sd (broadband)
  websocket_ticker  20.60 ± 6.18 s      webtunnel        3.44 ± 3.10 s
  video_streaming    2.15 ± 0.88 s      websocket_chat   1.53 ± 0.40 s
  web_assets         1.18 ± 0.31 s      direct_browsing  1.05 ± 0.36 s
```

---

## 6. Packet-size structure (F-06, and the one real result)

Most frequent upstream TLS record sizes, broadband, aggregated over 60–120 captures:

```
webtunnel         558×1904   1072×253   1448×106   138×17   652×13   1586×6
                  → 81.4% of all upstream records are exactly 558 B  (n = 2,338)

webtunnel down    558×1375   2896×473   4118×457   1222×238  3642×190  1586×127

websocket_ticker  46×201  47×190  45×145  517×60  80×60  304×60  48×58  44×54  30×52
websocket_chat    52×74   53×74   517×60  302×60  80×60  51×48   30×47  54×45  24×45
ws_ticker down    265×89  245×70  216×68  217×66  230×65  219×64  285×64  297×63
```

### The 558-byte arithmetic

```
558 = 5   TLS record header
    + 536 plaintext  ( = 514-byte Tor cell + 22 bytes WebSocket/HTTPT framing )
    + 1   TLS 1.3 inner content type
    + 16  AEAD authentication tag

confirmed by the 2× multiple:  1072 = 2 × 536  (two coalesced cells)
```

Class-conditional medians, extraction variant V3:

```
class                 ratio_up_bytes  ratio_up_pkts  up_len_p50  len_p50  total_bytes  total_pkts
webtunnel                 0.141          0.232         558.0     1119.0       78,486        71
direct_web_browsing       0.323          0.375          65.5       57.0        3,131        16
websocket_ticker          0.017          0.082          46.0      248.5       43,036       188
websocket_chat            0.403          0.455          52.0       48.0          618        11
video_streaming           0.000          0.005          66.0     1448.0    1,624,312     1,143
web_assets                0.001          0.013          74.0     1448.0    1,254,747       919

up_len_p50 support:
  webtunnel   p1 = 558.0   p50 = 558.0   p99 = 558.0     ← zero variance
  negatives   p1 =  40.5   p50 =  65.5   p99 =  81.0     ← zero overlap with the above
```

Reproduce: `audit/pcap_forensics.py --mode histogram`

---

## 7. Separability (F-09)

Gradient boosting, 48 features, the repo's own session split, on independently rebuilt matrices:

```
EXTRACTION VARIANT                              n_train  n_test    acc     ROC-AUC  AP      FPR@TPR95
V0  all packets merged (repo pipeline)            5,706   1,231  100.00%   1.0000  1.0000    0.0000
V1  single target TCP flow only                   5,706   1,231  100.00%   1.0000  1.0000    0.0000
V2  single flow, first 10 packets dropped         5,493   1,189  100.00%   1.0000  1.0000    0.0000
V3  V2 + segmentation offload undone              5,608   1,210  100.00%   1.0000  1.0000    0.0000

FEATURE SUBSET ON V3
sizes only  (28 feats)                                           100.00%   1.0000  1.0000    0.0000
timing only (11 feats)                                            99.92%   1.0000  1.0000    0.0000
volume only ( 4 feats: 2 ratios + total_pkts + total_bytes)      100.00%   1.0000  1.0000    0.0000
no volume, no duration features                                  100.00%   1.0000  1.0000    0.0000

CROSS-PROFILE (train broadband id<=350)
V3 → broadband  acc  99.77%  AUC 1.0000  recall  98.6%
V3 → lte        acc  99.77%  AUC 0.9998  recall  98.6%
V3 → lossy      acc 100.00%  AUC 1.0000  recall 100.0%
```

Single-feature depth-1 stumps:

```
ON V3 (independently rebuilt, leak-controlled extraction)
  up_len_mean            100.00%   AUC 1.0000   thr 452
  up_len_p10             100.00%   AUC 1.0000   thr  97
  up_len_p25             100.00%   AUC 1.0000   thr 316
  up_len_p50             100.00%   AUC 1.0000   thr 391.2
  up_len_p75             100.00%   AUC 1.0000   thr 531
  up_len_min              99.92%   AUC 0.9953   thr  54
  down_len_min            99.09%   AUC 0.9919   thr  81
  len_min                 99.59%   AUC 0.9901   thr  53

ON THE REPO'S OWN tabular_dataset.npz
  up_len_p25              98.97%   AUC 0.9868   thr 314
  up_len_mean             98.90%   AUC 0.9863   thr 513.3
  up_len_p10              98.90%   AUC 0.9846   thr 210
  up_len_p50              98.74%   AUC 0.9854   thr 547.8
  down_len_min            98.58%   AUC 0.9635   thr  79
```

The repo's own SHAP ranking (from `4_evaluation/plots/xgboost_feature_importance.png`) agrees:
`up_len_p25` carries ~3× the mean absolute attribution of the runner-up (`len_p10`), with
`down_len_min` fourth.

Reproduce: `audit/leakage_probe.py`

---

## 8. Countermeasure re-check (F-10)

Repository's own saved XGBoost model, own test split, own defence code:

```
condition                                                  accuracy   recall    AUC
(1) original features, saved model  — repo "before"          98.90%   100.00%  1.0000
(2) CONTROL: recomputed from the UNMODIFIED 200×2 tensor     86.66%    24.89%  0.9955
(3) Mode 1 adaptive padding, overhead 4.1% — repo "after"    88.63%    36.00%  0.9965
(4) Mode 2 cell coalescing, overhead 4.4% — repo "after"     88.87%    37.33%  0.9966

ADAPTIVE ADVERSARY — retrain XGBoost on defended traffic
Mode 1 padding, retrained                                   100.00%   100.00%  1.0000  AP 1.0000
Mode 2 coalescing, retrained                                100.00%   100.00%  1.0000  AP 1.0000
reference: retrained on UNdefended recomputed features       98.90%   100.00%  1.0000
```

Condition (2) is the control the repository never runs. It shows the reported 36.0 % is a
feature-pipeline artefact, not a defence effect — and that defended traffic is scored *higher*
than undefended traffic through the same path.

The repo's own `table_before_after_defense.tex` corroborates: 1D-CNN recall 100.0 % and
Transformer 99.6 % are **unchanged** by both defences. Only XGBoost moved.

Reproduce: `audit/defense_recheck.py`

---

## 9. Cross-validation group alignment (F-11)

```
len(sample_ids_all)                                            8,500
len(concat(train, val, test) ids)                              8,500
arrays identical?                                              False
positions where the CV group label is the correct session id   6.85%
unique group values                                            500 (1..500)
```

`X_tab = concat(X_train, X_val, X_test)` is a permutation of file order; `groups =
sample_ids_all` is in file order. `StratifiedGroupKFold` therefore groups on a label vector wrong
for 93 % of rows.

---

## 10. Statistical resolution (F-13)

```
test-set negatives                       1,042
smallest observable non-zero FPR         1/1042 = 9.6 × 10⁻⁴
DET curve is plotted down to             1 × 10⁻⁴          ← below the data's resolution
base-rate table projects FPR at          10⁻⁴ and 10⁻⁵

rule of three, 95% upper bound = 3/n, zero observed FPs:
  FPR ≤ 10⁻³    →  n = 3,000
  FPR ≤ 10⁻⁴    →  n = 30,000
  FPR ≤ 10⁻⁵    →  n = 300,000
```

Comparison: Wails et al. (NDSS 2024) evaluate on 60,000,000 flows to 600,000 destinations.
This thesis: 8,500 captures, 2 destinations, ~13 independent positive connections.

---

## 11. Environment / reproducibility (F-17)

```
project/requirements.txt pins           latest published (Sept 2026)      installs?
  xgboost>=3.4.1                          3.2.0                             no
  scikit-learn>=1.9.0                     1.7.2                             no
  numpy>=2.5.0, scipy>=1.18.0,            —                                 no
  pandas>=3.0.0, matplotlib>=3.11.0,
  shap>=0.52.0, torch>=2.13.0
```

`.gitignore` excludes `data/raw_pcap/` and `*.pcap`, so the 4.1 GB of evidence is not archived
anywhere reproducible.

---

## 12. Configuration inconsistencies (F-07, F-18)

```
common/config.py PROFILE_DISPLAY_NAMES        router/netem_profiles.sh (what actually runs)
  "Gigabit Fiber, 0% Loss, 2ms RTT"      vs   delay 20ms ±4ms normal, loss 0.05%, dup 0.02%
  "4G/LTE (30ms RTT, Jitter 5ms)"        vs   delay 45ms ±15ms paretonormal, loss 0.2%,
                                              reorder 0.5% 25%
  "Lossy WAN (2% Loss, 80ms, Jitter 15ms)" vs delay 90ms ±25ms paretonormal,
                                              loss state 0.02 0.30 0.01 0.10 (Gilbert–Elliot)
```

netem is applied as `tc qdisc … dev eth0 root` — **egress only**, no rate ceiling.

Other documentation mismatches: Mode 2 docstring says "~11–14 % overhead" vs 4.5 % in the README
and table; `table_class_breakdown.tex` labels WebTunnel "Tor over HTTP/2 WSS" (it is HTTP/1.1);
nginx allows `TLSv1.2` despite "TLS 1.3" claims throughout.
