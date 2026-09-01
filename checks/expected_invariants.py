"""Registry of features that ARE allowed to separate the classes strongly.

The leakage tripwire (G2) does not ban strong features -- it forces you to explain them.
Any feature whose single-feature stump AUC exceeds TRIPWIRE_AUC_LIMIT must appear here with a
protocol-level derivation, or the testbed has to change.

v2.1 PRUNE (audit V-01 / section 4).
------------------------------------
v2.0 turned G2 green by adding six features to this registry:
    up_len_max, up_len_std, up_len_p90, up_len_mean, len_p90, down_len_p90
None of them had an arithmetic derivation.

  * `up_len_max` separated at AUC 1.0000 with threshold 951 B because three literal constants
    in main.go capped every negative class's upstream records at 830 B.  That is a fact about
    the generator, not about Tor.  Fixed in the generator (P0.5); removed from this registry.
  * `len_p90` and `down_len_p90` were exactly 1500.0 for video_streaming and web_assets --
    the `min(pkt_len, 1500)` clamp firing on TSO super-packets.  That is a capture artefact.
    Fixed by real TCP reassembly (P0.7); removed from this registry.
  * `up_len_std`, `up_len_p90` and `up_len_mean` are downstream consequences of the same two
    artefacts plus the genuine lattice.  They are not independent derivations, so they do not
    get their own entries: the lattice below is the claim, and it is the only claim.

Exactly ONE invariant is registered, and it is the thesis's contribution.
"""

from common.contracts import LATTICE_BASE, LATTICE_OFFSET, TOR_CELL_BYTES

TRIPWIRE_AUC_LIMIT = 0.90

#: feature name -> written, arithmetic justification
EXPECTED_INVARIANTS = {
    "up_lattice_frac": (
        "Tor cell quantization lattice. A WebTunnel upstream TLS record carrying k Tor cells has "
        f"length L = {LATTICE_OFFSET} + {TOR_CELL_BYTES}k, where "
        f"{LATTICE_OFFSET} = 5 (TLS record header) + 22 (WebSocket/HTTPT framing) "
        "+ 1 (TLS 1.3 inner content type) + 16 (AEAD tag), and 514 B is the fixed Tor cell. "
        f"k=1 -> {LATTICE_BASE} B, k=2 -> 1072 B, k=3 -> 1586 B, k=4 -> 2100 B, k=6 -> 3128 B, "
        "k=7 -> 3642 B -- all seven observed on the wire. This feature is the fraction of a "
        "flow's upstream records that satisfy (L - 44) mod 514 == 0. It is a protocol invariant "
        "of Tor's fixed-size cell scheduling, not a laboratory artefact: it survives handshake "
        "removal, TCP reassembly, single-socket demultiplexing, offload removal and every "
        "network profile. It is also directly exploitable without machine learning -- see "
        "3_models/lattice_rule.py. Derivation: tor-spec.txt section 3 (CELL_LEN = 514 for "
        "link protocol v4+), RFC 8446 section 5.2 (inner content type + AEAD expansion), "
        "RFC 6455 section 5.2 (frame header). See docs/04-v2-audit.md section 4.4."
    ),
    "down_lattice_frac": (
        "The same Tor cell lattice, downstream. The bridge relays cells back to the client "
        "under identical framing, so downstream records satisfy the same L = 44 + 514k relation "
        "(3642 B = k7 and 3128 B = k6 are both common). It is registered separately from the "
        "upstream feature only because the two directions have different coalescing behaviour: "
        "downstream bursts coalesce more cells per record, so the fraction is lower "
        "(57.5% vs 74.1% measured). Same derivation, same citation."
    ),
}

#: Features that are KNOWN laboratory artefacts. If any of these fires, the corpus is broken.
#: They are listed so the gate names the failure precisely instead of just printing a number.
KNOWN_ARTEFACTS = {
    "up_len_max": (
        "The negative generators' upstream payload ceiling. v2.0 capped negative upstream "
        "payloads at 400+rand(400) / 350+rand(400) B, so no negative could emit a TLS record "
        "above 830 B while WebTunnel coalesces to 1072 B and beyond. If this fires, the "
        "generator's payload distribution does not reach the MSS -- see main.go payloadSize()."
    ),
    "len_p90": (
        "The min(pkt_len, 1500) clamp on TSO super-packets. If this reads exactly 1500 for any "
        "class, segmentation offload is still on and/or TCP reassembly is not running (F-05, V-04)."
    ),
    "down_len_p90": "Same clamp as len_p90, downstream. See F-05 / V-04.",
    "down_len_min": (
        "The 6-byte TLS 1.3 ChangeCipherSpec record (or the 80-byte TLS 1.2 one). Present in "
        "flows that contain a handshake and absent from mid-connection windows. If this fires, "
        "the collector is not opening a fresh connection per sample (F-02)."
    ),
    "total_bytes": "Session-budget parity failed -- see gate G4 (F-06).",
    "total_pkts": "Session-budget parity failed -- see gate G4 (F-06).",
    "ratio_up_bytes": "Session-budget parity failed -- see gate G4 (F-06).",
    "ratio_up_pkts": "Session-budget parity failed -- see gate G4 (F-06).",
    "iat_max": "Capture-window duration is acting as a class label -- see gate G4 (F-06).",
    "clienthello_len": "TLS stack parity failed -- see gate G1 (F-03).",
}
