"""
v2 data contract for the WebTunnel detectability testbed.

WHY THIS EXISTS
---------------
The v1 pipeline let `sanitizer.py` *infer* what it was looking at: which host was the client
(from the first packet in the file), which packets belonged to the flow (all of them), and where
the TLS handshake ended (the first 0x17 byte). Each of those inferences is wrong somewhere in the
v1 corpus, and nothing in the pipeline could notice. See docs/01-audit-findings.md F-01..F-04.

The collector, by contrast, KNOWS all of it. So it writes it down, and the parser stops guessing.

TWO RULES THAT MAKE THIS SELF-ENFORCING
---------------------------------------
1. A flow whose 5-tuple does not match `CaptureManifest.target_5tuple` is DISCARDED, not
   analysed. That single rule kills cross-class contamination, multicast noise, and the
   direction-inversion bug.
2. `FlowRecord.conn_id` is the ONLY legal grouping key for train/val/test splits and for
   StratifiedGroupKFold. Never group by sample index (that is what produced F-11).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

CLASSES: Tuple[str, ...] = (
    "webtunnel",
    "direct_web_browsing",
    "websocket_ticker",
    "websocket_chat",
    "video_streaming",
    "web_assets",
    "quic_http3",          # NEW in v2 — closes the assignment's explicit QUIC requirement
)

#: Client behaviour, applied to EVERY class including the positive one.
#: v1 had five behaviours across the negatives and exactly one for WebTunnel (F-06 / roadmap 4.1).
BEHAVIOURS: Tuple[str, ...] = ("browse", "bulk", "interactive")

PROFILES: Tuple[str, ...] = ("broadband", "lte", "lossy")

#: The single TLS fingerprint every class must present. Enforced by gate G1.
CLIENT_STACK: str = "utls-HelloChrome_Auto"

FiveTuple = Tuple[str, int, str, int, str]  # (src_ip, src_port, dst_ip, dst_port, "tcp"|"udp")


# ---------------------------------------------------------------------------
# Emitted by the collector, one per capture, alongside the .pcap
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaptureManifest:
    """Ground truth for a single capture. Written by the collector; never inferred."""

    capture_id: str                 # e.g. "20261007T2143Z_a91f3c"
    pcap_path: str

    # --- what was generated -------------------------------------------------
    label: str                      # one of CLASSES
    behaviour: str                  # one of BEHAVIOURS
    profile: str                    # one of PROFILES
    dest_id: str                    # "vhost-03" | "bridge-02" — enables destination-split
    client_stack: str = CLIENT_STACK

    # --- the flow the analysis is allowed to look at ------------------------
    target_5tuple: Optional[FiveTuple] = None
    client_ip: str = ""             # authoritative; NEVER derive direction from packet order

    # --- session budget (drawn from a distribution SHARED by all classes) ----
    # This is what makes total_bytes / total_pkts / duration non-informative by design.
    target_duration_s: float = 0.0
    target_bytes_up: int = 0
    target_bytes_down: int = 0

    # --- provenance ---------------------------------------------------------
    t_start: float = 0.0
    t_end: float = 0.0
    generator_seed: int = 0
    netem_params: Dict[str, str] = field(default_factory=dict)
    git_commit: str = ""
    epoch: str = "A"                # capture campaign epoch -> free temporal split
    notes: str = ""

    @property
    def is_positive(self) -> bool:
        return self.label == "webtunnel"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(text: str) -> "CaptureManifest":
        d = json.loads(text)
        if d.get("target_5tuple") is not None:
            d["target_5tuple"] = tuple(d["target_5tuple"])
        return CaptureManifest(**d)


# ---------------------------------------------------------------------------
# Produced by the flow builder, consumed by everything above it
# ---------------------------------------------------------------------------

@dataclass
class FlowRecord:
    """One TCP (or QUIC) connection. THE atom of the v2 dataset.

    `records` holds TLS-record-level observations after TCP desegmentation, so it is
    independent of segmentation offload on the capture host (v1 failure F-05).
    """

    flow_uid: str                   # sha256(capture_id | 5-tuple | syn_ts)[:16]
    conn_id: str                    # identity of the underlying connection — THE grouping key
    capture_id: str

    label: str
    behaviour: str
    profile: str
    dest_id: str
    client_stack: str
    epoch: str

    # (t_relative_seconds, direction:+1 client->server / -1 server->client, tls_record_len)
    records: List[Tuple[float, int, int]] = field(default_factory=list)

    #: index into `records` of the first true application record
    #: (the client's first record AFTER its Finished — not the first 0x17 byte, see F-12)
    hs_end_idx: int = 0

    # denormalised for convenience / gate checks
    n_up: int = 0
    n_down: int = 0
    bytes_up: int = 0
    bytes_down: int = 0
    duration: float = 0.0

    #: length of the ClientHello record, if this flow contains one. Gate G1 asserts that the
    #: distribution of this value is identical across classes.
    clienthello_len: Optional[int] = None
    ja4: Optional[str] = None

    git_commit: str = ""
    generator_seed: int = 0

    @property
    def is_positive(self) -> bool:
        return self.label == "webtunnel"

    @property
    def app_records(self) -> List[Tuple[float, int, int]]:
        """Post-handshake records only. Use this for any 'handshake-free' experiment."""
        return self.records[self.hs_end_idx:]

    @staticmethod
    def make_uid(capture_id: str, five_tuple: FiveTuple, syn_ts: float) -> str:
        raw = f"{capture_id}|{'|'.join(map(str, five_tuple))}|{syn_ts:.6f}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def make_conn_id(five_tuple: FiveTuple, syn_ts: float) -> str:
        """Connection identity WITHOUT the capture id.

        Two captures of the same socket therefore collide on purpose — that is exactly the
        v1 failure (F-01) we want the split logic to be able to see.
        """
        raw = f"{'|'.join(map(str, five_tuple))}|{syn_ts:.3f}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Split protocol
# ---------------------------------------------------------------------------

#: The four generalisation axes the v2 evaluation reports. All are keyed on conn_id.
SPLIT_AXES: Tuple[str, ...] = (
    "unseen_connection",   # baseline: no conn_id shared between folds
    "unseen_destination",  # test dest_ids never seen in training  (roadmap 5.2, never done in v1)
    "unseen_profile",      # train broadband, test lte + lossy
    "unseen_epoch",        # train campaign A, test campaign B     (genuine temporal split)
)


def assert_split_disjoint(splits: Dict[str, Sequence[str]]) -> None:
    """Raise if any conn_id appears in more than one split. Call this from build_dataset."""
    seen: Dict[str, str] = {}
    for split_name, conn_ids in splits.items():
        for cid in conn_ids:
            if cid in seen and seen[cid] != split_name:
                raise AssertionError(
                    f"conn_id {cid!r} appears in both {seen[cid]!r} and {split_name!r}. "
                    "This is exactly the v1 leak (F-01): the same TCP connection in train and test."
                )
            seen[cid] = split_name


def assert_groups_aligned(n_rows: int, groups: Sequence, feature_matrix_name: str = "X") -> None:
    """Raise if the group vector cannot possibly be aligned with the feature matrix.

    v1 built `X = concat(X_train, X_val, X_test)` (a permutation of file order) but passed
    `groups = sample_ids_all` (file order). Only 6.85% of positions matched. See F-11.
    """
    if len(groups) != n_rows:
        raise AssertionError(
            f"group vector has {len(groups)} entries but {feature_matrix_name} has {n_rows} rows. "
            "Carry groups through the SAME permutation as the features."
        )
