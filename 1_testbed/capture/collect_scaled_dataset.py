#!/usr/bin/env python3
"""
v2.1 capture collector.

Every change here maps to a finding in docs/04-v2-audit.md.

P0.1  Fresh bridge socket per WebTunnel sample.
      `pkill -HUP tor` only reloads the config; it never closes an OR connection, which is why
      client port 56446 appeared in 234 of 336 v2.0 WebTunnel captures and spanned all three
      splits.  The daemon is now fully stopped and restarted, and the capture is REFUSED if the
      resulting client port has already been seen.

P0.2  Real 5-tuple, never a fabricated one.
      Negatives: the generator reports its own local socket.
      WebTunnel: the generator cannot see the bridge socket (it dials a SOCKS proxy), so
      start_tor.sh snapshots `ss` inside the container and the collector writes THAT.
      A capture whose 5-tuple is unknown or ambiguous is dropped with a reason, not guessed.

P0.6  Offload state is verified, not assumed, and recorded in every manifest.

G4    Session budgets are PAIRED: one (duration, bytes_up, bytes_down) triple is drawn per
      (sample_id, profile) and handed to EVERY class, so budget parity becomes a matched test.

V-03  manifest.jsonl is truncated per run (the old one is kept as .bak), so a rerun can never
      leave 955 duplicate rows behind.

V-05  --behaviour is passed to the generator and actually changes the traffic.

F-08  The design matrix is shuffled across class AND profile, so neither is confounded with
      wall-clock time.
"""
import os
import sys
import json
import time
import random
import shutil
import argparse
import subprocess
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.contracts import (          # noqa: E402
    ALPN_PARITY, BEHAVIOURS, PROFILES, PROVENANCE_AUTHORITATIVE, CaptureManifest,
)

COMPOSE_FILE = os.path.join("1_testbed", "docker-compose.yml")
RAW_PCAP_DIR = os.path.join("data", "raw_pcap")
MANIFEST_PATH = os.path.join("data", "manifest.jsonl")

BRIDGE_IP, BRIDGE_PORT = "172.20.0.10", 443
LEGIT_IP, LEGIT_PORT = "172.20.0.20", 8443

# (class, generator mode, destination id, server ip, server port)
TRAFFIC_CLASSES: List[Tuple[str, str, str, str, int]] = [
    ("webtunnel",           "webtunnel",           "bridge-01", BRIDGE_IP, BRIDGE_PORT),
    ("direct_web_browsing", "direct_web_browsing", "vhost-01",  LEGIT_IP,  LEGIT_PORT),
    ("websocket_ticker",    "websocket_ticker",    "vhost-01",  LEGIT_IP,  LEGIT_PORT),
    ("websocket_chat",      "websocket_chat",      "vhost-01",  LEGIT_IP,  LEGIT_PORT),
    ("video_streaming",     "video_streaming",     "vhost-01",  LEGIT_IP,  LEGIT_PORT),
    ("web_assets",          "web_assets",          "vhost-01",  LEGIT_IP,  LEGIT_PORT),
]


def sh(cmd: str, timeout: int = 180) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", "timeout")


def dexec(cmd: str, detach: bool = False, timeout: int = 180) -> subprocess.CompletedProcess:
    flag = "-d " if detach else ""
    return sh(f"docker compose -f {COMPOSE_FILE} exec {flag}-T client {cmd}", timeout=timeout)


def last_json(text: str) -> Dict[str, Any]:
    for line in reversed((text or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def git_commit_and_cleanliness() -> Tuple[str, bool]:
    commit = sh("git rev-parse --short HEAD").stdout.strip() or "unknown"
    dirty = bool(sh("git status --porcelain").stdout.strip())
    return commit, not dirty


def verify_offloads() -> Dict[str, Any]:
    """P0.6 -- assert the capture interface really has tso/gso/gro off."""
    res = dexec("/usr/local/bin/offload_off.sh")
    info = last_json(res.stdout)
    if not info.get("offloads_disabled"):
        print(f"  !! OFFLOAD DISABLE FAILED: {info or res.stderr.strip()}")
    return info


def netem_params(profile: str) -> Dict[str, str]:
    """The parameters actually applied, so config.PROFILE_DISPLAY_NAMES can be generated
    from them instead of being written by hand and drifting (F-07)."""
    return {
        "broadband": {"delay": "20ms", "jitter": "4ms", "loss": "0.05%", "rate": "200mbit"},
        "lte":       {"delay": "45ms", "jitter": "15ms", "loss": "0.2%",  "rate": "40mbit"},
        "lossy":     {"delay": "90ms", "jitter": "25ms", "loss": "gilbert-elliot 2%", "rate": "8mbit"},
    }[profile]


def apply_netem(profile: str) -> None:
    dexec(f"/app/netem_profiles.sh {profile}")
    time.sleep(0.3)


def draw_paired_budget(sample_id: int, profile: str, campaign_seed: int) -> Dict[str, Any]:
    """G4 -- ONE budget per (sample_id, profile), handed to EVERY class.

    v2.0 drew a different budget per class from a shared distribution and then ran a two-sample
    KS test on it, which fails at n=500 for any tiny difference.  A paired draw makes the null
    exactly true by construction, so a G4 failure means a generator did not honour its budget --
    which is the thing worth testing.
    """
    rng = random.Random(f"{campaign_seed}|{profile}|{sample_id}")
    return {
        "budget_id": f"{profile}-{sample_id:05d}",
        "duration_s": round(rng.uniform(2.0, 4.5), 2),
        "bytes_up": int(rng.uniform(15000, 60000)),
        "bytes_down": int(rng.uniform(30000, 150000)),
    }


class Collector:
    def __init__(self, args):
        self.args = args
        self.commit, self.clean = git_commit_and_cleanliness()
        self.offload = verify_offloads()
        self.seen_sockets: Dict[str, str] = {}      # "ip:port" -> capture_id that first used it
        self.drops: Dict[str, Counter] = defaultdict(Counter)
        self.written = 0

    # -- capture window ---------------------------------------------------
    def _start_tcpdump(self, pcap: str, server_ip: str, server_port: int) -> None:
        bpf = f"tcp and host {server_ip} and port {server_port}"
        dexec(f"tcpdump -i eth0 -s 0 -U -w {pcap} '{bpf}'", detach=True)
        time.sleep(0.35)

    def _stop_tcpdump(self) -> None:
        dexec("pkill -f tcpdump")
        time.sleep(0.25)

    # -- one sample -------------------------------------------------------
    def capture(self, cls: str, mode: str, dest_id: str, server_ip: str, server_port: int,
                profile: str, behaviour: str, sample_id: int, idx: int, total: int,
                t0: float) -> None:

        capture_id = f"{cls}_{profile}_{sample_id:04d}"
        host_pcap = os.path.join(RAW_PCAP_DIR, f"{capture_id}.pcap")
        cont_pcap = f"/data/raw_pcap/{capture_id}.pcap"
        budget = draw_paired_budget(sample_id, profile, self.args.campaign_seed)
        seed = abs(hash((self.args.campaign_seed, cls, profile, sample_id))) % (2 ** 31)

        five_tuple: Optional[List[Any]] = None
        client_ip = ""
        drop_reason = ""
        ok = True
        gen: Dict[str, Any] = {}

        # 1. WebTunnel only: tear the daemon down BEFORE the capture window opens, so the
        #    SYN of the new bridge connection lands inside it (P0.1).
        if cls == "webtunnel":
            stop = last_json(dexec("/usr/local/bin/stop_tor.sh", timeout=60).stdout)
            if not stop.get("stopped"):
                ok, drop_reason = False, "tor_stop_failed"

        # 2. Capture window opens.
        if ok:
            self._start_tcpdump(cont_pcap, server_ip, server_port)

        # 3. WebTunnel only: bring Tor up inside the window and read the REAL socket (P0.2).
        t_start = time.time()
        if ok and cls == "webtunnel":
            started = last_json(dexec("/usr/local/bin/start_tor.sh", timeout=150).stdout)
            if not started.get("ok"):
                ok, drop_reason = False, started.get("drop_reason", "tor_start_failed")
            else:
                client_ip = started["client_ip"]
                five_tuple = [client_ip, int(started["client_port"]),
                              started["server_ip"], int(started["server_port"]), "tcp"]

        # 4. Generate.
        if ok:
            cmd = (f"/usr/local/bin/traffic-generator --mode {mode} "
                   f"--server {server_ip}:{server_port} --sni legitimate-servers "
                   f"--behaviour {behaviour} "
                   f"--target-duration {budget['duration_s']} "
                   f"--target-bytes-up {budget['bytes_up']} "
                   f"--target-bytes-down {budget['bytes_down']} "
                   f"--seed {seed}")
            gen = last_json(dexec(cmd, timeout=120).stdout)
            if not gen.get("ok"):
                ok, drop_reason = False, (gen.get("error") or "generator_failed")[:120]
            elif gen.get("tuple_known"):
                client_ip = gen.get("client_ip", "")
                five_tuple = [client_ip, int(gen.get("client_port", 0)),
                              gen.get("server_ip", server_ip), int(gen.get("server_port", server_port)),
                              "tcp"]
        t_end = time.time()

        self._stop_tcpdump()

        # 5. Provenance checks that refuse rather than guess.
        if ok and (not five_tuple or int(five_tuple[1]) == 0):
            ok, drop_reason = False, "no_client_port"

        if ok:
            sock = f"{five_tuple[0]}:{five_tuple[1]}"
            prev = self.seen_sockets.get(sock)
            if prev is not None:
                # P0.1 verification: a reused ephemeral port means the socket was NOT fresh.
                ok, drop_reason = False, f"socket_reused_from:{prev}"
            else:
                self.seen_sockets[sock] = capture_id

        size = os.path.getsize(host_pcap) if os.path.exists(host_pcap) else 0
        if ok and size < 500:
            ok, drop_reason = False, f"pcap_too_small:{size}"

        manifest = CaptureManifest(
            capture_id=capture_id,
            pcap_path=host_pcap,
            label=cls,
            behaviour=behaviour,
            profile=profile,
            dest_id=dest_id,
            client_stack=gen.get("client_stack", "utls-HelloChrome_Auto"),
            target_5tuple=tuple(five_tuple) if five_tuple else None,
            client_ip=client_ip,
            budget_id=budget["budget_id"],
            target_duration_s=budget["duration_s"],
            target_bytes_up=budget["bytes_up"],
            target_bytes_down=budget["bytes_down"],
            provenance=PROVENANCE_AUTHORITATIVE,
            alpn_offered=tuple(gen.get("alpn_offered") or ALPN_PARITY),
            mss=int(self.offload.get("mss", 0)),
            offloads_disabled=bool(self.offload.get("offloads_disabled", False)),
            t_start=t_start,
            t_end=t_end,
            generator_seed=seed,
            netem_params=netem_params(profile),
            git_commit=self.commit,
            epoch=self.args.epoch,
            ok=ok,
            drop_reason=drop_reason,
            notes=gen.get("error", ""),
        )

        with open(os.path.join(RAW_PCAP_DIR, f"{capture_id}.manifest.json"), "w",
                  encoding="utf-8") as f:
            f.write(manifest.to_json())
        with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(json.loads(manifest.to_json())) + "\n")

        if ok:
            self.written += 1
        else:
            self.drops[cls][drop_reason.split(":")[0]] += 1

        elapsed = time.time() - t0
        eta = (elapsed / max(1, idx)) * (total - idx) / 60.0
        tag = "OK  " if ok else "DROP"
        detail = f"{size}B" if ok else drop_reason
        print(f"  [{idx}/{total} {100*idx/total:5.1f}% ETA {eta:5.1f}m] {tag} "
              f"{capture_id:<38} {behaviour:<11} {detail}")

    # -- campaign ---------------------------------------------------------
    def run(self) -> None:
        os.makedirs(RAW_PCAP_DIR, exist_ok=True)
        if os.path.exists(MANIFEST_PATH) and os.path.getsize(MANIFEST_PATH):
            shutil.move(MANIFEST_PATH, MANIFEST_PATH + ".bak")   # V-03
            print(f"  previous manifest moved to {MANIFEST_PATH}.bak")
        open(MANIFEST_PATH, "w").close()

        tasks = []
        for profile in PROFILES:
            for sid in range(1, self.args.samples_per_profile + 1):
                behaviour = BEHAVIOURS[sid % len(BEHAVIOURS)]
                for cls, mode, dest_id, sip, sport in TRAFFIC_CLASSES:
                    tasks.append((cls, mode, dest_id, sip, sport, profile, behaviour, sid))

        # F-08: shuffle across class AND profile.  v2.0 blocked by profile, so profile was
        # perfectly confounded with wall-clock time and host state.
        random.Random(self.args.campaign_seed).shuffle(tasks)

        total = len(tasks)
        print("=" * 78)
        print(f"  v2.1 capture campaign -- epoch {self.args.epoch} -- {total} captures")
        print(f"  commit {self.commit}{'' if self.clean else '  !! WORKING TREE DIRTY'}")
        print(f"  offloads_disabled={self.offload.get('offloads_disabled')}  mss={self.offload.get('mss')}")
        print(f"  budgets are PAIRED per (sample_id, profile); matrix shuffled across class+profile")
        print("=" * 78)
        if not self.clean:
            print("  WARNING: git_commit in the manifests will not describe the code that ran.")

        t0 = time.time()
        current_profile = None
        for i, (cls, mode, dest_id, sip, sport, profile, behaviour, sid) in enumerate(tasks, 1):
            if profile != current_profile:
                apply_netem(profile)
                current_profile = profile
            self.capture(cls, mode, dest_id, sip, sport, profile, behaviour, sid, i, total, t0)

        apply_netem("reset")
        self.report(total, time.time() - t0)

    def report(self, total: int, elapsed: float) -> None:
        print("\n" + "=" * 78)
        print(f"  DONE in {elapsed/60:.1f} min -- {self.written}/{total} usable captures "
              f"({100*self.written/max(1,total):.1f}%)")
        print(f"  distinct client sockets: {len(self.seen_sockets)}")
        print("\n  attrition by class and reason (G6 requires this table):")
        for cls in sorted(self.drops):
            for reason, n in self.drops[cls].most_common():
                print(f"    {cls:<22} {reason:<34} {n:5d}")
        if not self.drops:
            print("    (none)")
        with open(os.path.join("data", "attrition.json"), "w", encoding="utf-8") as f:
            json.dump({c: dict(v) for c, v in self.drops.items()}, f, indent=2)
        print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="v2.1 scaled dataset collector")
    ap.add_argument("--pilot", action="store_true", help="2,016-capture pilot (112 per class per profile)")
    ap.add_argument("--samples-per-profile", type=int, default=112)
    ap.add_argument("--epoch", default="A")
    ap.add_argument("--campaign-seed", type=int, default=20260901)
    args = ap.parse_args()
    if args.pilot:
        args.samples_per_profile = 112
    Collector(args).run()


if __name__ == "__main__":
    main()
