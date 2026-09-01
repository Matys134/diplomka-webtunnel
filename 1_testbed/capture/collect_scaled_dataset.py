#!/usr/bin/env python3
"""
v2 Multi-Environment Scaled Dataset Collector:
- Fresh TCP connection per sample (conn_id isolation).
- uTLS ClientHello stack parity (HelloChrome_Auto).
- Shared session budgets across all classes (eliminates volumetric leaks).
- BPF-filtered tcpdump with offloads disabled.
- Sidecar CaptureManifest (manifest.jsonl).
- Randomized matrix interleaving.
"""
import os
import sys
import time
import json
import subprocess
import argparse
import random
import hashlib
from typing import Dict, Any, List

COMPOSE_FILE = "1_testbed/docker-compose.yml"
RAW_PCAP_DIR = "data/raw_pcap"
MANIFEST_PATH = "data/manifest.jsonl"

TRAFFIC_CLASSES = [
    ("webtunnel", "webtunnel"),
    ("direct_web_browsing", "direct_web_browsing"),
    ("websocket_ticker", "websocket_ticker"),
    ("websocket_chat", "websocket_chat"),
    ("video_streaming", "video_streaming"),
    ("web_assets", "web_assets"),
]

BEHAVIOURS = ["browse", "bulk", "interactive"]
NETEM_PROFILES = ["broadband", "lte", "lossy"]


def run_cmd(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def get_git_commit() -> str:
    try:
        res = run_cmd("git rev-parse --short HEAD")
        return res.stdout.strip()
    except Exception:
        return "unknown"


def disable_offloads():
    run_cmd(f"docker compose -f {COMPOSE_FILE} exec client ethtool -K eth0 tso off gso off gro off 2>/dev/null || true")


def apply_netem(profile: str):
    print(f"\n>>> [tc-netem] Switching network profile to: {profile.upper()} <<<")
    cmd = f"docker compose -f {COMPOSE_FILE} exec client /app/netem_profiles.sh {profile}"
    res = run_cmd(cmd)
    print(res.stdout.strip())
    time.sleep(1)


def draw_session_budget(seed: int) -> Dict[str, Any]:
    """Draws duration, bytes_up, and bytes_down from a shared distribution."""
    rng = random.Random(seed)
    dur = round(rng.uniform(2.0, 4.5), 2)
    b_up = int(rng.uniform(15000, 60000))
    b_down = int(rng.uniform(30000, 150000))
    return {"duration_s": dur, "bytes_up": b_up, "bytes_down": b_down}


def capture_session(
    class_name: str,
    gen_mode: str,
    profile: str,
    behaviour: str,
    sample_id: int,
    epoch: str,
    git_commit: str,
    current_count: int,
    total_count: int,
    start_time: float
) -> bool:
    capture_id = f"{class_name}_{profile}_{sample_id:04d}"
    pcap_filename = f"{capture_id}.pcap"
    manifest_filename = f"{capture_id}.manifest.json"
    host_pcap_path = os.path.join(RAW_PCAP_DIR, pcap_filename)
    host_manifest_path = os.path.join(RAW_PCAP_DIR, manifest_filename)
    container_pcap_path = f"/data/raw_pcap/{pcap_filename}"

    # Draw budget for this session
    seed = sample_id * 1000 + abs(hash(class_name + profile)) % 1000
    budget = draw_session_budget(seed)

    # 1. Fresh connection enforcement
    if class_name == "webtunnel":
        # Force Tor to open fresh TLS bridge socket
        run_cmd(f"docker compose -f {COMPOSE_FILE} exec client pkill -HUP tor")
        time.sleep(0.4)

    # 2. Start BPF-filtered tcpdump
    tcpdump_cmd = f"docker compose -f {COMPOSE_FILE} exec -d client tcpdump -i eth0 -s 0 -w {container_pcap_path} tcp"
    run_cmd(tcpdump_cmd)
    time.sleep(0.3)

    # 3. Trigger uTLS generator
    t_start = time.time()
    gen_cmd = (
        f"docker compose -f {COMPOSE_FILE} exec client /usr/local/bin/traffic-generator "
        f"--mode {gen_mode} --target-duration {budget['duration_s']} "
        f"--target-bytes-up {budget['bytes_up']} --target-bytes-down {budget['bytes_down']} "
        f"--seed {seed}"
    )
    res = run_cmd(gen_cmd)
    t_end = time.time()

    # 4. Stop tcpdump
    run_cmd(f"docker compose -f {COMPOSE_FILE} exec client pkill -f tcpdump")
    time.sleep(0.2)

    # Parse generator output
    gen_meta = {}
    try:
        gen_meta = json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        pass

    target_5tuple = None
    if gen_meta.get("client_ip") and gen_meta.get("server_ip"):
        target_5tuple = [
            gen_meta["client_ip"],
            gen_meta.get("client_port", 0),
            gen_meta["server_ip"],
            gen_meta.get("server_port", 443 if class_name == "webtunnel" else 8443),
            "tcp"
        ]

    # Construct CaptureManifest
    manifest_data = {
        "capture_id": capture_id,
        "pcap_path": host_pcap_path,
        "label": class_name,
        "behaviour": behaviour,
        "profile": profile,
        "dest_id": "vhost-01" if class_name != "webtunnel" else "bridge-01",
        "client_stack": "utls-HelloChrome_Auto",
        "target_5tuple": target_5tuple,
        "client_ip": gen_meta.get("client_ip", "172.20.0.30"),
        "target_duration_s": budget["duration_s"],
        "target_bytes_up": budget["bytes_up"],
        "target_bytes_down": budget["bytes_down"],
        "t_start": t_start,
        "t_end": t_end,
        "generator_seed": seed,
        "git_commit": git_commit,
        "epoch": epoch,
        "notes": gen_meta.get("error", "")
    }

    # Save per-capture manifest and append to global manifest.jsonl
    with open(host_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_data) + "\n")

    elapsed = time.time() - start_time
    avg_per_sample = elapsed / max(1, current_count)
    eta_min = (avg_per_sample * (total_count - current_count)) / 60.0
    pct = (current_count / total_count) * 100.0

    if os.path.exists(host_pcap_path) and os.path.getsize(host_pcap_path) > 500:
        print(f"  [{current_count}/{total_count} ({pct:5.1f}%) | ETA: {eta_min:4.1f} min] [{profile.upper()}] [{sample_id:04d}] OK: {pcap_filename} ({os.path.getsize(host_pcap_path)} B)")
        return True
    else:
        sz = os.path.getsize(host_pcap_path) if os.path.exists(host_pcap_path) else 0
        print(f"  [{current_count}/{total_count} ({pct:5.1f}%) | ETA: {eta_min:4.1f} min] [{profile.upper()}] [{sample_id:04d}] WARN: size={sz} B")
        return False


def main():
    parser = argparse.ArgumentParser(description="v2 Scaled Dataset Collector")
    parser.add_argument("--pilot", action="store_true", help="Run 2,000-flow pilot capture campaign")
    parser.add_argument("--samples-per-profile", type=int, default=112, help="Number of samples per class per profile (112 * 18 = 2016 for pilot, 500 for full)")
    parser.add_argument("--epoch", default="A", help="Campaign epoch (A or B)")
    args = parser.parse_args()

    if args.pilot:
        args.samples_per_profile = 112  # 112 * 6 classes * 3 profiles = 2,016 samples

    os.makedirs(RAW_PCAP_DIR, exist_ok=True)
    git_commit = get_git_commit()
    disable_offloads()

    # Generate design matrix
    tasks = []
    for profile in NETEM_PROFILES:
        for class_name, gen_mode in TRAFFIC_CLASSES:
            for sid in range(1, args.samples_per_profile + 1):
                beh = BEHAVIOURS[sid % len(BEHAVIOURS)]
                tasks.append((class_name, gen_mode, profile, beh, sid))

    # Interleave tasks across classes within each profile to eliminate block bias
    random.seed(42)
    # Group by profile to minimize netem switching overhead
    by_profile_tasks = {}
    for t in tasks:
        p = t[2]
        by_profile_tasks.setdefault(p, []).append(t)

    for p in by_profile_tasks:
        random.shuffle(by_profile_tasks[p])

    ordered_tasks = []
    for p in NETEM_PROFILES:
        ordered_tasks.extend(by_profile_tasks.get(p, []))

    total = len(ordered_tasks)
    print(f"=== Starting v2 Collection Campaign (Epoch {args.epoch}): {total} PCAPs ===")
    print(f"Commit: {git_commit} | Offloads disabled | Stack: utls-HelloChrome_Auto")

    # Clear old manifest if starting fresh
    if not os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "w") as f:
            pass

    start_time = time.time()
    counter = 0
    current_profile = None

    for class_name, gen_mode, profile, beh, sid in ordered_tasks:
        if profile != current_profile:
            apply_netem(profile)
            current_profile = profile

        counter += 1
        capture_session(
            class_name, gen_mode, profile, beh, sid,
            args.epoch, git_commit, counter, total, start_time
        )
        time.sleep(random.uniform(0.08, 0.2))

    apply_netem("reset")
    total_elapsed = time.time() - start_time
    print(f"\n=== [DONE] v2 Dataset Collection completed in {total_elapsed/60:.2f} minutes! ===")


if __name__ == "__main__":
    main()
