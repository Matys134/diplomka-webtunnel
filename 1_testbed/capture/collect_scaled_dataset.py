#!/usr/bin/env python3
"""
Multi-Environment Scaled Dataset Collector:
Captures raw PCAP traffic for 6 classes under 3 NetEm network profiles (Broadband, LTE, Lossy WAN).
Supports resume / incremental scaling and prints real-time progress with ETA.
"""
import os
import time
import subprocess
import argparse
import random

COMPOSE_FILE = "1_testbed/docker-compose.yml"
RAW_PCAP_DIR = "data/raw_pcap"

TRAFFIC_CLASSES = [
    ("webtunnel", "webtunnel"),
    ("direct_web_browsing", "direct_browsing"),
    ("websocket_ticker", "ws_ticker"),
    ("websocket_chat", "ws_chat"),
    ("video_streaming", "video"),
    ("web_assets", "web_assets"),
]

NETEM_PROFILES = ["broadband", "lte", "lossy"]


def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def apply_netem(profile):
    print(f"\n>>> [tc-netem] Switching network profile to: {profile.upper()} <<<")
    cmd = f"docker compose -f {COMPOSE_FILE} exec client /app/netem_profiles.sh {profile}"
    res = run_cmd(cmd)
    print(res.stdout.strip())
    time.sleep(1)


def capture_session(class_name, traffic_type, profile, sample_id, current_count, total_count, start_time):
    pcap_filename = f"{class_name}_{profile}_{sample_id:04d}.pcap"
    container_pcap_path = f"/data/raw_pcap/{pcap_filename}"
    host_pcap_path = os.path.join(RAW_PCAP_DIR, pcap_filename)

    # Resume check: if PCAP already exists and is valid (> 500 B), skip capturing
    if os.path.exists(host_pcap_path) and os.path.getsize(host_pcap_path) > 500:
        elapsed = time.time() - start_time
        pct = (current_count / total_count) * 100.0
        print(f"  [{current_count}/{total_count} ({pct:5.1f}%)] [{profile.upper()}] [{sample_id:04d}] EXISTS: {pcap_filename} ({os.path.getsize(host_pcap_path)} B)")
        return True

    # 1. Start background tcpdump inside client container
    tcpdump_cmd = f"docker compose -f {COMPOSE_FILE} exec -d client tcpdump -i eth0 -s 0 -w {container_pcap_path}"
    run_cmd(tcpdump_cmd)
    time.sleep(0.35)

    # 2. Trigger traffic generation inside client
    gen_cmd = f"docker compose -f {COMPOSE_FILE} exec client python3 traffic_generator.py --mode {traffic_type}"
    run_cmd(gen_cmd)
    time.sleep(0.55)

    # 3. Stop tcpdump
    stop_cmd = f"docker compose -f {COMPOSE_FILE} exec client pkill -f tcpdump"
    run_cmd(stop_cmd)
    time.sleep(0.25)

    elapsed = time.time() - start_time
    avg_per_sample = elapsed / max(1, current_count)
    remaining_sec = avg_per_sample * (total_count - current_count)
    pct = (current_count / total_count) * 100.0
    eta_min = remaining_sec / 60.0

    if os.path.exists(host_pcap_path) and os.path.getsize(host_pcap_path) > 500:
        print(f"  [{current_count}/{total_count} ({pct:5.1f}%) | ETA: {eta_min:4.1f} min] [{profile.upper()}] [{sample_id:04d}] CAPTURED: {pcap_filename} ({os.path.getsize(host_pcap_path)} B)")
        return True
    else:
        sz = os.path.getsize(host_pcap_path) if os.path.exists(host_pcap_path) else 0
        print(f"  [{current_count}/{total_count} ({pct:5.1f}%) | ETA: {eta_min:4.1f} min] [{profile.upper()}] [{sample_id:04d}] WARN: size={sz} B")
        return False


def main():
    parser = argparse.ArgumentParser(description="Multi-Environment Scaled Dataset Collector")
    parser.add_argument("--samples-per-profile", type=int, default=500, help="Number of samples per class per network profile")
    args = parser.parse_args()

    os.makedirs(RAW_PCAP_DIR, exist_ok=True)
    total = len(NETEM_PROFILES) * len(TRAFFIC_CLASSES) * args.samples_per_profile
    print(f"=== Starting Scaled Collection: {len(NETEM_PROFILES)} profiles x {len(TRAFFIC_CLASSES)} classes x {args.samples_per_profile} samples = {total} PCAPs ===")

    start_time = time.time()
    counter = 0

    for profile in NETEM_PROFILES:
        apply_netem(profile)
        for class_name, traffic_type in TRAFFIC_CLASSES:
            print(f"\n--- Generating class: {class_name} under {profile} ({args.samples_per_profile} samples) ---")
            for i in range(1, args.samples_per_profile + 1):
                counter += 1
                capture_session(class_name, traffic_type, profile, i, counter, total, start_time)
                time.sleep(random.uniform(0.08, 0.25))

    apply_netem("reset")
    total_elapsed = time.time() - start_time
    print(f"\n=== [DONE] Scaled dataset collection completed in {total_elapsed/60:.2f} minutes! ===")


if __name__ == "__main__":
    main()
