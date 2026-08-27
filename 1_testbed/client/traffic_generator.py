import asyncio
import argparse
import random
import time
import ssl
import requests
import websockets
import json
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SOCKS_PROXY = "socks5h://127.0.0.1:9050"

WEBTUNNEL_TARGETS = [
    "https://check.torproject.org",
    "https://en.wikipedia.org/wiki/Main_Page",
    "https://duckduckgo.com",
    "https://legitimate-servers:8443/api/v1/feed",
    "https://legitimate-servers:8443/web/assets/bundle.js",
]

# SSL Context for legitimate TLS connections
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

def run_webtunnel_request(num_requests: int = None):
    """Sends diverse randomized requests via WebTunnel Tor SOCKS5 proxy."""
    session = requests.Session()
    session.proxies = {"http": SOCKS_PROXY, "https": SOCKS_PROXY}
    
    if num_requests is None:
        num_requests = random.randint(1, 3)
        
    for i in range(num_requests):
        target = random.choice(WEBTUNNEL_TARGETS)
        try:
            resp = session.get(target, timeout=20, verify=False)
        except Exception:
            pass
        time.sleep(random.uniform(0.3, 1.5))

async def run_legitimate_ws_ticker(ws_url: str, duration_sec: float = None):
    """Connects to legitimate WebSocket ticker over WSS (TLS 1.3) and receives bursts."""
    if duration_sec is None:
        duration_sec = random.uniform(2.0, 4.5)
        
    try:
        async with websockets.connect(ws_url, ssl=SSL_CTX) as ws:
            start_t = time.time()
            while time.time() - start_t < duration_sec:
                _ = await ws.recv()
    except Exception:
        pass

async def run_legitimate_ws_chat(ws_url: str, num_messages: int = None):
    """Simulates interactive keystrokes and chat replies over WSS (TLS 1.3)."""
    if num_messages is None:
        num_messages = random.randint(4, 10)
        
    try:
        async with websockets.connect(ws_url, ssl=SSL_CTX) as ws:
            for i in range(num_messages):
                msg = f"msg_{i}_{'k' * random.randint(5, 80)}"
                await ws.send(msg)
                _ = await ws.recv()
                await asyncio.sleep(random.uniform(0.05, 0.4))
    except Exception:
        pass

def run_legitimate_video_streaming(base_url: str, num_segments: int = None):
    """Simulates adaptive video playback over HTTPS (TLS 1.3) (burst-and-idle pattern)."""
    if num_segments is None:
        num_segments = random.randint(2, 5)
        
    for i in range(num_segments):
        url = f"{base_url}/video/segment_{i}.m4s"
        try:
            _ = requests.get(url, timeout=10, verify=False)
        except Exception:
            pass
        time.sleep(random.uniform(0.5, 1.2))

def run_legitimate_web_assets(base_url: str, num_assets: int = None):
    """Simulates web page asset downloading over HTTPS (TLS 1.3)."""
    if num_assets is None:
        num_assets = random.randint(5, 15)
    for i in range(num_assets):
        url = f"{base_url}/web/assets/item_{i}.bin"
        try:
            _ = requests.get(url, timeout=5, verify=False)
        except Exception:
            pass
        time.sleep(random.uniform(0.02, 0.15))

def main():
    parser = argparse.ArgumentParser(description="Testbed Traffic Generator (TLS 1.3 Enabled)")
    parser.add_argument("--mode", choices=["webtunnel", "ws_ticker", "ws_chat", "video", "web_assets", "all"], default="all")
    parser.add_argument("--ws-server", default="wss://legitimate-servers:8443")
    parser.add_argument("--http-server", default="https://legitimate-servers:8443")
    args = parser.parse_args()

    if args.mode in ["webtunnel", "all"]:
        run_webtunnel_request()
    if args.mode in ["ws_ticker", "all"]:
        asyncio.run(run_legitimate_ws_ticker(f"{args.ws_server}/ws/ticker"))
    if args.mode in ["ws_chat", "all"]:
        asyncio.run(run_legitimate_ws_chat(f"{args.ws_server}/ws/chat"))
    if args.mode in ["video", "all"]:
        run_legitimate_video_streaming(args.http_server)
    if args.mode in ["web_assets", "all"]:
        run_legitimate_web_assets(args.http_server)

if __name__ == "__main__":
    main()
