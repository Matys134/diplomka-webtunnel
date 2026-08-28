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

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

# SSL Context for legitimate TLS connections
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

def run_webtunnel_request(num_requests: int = None, http_server: str = "https://legitimate-servers:8443"):
    """Sends diverse randomized requests via WebTunnel Tor SOCKS5 proxy."""
    session = requests.Session()
    session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
    session.proxies = {"http": SOCKS_PROXY, "https": SOCKS_PROXY}
    
    if num_requests is None:
        num_requests = random.randint(1, 3)
        
    for i in range(num_requests):
        target = random.choice(WEBTUNNEL_TARGETS)
        try:
            # 70% GET, 30% POST with small payload
            if random.random() < 0.7 or "legitimate-servers" not in target:
                resp = session.get(target, timeout=20, verify=False)
            else:
                payload = {"data": "W" * random.randint(200, 800), "client_ts": time.time()}
                resp = session.post(f"{http_server}/api/v1/telemetry", json=payload, timeout=20, verify=False)
        except Exception:
            pass
        time.sleep(random.uniform(0.3, 1.5))

def run_direct_web_browsing(num_requests: int = None, http_server: str = "https://legitimate-servers:8443"):
    """Sends direct HTTPS requests with realistic browsing mix (GET + Upload/POST) over TLS 1.3."""
    session = requests.Session()
    session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
    
    if num_requests is None:
        num_requests = random.randint(1, 3)
        
    for i in range(num_requests):
        target = random.choice(WEBTUNNEL_TARGETS)
        try:
            resp = session.get(target, timeout=15, verify=False)
        except Exception:
            pass
        time.sleep(random.uniform(0.2, 0.8))
        
    # Simulate realistic modern web upload / telemetry (300 - 1200 Bytes upstream payload)
    if random.random() < 0.6:
        try:
            upload_size = random.randint(300, 1200)
            payload = {"telemetry": "D" * upload_size, "event": "page_view", "ts": time.time()}
            _ = session.post(f"{http_server}/api/v1/telemetry", json=payload, timeout=10, verify=False)
        except Exception:
            pass

async def run_legitimate_ws_ticker(ws_url: str, duration_sec: float = None):
    """Connects to legitimate WebSocket ticker over WSS (TLS 1.3) and receives bursts with client heartbeats."""
    if duration_sec is None:
        duration_sec = random.uniform(2.0, 4.5)
        
    try:
        async with websockets.connect(ws_url, ssl=SSL_CTX) as ws:
            start_t = time.time()
            while time.time() - start_t < duration_sec:
                # Occasional client subscription update (150 - 450 Bytes)
                if random.random() < 0.15:
                    sub_msg = {"action": "subscribe", "channels": ["trades", "depth", "kline"], "client_id": "x" * random.randint(100, 350)}
                    await ws.send(json.dumps(sub_msg))
                _ = await ws.recv()
    except Exception:
        pass

async def run_legitimate_ws_chat(ws_url: str, num_messages: int = None):
    """Simulates interactive chat with realistic message lengths (200 - 800 Bytes upstream) over WSS."""
    if num_messages is None:
        num_messages = random.randint(4, 10)
        
    try:
        async with websockets.connect(ws_url, ssl=SSL_CTX) as ws:
            for i in range(num_messages):
                msg_len = random.randint(200, 800)
                msg = f"msg_{i}_{'k' * msg_len}"
                await ws.send(msg)
                _ = await ws.recv()
                await asyncio.sleep(random.uniform(0.05, 0.4))
    except Exception:
        pass

def run_legitimate_video_streaming(base_url: str, num_segments: int = None):
    """Simulates adaptive video playback over HTTPS (TLS 1.3) (burst-and-idle pattern)."""
    if num_segments is None:
        num_segments = random.randint(2, 5)
        
    session = requests.Session()
    session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
    for i in range(num_segments):
        url = f"{base_url}/video/segment_{i}.m4s"
        try:
            _ = session.get(url, timeout=10, verify=False)
        except Exception:
            pass
        time.sleep(random.uniform(0.5, 1.2))

def run_legitimate_web_assets(base_url: str, num_assets: int = None):
    """Simulates web page asset downloading over HTTPS (TLS 1.3)."""
    if num_assets is None:
        num_assets = random.randint(5, 15)
    session = requests.Session()
    session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
    for i in range(num_assets):
        url = f"{base_url}/web/assets/item_{i}.bin"
        try:
            _ = session.get(url, timeout=5, verify=False)
        except Exception:
            pass
        time.sleep(random.uniform(0.02, 0.15))

def main():
    parser = argparse.ArgumentParser(description="Testbed Traffic Generator (TLS 1.3 Enabled)")
    parser.add_argument("--mode", choices=["webtunnel", "direct_browsing", "ws_ticker", "ws_chat", "video", "web_assets", "all"], default="all")
    parser.add_argument("--ws-server", default="wss://legitimate-servers:8443")
    parser.add_argument("--http-server", default="https://legitimate-servers:8443")
    args = parser.parse_args()

    if args.mode in ["webtunnel", "all"]:
        run_webtunnel_request()
    if args.mode in ["direct_browsing", "all"]:
        run_direct_web_browsing()
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
