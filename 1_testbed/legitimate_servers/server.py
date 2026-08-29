import os
import asyncio
import json
import random
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, Response
import uvicorn

app = FastAPI(title="Legitimate Hard Negatives Mock Server")

# 1. WebSocket Live Ticker (Real-World Financial Depth / L2 Orderbook Snapshots)
@app.websocket("/ws/ticker")
async def websocket_ticker(websocket: WebSocket):
    await websocket.accept()
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", "XRPUSDT"]
    try:
        while True:
            mode = random.choices(["small_tick", "orderbook_snapshot", "kline_batch"], weights=[0.40, 0.45, 0.15])[0]
            if mode == "small_tick":
                # Small tick update (60 - 120 Bytes)
                msg = {
                    "e": "trade",
                    "s": random.choice(symbols),
                    "p": round(random.uniform(1.0, 95000.0), 2),
                    "q": round(random.uniform(0.001, 15.0), 4),
                    "t": int(time.time() * 1000)
                }
            elif mode == "orderbook_snapshot":
                # L2 Orderbook depth snapshot (350 - 750 Bytes) - directly overlaps with Tor cell framing!
                num_levels = random.randint(8, 18)
                bids = [[round(random.uniform(90000.0, 95000.0), 2), round(random.uniform(0.1, 5.0), 3)] for _ in range(num_levels)]
                asks = [[round(random.uniform(95000.0, 99000.0), 2), round(random.uniform(0.1, 5.0), 3)] for _ in range(num_levels)]
                msg = {
                    "e": "depthUpdate",
                    "s": random.choice(symbols),
                    "u": random.randint(100000, 999999),
                    "bids": bids,
                    "asks": asks,
                    "ts": int(time.time() * 1000)
                }
            else:
                # Kline batch / historical candle burst (500 - 1100 Bytes)
                candles = [{"t": int(time.time()*1000) - i*60000, "o": 94000.0, "c": 94200.0, "v": 12.5} for i in range(random.randint(6, 14))]
                msg = {"e": "kline_batch", "s": random.choice(symbols), "candles": candles}
                
            await websocket.send_text(json.dumps(msg))
            await asyncio.sleep(random.uniform(0.05, 0.30))
    except WebSocketDisconnect:
        pass

# 2. WebSocket Interactive Chat / Collaboration (Rich Messages, Embeds, State Sync)
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await asyncio.sleep(random.uniform(0.02, 0.15))
            
            # Mix of response types: rich message, embed, avatar/profile sync, typing event
            resp_type = random.choices(["rich_msg", "state_sync", "typing_ack"], weights=[0.55, 0.30, 0.15])[0]
            if resp_type == "rich_msg":
                # Rich text message with embeds and reactions (320 - 820 Bytes)
                payload_len = random.randint(250, 750)
                reply = {
                    "type": "message_create",
                    "channel_id": "1234567890",
                    "author": {"id": "user_987", "name": "collaborator", "avatar": "hash_" + "a"*32},
                    "content": "R" * payload_len,
                    "embeds": [{"title": "Shared Resource", "url": "https://example.com/item", "description": "d"*120}],
                    "reactions": [{"emoji": "👍", "count": 3}],
                    "server_time": time.time()
                }
            elif resp_type == "state_sync":
                # Presence / Workspace canvas state synchronization (400 - 900 Bytes)
                reply = {
                    "type": "presence_update",
                    "status": "online",
                    "canvas_deltas": [{"x": random.randint(0, 1920), "y": random.randint(0, 1080), "op": "draw", "buf": "b" * random.randint(200, 600)}],
                    "timestamp": time.time()
                }
            else:
                # Small typing ack (80 - 150 Bytes)
                reply = {"type": "typing_ack", "user_id": "user_987", "ts": time.time()}
                
            await websocket.send_text(json.dumps(reply))
    except WebSocketDisconnect:
        pass

# 3. Adaptive DASH / HLS Video Segment & ABR Control Endpoint
@app.get("/video/manifest.mpd")
async def video_manifest():
    # Dynamic MPD manifest (400 - 800 Bytes)
    manifest = f"<MPD minBufferTime='PT1.5S'><Period><AdaptationSet mimeType='video/mp4' bandwidth='{random.randint(2000, 8000)}k' /></Period></MPD>"
    return Response(content=manifest, media_type="application/dash+xml")

@app.get("/video/abr_control")
async def video_abr_control():
    # ABR bandwidth estimation and rate control packet (350 - 650 Bytes)
    abr_data = {
        "status": "ok",
        "recommended_bitrate": random.choice([2500000, 4500000, 8000000]),
        "buffer_target_sec": 4.0,
        "segment_duration_sec": 2.0,
        "quality_profile": "1080p60_av1",
        "telemetry_sync": "t" * random.randint(150, 450)
    }
    return Response(content=json.dumps(abr_data), media_type="application/json")

@app.get("/video/segment_{segment_id}.m4s")
async def video_segment(segment_id: int):
    # Variable audio/video chunk sizes (audio: 20-80KB, video: 200-1500KB)
    if segment_id % 3 == 0:
        chunk_size = random.randint(25 * 1024, 75 * 1024)  # Audio segment
    else:
        chunk_size = random.randint(180 * 1024, 1400 * 1024) # Video segment
    data = b"\x00" * chunk_size
    return Response(content=data, media_type="video/iso.segment")

# 4. Multi-Asset Web Endpoint (Mixed Assets: CSS, Font subsets, WASM, JSON bundles)
@app.get("/web/assets/{asset_id}")
async def web_asset(asset_id: str):
    # Diverse asset sizes: icons/fonts (350B - 4KB), bundles (20KB - 250KB)
    if "icon" in asset_id or "meta" in asset_id:
        asset_size = random.randint(350, 1800)  # Small icon / manifest (350 - 1800 B)
    else:
        asset_size = random.randint(4 * 1024, 250 * 1024)
    return Response(content=b"A" * asset_size, media_type="application/octet-stream")

# 5. REST & GraphQL API Endpoints (Realistic SPA Payloads)
@app.get("/api/v1/feed")
async def rest_feed():
    num_items = random.randint(5, 18)
    items = [{"id": i, "title": f"Article {i}", "summary": "S" * random.randint(30, 80), "tags": ["tech", "web"]} for i in range(num_items)]
    return {"status": "active", "items": items, "count": num_items, "server_ts": time.time()}

@app.post("/api/v1/graphql")
async def graphql_endpoint(request: Request):
    body = await request.body()
    # Return structured GraphQL response matching query scale (350 - 850 Bytes)
    resp = {
        "data": {
            "user": {"id": "usr_42", "name": "test_user", "preferences": {"theme": "dark", "lang": "cs"}},
            "notifications": [{"id": i, "text": "Notice " + "x"*random.randint(20, 60), "read": False} for i in range(random.randint(2, 6))]
        },
        "extensions": {"tracing": {"duration": random.randint(15, 45)}}
    }
    return Response(content=json.dumps(resp), media_type="application/json")

@app.post("/api/v1/telemetry")
async def post_telemetry(request: Request):
    body = await request.body()
    return {"status": "received", "bytes": len(body), "timestamp": time.time()}

@app.post("/api/v1/upload")
async def post_upload(request: Request):
    body = await request.body()
    return {"status": "ok", "uploaded_size": len(body)}

@app.get("/")
async def root():
    return HTMLResponse("<h1>Legitimate Service Cluster (TLS Encrypted)</h1><p>Mock server for Hard Negatives.</p>")

if __name__ == "__main__":
    ssl_key = "/app/certs/server.key" if os.path.exists("/app/certs/server.key") else None
    ssl_cert = "/app/certs/server.crt" if os.path.exists("/app/certs/server.crt") else None
    
    # Run with HTTP/2 and WebSockets over TLS on port 8443
    from hypercorn.config import Config
    from hypercorn.asyncio import serve
    
    config = Config()
    config.bind = ["0.0.0.0:8443"]
    if ssl_key and ssl_cert:
        config.keyfile = ssl_key
        config.certfile = ssl_cert
        config.alpn_protocols = ["h2", "http/1.1"]
    config.accesslog = "-"
    config.loglevel = "WARNING"
    
    asyncio.run(serve(app, config))
