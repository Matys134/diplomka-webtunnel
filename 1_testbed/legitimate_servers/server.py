import os
import asyncio
import json
import random
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, Response
import uvicorn

app = FastAPI(title="Legitimate Hard Negatives Mock Server")

# 1. WebSocket Live Ticker (Mimics Financial Streaming / Bursts)
@app.websocket("/ws/ticker")
async def websocket_ticker(websocket: WebSocket):
    await websocket.accept()
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", "XRPUSDT"]
    try:
        while True:
            burst_size = random.choices([1, 2, 4, 8], weights=[0.5, 0.3, 0.15, 0.05])[0]
            for _ in range(burst_size):
                msg = {
                    "stream": "trade",
                    "data": {
                        "symbol": random.choice(symbols),
                        "price": round(random.uniform(1.0, 95000.0), 2),
                        "quantity": round(random.uniform(0.001, 15.0), 4),
                        "timestamp": int(time.time() * 1000)
                    }
                }
                await websocket.send_text(json.dumps(msg))
            await asyncio.sleep(random.uniform(0.03, 0.35))
    except WebSocketDisconnect:
        pass

# 2. WebSocket Interactive Chat / Shell (Interactive Keystrokes)
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await asyncio.sleep(random.uniform(0.02, 0.18))
            reply = {
                "status": "ok",
                "echo": data,
                "server_time": time.time(),
                "payload": "x" * random.randint(30, 600)
            }
            await websocket.send_text(json.dumps(reply))
    except WebSocketDisconnect:
        pass

# 3. Adaptive DASH / HLS Video Segment Endpoint
@app.get("/video/segment_{segment_id}.m4s")
async def video_segment(segment_id: int):
    chunk_size = random.randint(150 * 1024, 1800 * 1024)
    data = b"\x00" * chunk_size
    return Response(content=data, media_type="video/iso.segment")

# 4. Multi-Asset Web Endpoint (Simulates Web Browsing Assets)
@app.get("/web/assets/{asset_id}")
async def web_asset(asset_id: str):
    asset_size = random.randint(2 * 1024, 250 * 1024)
    return Response(content=b"A" * asset_size, media_type="application/octet-stream")

@app.get("/api/v1/feed")
async def rest_feed():
    return {"status": "active", "items": [{"id": i, "content": f"Sample item {i}"} for i in range(20)]}

# 5. Realistic HTTP POST Upload / Telemetry endpoints
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
