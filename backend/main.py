import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from websockets.asyncio.client import connect


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger("stopbuy-backend")

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
AGENT_WS_URL = os.getenv("AGENT_WS_URL", "ws://127.0.0.1:8765")
AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "120"))

app = FastAPI(title="StopBuy2.0 Backend", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")


@app.get("/", response_model=None)
async def index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"name": "StopBuy2.0 Backend", "docs": "/docs", "websocket": "/ws/{session_id}"}


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "agent_ws_url": AGENT_WS_URL}


async def send_json(websocket: WebSocket, message: Dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(message, ensure_ascii=False))


async def fallback_result(session_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    product = data.get("product") or {}
    product_name = product.get("name") or ("URL 입력 상품" if data.get("product_url") else "이미지 입력 상품")
    return {
        "type": "result",
        "session_id": session_id,
        "data": {
            "product": {
                "name": product_name,
                "brand": product.get("brand"),
                "category": product.get("category"),
                "price": product.get("price") or 0,
                "rating": product.get("rating") or 3.5,
                "review_count": product.get("review_count") or 0,
                "return_rate": product.get("return_rate") or 5.0,
            },
            "product_name": product_name,
            "regret_score": 0.45,
            "regret_level": "medium",
            "model_regret_score": 0.42,
            "cause_score": 0.5,
            "threshold": 0.4,
            "should_reconsider": True,
            "regret_causes": [
                {
                    "code": "AGENT_UNAVAILABLE",
                    "title": "Agent 미연결",
                    "message": "AI Agent에 연결하지 못해 데모 분석 결과를 표시합니다.",
                    "severity": "medium",
                    "impact_score": 0.5,
                }
            ],
            "regret_reasons": ["AI Agent에 연결하지 못해 데모 분석 결과를 표시합니다."],
            "alternatives": [],
            "summary": "AI Agent 연결 후 실제 분석 결과를 확인할 수 있습니다.",
            "_demo": True,
        },
    }


async def relay_to_agent(frontend_ws: WebSocket, session_id: str, payload: Dict[str, Any]) -> None:
    try:
        await send_json(frontend_ws, {
            "type": "progress",
            "session_id": session_id,
            "progress": 10,
            "message": "Backend가 AI Agent에 연결하고 있습니다.",
        })

        async with connect(AGENT_WS_URL, open_timeout=10, ping_interval=20, ping_timeout=10) as agent_ws:
            await agent_ws.send(json.dumps({
                "type": "request",
                "session_id": session_id,
                "data": payload,
            }, ensure_ascii=False))

            while True:
                raw = await asyncio.wait_for(agent_ws.recv(), timeout=AGENT_TIMEOUT)
                message = json.loads(raw)
                await send_json(frontend_ws, message)
                if message.get("type") in ("result", "error"):
                    break
    except Exception as exc:
        logger.warning("agent relay failed: %s", exc)
        await send_json(frontend_ws, await fallback_result(session_id, payload))


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await send_json(websocket, {
                    "type": "error",
                    "session_id": session_id,
                    "message": "잘못된 JSON 메시지입니다.",
                })
                continue

            msg_type = message.get("type")
            if msg_type == "ping":
                await send_json(websocket, {"type": "pong", "session_id": session_id})
                continue
            if msg_type != "request":
                await send_json(websocket, {
                    "type": "error",
                    "session_id": session_id,
                    "message": f"지원하지 않는 메시지 타입입니다: {msg_type}",
                })
                continue

            await relay_to_agent(websocket, session_id, message.get("data") or {})
    except WebSocketDisconnect:
        logger.info("frontend disconnected: session=%s", session_id)
