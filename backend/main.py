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

from backend.env_loader import load_dotenv


load_dotenv()

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
LOG_PAYLOAD_MAX_LENGTH = int(os.getenv("LOG_PAYLOAD_MAX_LENGTH", "4000"))

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


def redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        for key, item in value.items():
            if key in {"image_base64", "base64"} and isinstance(item, str):
                redacted[key] = f"<base64 length={len(item)}>"
            elif key in {"image_url", "product_url", "source_url"} and isinstance(item, str) and item.startswith("data:"):
                redacted[key] = f"<data-url length={len(item)}>"
            else:
                redacted[key] = redact_payload(item)
        return redacted
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    return value


def log_agent_payload(direction: str, payload: Any) -> None:
    try:
        text = json.dumps(redact_payload(payload), ensure_ascii=False, default=str, indent=2)
    except Exception:
        text = repr(payload)

    if len(text) > LOG_PAYLOAD_MAX_LENGTH:
        text = f"{text[:LOG_PAYLOAD_MAX_LENGTH]}... <truncated length={len(text)}>"

    logger.info("backend agent %s payload: %s", direction, text)


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
            "alternatives": [
                {
                    "product_id": 1,
                    "name": "Galaxy S24 FE",
                    "brand": "Samsung",
                    "category": product.get("category") or "스마트폰",
                    "price": 699000,
                    "rating": 4.4,
                    "return_rate": 3.2,
                    "image_url": "https://fdn2.gsmarena.com/vv/bigpic/samsung-galaxy-s24-fe.jpg",
                    "regret_score": 0.22,
                    "match_score": 0.91,
                    "recommendation_reason": "높은 평점과 낮은 반품률로 만족도가 검증된 상품입니다.",
                },
                {
                    "product_id": 2,
                    "name": "iPhone 15",
                    "brand": "Apple",
                    "category": product.get("category") or "스마트폰",
                    "price": 1250000,
                    "rating": 4.6,
                    "return_rate": 2.1,
                    "image_url": "https://fdn2.gsmarena.com/vv/pics/apple/apple-iphone-15-1.jpg",
                    "regret_score": 0.15,
                    "match_score": 0.88,
                    "recommendation_reason": "사용자 만족도와 제품 안정성이 높은 프리미엄 선택지입니다.",
                },
                {
                    "product_id": 3,
                    "name": "Pixel 8a",
                    "brand": "Google",
                    "category": product.get("category") or "스마트폰",
                    "price": 649000,
                    "rating": 4.4,
                    "return_rate": 2.8,
                    "image_url": "https://fdn2.gsmarena.com/vv/bigpic/google-pixel-8a.jpg",
                    "regret_score": 0.19,
                    "match_score": 0.86,
                    "recommendation_reason": "합리적인 가격에 안정적인 사용 경험을 제공합니다.",
                },
                {
                    "product_id": 4,
                    "name": "Xiaomi 14T",
                    "brand": "Xiaomi",
                    "category": product.get("category") or "스마트폰",
                    "price": 599000,
                    "rating": 4.2,
                    "return_rate": 3.4,
                    "image_url": "https://fdn2.gsmarena.com/vv/bigpic/xiaomi-14t.jpg",
                    "regret_score": 0.24,
                    "match_score": 0.82,
                    "recommendation_reason": "가격 대비 성능이 좋아 예산 부담을 낮출 수 있습니다.",
                },
                {
                    "product_id": 5,
                    "name": "OnePlus 12R",
                    "brand": "OnePlus",
                    "category": product.get("category") or "스마트폰",
                    "price": 749000,
                    "rating": 4.3,
                    "return_rate": 3.0,
                    "image_url": "https://fdn2.gsmarena.com/vv/bigpic/oneplus-12r.jpg",
                    "regret_score": 0.21,
                    "match_score": 0.84,
                    "recommendation_reason": "성능과 배터리 만족도가 균형 잡힌 대체상품입니다.",
                },
            ],
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
            agent_request = {
                "type": "request",
                "session_id": session_id,
                "data": payload,
            }
            log_agent_payload("send", agent_request)
            await agent_ws.send(json.dumps(agent_request, ensure_ascii=False))

            while True:
                raw = await asyncio.wait_for(agent_ws.recv(), timeout=AGENT_TIMEOUT)
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    log_agent_payload("receive_raw", raw)
                    raise
                log_agent_payload("receive", message)
                await send_json(frontend_ws, message)
                if message.get("type") in ("result", "error", "product_candidates"):
                    break
    except Exception as exc:
        logger.warning("agent relay failed: %s", exc)
        await send_json(frontend_ws, {
            "type": "error",
            "session_id": session_id,
            "message": f"AI Agent 분석 결과를 가져오지 못했습니다: {exc}",
        })


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
