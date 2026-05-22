import asyncio
import json
import logging
import os
from typing import Any, Dict

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from product_extractor import extract_product_info
from predictor import RegretPredictor


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger("stopbuy-agent")

AGENT_HOST = os.getenv("AGENT_HOST", "127.0.0.1")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8765"))
REGRET_THRESHOLD = float(os.getenv("REGRET_THRESHOLD", "0.4"))

predictor = RegretPredictor(threshold=REGRET_THRESHOLD)


async def send_message(ws: ServerConnection, message: Dict[str, Any]) -> None:
    await ws.send(json.dumps(message, ensure_ascii=False))


async def send_progress(ws: ServerConnection, session_id: str, progress: int, message: str) -> None:
    await send_message(ws, {
        "type": "progress",
        "session_id": session_id,
        "progress": progress,
        "message": message,
    })


async def handle_request(ws: ServerConnection, session_id: str, data: Dict[str, Any]) -> None:
    try:
        await send_progress(ws, session_id, 25, "상품 정보를 추출하고 있습니다.")
        product = await extract_product_info(data)

        await send_progress(ws, session_id, 55, "구매 후회 가능성을 예측하고 있습니다.")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: predictor.predict(user=data.get("user") or {}, product=product),
        )

        await send_progress(ws, session_id, 85, "대체상품 추천 결과를 정리하고 있습니다.")
        await asyncio.sleep(0.2)
        await send_message(ws, {
            "type": "result",
            "session_id": session_id,
            "data": result,
        })
    except Exception as exc:
        logger.exception("analysis failed: session=%s", session_id)
        await send_message(ws, {
            "type": "error",
            "session_id": session_id,
            "message": f"Agent 분석 중 오류가 발생했습니다: {exc}",
        })


async def handle_client(ws: ServerConnection) -> None:
    logger.info("backend connected: %s", ws.remote_address)
    try:
        async for raw in ws:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await send_message(ws, {"type": "error", "message": "잘못된 JSON 메시지입니다."})
                continue

            msg_type = message.get("type")
            session_id = message.get("session_id") or "unknown"
            if msg_type == "ping":
                await send_message(ws, {"type": "pong", "session_id": session_id})
            elif msg_type == "request":
                asyncio.create_task(handle_request(ws, session_id, message.get("data") or {}))
            else:
                await send_message(ws, {
                    "type": "error",
                    "session_id": session_id,
                    "message": f"지원하지 않는 메시지 타입입니다: {msg_type}",
                })
    except ConnectionClosed:
        logger.info("backend disconnected")


async def main() -> None:
    logger.info("StopBuy2.0 Agent listening on ws://%s:%s", AGENT_HOST, AGENT_PORT)
    async with serve(handle_client, AGENT_HOST, AGENT_PORT, ping_interval=20, ping_timeout=10):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
