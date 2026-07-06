import asyncio
import json
import logging
import os
from typing import Any, Dict

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from env_loader import load_dotenv

load_dotenv()

from product_extractor import extract_alternative_candidates, extract_product_candidates_from_image, extract_product_info
from predictor import RegretPredictor, resolve_llm_model_config, warm_up_preference_llm


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger("stopbuy-agent")

AGENT_HOST = os.getenv("AGENT_HOST", "127.0.0.1")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8765"))
REGRET_THRESHOLD = float(os.getenv("REGRET_THRESHOLD", "0.4"))
LOG_PAYLOAD_MAX_LENGTH = int(os.getenv("LOG_PAYLOAD_MAX_LENGTH", "4000"))
MODEL_PATH = os.getenv("REGRET_MODEL_PATH")
DATASET_PATH = os.getenv("REGRET_DATASET_PATH")
PREFERENCE_LLM_WARMUP_REQUIRED = os.getenv("PREFERENCE_LLM_WARMUP_REQUIRED", "false").lower() in {"1", "true", "yes", "y"}

predictor = RegretPredictor(
    model_path=MODEL_PATH,
    dataset_path=DATASET_PATH,
    threshold=REGRET_THRESHOLD,
)


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


def log_payload(direction: str, payload: Dict[str, Any]) -> None:
    try:
        text = json.dumps(redact_payload(payload), ensure_ascii=False, default=str, indent=2)
    except Exception:
        text = repr(payload)

    if LOG_PAYLOAD_MAX_LENGTH > 0 and len(text) > LOG_PAYLOAD_MAX_LENGTH:
        text = f"{text[:LOG_PAYLOAD_MAX_LENGTH]}... <truncated length={len(text)}>"

    logger.info("agent %s payload: %s", direction, text)


async def send_message(ws: ServerConnection, message: Dict[str, Any]) -> None:
    log_payload("send", message)
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
        if data.get("input_type") == "image" and not data.get("product"):
            await send_progress(ws, session_id, 25, "이미지에서 상품 정보를 추출하고 있습니다.")
            candidate_result = await extract_product_candidates_from_image(data)
            await send_progress(ws, session_id, 70, "네이버 쇼핑에서 유사 상품을 검색하고 있습니다.")
            await send_message(ws, {
                "type": "product_candidates",
                "session_id": session_id,
                "data": candidate_result,
            })
            return

        if data.get("input_type") == "url":
            product_url = str(data.get("product_url") or "").lower()
            if "coupang.com" in product_url:
                await send_progress(ws, session_id, 25, "쿠팡 상품 정보를 추출하고 있습니다. 접근이 제한되면 브라우저 방식으로 다시 확인합니다.")
            elif "gmarket" in product_url:
                await send_progress(ws, session_id, 25, "G마켓 상품 정보를 추출하고 있습니다. 접근이 제한되면 상품코드와 검색 결과로 보강합니다.")
            else:
                await send_progress(ws, session_id, 25, "상품 정보를 추출하고 있습니다.")
        else:
            await send_progress(ws, session_id, 25, "상품 정보를 추출하고 있습니다.")
        product = await extract_product_info(data)
        if data.get("input_type") == "url" and isinstance(product, dict) and product.get("product_info_missing"):
            await send_message(ws, {
                "type": "error",
                "session_id": session_id,
                "message": "상품정보를 가져오지 못했습니다. 네이버 로그인 또는 접근 제한으로 상품 상세정보를 확인할 수 없습니다. 다른 상품 URL을 입력하거나 이미지를 업로드하여 재시도하세요.",
                "data": {
                    "product_url": product.get("product_url") or product.get("source_url"),
                    "product_info_source": product.get("product_info_source"),
                    "url_fetch_error": product.get("url_fetch_error"),
                    "search_query": product.get("search_query"),
                },
            })
            return
        if data.get("input_type") == "manual" and isinstance(product, dict):
            has_rating = product.get("rating") not in (None, "")
            has_review_count = product.get("review_count") not in (None, "")
            if has_rating or has_review_count:
                product["review_data_available"] = True

        llm_config = resolve_llm_model_config()
        if llm_config.get("provider") in {"local", "local_hf", "hf", "transformers"}:
            await send_progress(
                ws,
                session_id,
                55,
                f"로컬 LLM 모델을 로딩하고 있습니다. 최초 실행은 오래 걸릴 수 있습니다. ({llm_config.get('model_id')})",
            )
        else:
            await send_progress(ws, session_id, 55, "구매 후회 가능성을 예측하고 있습니다.")
        await send_progress(ws, session_id, 70, "네이버 쇼핑에서 조건에 맞는 대체상품을 검색하고 있습니다.")
        alternative_result = await extract_alternative_candidates(
            data.get("user") or {},
            product,
            max(predictor.max_alternative_products * 3, predictor.max_alternative_products),
        )
        product["alternative_candidate_queries"] = alternative_result.get("queries") or []
        product["alternative_candidate_errors"] = alternative_result.get("errors") or []
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: predictor.predict(
                user=data.get("user") or {},
                product=product,
                alternative_candidates=alternative_result.get("candidates") or [],
            ),
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
            "message": f"Agent 처리 중 오류가 발생했습니다: {exc}",
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

            log_payload("recv", message)
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
    loop = asyncio.get_running_loop()
    try:
        warmup_result = await loop.run_in_executor(None, warm_up_preference_llm)
        if warmup_result.get("warmed"):
            logger.info(
                "preference LLM is ready: requested=%s provider=%s model_id=%s",
                warmup_result.get("requested"),
                warmup_result.get("provider"),
                warmup_result.get("model_id"),
            )
    except Exception:
        logger.exception("preference LLM warm-up failed")
        if PREFERENCE_LLM_WARMUP_REQUIRED:
            raise

    logger.info("StopBuy2.0 Agent listening on ws://%s:%s", AGENT_HOST, AGENT_PORT)
    async with serve(handle_client, AGENT_HOST, AGENT_PORT, ping_interval=20, ping_timeout=10):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
