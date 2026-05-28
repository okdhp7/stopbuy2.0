import asyncio
import base64
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from env_loader import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
NAVER_SHOPPING_DISPLAY = int(os.getenv("NAVER_SHOPPING_DISPLAY", "10"))
NAVER_SHOPPING_ENDPOINT = "https://openapi.naver.com/v1/search/shop.json"
NAVER_LOG_PAYLOAD_MAX_LENGTH = int(os.getenv("NAVER_LOG_PAYLOAD_MAX_LENGTH", "8000"))
NAVER_REVIEW_ENRICH_ENABLED = os.getenv("NAVER_REVIEW_ENRICH_ENABLED", "true").lower() not in {"0", "false", "no"}
NAVER_REVIEW_ENRICH_LIMIT = int(os.getenv("NAVER_REVIEW_ENRICH_LIMIT", "5"))
NAVER_REVIEW_SAMPLE_LIMIT = int(os.getenv("NAVER_REVIEW_SAMPLE_LIMIT", "3"))
logger = logging.getLogger("stopbuy-agent.naver")


CATEGORY_KEYWORDS = {
    "스마트폰": ["iphone", "galaxy", "pixel", "phone", "스마트폰", "아이폰", "갤럭시"],
    "노트북": ["laptop", "notebook", "macbook", "gram", "노트북", "맥북", "그램"],
    "이어폰/헤드폰": ["airpods", "headphone", "earbuds", "이어폰", "헤드폰", "버즈"],
    "TV": ["tv", "oled", "qled", "television", "티비", "텔레비전"],
    "생활가전": ["청소기", "냉장고", "세탁기", "공기청정기", "vacuum", "washer"],
}


## 문자열의 연속 공백을 정리하고 비어 있으면 None을 반환합니다.
def _clean_text(
    value: Optional[str],
) -> Optional[str]:
    if not value:
        return None
    return re.sub(r"\s+", " ", value).strip()


## 텍스트에서 첫 번째 숫자 또는 콤마 포함 숫자를 추출해 float로 변환합니다.
def _number_from_text(
    text: str,
) -> Optional[float]:
    match = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)", text)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


## 상품명/설명 텍스트에서 키워드 기반으로 상품 카테고리를 추론합니다.
def detect_category(
    text: str,
) -> Optional[str]:
    lowered = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return category
    return None


## 상품 URL의 호스트명을 기준으로 쇼핑몰 식별자를 반환합니다.
def detect_shop(
    url: str,
) -> str:
    host = urlparse(url).netloc.lower()
    if "coupang" in host:
        return "coupang"
    if "naver" in host:
        return "naver"
    if "11st" in host:
        return "11st"
    if "gmarket" in host:
        return "gmarket"
    if "amazon" in host:
        return "amazon"
    return "generic"


## HTML의 JSON-LD Product 데이터를 읽어 상품 정보를 추출합니다.
def parse_json_ld(
    soup: BeautifulSoup,
) -> Dict[str, Any]:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            candidates = data if isinstance(data, list) else [data]
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                graph = item.get("@graph")
                if isinstance(graph, list):
                    candidates.extend(graph)
                    continue
                if item.get("@type") not in ("Product", ["Product"]):
                    continue
                result: Dict[str, Any] = {}
                if item.get("name"):
                    result["name"] = item["name"]
                if item.get("description"):
                    result["description"] = item["description"]
                brand = item.get("brand")
                if isinstance(brand, dict):
                    result["brand"] = brand.get("name")
                elif brand:
                    result["brand"] = str(brand)
                image = item.get("image")
                if isinstance(image, list) and image:
                    result["image_url"] = image[0]
                elif image:
                    result["image_url"] = image
                offers = item.get("offers")
                if isinstance(offers, list) and offers:
                    offers = offers[0]
                if isinstance(offers, dict):
                    price = offers.get("price") or offers.get("lowPrice")
                    if price:
                        result["price"] = _number_from_text(str(price))
                rating = item.get("aggregateRating")
                if isinstance(rating, dict):
                    if rating.get("ratingValue"):
                        result["rating"] = float(rating["ratingValue"])
                    if rating.get("reviewCount"):
                        result["review_count"] = int(float(rating["reviewCount"]))
                if result:
                    return result
        except Exception:
            continue
    return {}


## URL 페이지를 가져와 메타태그, JSON-LD, 본문 텍스트에서 상품 정보를 추출합니다.
async def extract_from_url(
    url: str,
) -> Dict[str, Any]:
    headers = {
        "User-Agent": "Mozilla/5.0 StopBuy2.0 Product Analyzer",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    try:
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True, headers=headers) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text
    except Exception:
        return {
            "name": "URL 입력 상품",
            "source_url": url,
            "shop": detect_shop(url),
            "description": "URL에서 상품 정보를 자동 추출하지 못했습니다. 기본값으로 분석합니다.",
        }

    soup = BeautifulSoup(html, "lxml")
    product = parse_json_ld(soup)

    if not product.get("name"):
        og_title = soup.find("meta", property="og:title")
        title = og_title.get("content") if og_title else None
        if not title and soup.title:
            title = soup.title.get_text(" ", strip=True)
        product["name"] = _clean_text(title) or "URL 입력 상품"

    if not product.get("description"):
        og_desc = soup.find("meta", property="og:description")
        product["description"] = _clean_text(og_desc.get("content")) if og_desc else None

    if not product.get("image_url"):
        og_image = soup.find("meta", property="og:image")
        product["image_url"] = og_image.get("content") if og_image else None

    visible_text = soup.get_text(" ", strip=True)[:8000]
    if not product.get("price"):
        product["price"] = _number_from_text(visible_text) or 0
    if not product.get("rating"):
        product["rating"] = 3.5
    if not product.get("review_count"):
        product["review_count"] = 0
    if not product.get("return_rate"):
        product["return_rate"] = 5.0
    if not product.get("category"):
        product["category"] = detect_category(f"{product.get('name', '')} {product.get('description', '')} {visible_text}")

    product["source_url"] = url
    product["shop"] = detect_shop(url)
    return product


## base64 이미지에서 OpenAI 비전 모델을 사용해 상품 정보를 추출합니다.
async def extract_from_image(
    image_base64: str,
) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {
            "name": "이미지 입력 상품",
            "category": None,
            "price": 0,
            "rating": 3.5,
            "review_count": 0,
            "return_rate": 5.0,
            "description": "이미지 분석을 위해 OPENAI_API_KEY가 필요합니다. 기본값으로 분석합니다.",
        }

    try:
        from openai import AsyncOpenAI

        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        # Validate that the input is base64 before sending it onward.
        base64.b64decode(image_base64[:200] + "===")

        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "이미지 속 구매 후보 상품 정보를 JSON으로 추출하세요. "
                                "키는 name, brand, category, price, rating, review_count, return_rate, description만 사용하세요. "
                                "모르는 값은 null로 둡니다."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "low"},
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        parsed.setdefault("name", "이미지 입력 상품")
        parsed.setdefault("rating", 3.5)
        parsed.setdefault("review_count", 0)
        parsed.setdefault("return_rate", 5.0)
        return parsed
    except Exception:
        return {
            "name": "이미지 입력 상품",
            "category": None,
            "price": 0,
            "rating": 3.5,
            "review_count": 0,
            "return_rate": 5.0,
            "description": "이미지 분석 중 오류가 발생해 기본값으로 분석합니다.",
        }


## 입력 타입에 따라 URL, 이미지, 수동 입력 중 적절한 상품 추출 경로를 선택합니다.

## HTML 태그와 엔티티를 제거해 네이버 쇼핑 결과의 상품명을 화면 표시용으로 정리합니다.
def clean_shopping_text(
    value: Optional[str],
) -> Optional[str]:
    if not value:
        return None
    text = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
    return _clean_text(text)


## 이미지 분석으로 얻은 상품 정보에서 네이버 쇼핑 검색에 사용할 질의어를 만듭니다.
def build_image_search_query(
    product: Dict[str, Any],
) -> str:
    candidates = [
        product.get("brand"),
        product.get("name"),
        product.get("category"),
    ]
    query = " ".join(str(value).strip() for value in candidates if value)
    generic_words = {"이미지", "입력", "상품", "image", "product", "unknown", "null"}
    tokens = [token for token in re.split(r"\s+", query) if token and token.lower() not in generic_words]
    if tokens:
        return " ".join(tokens[:8])

    description = product.get("description")
    if description:
        return " ".join(str(description).split()[:8])
    return ""


## 네이버 쇼핑 API 응답 항목을 StopBuy 예측 입력과 화면 표시가 가능한 상품 후보 구조로 변환합니다.
def normalize_naver_shopping_item(
    item: Dict[str, Any],
    index: int,
    query: str,
) -> Dict[str, Any]:
    raw_price = item.get("lprice") or item.get("hprice") or 0
    try:
        price = int(float(raw_price))
    except (TypeError, ValueError):
        price = 0

    category = next(
        (
            clean_shopping_text(item.get(key))
            for key in ("category4", "category3", "category2", "category1")
            if item.get(key)
        ),
        None,
    )
    brand = clean_shopping_text(item.get("brand") or item.get("maker"))
    name = clean_shopping_text(item.get("title")) or "네이버 쇼핑 상품"
    link = item.get("link")
    catalog_match = re.search(r"/catalog/(\d+)", str(link or ""))
    catalog_id = catalog_match.group(1) if catalog_match else None

    return {
        "product_id": f"naver-{item.get('productId') or index}",
        "naver_product_id": item.get("productId"),
        "naver_product_type": item.get("productType"),
        "naver_catalog_id": catalog_id,
        "name": name,
        "brand": brand,
        "category": category,
        "price": price,
        "rating": None,
        "review_count": None,
        "return_rate": None,
        "review_data_available": False,
        "review_source": None,
        "review_texts": [],
        "image_url": item.get("image"),
        "source_url": link,
        "product_url": link,
        "mall_name": clean_shopping_text(item.get("mallName")),
        "search_query": query,
        "description": f"네이버 쇼핑에서 '{query}'로 검색된 상품입니다.",
    }


## 네이버 쇼핑 API로 상품 후보를 검색합니다. 키가 없거나 검색어가 비어 있으면 빈 목록을 반환합니다.

## 숫자 텍스트를 정수 개수로 변환합니다. 1.2만, 3천 같은 한국식 단위도 처리합니다.
def parse_count_value(
    value: Any,
) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0, int(value))
    text = str(value).replace(",", "").strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?|?|k|K)?", text)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2)
    if unit == "?":
        number *= 10000
    elif unit == "?" or unit in {"k", "K"}:
        number *= 1000
    return max(0, int(number))


## 숫자 텍스트를 평점 값으로 변환하고 0~5 범위 값만 인정합니다.
def parse_rating_value(
    value: Any,
) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        rating = float(value)
    else:
        match = re.search(r"([0-5](?:\.\d+)?)", str(value))
        if not match:
            return None
        rating = float(match.group(1))
    if 0.0 <= rating <= 5.0:
        return round(rating, 2)
    return None


## 중첩 JSON에서 리뷰 관련 숫자와 리뷰 문구 후보를 재귀적으로 찾습니다.
def collect_review_values(
    value: Any,
    result: Dict[str, Any],
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ["reviewcount", "review_count", "reviewcnt", "totalreview", "reviewtotal"]):
                count = parse_count_value(item)
                if count is not None and result.get("review_count") is None:
                    result["review_count"] = count
            if any(token in lowered for token in ["ratingvalue", "averagereviewscore", "reviewscore", "starscore", "starpoint"]):
                rating = parse_rating_value(item)
                if rating is not None and result.get("rating") is None:
                    result["rating"] = rating
            if "review" in lowered and any(token in lowered for token in ["content", "contents", "text", "body"]):
                text_value = clean_shopping_text(str(item))
                if text_value and len(text_value) >= 12:
                    result.setdefault("review_texts", []).append(text_value[:300])
            collect_review_values(item, result)
    elif isinstance(value, list):
        for item in value:
            collect_review_values(item, result)


## HTML/스크립트에서 평점, 후기개수, 리뷰 샘플을 추출합니다.
def extract_review_info_from_html(
    html: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"rating": None, "review_count": None, "review_texts": []}
    soup = BeautifulSoup(html, "lxml")

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            collect_review_values(data, result)
        except Exception:
            continue

    script_text = "\n".join(script.get_text(" ", strip=True) for script in soup.find_all("script"))[:1_500_000]
    regex_pairs = [
        ("review_count", r'"(?:reviewCount|review_count|reviewCnt|totalReviewCount|reviewTotalCount)"\s*:\s*"?([0-9,.]+)"?'),
        ("rating", r'"(?:ratingValue|averageReviewScore|reviewScore|starScore|starPoint)"\s*:\s*"?([0-5](?:\.[0-9]+)?)"?'),
    ]
    for field, pattern in regex_pairs:
        if result.get(field) is not None:
            continue
        match = re.search(pattern, script_text, re.IGNORECASE)
        if not match:
            continue
        if field == "review_count":
            result[field] = parse_count_value(match.group(1))
        else:
            result[field] = parse_rating_value(match.group(1))

    visible_text = soup.get_text(" ", strip=True)[:300_000]
    if result.get("rating") is None:
        match = re.search(r"(?:평점|별점)\s*([0-5](?:\.[0-9]+)?)", visible_text)
        if match:
            result["rating"] = parse_rating_value(match.group(1))
    if result.get("review_count") is None:
        match = re.search(r"(?:리뷰|후기|상품평)\s*([0-9,.]+\s*(?:만|천|k|K)?)\s*(?:개|건)?", visible_text)
        if match:
            result["review_count"] = parse_count_value(match.group(1))

    for candidate in re.split(r"(?<=[.!?])\s+|\s{2,}", visible_text):
        cleaned = clean_shopping_text(candidate)
        if not cleaned or len(cleaned) < 20:
            continue
        if any(keyword in cleaned for keyword in ["리뷰", "후기", "만족", "불만", "배송", "품질", "사용"]):
            result["review_texts"].append(cleaned[:300])
        if len(result["review_texts"]) >= NAVER_REVIEW_SAMPLE_LIMIT:
            break

    result["review_texts"] = list(dict.fromkeys(result.get("review_texts") or []))[:NAVER_REVIEW_SAMPLE_LIMIT]
    return result


## 네이버 후보 상품의 상세/카탈로그 페이지를 열어 평점과 후기개수를 보강합니다.
async def enrich_naver_review_info(
    client: httpx.AsyncClient,
    product: Dict[str, Any],
) -> Dict[str, Any]:
    urls: List[str] = []
    for key in ("source_url", "product_url"):
        url = product.get(key)
        if url:
            urls.append(str(url))

    catalog_id = product.get("naver_catalog_id")
    if catalog_id:
        urls.append(f"https://search.shopping.naver.com/catalog/{catalog_id}")

    attempted_urls: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        try:
            response = await client.get(url)
            attempted_urls.append({
                "url": url,
                "status_code": response.status_code,
                "final_url": str(response.url),
                "content_length": len(response.text),
            })
            response.raise_for_status()
            info = extract_review_info_from_html(response.text)
        except Exception as exc:
            if attempted_urls:
                attempted_urls[-1]["error"] = str(exc)
            logger.info("naver review enrich failed: url=%s error=%s", url, exc)
            continue

        has_review_data = any([
            info.get("rating") is not None,
            info.get("review_count") is not None,
            bool(info.get("review_texts")),
        ])
        if not has_review_data:
            attempted_urls[-1]["review_data_found"] = False
            continue

        attempted_urls[-1]["review_data_found"] = True
        if info.get("rating") is not None:
            product["rating"] = info["rating"]
        if info.get("review_count") is not None:
            product["review_count"] = info["review_count"]
        if info.get("review_texts"):
            product["review_texts"] = info["review_texts"]
        product["review_data_available"] = True
        product["review_source"] = url
        break

    product["review_attempted_urls"] = attempted_urls

    try:
        logger.info(
            "naver review enriched product: %s",
            json.dumps({
                "name": product.get("name"),
                "rating": product.get("rating"),
                "review_count": product.get("review_count"),
                "review_data_available": product.get("review_data_available"),
                "review_source": product.get("review_source"),
                "review_texts": product.get("review_texts"),
                "review_attempted_urls": attempted_urls,
                "naver_product_id": product.get("naver_product_id"),
                "naver_catalog_id": product.get("naver_catalog_id"),
            }, ensure_ascii=False, indent=2, default=str),
        )
    except Exception:
        logger.info("naver review enriched product: %r", product)
    return product

async def search_naver_shopping(
    query: str,
    display: Optional[int] = None,
) -> List[Dict[str, Any]]:
    query = _clean_text(query) or ""
    if not query or not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return []

    limit = max(1, min(display or NAVER_SHOPPING_DISPLAY, 20))
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": query, "display": limit, "start": 1, "sort": "sim"}

    async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
        response = await client.get(NAVER_SHOPPING_ENDPOINT, params=params)
        response.raise_for_status()
        payload = response.json()

    try:
        log_payload = json.dumps(
            {
                "request": {
                    "endpoint": NAVER_SHOPPING_ENDPOINT,
                    "params": params,
                },
                "response": payload,
            },
            ensure_ascii=False,
            default=str,
            indent=2,
        )
    except Exception:
        log_payload = repr(payload)

    if len(log_payload) > NAVER_LOG_PAYLOAD_MAX_LENGTH:
        log_payload = f"{log_payload[:NAVER_LOG_PAYLOAD_MAX_LENGTH]}... <truncated length={len(log_payload)}>"
    logger.info("naver shopping payload: %s", log_payload)

    items = payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return []

    products = [normalize_naver_shopping_item(item, index + 1, query) for index, item in enumerate(items)]
    if NAVER_REVIEW_ENRICH_ENABLED and products:
        review_headers = {
            "User-Agent": "Mozilla/5.0 StopBuy2.0 Review Enricher",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        enrich_limit = max(0, min(NAVER_REVIEW_ENRICH_LIMIT, len(products)))
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True, headers=review_headers) as review_client:
            enriched = await asyncio.gather(
                *(enrich_naver_review_info(review_client, product) for product in products[:enrich_limit]),
                return_exceptions=True,
            )
        for index, item in enumerate(enriched):
            if isinstance(item, dict):
                products[index] = item
            elif isinstance(item, Exception):
                logger.info("naver review enrich task failed: %s", item)
    return products


## 이미지에서 상품 정보를 추출한 뒤 네이버 쇼핑 후보 목록을 함께 조회합니다.
async def extract_product_candidates_from_image(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    product = await extract_from_image(data.get("image_base64") or "")
    query = build_image_search_query(product)
    candidates: List[Dict[str, Any]] = []
    search_error = None

    try:
        candidates = await search_naver_shopping(query, NAVER_SHOPPING_DISPLAY)
    except Exception as exc:
        search_error = str(exc)

    return {
        "query": query,
        "extracted_product": product,
        "candidates": candidates,
        "search_error": search_error,
        "naver_configured": bool(NAVER_CLIENT_ID and NAVER_CLIENT_SECRET),
    }

async def extract_product_info(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    input_type = data.get("input_type", "manual")
    if input_type == "url" and data.get("product_url"):
        return await extract_from_url(data["product_url"])
    if input_type == "image" and data.get("image_base64"):
        return await extract_from_image(data["image_base64"])
    product = data.get("product") or {}
    return product if isinstance(product, dict) else {}
