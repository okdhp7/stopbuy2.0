import base64
import json
import os
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from env_loader import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")


CATEGORY_KEYWORDS = {
    "스마트폰": ["iphone", "galaxy", "pixel", "phone", "스마트폰", "아이폰", "갤럭시"],
    "노트북": ["laptop", "notebook", "macbook", "gram", "노트북", "맥북", "그램"],
    "이어폰/헤드폰": ["airpods", "headphone", "earbuds", "이어폰", "헤드폰", "버즈"],
    "TV": ["tv", "oled", "qled", "television", "티비", "텔레비전"],
    "생활가전": ["청소기", "냉장고", "세탁기", "공기청정기", "vacuum", "washer"],
}


def _clean_text(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return re.sub(r"\s+", " ", value).strip()


def _number_from_text(text: str) -> Optional[float]:
    match = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)", text)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def detect_category(text: str) -> Optional[str]:
    lowered = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return category
    return None


def detect_shop(url: str) -> str:
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


def parse_json_ld(soup: BeautifulSoup) -> Dict[str, Any]:
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


async def extract_from_url(url: str) -> Dict[str, Any]:
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


async def extract_from_image(image_base64: str) -> Dict[str, Any]:
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


async def extract_product_info(data: Dict[str, Any]) -> Dict[str, Any]:
    input_type = data.get("input_type", "manual")
    if input_type == "url" and data.get("product_url"):
        return await extract_from_url(data["product_url"])
    if input_type == "image" and data.get("image_base64"):
        return await extract_from_image(data["image_base64"])
    product = data.get("product") or {}
    return product if isinstance(product, dict) else {}
