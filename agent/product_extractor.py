import asyncio
import base64
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from env_loader import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = os.getenv("VISION_LLM_MODEL_NAME", "gpt-4o-mini")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
NAVER_SHOPPING_DISPLAY = int(os.getenv("NAVER_SHOPPING_DISPLAY", "10"))
NAVER_SHOPPING_ENDPOINT = "https://openapi.naver.com/v1/search/shop.json"
NAVER_LOG_PAYLOAD_MAX_LENGTH = int(os.getenv("NAVER_LOG_PAYLOAD_MAX_LENGTH", "8000"))
NAVER_REVIEW_ENRICH_ENABLED = os.getenv("NAVER_REVIEW_ENRICH_ENABLED", "true").lower() not in {"0", "false", "no"}
NAVER_REVIEW_ENRICH_LIMIT = int(os.getenv("NAVER_REVIEW_ENRICH_LIMIT", "5"))
NAVER_REVIEW_SAMPLE_LIMIT = int(os.getenv("NAVER_REVIEW_SAMPLE_LIMIT", "3"))
COUPANG_BROWSER_FALLBACK_ENABLED = os.getenv("COUPANG_BROWSER_FALLBACK_ENABLED", "true").lower() not in {"0", "false", "no", "n"}
COUPANG_BROWSER_TIMEOUT_SECONDS = float(os.getenv("COUPANG_BROWSER_TIMEOUT_SECONDS", "20"))
COUPANG_BROWSER_HEADLESS = os.getenv("COUPANG_BROWSER_HEADLESS", "true").lower() not in {"0", "false", "no", "n"}
logger = logging.getLogger("stopbuy-agent.naver")


CATEGORY_KEYWORDS = {
    "스마트폰": ["iphone", "galaxy", "pixel", "phone", "스마트폰", "아이폰", "갤럭시"],
    "노트북": ["laptop", "notebook", "macbook", "gram", "노트북", "맥북", "그램"],
    "이어폰/헤드폰": ["airpods", "headphone", "earbuds", "이어폰", "헤드폰", "버즈"],
    "TV": ["tv", "oled", "qled", "television", "티비", "텔레비전"],
    "생활가전": ["청소기", "냉장고", "세탁기", "공기청정기", "vacuum", "washer"],
    "전기면도기": ["면도기", "전기면도기", "shaver", "razor", "쉐이버"],
}

GENERIC_URL_PRODUCT_NAMES = {"URL 입력 상품", "상품", "product", "unknown", "null"}
GENERIC_PAGE_TITLES = {
    "G마켓 - 쇼핑을 바꾸는 쇼핑",
    "G마켓",
    "Access Denied",
}
PRICE_KEY_PRIORITY = {
    "finalprice": 0,
    "final_price": 0,
    "saleprice": 1,
    "sale_price": 1,
    "discountprice": 1,
    "discount_price": 1,
    "discountedsaleprice": 1,
    "discounted_sale_price": 1,
    "mobilediscountedsaleprice": 1,
    "mobile_discounted_sale_price": 1,
    "dispdiscountedsaleprice": 1,
    "disp_discounted_sale_price": 1,
    "couponprice": 1,
    "coupon_price": 1,
    "lprice": 1,
    "lowprice": 1,
    "low_price": 1,
    "sellingprice": 2,
    "selling_price": 2,
    "price": 3,
    "regularprice": 4,
    "regular_price": 4,
    "originprice": 4,
    "originalprice": 4,
    "listprice": 4,
    "hprice": 5,
    "highprice": 5,
}
BRAND_KEYWORDS = {
    "삼성": ["삼성", "삼성전자", "samsung", "galaxy", "갤럭시"],
    "애플": ["애플", "apple", "iphone", "아이폰", "ipad", "아이패드", "macbook", "맥북", "airpods", "에어팟"],
    "LG": ["lg", "엘지", "lg전자", "gram", "그램"],
    "소니": ["sony", "소니"],
    "레노버": ["lenovo", "레노버"],
    "샤오미": ["xiaomi", "샤오미", "redmi", "레드미"],
    "구글": ["google", "구글", "pixel", "픽셀"],
    "HP": ["hp", "hpe", "hewlett", "proliant", "휴렛팩커드"],
    "델": ["dell", "델", "xps"],
    "ASUS": ["asus", "에이수스", "젠북", "zenbook"],
    "Bose": ["bose", "보스"],
    "JBL": ["jbl"],
    "필립스": ["필립스", "philips"],
    "브라운": ["브라운", "braun"],
    "파나소닉": ["파나소닉", "panasonic"],
    "더함": ["더함", "thehaam", "the ham"],
}
BRAND_SLUG_ALIASES = {
    "philips": "필립스",
    "samsung": "삼성",
    "apple": "애플",
    "lg": "LG",
    "sony": "소니",
    "braun": "브라운",
    "panasonic": "파나소닉",
    "xiaomi": "샤오미",
    "lenovo": "레노버",
    "thehaam": "더함",
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


## 가격 키 이름에 따라 할인가/최종가를 정가보다 우선하도록 우선순위를 계산합니다.
def price_key_priority(
    key: Any,
) -> int:
    normalized = re.sub(r"[^0-9a-zA-Z_]+", "", str(key or "")).lower()
    return PRICE_KEY_PRIORITY.get(normalized, 9)


## 가격 후보를 결과에 반영하되 할인가와 최종가를 더 높은 우선순위로 유지합니다.
def apply_price_candidate(
    result: Dict[str, Any],
    key: Any,
    value: Any,
) -> None:
    price = _number_from_text(str(value))
    if not price:
        return
    priority = price_key_priority(key)
    current_priority = int(result.get("_price_priority", 99))
    current_price = _number_from_text(str(result.get("price"))) or 0
    if result.get("price") is None or priority < current_priority or (priority == current_priority and current_price and price < current_price):
        result["price"] = price
        result["_price_priority"] = priority
        result["price_source_key"] = str(key)


## HTML 속성값과 스크립트의 유니코드 이스케이프를 안전하게 풀어냅니다.
def decode_html_value(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None
    raw_text = str(value)
    if "<" in raw_text and ">" in raw_text:
        text = BeautifulSoup(raw_text, "html.parser").get_text(" ", strip=True)
    else:
        text = raw_text
    try:
        text = json.loads(f'"{text}"')
    except Exception:
        pass
    return _clean_text(text)


## HTML 메타 태그에서 후보 속성명에 해당하는 content 값을 찾습니다.
def get_meta_content(
    soup: BeautifulSoup,
    *names: str,
) -> Optional[str]:
    for name in names:
        tag = (
            soup.find("meta", property=name)
            or soup.find("meta", attrs={"name": name})
            or soup.find("meta", attrs={"itemprop": name})
        )
        if tag and tag.get("content"):
            return decode_html_value(tag.get("content"))
    return None


## 상품명에 붙은 쇼핑몰/페이지 부가 문구를 제거합니다.
def clean_product_title(
    value: Optional[str],
) -> Optional[str]:
    text = _clean_text(value)
    if not text:
        return None
    for separator in [" : 네이버쇼핑", " - 네이버쇼핑", " | 네이버", " | 쿠팡", " - 쿠팡", " - 11번가", " - G마켓"]:
        text = text.replace(separator, "")
    text = re.sub(r"\s*[-|:]\s*(스마트스토어|네이버\s*쇼핑|쿠팡|11번가|G마켓|옥션)\s*$", "", text, flags=re.IGNORECASE)
    return _clean_text(text)


## 상품명/브랜드 텍스트에서 대표 브랜드명을 추론합니다.
def infer_brand_from_text(
    *values: Any,
) -> Optional[str]:
    source = " ".join(str(value) for value in values if value).lower()
    if not source:
        return None
    for brand, keywords in BRAND_KEYWORDS.items():
        if any(keyword.lower() in source for keyword in keywords):
            return brand
    return None


## 네이버 브랜드스토어/스마트스토어 URL에서 브랜드 슬러그와 상품 ID를 추출합니다.
def extract_naver_store_url_info(
    url: str,
) -> Dict[str, Optional[str]]:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path_parts = [unquote(part) for part in parsed.path.split("/") if part]
    result: Dict[str, Optional[str]] = {"brand_slug": None, "brand": None, "product_id": None}
    if "brand.naver.com" not in host and "smartstore.naver.com" not in host:
        return result

    if path_parts:
        slug = path_parts[0].strip()
        if slug.lower() not in {"main", "products", "product"}:
            result["brand_slug"] = slug
            result["brand"] = BRAND_SLUG_ALIASES.get(slug.lower()) or infer_brand_from_text(slug)

    for index, part in enumerate(path_parts):
        if part.lower() in {"products", "product"} and index + 1 < len(path_parts):
            product_id = path_parts[index + 1]
            if product_id.isdigit():
                result["product_id"] = product_id
                break
    return result


## 네이버 상품 URL에서 products 또는 catalog 뒤의 숫자 상품 ID를 추출합니다.
def extract_naver_product_id_from_url(
    url: Any,
) -> Optional[str]:
    text = str(url or "")
    match = re.search(r"/(?:products|product|catalog)/(\d+)", text)
    return match.group(1) if match else None


## 쿠팡 상품 URL에서 상품 ID와 vendorItemId 같은 URL 식별자를 추출합니다.
def extract_coupang_url_info(
    url: str,
) -> Dict[str, Optional[str]]:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    query_values = parse_qs(parsed.query)
    result: Dict[str, Optional[str]] = {"product_id": None, "item_id": None, "vendor_item_id": None}
    if "coupang.com" not in host:
        return result

    match = re.search(r"/(?:vp/)?products/(\d+)", parsed.path)
    if match:
        result["product_id"] = match.group(1)
    if query_values.get("itemId"):
        result["item_id"] = str(query_values["itemId"][0])
    if query_values.get("vendorItemId"):
        result["vendor_item_id"] = str(query_values["vendorItemId"][0])
    return result


## G마켓 상품 URL에서 상품코드와 검색 힌트를 추출합니다.
def extract_gmarket_url_info(
    url: str,
) -> Dict[str, Optional[str]]:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    query_values = parse_qs(parsed.query)
    result: Dict[str, Optional[str]] = {"goods_code": None, "keyword": None}
    if "gmarket" not in host:
        return result

    for key in ("goodscode", "goodsCode", "goods_code", "gdsc_cd"):
        values = query_values.get(key)
        if values:
            result["goods_code"] = str(values[0]).strip()
            break

    for key in ("keyword", "searchKeyword", "search", "q"):
        values = query_values.get(key)
        if values:
            result["keyword"] = clean_product_title(unquote(str(values[0])))
            break
    return result


## G마켓 URL 또는 네이버 쇼핑 경유 URL에서 G마켓 상품번호를 추출합니다.
def extract_gmarket_product_id_from_url(
    url: Any,
) -> Optional[str]:
    parsed = urlparse(str(url or ""))
    query_values = parse_qs(parsed.query)
    for key in ("goodscode", "goodsCode", "goods_code", "gdsc_cd", "item-no", "itemNo"):
        values = query_values.get(key)
        if values:
            value = str(values[0]).strip()
            if value.isdigit():
                return value
    match = re.search(r"(?:goodscode|goodsCode|item-no|itemNo)[=/](\d+)", str(url or ""))
    return match.group(1) if match else None


## URL의 검색 쿼리와 경로에서 상품 검색에 쓸 후보 문구를 추출합니다.
def extract_url_query_hint(
    url: str,
) -> Optional[str]:
    parsed = urlparse(url)
    query_values = parse_qs(parsed.query)
    for key in ("nl-query", "query", "q", "keyword", "search", "searchKeyword"):
        values = query_values.get(key)
        if values:
            cleaned = clean_product_title(unquote(str(values[0])))
            if cleaned:
                return cleaned

    naver_info = extract_naver_store_url_info(url)
    if naver_info.get("brand"):
        return naver_info["brand"]

    coupang_info = extract_coupang_url_info(url)
    if coupang_info.get("product_id"):
        return " ".join(
            value
            for value in [
                "쿠팡",
                coupang_info.get("product_id"),
                coupang_info.get("vendor_item_id"),
            ]
            if value
        )

    gmarket_info = extract_gmarket_url_info(url)
    if gmarket_info.get("keyword"):
        return gmarket_info["keyword"]
    if gmarket_info.get("goods_code"):
        return " ".join(["G마켓", str(gmarket_info["goods_code"])])

    path_parts = [unquote(part) for part in parsed.path.split("/") if part]
    meaningful_parts = [
        part
        for part in path_parts
        if not part.isdigit()
        and part.lower() not in {"products", "product", "catalog", "main", "smartstore"}
    ]
    if meaningful_parts:
        return clean_product_title(" ".join(meaningful_parts[:3]))
    return None


## URL 추출 결과가 기본값 수준인지 판단합니다.
def is_generic_product_name(
    value: Any,
) -> bool:
    text = str(value or "").strip()
    return not text or text in GENERIC_URL_PRODUCT_NAMES


## 검색/병합 품질 평가에 사용할 토큰 집합을 만듭니다.
def text_tokens(
    *values: Any,
) -> Set[str]:
    text = " ".join(str(value) for value in values if value).lower()
    return {token for token in re.split(r"[^0-9a-zA-Z가-힣]+", text) if len(token) >= 2}


## 두 상품 텍스트의 단순 토큰 유사도를 계산합니다.
def token_similarity(
    left: Any,
    right: Any,
) -> float:
    left_tokens = text_tokens(left)
    right_tokens = text_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


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
                item_type = item.get("@type")
                if item_type != "Product" and not (isinstance(item_type, list) and "Product" in item_type):
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
                    for key, value in offers.items():
                        if price_key_priority(key) < 9:
                            apply_price_candidate(result, key, value)
                rating = item.get("aggregateRating")
                if isinstance(rating, dict):
                    if rating.get("ratingValue"):
                        result["rating"] = float(rating["ratingValue"])
                    if rating.get("reviewCount"):
                        result["review_count"] = int(float(rating["reviewCount"]))
                if result:
                    return {key: value for key, value in result.items() if not key.startswith("_") and key != "price_source_key"}
        except Exception:
            continue
    return {}


## 중첩된 스크립트 JSON에서 상품명/브랜드/가격 후보를 재귀적으로 찾습니다.
def collect_product_values(
    value: Any,
    result: Dict[str, Any],
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if result.get("name") is None and lowered in {"name", "productname", "product_name", "producttitle", "title"} and isinstance(item, (str, int, float)):
                cleaned = clean_product_title(clean_shopping_text(str(item)))
                if cleaned and len(cleaned) >= 2:
                    result["name"] = cleaned
            if result.get("brand") is None and lowered in {"brand", "brandname", "brand_name", "maker", "manufacturer"}:
                if isinstance(item, dict):
                    item = item.get("name") or item.get("brandName")
                cleaned = clean_shopping_text(str(item)) if isinstance(item, (str, int, float)) else None
                if cleaned and len(cleaned) >= 2:
                    result["brand"] = cleaned
            if lowered in PRICE_KEY_PRIORITY and isinstance(item, (str, int, float)):
                apply_price_candidate(result, lowered, item)
            if result.get("image_url") is None and lowered in {"image", "imageurl", "representimageurl", "thumbnail", "thumbnailurl"}:
                if isinstance(item, list) and item:
                    item = item[0]
                if isinstance(item, str) and item.startswith(("http://", "https://")):
                    result["image_url"] = item
            if result.get("description") is None and lowered in {"description", "productdescription", "desc"} and isinstance(item, (str, int, float)):
                cleaned = clean_shopping_text(str(item))
                if cleaned and len(cleaned) >= 10:
                    result["description"] = cleaned[:500]
            collect_product_values(item, result)
    elif isinstance(value, list):
        for item in value:
            collect_product_values(item, result)


## Next.js와 일반 스크립트 JSON에서 상품 후보 정보를 추출합니다.
def parse_script_product_data(
    soup: BeautifulSoup,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"name": None, "brand": None, "price": None, "image_url": None, "description": None}
    script_candidates = []
    next_data = soup.find("script", id="__NEXT_DATA__")
    if next_data and next_data.string:
        script_candidates.append(next_data.string)
    for script in soup.find_all("script", type="application/json"):
        if script.string:
            script_candidates.append(script.string)

    for text in script_candidates[:8]:
        try:
            parsed = json.loads(text)
        except Exception:
            continue
        collect_product_values(parsed, result)
    return {key: item for key, item in result.items() if item not in (None, "") and not key.startswith("_") and key != "price_source_key"}


## HTML 스크립트 텍스트에서 지정된 키의 문자열 값을 찾습니다.
def find_script_string_value(
    script_text: str,
    *keys: str,
) -> Optional[str]:
    for key in keys:
        patterns = [
            rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"',
            rf"'{re.escape(key)}'\s*:\s*'((?:\\.|[^'\\])*)'",
        ]
        for pattern in patterns:
            match = re.search(pattern, script_text, re.IGNORECASE)
            if match:
                value = decode_html_value(match.group(1))
                if value and value.lower() not in {"null", "undefined"}:
                    return value
    return None


## HTML 스크립트 텍스트에서 지정된 키의 숫자 값을 찾습니다.
def find_script_number_value(
    script_text: str,
    *keys: str,
) -> Optional[float]:
    for key in keys:
        patterns = [
            rf'"{re.escape(key)}"\s*:\s*"?([0-9][0-9,.]*)"?',
            rf"'{re.escape(key)}'\s*:\s*'?([0-9][0-9,.]*)'?",
        ]
        for pattern in patterns:
            match = re.search(pattern, script_text, re.IGNORECASE)
            if match:
                value = _number_from_text(match.group(1))
                if value:
                    return value
    return None


## 페이지 메타 태그와 스크립트에서 공통 상품 정보를 추출합니다.
def parse_common_product_data(
    soup: BeautifulSoup,
    html: str,
) -> Dict[str, Any]:
    product = parse_json_ld(soup)
    script_product = parse_script_product_data(soup)
    for key, value in script_product.items():
        if key == "price":
            current_price = _number_from_text(str(product.get("price"))) or 0
            next_price = _number_from_text(str(value)) or 0
            if next_price and (not current_price or next_price <= current_price):
                product[key] = next_price
        elif not product.get(key):
            product[key] = value

    script_text = "\n".join(script.get_text(" ", strip=True) for script in soup.find_all("script"))[:2_000_000]
    if not product.get("name"):
        product["name"] = (
            clean_product_title(get_meta_content(soup, "og:title", "twitter:title", "title", "name"))
            or find_script_string_value(script_text, "productName", "product_name", "itemName", "item_name", "title")
        )
    if not product.get("brand"):
        product["brand"] = (
            get_meta_content(soup, "product:brand", "brand", "og:brand", "twitter:data1")
            or find_script_string_value(script_text, "brandName", "brand_name", "brand", "maker", "manufacturer")
        )
    if not product.get("category"):
        product["category"] = (
            get_meta_content(soup, "product:category", "category")
            or find_script_string_value(script_text, "categoryName", "category_name", "displayCategoryName", "category")
        )
    price_result: Dict[str, Any] = {"price": None}
    for key in [
        "product:sale_price:amount",
        "sale_price",
        "discount_price",
        "final_price",
        "product:price:amount",
        "og:price:amount",
        "price",
    ]:
        price_text = get_meta_content(soup, key)
        if price_text:
            apply_price_candidate(price_result, key, price_text)
    meta_price = _number_from_text(str(price_result.get("price"))) or 0
    current_price = _number_from_text(str(product.get("price"))) or 0
    if meta_price and (not current_price or meta_price <= current_price):
        product["price"] = meta_price
        current_price = meta_price
    script_price = find_script_number_value(
        script_text,
        "finalPrice",
        "discountedSalePrice",
        "mobileDiscountedSalePrice",
        "dispDiscountedSalePrice",
        "salePrice",
        "discountPrice",
        "couponPrice",
        "sellingPrice",
        "lprice",
        "lowPrice",
        "price",
    )
    if script_price and (not current_price or script_price <= current_price):
        product["price"] = script_price
    if not product.get("image_url"):
        product["image_url"] = (
            get_meta_content(soup, "og:image", "twitter:image", "image")
            or find_script_string_value(script_text, "imageUrl", "image_url", "thumbnailUrl", "thumbnail")
        )
    if not product.get("description"):
        product["description"] = (
            get_meta_content(soup, "og:description", "twitter:description", "description")
            or find_script_string_value(script_text, "description", "productDescription")
        )
    return {key: item for key, item in product.items() if item not in (None, "")}


## 쿠팡 상품 페이지에서 상품명, 브랜드, 카테고리, 가격을 우선 추출합니다.
def parse_coupang_product_data(
    soup: BeautifulSoup,
    html: str,
    url: str,
) -> Dict[str, Any]:
    product = parse_common_product_data(soup, html)
    coupang_info = extract_coupang_url_info(url)
    product["coupang_product"] = coupang_info
    product["shop"] = "coupang"
    product["product_info_source"] = "coupang_url"

    if not product.get("name") and soup.title:
        product["name"] = clean_product_title(soup.title.get_text(" ", strip=True))
    if product.get("name"):
        product["name"] = clean_product_title(product.get("name"))
    if not product.get("brand"):
        product["brand"] = infer_brand_from_text(product.get("name"), product.get("description"))
    if not product.get("category"):
        product["category"] = detect_category(f"{product.get('name', '')} {product.get('description', '')}")
    return product


## G마켓 상품 페이지에서 상품명, 브랜드, 카테고리, 가격을 우선 추출합니다.
def parse_gmarket_product_data(
    soup: BeautifulSoup,
    html: str,
    url: str,
) -> Dict[str, Any]:
    product = parse_common_product_data(soup, html)
    gmarket_info = extract_gmarket_url_info(url)
    product["gmarket_product"] = gmarket_info
    product["shop"] = "gmarket"
    product["product_info_source"] = "gmarket_url"

    if not product.get("name"):
        title = get_meta_content(soup, "og:title", "twitter:title", "title", "name")
        if not title and soup.title:
            title = soup.title.get_text(" ", strip=True)
        product["name"] = clean_product_title(title)
    if product.get("name") in GENERIC_PAGE_TITLES:
        product["name"] = None

    if not product.get("name"):
        for selector in [
            "h1.itemtit",
            ".itemtit",
            ".box__item-title",
            ".item-topinfo_headline",
            "[class*='item-title']",
            "h1",
        ]:
            tag = soup.select_one(selector)
            if tag:
                product["name"] = clean_product_title(tag.get_text(" ", strip=True))
                if product["name"]:
                    break

    if not product.get("price"):
        for selector in [
            ".price_discount",
            ".box__price-coupon strong",
            ".box__price-coupon",
            ".box__price-seller strong",
            ".box__price-seller",
            ".price_real",
            ".price",
            "[class*='discount'][class*='price']",
            "[class*='sale'][class*='price']",
            "[class*='price'] strong",
            "[class*='price']",
        ]:
            tag = soup.select_one(selector)
            if tag:
                price = _number_from_text(tag.get_text(" ", strip=True))
                if price:
                    product["price"] = int(price)
                    break

    if not product.get("brand"):
        product["brand"] = get_meta_content(soup, "product:brand", "brand", "og:brand")
    if not product.get("brand"):
        product["brand"] = infer_brand_from_text(product.get("name"), product.get("description"), url)
    if not product.get("category"):
        product["category"] = detect_category(f"{product.get('name', '')} {product.get('description', '')}")
    return {key: item for key, item in product.items() if item not in (None, "")}


## 상품정보가 구매후회예측에 쓰기 어려울 만큼 부족한지 판단합니다.
def is_incomplete_product_info(
    product: Dict[str, Any],
) -> bool:
    name = product.get("name")
    price = _number_from_text(str(product.get("price"))) or 0
    return is_generic_product_name(name) or name in GENERIC_PAGE_TITLES or not name or price <= 0


## 기존 추출 결과에 브라우저 fallback 결과를 병합합니다.
def merge_product_info(
    base: Dict[str, Any],
    addition: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(base)
    for key in ("name", "brand", "category", "price", "image_url", "description"):
        value = addition.get(key)
        if value in (None, ""):
            continue
        if key == "name":
            if is_generic_product_name(merged.get("name")) or not merged.get("name"):
                merged[key] = value
            continue
        if key == "price":
            current_price = _number_from_text(str(merged.get("price"))) or 0
            next_price = _number_from_text(str(value)) or 0
            if next_price and current_price <= 0:
                merged[key] = next_price
            continue
        if not merged.get(key):
            merged[key] = value
    for key, value in addition.items():
        if key not in merged and value not in (None, ""):
            merged[key] = value
    return merged


## Playwright 브라우저로 쿠팡 상품 페이지를 렌더링해 상품정보를 추출합니다.
async def extract_coupang_with_browser(
    url: str,
) -> Dict[str, Any]:
    if not COUPANG_BROWSER_FALLBACK_ENABLED:
        return {}

    timeout_ms = int(max(COUPANG_BROWSER_TIMEOUT_SECONDS, 3) * 1000)
    logger.info("coupang browser fallback start: url=%s", url)
    try:
        from playwright.async_api import async_playwright
    except Exception as exc:
        logger.info("coupang browser fallback unavailable: %s", exc)
        return {
            "browser_fallback_attempted": True,
            "browser_fallback_success": False,
            "browser_fallback_error": f"Playwright를 사용할 수 없습니다: {exc}",
        }

    browser = None
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=COUPANG_BROWSER_HEADLESS,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
            context = await browser.new_context(
                locale="ko-KR",
                timezone_id="Asia/Seoul",
                viewport={"width": 1365, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
                extra_http_headers={
                    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
                    "Referer": "https://www.coupang.com/",
                },
            )
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 8000))
            except Exception:
                pass
            await page.wait_for_timeout(1500)

            data = await page.evaluate(
                """
                () => {
                  const clean = (value) => (value || "").replace(/\\s+/g, " ").trim();
                  const text = (selectors) => {
                    for (const selector of selectors) {
                      const node = document.querySelector(selector);
                      const value = node ? clean(node.innerText || node.textContent || "") : "";
                      if (value) return value;
                    }
                    return "";
                  };
                  const attr = (selectors, name) => {
                    for (const selector of selectors) {
                      const node = document.querySelector(selector);
                      const value = node ? clean(node.getAttribute(name) || "") : "";
                      if (value) return value;
                    }
                    return "";
                  };
                  const meta = (names) => {
                    for (const name of names) {
                      const node = document.querySelector(`meta[property="${name}"], meta[name="${name}"], meta[itemprop="${name}"]`);
                      const value = node ? clean(node.getAttribute("content") || "") : "";
                      if (value) return value;
                    }
                    return "";
                  };
                  const title = clean(document.title || "");
                  const bodyText = clean(document.body ? document.body.innerText || "" : "");
                  return {
                    title,
                    body_text_sample: bodyText.slice(0, 1000),
                    name: text([
                      "h1.prod-buy-header__title",
                      ".prod-buy-header__title",
                      "h1[class*='title']",
                      "h1"
                    ]) || meta(["og:title", "twitter:title", "title"]),
                    price: text([
                      ".prod-coupon-price .total-price",
                      ".prod-sale-price .total-price",
                      ".prod-price .total-price",
                      "[class*='total-price']",
                      "[class*='sale-price']",
                      "[class*='price'] strong",
                      "[class*='price']"
                    ]) || meta(["product:price:amount", "og:price:amount", "price"]),
                    category: text([
                      ".breadcrumb a:last-child",
                      ".breadcrumb-link:last-child",
                      "[class*='breadcrumb'] a:last-child",
                      "[class*='breadcrumb'] li:last-child"
                    ]) || meta(["product:category", "category"]),
                    image_url: meta(["og:image", "twitter:image", "image"]) || attr(["img.prod-image__detail", "img[class*='prod-image']", "img"], "src"),
                    description: meta(["og:description", "description"])
                  };
                }
                """
            )
            await context.close()

        title = clean_product_title(data.get("title"))
        name = clean_product_title(data.get("name")) or title
        if name and "Access Denied" in name:
            name = None
        body_sample = str(data.get("body_text_sample") or "")
        blocked = any(keyword in f"{title or ''} {body_sample}" for keyword in ["Access Denied", "접근이 제한", "자동화"])
        product = {
            "name": name,
            "brand": infer_brand_from_text(name, data.get("description")),
            "category": clean_shopping_text(data.get("category")) or detect_category(f"{name or ''} {data.get('description') or ''}"),
            "price": _number_from_text(str(data.get("price") or "")) or 0,
            "image_url": data.get("image_url"),
            "description": clean_shopping_text(data.get("description")),
            "browser_fallback_attempted": True,
            "browser_fallback_success": bool(name and not blocked),
            "browser_fallback_blocked": blocked,
            "product_info_source": "coupang_browser",
        }
        logger.info(
            "coupang browser fallback result: %s",
            json.dumps({
                "success": product["browser_fallback_success"],
                "blocked": product["browser_fallback_blocked"],
                "name": product.get("name"),
                "brand": product.get("brand"),
                "category": product.get("category"),
                "price": product.get("price"),
            }, ensure_ascii=False, indent=2, default=str),
        )
        return product if product["browser_fallback_success"] else {
            "browser_fallback_attempted": True,
            "browser_fallback_success": False,
            "browser_fallback_blocked": blocked,
            "browser_fallback_error": "브라우저 방식에서도 쿠팡 상품정보를 확인하지 못했습니다.",
        }
    except Exception as exc:
        logger.info("coupang browser fallback failed: %s", exc)
        return {
            "browser_fallback_attempted": True,
            "browser_fallback_success": False,
            "browser_fallback_error": str(exc),
        }
    finally:
        if browser:
            try:
                await browser.close()
            except Exception:
                pass


## 네이버 쇼핑/브랜드스토어 페이지에서 상품명, 브랜드, 카테고리, 가격을 우선 추출합니다.
def parse_naver_product_data(
    soup: BeautifulSoup,
    html: str,
    url: str,
) -> Dict[str, Any]:
    product = parse_common_product_data(soup, html)
    naver_info = extract_naver_store_url_info(url)
    product["naver_store"] = naver_info
    product["shop"] = "naver"
    product["product_info_source"] = "naver_url"
    product["url_query"] = extract_url_query_hint(url)

    if not product.get("brand"):
        product["brand"] = naver_info.get("brand")
    if product.get("name"):
        product["name"] = clean_product_title(product.get("name"))
    if not product.get("brand"):
        product["brand"] = infer_brand_from_text(product.get("name"), product.get("description"), product.get("url_query"), url)
    if not product.get("category"):
        product["category"] = detect_category(f"{product.get('name', '')} {product.get('description', '')}")
    return product


## URL 추출 결과와 네이버 쇼핑 후보를 병합해 부족한 상품 정보를 보강합니다.
def merge_naver_candidate(
    product: Dict[str, Any],
    candidate: Dict[str, Any],
    allow_loose_identity_merge: bool = False,
) -> Dict[str, Any]:
    merged = dict(product)
    naver_store = merged.get("naver_store") if isinstance(merged.get("naver_store"), dict) else {}
    source_product_id = str(naver_store.get("product_id") or "")
    candidate_product_ids = {
        str(value)
        for value in [
            candidate.get("naver_product_id"),
            extract_naver_product_id_from_url(candidate.get("source_url")),
            extract_naver_product_id_from_url(candidate.get("product_url")),
        ]
        if value
    }
    same_naver_product = bool(source_product_id and source_product_id in candidate_product_ids)
    should_replace_name = (
        same_naver_product
        or is_generic_product_name(merged.get("name"))
        or bool(merged.get("name_from_url_hint"))
        or str(merged.get("name") or "").strip() == str(merged.get("url_query") or "").strip()
    )
    can_merge_identity = same_naver_product or allow_loose_identity_merge
    if can_merge_identity and should_replace_name and candidate.get("name"):
        merged["name"] = candidate["name"]
        merged["name_from_url_hint"] = False
    if can_merge_identity and (same_naver_product or not merged.get("brand")) and candidate.get("brand"):
        merged["brand"] = candidate["brand"]
    if can_merge_identity and (same_naver_product or not merged.get("category")) and candidate.get("category"):
        merged["category"] = candidate["category"]
    if can_merge_identity and (same_naver_product or not merged.get("image_url")) and candidate.get("image_url"):
        merged["image_url"] = candidate["image_url"]
    current_price = _number_from_text(str(merged.get("price"))) or 0
    candidate_price = _number_from_text(str(candidate.get("price"))) or 0
    if candidate_price and (same_naver_product or (allow_loose_identity_merge and (not current_price or current_price < 1000))):
        merged["price"] = candidate["price"]
    if can_merge_identity:
        if not merged.get("rating") and candidate.get("rating") is not None:
            merged["rating"] = candidate["rating"]
        if not merged.get("review_count") and candidate.get("review_count") is not None:
            merged["review_count"] = candidate["review_count"]
        if candidate.get("review_data_available"):
            merged["review_data_available"] = True
            merged["review_source"] = candidate.get("review_source")
            merged["review_texts"] = candidate.get("review_texts") or []
    merged["naver_enriched"] = can_merge_identity
    merged["naver_identity_merged"] = can_merge_identity
    merged["naver_candidate"] = {
        "name": candidate.get("name"),
        "brand": candidate.get("brand"),
        "price": candidate.get("price"),
        "source_url": candidate.get("source_url"),
        "mall_name": candidate.get("mall_name"),
    }
    return merged


## URL 추출 결과를 기반으로 네이버 쇼핑 검색어를 구성합니다.
def build_url_search_query(
    product: Dict[str, Any],
) -> str:
    candidates = [
        product.get("url_query"),
        product.get("brand"),
        None if is_generic_product_name(product.get("name")) else product.get("name"),
        product.get("category"),
        product.get("description"),
    ]
    query = " ".join(str(value).strip() for value in candidates if value)
    tokens = [token for token in re.split(r"\s+", query) if token and token.lower() not in {"url", "상품", "입력", "product"}]
    return " ".join(tokens[:10])


## 네이버 상품 URL 보강에 사용할 검색어 후보를 우선순위대로 구성합니다.
def build_naver_search_queries(
    product: Dict[str, Any],
) -> List[str]:
    naver_store = product.get("naver_store") if isinstance(product.get("naver_store"), dict) else {}
    gmarket_product = product.get("gmarket_product") if isinstance(product.get("gmarket_product"), dict) else {}
    candidates = [
        product.get("url_query"),
        " ".join(str(value) for value in [product.get("brand"), product.get("name")] if value and not is_generic_product_name(value)),
        None if is_generic_product_name(product.get("name")) else product.get("name"),
        product.get("brand"),
        naver_store.get("brand"),
        gmarket_product.get("keyword"),
    ]
    product_id = naver_store.get("product_id")
    if product_id and any(candidates):
        candidates.append(" ".join(str(value) for value in [product.get("brand") or naver_store.get("brand"), product_id] if value))
    if product_id:
        candidates.append(product_id)
    goods_code = gmarket_product.get("goods_code")
    if goods_code and any(candidates):
        candidates.append(" ".join(str(value) for value in [product.get("brand"), product.get("name"), goods_code] if value and not is_generic_product_name(value)))
    if goods_code:
        candidates.append(goods_code)

    queries: List[str] = []
    for candidate in candidates:
        query = _clean_text(str(candidate)) if candidate else None
        if not query:
            continue
        tokens = [token for token in re.split(r"\s+", query) if token and token.lower() not in {"url", "상품", "입력", "product", "none", "null"}]
        normalized = " ".join(tokens[:10])
        if normalized and normalized not in queries:
            queries.append(normalized)
    return queries


## 네이버 쇼핑 API 결과 중 URL 추출 상품과 가장 유사한 후보를 선택합니다.
async def enrich_product_with_naver_search(
    product: Dict[str, Any],
) -> Dict[str, Any]:
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return product
    queries = build_naver_search_queries(product)
    if not queries:
        logger.info(
            "url naver enrichment skipped: %s",
            json.dumps({
                "reason": "empty_query",
                "name": product.get("name"),
                "brand": product.get("brand"),
                "naver_store": product.get("naver_store"),
                "source_url": product.get("source_url"),
            }, ensure_ascii=False, indent=2, default=str),
        )
        return product
    naver_store = product.get("naver_store") if isinstance(product.get("naver_store"), dict) else {}
    gmarket_product = product.get("gmarket_product") if isinstance(product.get("gmarket_product"), dict) else {}
    source_product_id = str(naver_store.get("product_id") or "")
    source_gmarket_id = str(gmarket_product.get("goods_code") or "")
    has_gmarket_keyword = bool(gmarket_product.get("keyword"))
    has_real_product_name = bool(product.get("name") and not is_generic_product_name(product.get("name")) and product.get("name") not in GENERIC_PAGE_TITLES)
    candidates: List[Dict[str, Any]] = []
    used_query = queries[0]
    matched_candidate: Optional[Dict[str, Any]] = None
    for query in queries:
        try:
            query_candidates = await search_naver_shopping(query, min(NAVER_SHOPPING_DISPLAY, 5))
        except Exception as exc:
            logger.info("url naver enrichment failed: query=%s error=%s", query, exc)
            continue
        if query_candidates and not candidates:
            candidates = query_candidates
            used_query = query
        if source_product_id and query_candidates:
            for candidate in query_candidates:
                candidate_product_ids = {
                    str(value)
                    for value in [
                        candidate.get("naver_product_id"),
                        extract_naver_product_id_from_url(candidate.get("source_url")),
                        extract_naver_product_id_from_url(candidate.get("product_url")),
                    ]
                    if value
                }
                if source_product_id in candidate_product_ids:
                    matched_candidate = candidate
                    candidates = query_candidates
                    used_query = query
                    break
        if source_gmarket_id and query_candidates:
            for candidate in query_candidates:
                candidate_gmarket_ids = {
                    str(value)
                    for value in [
                        extract_gmarket_product_id_from_url(candidate.get("source_url")),
                        extract_gmarket_product_id_from_url(candidate.get("product_url")),
                    ]
                    if value
                }
                if source_gmarket_id in candidate_gmarket_ids:
                    matched_candidate = candidate
                    candidates = query_candidates
                    used_query = query
                    break
        if matched_candidate:
            break
        if query_candidates and not source_product_id:
            candidates = query_candidates
            used_query = query
            break
    if not candidates:
        logger.info(
            "url naver enrichment no candidates: %s",
            json.dumps({
                "queries": queries,
                "name": product.get("name"),
                "brand": product.get("brand"),
                "naver_store": product.get("naver_store"),
                "source_url": product.get("source_url"),
            }, ensure_ascii=False, indent=2, default=str),
        )
        return product

    reference_text = " ".join(
        str(value)
        for value in [
            product.get("url_query"),
            None if is_generic_product_name(product.get("name")) else product.get("name"),
            product.get("brand"),
        ]
        if value
    )
    best_candidate = matched_candidate or max(
        candidates,
        key=lambda item: token_similarity(
            reference_text,
            f"{item.get('name', '')} {item.get('brand', '')}",
        ),
    )
    score = token_similarity(
        reference_text,
        f"{best_candidate.get('name', '')} {best_candidate.get('brand', '')}",
    )
    if source_gmarket_id and not matched_candidate and not has_gmarket_keyword and not has_real_product_name:
        logger.info(
            "url naver enrichment skipped: %s",
            json.dumps({
                "reason": "gmarket_code_only_without_exact_match",
                "goods_code": source_gmarket_id,
                "query": used_query,
                "candidate_name": best_candidate.get("name"),
                "candidate_source_url": best_candidate.get("source_url"),
            }, ensure_ascii=False, indent=2, default=str),
        )
        return product
    if score < 0.05 and not is_generic_product_name(product.get("name")) and not product.get("name_from_url_hint"):
        return product
    allow_loose_identity_merge = bool(matched_candidate)
    enriched = merge_naver_candidate(product, best_candidate, allow_loose_identity_merge=allow_loose_identity_merge)
    enriched["naver_enrichment_query"] = used_query
    enriched["naver_enrichment_queries"] = queries
    enriched["naver_enrichment_score"] = round(score, 4)
    logger.info(
        "url product enriched by naver: %s",
        json.dumps({
            "query": used_query,
            "queries": queries,
            "score": enriched["naver_enrichment_score"],
            "name": enriched.get("name"),
            "brand": enriched.get("brand"),
            "price": enriched.get("price"),
        }, ensure_ascii=False, indent=2, default=str),
    )
    return enriched


## URL 페이지에서 뽑은 텍스트를 LLM으로 정리해 부족한 상품명/브랜드를 보강합니다.
async def enrich_product_with_llm(
    product: Dict[str, Any],
    page_text: str,
) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return product
    if not is_generic_product_name(product.get("name")) and product.get("brand"):
        return product

    try:
        from openai import AsyncOpenAI

        payload = {
            "현재상품정보": {
                "name": product.get("name"),
                "brand": product.get("brand"),
                "category": product.get("category"),
                "price": product.get("price"),
                "description": product.get("description"),
                "source_url": product.get("source_url"),
            },
            "페이지텍스트": page_text[:4000],
        }
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model=LLM_MODEL_NAME,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "당신은 쇼핑몰 URL 페이지에서 상품명과 브랜드를 보수적으로 추출하는 분석가입니다. 반드시 JSON 객체만 출력하세요.",
                },
                {
                    "role": "user",
                    "content": (
                        "아래 정보에서 실제 판매 상품의 name, brand, category, price, description을 추출하세요. "
                        "확실하지 않은 값은 null로 두세요.\n"
                        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
                    ),
                },
            ],
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
    except Exception as exc:
        logger.info("url llm enrichment failed: error=%s", exc)
        return product

    enriched = dict(product)
    for key in ("name", "brand", "category", "description"):
        value = clean_shopping_text(str(parsed.get(key))) if parsed.get(key) is not None else None
        if value and (not enriched.get(key) or (key == "name" and is_generic_product_name(enriched.get("name")))):
            enriched[key] = value
    if not enriched.get("price") and parsed.get("price") is not None:
        enriched["price"] = _number_from_text(str(parsed.get("price"))) or enriched.get("price")
    enriched["llm_enriched"] = True
    logger.info(
        "url product enriched by llm: %s",
        json.dumps({
            "name": enriched.get("name"),
            "brand": enriched.get("brand"),
            "category": enriched.get("category"),
            "price": enriched.get("price"),
        }, ensure_ascii=False, indent=2, default=str),
    )
    return enriched


## URL 본문 수집에 실패했을 때 URL 자체의 단서와 네이버 검색으로 상품 정보를 보강합니다.
async def fallback_product_from_url(
    url: str,
    error: Optional[Exception] = None,
) -> Dict[str, Any]:
    shop = detect_shop(url)
    query_hint = extract_url_query_hint(url)
    naver_info = extract_naver_store_url_info(url)
    coupang_info = extract_coupang_url_info(url)
    gmarket_info = extract_gmarket_url_info(url)
    inferred_brand = naver_info.get("brand") or infer_brand_from_text(query_hint, url)
    product: Dict[str, Any] = {
        "name": "URL 입력 상품",
        "brand": inferred_brand,
        "source_url": url,
        "product_url": url,
        "shop": shop,
        "product_info_missing": True,
        "description": "URL에서 상품 정보를 자동 추출하지 못해 URL 단서와 네이버 쇼핑 검색으로 보강합니다.",
        "url_query": query_hint,
        "name_from_url_hint": bool(query_hint),
        "naver_store": naver_info,
        "coupang_product": coupang_info,
        "gmarket_product": gmarket_info,
        "product_info_source": f"{shop}_url_fallback",
    }
    if shop == "coupang":
        product["description"] = "쿠팡 URL에서 상품 본문을 자동 추출하지 못했습니다. 쿠팡 페이지 접근 제한 또는 동적 렌더링일 수 있습니다."
    if shop == "gmarket":
        product["description"] = "G마켓 URL에서 상품 본문을 자동 추출하지 못했습니다. G마켓 상품코드와 URL 단서를 이용해 네이버 쇼핑 검색으로 보강합니다."
    if error:
        product["url_fetch_error"] = str(error)
        logger.info(
            "url fetch failed, fallback extraction started: %s",
            json.dumps({
                "url": url,
                "shop": shop,
                "query_hint": query_hint,
                "brand": product.get("brand"),
                "naver_store": naver_info,
                "coupang_product": coupang_info,
                "gmarket_product": gmarket_info,
                "error": str(error),
            }, ensure_ascii=False, indent=2, default=str),
        )
    if shop == "coupang":
        browser_product = await extract_coupang_with_browser(url)
        product = merge_product_info(product, browser_product)
    if shop in {"naver", "gmarket"}:
        product = await enrich_product_with_naver_search(product)
    if not product.get("brand"):
        product["brand"] = infer_brand_from_text(product.get("name"), product.get("description"), query_hint, url)
    if not product.get("category"):
        product["category"] = detect_category(f"{product.get('name', '')} {product.get('description', '')}")
    if query_hint and not product.get("search_query"):
        product["search_query"] = query_hint
    product.setdefault("price", 0)
    if not (_number_from_text(str(product.get("price"))) or 0):
        product["price_missing"] = True
    product.setdefault("rating", 3.5)
    product.setdefault("review_count", 0)
    product.setdefault("return_rate", 5.0)
    return product


## URL 추출 결과에 공통 기본값과 도메인별 보강을 적용합니다.
async def finalize_url_product(
    product: Dict[str, Any],
    url: str,
    visible_text: str,
) -> Dict[str, Any]:
    shop = detect_shop(url)
    product["source_url"] = url
    product["product_url"] = url
    product["shop"] = shop
    product.setdefault("url_query", extract_url_query_hint(url))

    if not product.get("brand"):
        product["brand"] = infer_brand_from_text(product.get("name"), product.get("description"), product.get("url_query"), visible_text[:1000], url)
    if not product.get("price"):
        product["price"] = 0
        product["price_missing"] = True
    if not product.get("rating"):
        product["rating"] = 3.5
    if not product.get("review_count"):
        product["review_count"] = 0
    if not product.get("return_rate"):
        product["return_rate"] = 5.0
    if not product.get("category"):
        product["category"] = detect_category(f"{product.get('name', '')} {product.get('description', '')} {visible_text}")

    if is_generic_product_name(product.get("name")) and product.get("url_query"):
        product["name_from_url_hint"] = True
    if shop == "naver":
        product = await enrich_product_with_naver_search(product)
    if shop == "coupang" and is_incomplete_product_info(product):
        browser_product = await extract_coupang_with_browser(url)
        product = merge_product_info(product, browser_product)
    if shop == "gmarket" and is_incomplete_product_info(product):
        product = await enrich_product_with_naver_search(product)
    if shop != "coupang":
        product = await enrich_product_with_llm(product, visible_text)
    if not product.get("brand"):
        product["brand"] = infer_brand_from_text(product.get("name"), product.get("description"), product.get("url_query"), url)
    return product


## URL 페이지를 가져와 메타태그, JSON-LD, 본문 텍스트에서 상품 정보를 추출합니다.
async def extract_from_url(
    url: str,
) -> Dict[str, Any]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    try:
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True, headers=headers) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text
    except Exception as exc:
        return await fallback_product_from_url(url, exc)

    soup = BeautifulSoup(html, "lxml")
    shop = detect_shop(url)
    if shop == "coupang":
        product = parse_coupang_product_data(soup, html, url)
    elif shop == "naver":
        product = parse_naver_product_data(soup, html, url)
    elif shop == "gmarket":
        product = parse_gmarket_product_data(soup, html, url)
    else:
        product = parse_common_product_data(soup, html)

    naver_info = extract_naver_store_url_info(url)
    gmarket_info = extract_gmarket_url_info(url)
    product["url_query"] = extract_url_query_hint(url)
    product["naver_store"] = naver_info
    product["gmarket_product"] = gmarket_info

    if not product.get("name"):
        title = get_meta_content(soup, "og:title", "twitter:title", "title", "name")
        if not title and soup.title:
            title = soup.title.get_text(" ", strip=True)
        product["name"] = clean_product_title(title) or "URL 입력 상품"

    if not product.get("description"):
        product["description"] = get_meta_content(soup, "og:description", "twitter:description", "description")

    if not product.get("brand"):
        product["brand"] = get_meta_content(
            soup,
            "product:brand",
            "brand",
            "og:brand",
            "twitter:data1",
        )
    if not product.get("brand"):
        product["brand"] = naver_info.get("brand")

    visible_text = soup.get_text(" ", strip=True)[:8000]
    return await finalize_url_product(product, url, visible_text)


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
    if "<" in value and ">" in value:
        text = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
    else:
        text = value
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


## 분석 대상 상품과 구매조건을 이용해 대체상품 검색어 후보를 만듭니다.
def build_alternative_search_queries(
    user: Dict[str, Any],
    product: Dict[str, Any],
) -> List[str]:
    category = clean_shopping_text(str(product.get("category") or ""))
    brand = clean_shopping_text(str(product.get("brand") or ""))
    name = clean_shopping_text(str(product.get("name") or ""))
    purpose = clean_shopping_text(str(user.get("usage_purpose") or ""))
    factors = clean_shopping_text(str(user.get("important_factors") or ""))
    preferred_brands = user.get("preferred_brands")
    if isinstance(preferred_brands, list):
        preferred_brand_text = " ".join(str(item) for item in preferred_brands if item)
    else:
        preferred_brand_text = clean_shopping_text(str(preferred_brands or "")) or ""

    modifier_tokens: List[str] = []
    condition_text = f"{purpose} {factors}".lower()
    if any(token in condition_text for token in ["가성비", "저렴", "예산", "가격", "cheap", "value"]):
        modifier_tokens.extend(["가성비", "저렴한"])
    if any(token in condition_text for token in ["선물", "gift"]):
        modifier_tokens.append("선물")
    if any(token in condition_text for token in ["프리미엄", "고급", "premium"]):
        modifier_tokens.append("프리미엄")
    if any(token in condition_text for token in ["대용량", "용량", "많은", "bulk"]):
        modifier_tokens.append("대용량")

    broad_categories = {"전자기기", "기타", "상품", "생활용품"}
    base_terms = [term for term in [category, name, brand] if term and not is_generic_product_name(term)]
    queries: List[str] = []
    if name:
        name_tokens = [token for token in re.split(r"\s+", name) if token and token.lower() not in {"url", "상품", "입력"}]
        if name_tokens:
            queries.append(" ".join(name_tokens[:6]))
            if brand:
                queries.append(" ".join([brand, *name_tokens[:5]]))
    if category and category not in broad_categories:
        queries.append(" ".join([category, *modifier_tokens[:2]]).strip())
    if preferred_brand_text and category and category not in broad_categories:
        queries.append(" ".join([preferred_brand_text, category, *modifier_tokens[:1]]).strip())
    if base_terms:
        queries.append(" ".join(base_terms[:2]))

    normalized_queries: List[str] = []
    for query in queries:
        cleaned = _clean_text(query)
        if cleaned and cleaned not in normalized_queries:
            normalized_queries.append(cleaned)
    return normalized_queries[:4]


## 네이버 쇼핑에서 구매조건에 맞는 대체상품 후보를 검색합니다.
async def extract_alternative_candidates(
    user: Dict[str, Any],
    product: Dict[str, Any],
    limit: int,
) -> Dict[str, Any]:
    queries = build_alternative_search_queries(user, product)
    candidates: List[Dict[str, Any]] = []
    errors: List[str] = []
    seen: Set[str] = set()
    target_name = str(product.get("name") or "").strip().lower()
    target_url = str(product.get("source_url") or product.get("product_url") or "").strip()
    reference_text = " ".join(
        str(value)
        for value in [product.get("name"), product.get("brand"), product.get("category"), user.get("usage_purpose"), user.get("important_factors")]
        if value
    )

    for query in queries:
        try:
            products = await search_naver_shopping(query, max(limit * 2, NAVER_SHOPPING_DISPLAY))
        except Exception as exc:
            errors.append(f"{query}: {exc}")
            continue

        for item in products:
            item_url = str(item.get("source_url") or item.get("product_url") or "")
            item_name = str(item.get("name") or "").strip()
            dedupe_key = item_url or item_name.lower()
            if not dedupe_key or dedupe_key in seen:
                continue
            if target_url and item_url and item_url == target_url:
                continue
            if target_name and item_name.lower() == target_name:
                continue
            price_value = _number_from_text(str(item.get("price") or ""))
            if not item_name or not price_value:
                continue
            similarity = token_similarity(
                reference_text,
                f"{item.get('name', '')} {item.get('brand', '')} {item.get('category', '')} {item.get('description', '')}",
            )
            if reference_text and similarity < 0.03:
                continue
            seen.add(dedupe_key)
            item["alternative_source"] = "naver_shopping"
            item["alternative_search_query"] = query
            item["alternative_relevance_score"] = round(similarity, 4)
            candidates.append(item)

    return {
        "queries": queries,
        "candidates": sorted(
            candidates,
            key=lambda item: _number_from_text(str(item.get("alternative_relevance_score") or "0")) or 0.0,
            reverse=True,
        )[:limit],
        "errors": errors,
        "naver_configured": bool(NAVER_CLIENT_ID and NAVER_CLIENT_SECRET),
    }


## 네이버 쇼핑 API 응답 항목을 StopBuy 예측 입력과 화면 표시가 가능한 상품 후보 구조로 변환합니다.
def normalize_naver_shopping_item(
    item: Dict[str, Any],
    index: int,
    query: str,
) -> Dict[str, Any]:
    price_result: Dict[str, Any] = {"price": None}
    for key in ("lprice", "salePrice", "discountPrice", "lowPrice", "price", "hprice"):
        if item.get(key) not in (None, ""):
            apply_price_candidate(price_result, key, item.get(key))
    price = int(price_result.get("price") or 0)

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
    match = re.search(r"(\d+(?:\.\d+)?)\s*(만|천|k|K)?", text)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2)
    if unit == "만":
        number *= 10000
    elif unit == "천" or unit in {"k", "K"}:
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

    if NAVER_LOG_PAYLOAD_MAX_LENGTH > 0 and len(log_payload) > NAVER_LOG_PAYLOAD_MAX_LENGTH:
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
