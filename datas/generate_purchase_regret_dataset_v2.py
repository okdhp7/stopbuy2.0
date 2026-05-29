
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))

from predictor import TARGET_COLUMN, build_training_row, purchase_decision_features



DEFAULT_ROWS = 2500
DEFAULT_SEED = 42

COL_BUDGET = "예산"
COL_PURPOSE = "사용목적"
COL_FACTORS = "중요요소"
COL_PREF_BRAND = "선호브랜드"
COL_PRODUCT_NAME = "상품명"
COL_PRODUCT_DESC = "상품설명"
COL_CURRENT_PRICE = "현재가"
COL_LOWEST_PRICE = "최저가"
COL_ALT_LOWEST_PRICE = "대체상품최저가"
COL_RATING = "평점"
COL_REVIEW_COUNT = "리뷰수"
COL_NEG_REVIEW = "리뷰부정키워드수"
COL_SOURCE = "상품정보출처"
COL_RATING_COLLECTED = "평점수집여부"
COL_REVIEW_COLLECTED = "리뷰수집여부"
COL_ACTUAL_TARGET = "실제구매후회값"

WEIGHTS = {
    "예산초과율": 0.22,
    "사용목적부적합": 0.18,
    "중요요소불일치": 0.16,
    "대체상품우위위험": 0.12,
    "브랜드선호불일치": 0.08,
    "가격신뢰위험": 0.07,
    "리뷰부정위험": 0.07,
    "상품신뢰부족": 0.05,
    "구매필요성부족": 0.03,
    "행동위험": 0.02,
}

CATEGORIES = [
    {"category": "식품", "brands": ["CJ", "오뚜기", "동원"], "purposes": ["생활", "식사", "가족"], "factors": ["가격", "신선도", "배송"], "base": 0.12, "necessity": 0.82, "prices": (3000, 90000)},
    {"category": "생활용품", "brands": ["다이소", "3M", "유한킴벌리"], "purposes": ["생활", "청소", "가정"], "factors": ["가격", "내구성", "용량"], "base": 0.14, "necessity": 0.78, "prices": (5000, 150000)},
    {"category": "패션", "brands": ["나이키", "아디다스", "ZARA", "COS"], "purposes": ["출근", "데이트", "여행"], "factors": ["디자인", "착용감", "브랜드"], "base": 0.35, "necessity": 0.42, "prices": (25000, 1500000)},
    {"category": "화장품", "brands": ["이니스프리", "설화수", "라네즈"], "purposes": ["피부관리", "선물", "데일리"], "factors": ["성분", "가격", "후기"], "base": 0.28, "necessity": 0.48, "prices": (10000, 800000)},
    {"category": "전자기기", "brands": ["삼성", "LG", "애플", "소니", "레노버"], "purposes": ["업무", "공부", "게임", "콘텐츠"], "factors": ["성능", "배터리", "휴대성"], "base": 0.38, "necessity": 0.58, "prices": (90000, 9000000)},
    {"category": "가구", "brands": ["이케아", "한샘", "리바트"], "purposes": ["이사", "인테리어", "생활"], "factors": ["내구성", "크기", "디자인"], "base": 0.32, "necessity": 0.60, "prices": (50000, 5000000)},
    {"category": "명품", "brands": ["구찌", "루이비통", "샤넬", "프라다"], "purposes": ["선물", "보상", "소장"], "factors": ["브랜드", "희소성", "디자인"], "base": 0.55, "necessity": 0.20, "prices": (300000, 9000000)},
    {"category": "여행", "brands": ["대한항공", "하나투어", "마이리얼트립"], "purposes": ["휴식", "가족", "기념일"], "factors": ["가격", "일정", "숙소"], "base": 0.42, "necessity": 0.35, "prices": (150000, 5000000)},
    {"category": "교육", "brands": ["클래스101", "패스트캠퍼스", "메가스터디"], "purposes": ["공부", "취업", "자기계발"], "factors": ["강의품질", "가격", "커리큘럼"], "base": 0.24, "necessity": 0.74, "prices": (30000, 1500000)},
    {"category": "건강관리", "brands": ["정관장", "센트룸", "종근당"], "purposes": ["건강", "부모님", "회복"], "factors": ["성분", "후기", "안전성"], "base": 0.22, "necessity": 0.72, "prices": (25000, 3000000)},
]

JOBS = [
    ("학생", 700000, 19, 29),
    ("취업준비생", 1000000, 22, 33),
    ("사무직", 4200000, 24, 55),
    ("전문직", 8500000, 28, 60),
    ("공무원", 4600000, 25, 60),
    ("자영업", 5600000, 30, 65),
    ("프리랜서", 4300000, 24, 55),
    ("대기업직원", 7200000, 26, 57),
    ("주부", 2600000, 30, 61),
    ("은퇴자", 2800000, 58, 75),
]


def stable_float(*values: Any, low: float = 0.0, high: float = 1.0) -> float:
    raw = "|".join(map(str, values))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12 - 1)
    return low + (high - low) * value


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def choose(items: List[Any], *salt: Any) -> Any:
    return items[int(stable_float(*salt) * len(items)) % len(items)]


def price_from_range(low: int, high: int, idx: int) -> int:
    q = stable_float("price", idx, low=0.04, high=0.96)
    return int(round(math.exp(math.log(low) + q * (math.log(high) - math.log(low))) / 1000) * 1000)


def build_user(idx: int) -> Dict[str, Any]:
    job, base_income, min_age, max_age = choose(JOBS, "job", idx)
    age = min_age + int(stable_float("age", idx) * (max_age - min_age + 1))
    income = int(round(base_income * stable_float("income", idx, low=0.72, high=1.35) / 10000) * 10000)
    budget = int(round(income * stable_float("budget", idx, low=0.03, high=0.22) / 1000) * 1000)
    return {
        "gender": "male" if stable_float("gender", idx) < 0.5 else "female",
        "age": age,
        "monthly_income": income,
        "job": job,
        "marital_status": "기혼" if stable_float("married", idx) < (0.15 if age < 30 else 0.65) else "미혼",
        "consumption_type": "진보적" if stable_float("type", idx) < 0.48 else "보수적",
        "budget": budget,
    }


def build_product(idx: int) -> Dict[str, Any]:
    profile = choose(CATEGORIES, "category", idx)
    brand = choose(profile["brands"], "brand", idx)
    low, high = profile["prices"]
    price = price_from_range(low, high, idx)
    lowest_price = int(round(price * stable_float("lowest", idx, low=0.72, high=0.98) / 1000) * 1000)
    alt_price = int(round(price * stable_float("alt", idx, low=0.62, high=1.08) / 1000) * 1000)
    rating_available = stable_float("rating_available", idx) > 0.22
    review_available = stable_float("review_available", idx) > 0.28
    rating = round(stable_float("rating", idx, low=3.2, high=4.9), 2) if rating_available else None
    review_count = int(stable_float("review_count", idx, low=0, high=4000)) if review_available else None
    source = choose(["naver_api", "user_input", "image_estimate", "manual", "crawler"], "source", idx)
    factors = profile["factors"]
    purposes = profile["purposes"]
    name = f"{brand} {profile['category']} {idx + 1}"
    description = f"{brand} {profile['category']} 상품. 주요 특징: {', '.join(factors + purposes)}."
    return {
        "name": name,
        "brand": brand,
        "category": profile["category"],
        "price": price,
        "market_low_price": lowest_price,
        "best_alternative_price": alt_price,
        "rating": rating,
        "review_count": review_count,
        "review_data_available": bool(rating_available or review_available),
        "review_texts": ["반품 품질 불만"] * int(stable_float("negative", idx, low=0, high=5)),
        "description": description,
        "source_url": "https://search.shopping.naver.com/",
        "source": source,
        "rating_collected": "Y" if rating_available else "N",
        "review_collected": "Y" if review_available else "N",
        "base_risk": profile["base"],
        "necessity_base": profile["necessity"],
        "purpose_pool": purposes,
        "factor_pool": factors,
    }


def build_preferences(idx: int, product: Dict[str, Any]) -> Dict[str, Any]:
    purpose = choose(product["purpose_pool"], "purpose", idx) if stable_float("purpose_fit", idx) < 0.62 else choose(["선물", "소장", "유행", "휴식"], "purpose_other", idx)
    factors = product["factor_pool"][:2] if stable_float("factor_fit", idx) < 0.58 else ["희소성", "감성"]
    preferred_brand = product["brand"] if stable_float("brand_fit", idx) < 0.46 else choose([b for c in CATEGORIES for b in c["brands"] if b != product["brand"]], "other_brand", idx)
    return {
        "usage_purpose": purpose,
        "important_factors": ", ".join(factors),
        "preferred_brands": [preferred_brand],
    }


def behavior(idx: int, product: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    burden = product["price"] / max(user["monthly_income"], 1)
    review_count = product["review_count"] or 0
    views = int(clamp(math.log1p(review_count) / math.log1p(4000), 0, 1) * 40 + stable_float("views", idx, low=2, high=18))
    comparisons = int(clamp(burden * 12 + stable_float("compare", idx, low=0, high=6), 0, 60))
    revisits = int(clamp(burden * 8 + stable_float("revisit", idx, low=0, high=5), 0, 60))
    hour = int(stable_float("night", idx, low=0, high=4)) if stable_float("hour", idx) < 0.12 else int(stable_float("day", idx, low=9, high=23))
    return {
        "purchase_hour": hour,
        "card_installment": "Y" if product["price"] >= 500000 or burden > 0.18 else "N",
        "_views": max(1, views),
        "_comparisons": comparisons,
        "_revisits": revisits,
    }


def score(row: Dict[str, Any], product: Dict[str, Any]) -> float:
    behavior_risk = clamp(
        (1.0 if row["구매시간대"] >= 22 or row["구매시간대"] <= 3 else 0.0) * 0.35
        + (1.0 if row["비교상품수"] <= 1 else 0.0) * 0.25
        + (1.0 if row["재방문횟수"] <= 1 else 0.0) * 0.20
        + (1.0 if row["카드할부여부"] == "Y" else 0.0) * 0.20
    )
    score_value = 0.12 + product["base_risk"] * 0.10
    score_value += 0.22 * row["예산초과율"]
    score_value += 0.18 * (1.0 - row["사용목적적합도"])
    score_value += 0.16 * (1.0 - row["중요요소일치도"])
    score_value += 0.12 * row["대체상품우위위험"]
    score_value += 0.08 * (1.0 - row["브랜드선호일치도"])
    score_value += 0.07 * row["가격신뢰위험"]
    score_value += 0.07 * row["리뷰부정위험"]
    score_value += 0.05 * (1.0 - row["상품신뢰도"])
    score_value += 0.03 * (1.0 - row["구매필요성점수"])
    score_value += 0.02 * behavior_risk
    score_value += stable_float("noise", row[COL_PRODUCT_NAME], row["나이"], low=-0.025, high=0.025)
    return round(clamp(score_value), 3)


def build_dataset(rows: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    records: List[Dict[str, Any]] = []
    for idx in range(rows):
        user = build_user(idx)
        product = build_product(idx)
        prefs = build_preferences(idx, product)
        action = behavior(idx, product, user)
        row = build_training_row({**user, **prefs, "purchase_hour": action["purchase_hour"]}, {**product, "card_installment": action["card_installment"]})
        row["상품조회수"] = action["_views"]
        row["비교상품수"] = action["_comparisons"]
        row["재방문횟수"] = action["_revisits"]
        row.update(purchase_decision_features({**user, **prefs}, product))
        row.update({
            COL_BUDGET: user["budget"],
            COL_PURPOSE: prefs["usage_purpose"],
            COL_FACTORS: prefs["important_factors"],
            COL_PREF_BRAND: prefs["preferred_brands"][0],
            COL_PRODUCT_NAME: product["name"],
            COL_PRODUCT_DESC: product["description"],
            COL_CURRENT_PRICE: product["price"],
            COL_LOWEST_PRICE: product["market_low_price"],
            COL_ALT_LOWEST_PRICE: product["best_alternative_price"],
            COL_RATING: product["rating"],
            COL_REVIEW_COUNT: product["review_count"],
            COL_NEG_REVIEW: len(product["review_texts"]),
            COL_SOURCE: product["source"],
            COL_RATING_COLLECTED: product["rating_collected"],
            COL_REVIEW_COLLECTED: product["review_collected"],
        })
        row[TARGET_COLUMN] = score(row, product)
        row[COL_ACTUAL_TARGET] = row[TARGET_COLUMN]
        records.append(row)
    return pd.DataFrame(records)


def write_schema(output_dir: Path) -> None:
    schema = {
        "target": TARGET_COLUMN,
        "model_input_weights": WEIGHTS,
        "data_collection_sources": {
            "상품명/현재가/최저가/브랜드/카테고리": "Naver Shopping Search API",
            "사용목적/중요요소/선호브랜드/예산": "frontend user input",
            "평점/리뷰수/리뷰부정키워드": "trusted review source, crawler, or partner API when available",
            "실제구매후회값": "post-purchase feedback; synthetic label only for initial training",
        },
    }
    (output_dir / "purchase_regret_training_dataset_v2_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v2 purchase regret training dataset.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = build_dataset(args.rows, args.seed)
    csv_path = output_dir / "purchase_regret_training_dataset_v2.csv"
    xlsx_path = output_dir / "purchase_regret_training_dataset_v2.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)
    write_schema(output_dir)
    print(json.dumps({"rows": len(df), "csv": str(csv_path), "xlsx": str(xlsx_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
