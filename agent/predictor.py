import argparse
import json
import logging
import math
import os
import random
import re
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from env_loader import load_dotenv

    load_dotenv()
except Exception:
    pass


logger = logging.getLogger("stopbuy-agent.predictor")

TARGET_COLUMN = "구매후회값"
RAW_NUMERIC_COLUMNS = [
    "나이",
    "월수입",
    "예산",
    "구매단가",
    "판매단가",
    "평점",
    "리뷰수",
    "상품조회수",
    "비교상품수",
    "구매시간대",
    "재방문횟수",
]
ENGINEERED_NUMERIC_COLUMNS = [
    "예산초과율",
    "예산사용율",
    "월수입대비가격부담율",
    "월수입대비예산부담율",
    "소비성향위험도",
    "연령대위험도",
    "직업소득안정성",
    "가구부담보정값",
    "충동구매위험도",
    "사용목적적합도",
    "중요요소일치도",
    "브랜드선호일치도",
    "대체상품우위위험",
    "가격신뢰위험",
    "리뷰부정위험",
    "상품신뢰도",
    "데이터신뢰도",
    "구매필요성점수",
]
MODEL_NUMERIC_COLUMNS = RAW_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS
CATEGORICAL_COLUMNS = [
    "성별",
    "직업",
    "결혼여부",
    "소비성향",
    "가격등급",
    "카테고리",
    "브랜드",
    "상품정보출처",
    "카드할부여부",
]
TRAINING_COLUMNS = MODEL_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
SOURCE_DATASET_COLUMNS = [
    "성별",
    "나이",
    "월수입",
    "직업",
    "결혼여부",
    "소비성향",
    "예산",
    "구매단가",
    "사용목적",
    "중요요소",
    "선호브랜드",
    "상품명",
    "브랜드",
    "카테고리",
    "판매단가",
    "평점",
    "리뷰수",
    "상품설명",
    "상품정보출처",
    "상품조회수",
    "비교상품수",
    "구매시간대",
    "재방문횟수",
    "카드할부여부",
]

DATASET_FILENAME = "purchase_regret_training_dataset_source.xlsx"
FALLBACK_DATASET_FILENAMES = [
    "purchase_regret_training_dataset_v2.xlsx",
    "purchase_regret_training_dataset_1500.xlsx",
]


## 로컬 실행과 Docker 실행 모두에서 학습 데이터 경로를 찾습니다.
def resolve_default_dataset_path() -> Path:
    env_path = os.getenv("REGRET_DATASET_PATH")
    if env_path:
        return Path(env_path)
    module_dir = Path(__file__).resolve().parent
    candidates = [
        module_dir / "datas" / DATASET_FILENAME,
        module_dir.parent / "datas" / DATASET_FILENAME,
    ]
    for filename in FALLBACK_DATASET_FILENAMES:
        candidates.extend([
            module_dir / "datas" / filename,
            module_dir.parent / "datas" / filename,
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "regret_model.pkl"
DEFAULT_DATASET_PATH = resolve_default_dataset_path()
DEFAULT_OPTIONS_PATH = Path(__file__).resolve().parent / "agent_options.json"
LLM_MODEL_NAME = os.getenv("PREFERENCE_LLM_MODEL_NAME") or os.getenv("LLM_MODEL_NAME", "gpt")
LLM_PREFERENCE_TIMEOUT_SECONDS = float(os.getenv("LLM_PREFERENCE_TIMEOUT_SECONDS", "12"))
LLM_PREFERENCE_BLEND_WEIGHT = float(os.getenv("LLM_PREFERENCE_BLEND_WEIGHT", "0.6"))
LLM_PREFERENCE_ENABLED = os.getenv("LLM_PREFERENCE_ENABLED", "true").lower() not in {"0", "false", "no", "n"}
LOCAL_LLM_MAX_NEW_TOKENS = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "256"))
PREFERENCE_LLM_WARMUP_ENABLED = os.getenv("PREFERENCE_LLM_WARMUP_ENABLED", "true").lower() not in {"0", "false", "no", "n"}
DEFAULT_AGENT_OPTIONS = {
    "alternative_regret_score_threshold": 30,
    "max_alternative_products": 12,
}

LLM_MODEL_ALIASES = {
    "gpt": {"provider": "openai", "model_id": "gpt-4o-mini-2024-07-18"},
    "gemini": {"provider": "gemini", "model_id": "gemini-2.0-flash"},
    "ax": {"provider": "local_hf", "model_id": "skt/A.X-4.0-Light"},
    "exaone": {"provider": "local_hf", "model_id": "LGAI-EXAONE/EXAONE-4.0-1.2B"},
    "kanana": {"provider": "local_hf", "model_id": "kakaocorp/kanana-1.5-8b-instruct-2505"},
    "llama": {"provider": "local_hf", "model_id": "meta-llama/Llama-3.1-8B-Instruct"},
    "mistral": {"provider": "local_hf", "model_id": "mistralai/Mistral-7B-Instruct-v0.2"},
    "gemma": {"provider": "local_hf", "model_id": "google/gemma-3-4b-it"},
    "deepseek": {"provider": "local_hf", "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"},
}
_LOCAL_TEXT_GENERATOR: Dict[str, Any] = {}
_LOCAL_TEXT_GENERATOR_LOCK = threading.Lock()

PRODUCT_IMAGE_URLS = {
    "galaxy s24 fe": "https://fdn2.gsmarena.com/vv/bigpic/samsung-galaxy-s24-fe.jpg",
    "iphone 15": "https://fdn2.gsmarena.com/vv/pics/apple/apple-iphone-15-1.jpg",
    "pixel 8a": "https://fdn2.gsmarena.com/vv/bigpic/google-pixel-8a.jpg",
    "xiaomi 14t": "https://fdn2.gsmarena.com/vv/bigpic/xiaomi-14t.jpg",
    "oneplus 12r": "https://fdn2.gsmarena.com/vv/bigpic/oneplus-12r.jpg",
    "nothing phone 2a": "https://fdn2.gsmarena.com/vv/bigpic/nothing-phone-2a.jpg",
    "macbook air m3": "https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/mba13-m3-midnight-gallery1-202402?wid=900&hei=600&fmt=jpeg&qlt=90&.v=1707416815196",
    "sony wh-1000xm5": "https://mma.prnewswire.com/media/1816079/Sony_WH_1000XM5_headphones.jpg?p=publish",
    "airpods pro 2": "https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/MQD83?wid=900&hei=600&fmt=png-alpha&.v=1660803972361",
    "bose quietcomfort ultra": "https://assets.bosecreative.com/transform/775c3e9a-fcd1-489f-a2f7-a57ac66464e1/SF_QCUH_deepplum_gallery_1_816x612_x2?io=width%3A816%2Cheight%3A667%2Ctransform%3Afit&quality=90",
    "galaxy buds3 pro": "https://api.samsungmobilepress.com/api/v1/file/A883670379D5943613666FC47FDF336B969C8DAB0EE5DB9678DE17F1835EB12061AC10F8380C857FB887A4DA4D5CFD522B5EE84E4BE72A7B91D06F877E9ADFD4AF4A4E99108B76E53EE5FD7DB9A33BF4EF185AD10D51B1C0FEDFF15A6BAE9B98F7E8754FA83F81FA804EAC2ED054D34AA29D19123081709FE794338202CFD7F1",
    "jbl tour pro 3": "https://jblstore.co.id/wp-content/uploads/2024/11/01.LS-JBL-Tour-Pro-3-Product-Image-Case-Open-Black-600x600.webp",
}


## 구매후회값을 0~1 범위로 예측하는 PyTorch 회귀 신경망입니다.
## Load recommendation options from agent_options.json, falling back to defaults.
def load_agent_options(
    path: Path = DEFAULT_OPTIONS_PATH,
) -> Dict[str, Any]:
    options = dict(DEFAULT_AGENT_OPTIONS)
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                options.update(loaded)
        except Exception as exc:
            warnings.warn(f"Failed to load agent options from {path}: {exc}")
    return options


## Convert a 0~100 threshold option into the model's 0.0~1.0 score range.
def normalize_option_threshold(
    value: Any,
    default: float,
) -> float:
    parsed = safe_float(value, default)
    if parsed > 1.0:
        parsed = parsed / 100.0
    return max(0.0, min(1.0, parsed))


## Convert option values to a non-negative integer count.
def normalize_option_count(
    value: Any,
    default: int,
) -> int:
    return max(0, safe_int(value, default))


class RegretTorchRegressor(nn.Module):

    ## 입력 피처 차원에 맞춰 회귀 신경망 계층을 구성합니다.
    def __init__(
        self,
        input_dim: int,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    ## 전처리된 피처 텐서를 받아 후회 예측값을 반환합니다.
    def forward(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(value).squeeze(-1)


## 문자열/숫자 입력을 안전하게 float로 변환하고 실패 시 기본값을 반환합니다.
def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        if value is None or value == "":
            return default
        parsed = float(str(value).replace(",", ""))
        if math.isnan(parsed):
            return default
        return parsed
    except Exception:
        return default


## 문자열/숫자 입력을 안전하게 int로 변환하고 실패 시 기본값을 반환합니다.
def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value).replace(",", "")))
    except Exception:
        return default


## 점수를 0.0~1.0 범위로 제한합니다.
def clamp_score(
    value: Any,
) -> float:
    return max(0.0, min(1.0, safe_float(value)))


## 후회 점수를 low/medium/high 단계로 변환합니다.
def regret_level(
    score: float,
) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


## 가격을 학습 데이터의 가격등급 범주로 변환합니다.
def price_grade(
    price: float,
) -> str:
    if price < 30_000:
        return "초저가"
    if price < 100_000:
        return "저가"
    if price < 500_000:
        return "중저가"
    if price < 2_000_000:
        return "고가"
    return "초고가"


## Estimate a conservative price when online extraction does not include one.
def estimate_missing_price(
    product: Dict[str, Any],
) -> float:
    name = str(product.get("name") or "").lower()
    category = str(product.get("category") or "").lower()
    brand = str(product.get("brand") or "").lower()
    source = f"{name} {category} {brand}"

    if any(keyword in source for keyword in ["server", "proliant", "microserver", "hpe", "hewlett"]):
        return 1_200_000
    if any(keyword in source for keyword in ["macbook", "notebook", "laptop"]):
        return 1_400_000
    if any(keyword in source for keyword in ["iphone", "galaxy", "pixel", "phone"]):
        return 900_000
    if any(keyword in source for keyword in ["airpods", "buds", "earphone", "headphone"]):
        return 250_000
    return 300_000


## 입력 카테고리 문자열을 학습 데이터와 가까운 표준 카테고리로 정규화합니다.
def normalize_category(
    category: Optional[str],
) -> str:
    text = str(category or "").lower()
    if any(keyword in text for keyword in ["server", "proliant", "microserver", "workstation", "nas", "서버", "워크스테이션"]):
        return "전자기기"
    if any(keyword in text for keyword in ["phone", "iphone", "galaxy", "pixel", "스마트폰", "휴대폰"]):
        return "전자기기"
    if any(keyword in text for keyword in ["notebook", "laptop", "macbook", "노트북", "컴퓨터"]):
        return "전자기기"
    if any(keyword in text for keyword in ["이어폰", "헤드폰", "airpods", "buds", "sony"]):
        return "전자기기"
    if any(keyword in text for keyword in ["화장품", "cosmetic", "skincare"]):
        return "화장품"
    if any(keyword in text for keyword in ["패션", "의류", "신발", "fashion"]):
        return "패션"
    if any(keyword in text for keyword in ["가구", "의자", "책상"]):
        return "가구"
    if any(keyword in text for keyword in ["식품", "food"]):
        return "식품"
    if any(keyword in text for keyword in ["여행", "travel"]):
        return "여행"
    if any(keyword in text for keyword in ["교육", "강의"]):
        return "교육"
    if any(keyword in text for keyword in ["건강", "영양제"]):
        return "건강관리"
    if any(keyword in text for keyword in ["명품", "luxury"]):
        return "명품"
    return category or "전자기기"


## 브랜드명과 상품명에서 학습 데이터에 맞는 대표 브랜드명을 추론합니다.
def normalize_brand(
    brand: Optional[str],
    name: Optional[str] = None,
) -> str:
    source = f"{brand or ''} {name or ''}".lower()
    brand_map = {
        "samsung": "삼성",
        "galaxy": "삼성",
        "apple": "애플",
        "iphone": "애플",
        "airpods": "애플",
        "lg": "LG",
        "sony": "소니",
        "lenovo": "레노버",
        "xiaomi": "샤오미",
        "oneplus": "원플러스",
        "nothing": "Nothing",
        "google": "구글",
        "hpe": "HP",
        "hewlett": "HP",
        "proliant": "HP",
    }
    for keyword, value in brand_map.items():
        if keyword in source:
            return value
    return brand or "기타"


## 온라인 입력의 사용자/상품 정보를 모델 학습 피처 한 행으로 변환합니다.

## 안전하게 0~1 범위 점수로 제한합니다.
def clamp_unit(
    value: float,
) -> float:
    return max(0.0, min(1.0, safe_float(value)))


## LLM이 숫자 대신 낸 정성 표현을 0~1 점수로 변환합니다.
def coerce_llm_unit_score(
    value: Any,
    default: float = 0.5,
) -> float:
    if value is None:
        return clamp_unit(default)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return clamp_unit(float(value))

    text = str(value).strip().lower()
    if not text:
        return clamp_unit(default)
    numeric = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if numeric:
        number = float(numeric.group(1))
        if number > 1.0 and number <= 100.0:
            number /= 100.0
        return clamp_unit(number)

    high_words = ["매우 적합", "아주 적합", "적합함", "높음", "강함", "high", "strong", "very suitable", "same"]
    medium_words = ["보통", "중간", "부분", "medium", "partial", "somewhat"]
    low_words = ["부적합", "낮음", "약함", "다름", "low", "weak", "different", "not"]
    if any(word in text for word in high_words):
        return 0.9
    if any(word in text for word in medium_words):
        return 0.55
    if any(word in text for word in low_words):
        return 0.1
    return clamp_unit(default)


## 규칙 기반 적합도와 LLM 적합도를 가중 평균으로 결합합니다.
def blend_llm_fit_score(
    rule_score: float,
    llm_score: Any,
) -> float:
    if llm_score is None:
        return clamp_unit(rule_score)
    llm_value = clamp_unit(safe_float(llm_score, rule_score))
    weight = clamp_unit(LLM_PREFERENCE_BLEND_WEIGHT)
    return clamp_unit(rule_score * (1.0 - weight) + llm_value * weight)


## 규칙 기반 브랜드 비교와 LLM의 동일 브랜드 판별 결과를 결합합니다.
def resolve_brand_fit_score(
    preferred_tokens: set[str],
    brand_tokens: set[str],
    llm_scores: Optional[Dict[str, Any]] = None,
) -> float:
    if not preferred_tokens:
        return 0.5

    rule_score = 1.0 if preferred_tokens & brand_tokens else 0.0
    if not llm_scores or not llm_scores.get("used_llm"):
        return rule_score

    confidence = clamp_unit(llm_scores.get("brand_confidence"))
    same_brand = bool(llm_scores.get("same_brand"))
    brand_fit = llm_scores.get("brand_fit")
    if brand_fit is not None:
        return max(rule_score, clamp_unit(brand_fit)) if confidence >= 0.6 else rule_score
    if same_brand and confidence >= 0.8:
        return 1.0
    if same_brand and confidence >= 0.6:
        return max(rule_score, 0.7)
    if not same_brand and confidence >= 0.8:
        return 0.0
    return rule_score


## 입력 토큰과 상품 토큰의 단순 중첩 유사도를 계산합니다.
def preference_similarity_score(
    input_value: Any,
    product: Dict[str, Any],
) -> float:
    input_tokens = tokenize_preference_text(input_value)
    if not input_tokens:
        return 0.5
    product_tokens = tokenize_preference_text(product_preference_text(product))
    if not product_tokens:
        return 0.0
    return preference_token_match_score(input_tokens, product_tokens)


## 상품 카테고리와 사용목적에서 구매 필요성 점수를 추정합니다.
def purchase_necessity_score(
    user: Dict[str, Any],
    product: Dict[str, Any],
) -> float:
    purpose_tokens = tokenize_preference_text(user.get("usage_purpose"))
    category = normalize_category(product.get("category"))
    text = product_preference_text(product).lower()
    practical_categories = {"식품", "생활용품", "교육", "건강관리"}
    discretionary_categories = {"명품", "패션", "화장품", "여행"}
    work_tokens = {"work", "business", "office", "server", "storage", "업무", "사업", "사무", "서버", "백업", "스토리지"}
    score = 0.5
    if category in practical_categories:
        score += 0.2
    if category in discretionary_categories:
        score -= 0.15
    if purpose_tokens:
        score += (preference_similarity_score(user.get("usage_purpose"), product) - 0.5) * 0.5
    if any(word in text for word in ["업무", "공부", "학습", "생활", "필수", "office", "study", "daily"]):
        score += 0.12
    if purpose_tokens & work_tokens and any(word in text for word in ["server", "proliant", "microserver", "workstation", "nas", "서버", "워크스테이션", "백업", "스토리지"]):
        score += 0.18
    if any(word in text for word in ["한정", "럭셔리", "명품", "선물", "limited", "premium"]):
        score -= 0.08
    return clamp_unit(score)


## 평점/리뷰수는 보조 신호로만 쓰도록 상품 신뢰도를 계산합니다.
def product_trust_score(
    product: Dict[str, Any],
) -> float:
    rating_missing = bool(product.get("rating_missing"))
    review_count_missing = bool(product.get("review_count_missing"))
    review_data_available = bool(product.get("review_data_available"))
    rating = safe_float(product.get("rating"), 3.5)
    review_count = safe_int(product.get("review_count"))
    rating_score = 0.5 if rating_missing else clamp_unit((rating - 3.0) / 2.0)
    volume_score = 0.35 if review_count_missing else clamp_unit(math.log1p(max(review_count, 0)) / math.log1p(1000))
    source_score = 0.65 if review_data_available else 0.25
    return clamp_unit(rating_score * 0.25 + volume_score * 0.25 + source_score * 0.50)


## 사용자 특성정보를 구매부담과 후회위험에 직접 연결되는 파생 피처로 변환합니다.
def user_profile_risk_features(
    user: Dict[str, Any],
    product: Dict[str, Any],
    effective_budget: float,
) -> Dict[str, float]:
    price = safe_float(product.get("price"))
    monthly_income = safe_float(user.get("monthly_income"))
    age = safe_int(user.get("age"), 40)
    job = str(user.get("job") or "").strip()
    marital_status = str(user.get("marital_status") or "").strip()
    consumption_type = str(user.get("consumption_type") or "").strip()
    category = normalize_category(product.get("category"))

    income_price_burden = price / max(monthly_income, 1.0) if monthly_income > 0 and price > 0 else 0.0
    income_budget_burden = effective_budget / max(monthly_income, 1.0) if monthly_income > 0 and effective_budget > 0 else 0.0

    consumption_risk_map = {
        "보수적": 0.30,
        "균형형": 0.45,
        "가성비형": 0.38,
        "프리미엄형": 0.48,
        "충동형": 0.82,
    }
    consumption_risk = consumption_risk_map.get(consumption_type, 0.45)

    if age <= 24:
        age_risk = 0.62
    elif age <= 34:
        age_risk = 0.50
    elif age <= 54:
        age_risk = 0.42
    elif age <= 69:
        age_risk = 0.48
    else:
        age_risk = 0.58

    job_stability_map = {
        "전문직": 0.85,
        "공무원": 0.82,
        "사무직": 0.72,
        "기술직": 0.70,
        "영업직": 0.58,
        "서비스직": 0.55,
        "자영업": 0.50,
        "프리랜서": 0.45,
        "주부": 0.48,
        "학생": 0.35,
        "무직": 0.22,
        "기타": 0.50,
    }
    job_stability = job_stability_map.get(job, 0.55)

    household_adjustment = 0.40
    if marital_status == "기혼":
        household_adjustment += 0.12
        if category in {"명품", "여행", "패션", "화장품"}:
            household_adjustment += 0.08
    elif marital_status == "미혼":
        household_adjustment -= 0.04

    impulse_risk = consumption_risk
    if consumption_type == "충동형" and price >= 500_000:
        impulse_risk += 0.12
    if monthly_income > 0 and price > monthly_income * 0.5:
        impulse_risk += 0.10

    return {
        "월수입대비가격부담율": min(income_price_burden, 3.0),
        "월수입대비예산부담율": min(income_budget_burden, 1.5),
        "소비성향위험도": clamp_unit(consumption_risk),
        "연령대위험도": clamp_unit(age_risk),
        "직업소득안정성": clamp_unit(job_stability),
        "가구부담보정값": clamp_unit(household_adjustment),
        "충동구매위험도": clamp_unit(impulse_risk),
    }


## 가격, 조건 적합도, 대체상품 가능성 등 구매후회 핵심 입력지표를 산출합니다.
def purchase_decision_features(
    user: Dict[str, Any],
    product: Dict[str, Any],
) -> Dict[str, float]:
    price = safe_float(product.get("price"))
    budget = safe_float(user.get("budget"))
    purchase_unit_price = safe_float(
        user.get("purchase_unit_price")
        or product.get("purchase_unit_price")
        or product.get("target_price")
        or price
    )
    if budget <= 0:
        monthly_income = safe_float(user.get("monthly_income"))
        budget = monthly_income * 0.08 if monthly_income > 0 else 0.0
    profile_features = user_profile_risk_features(user, product, budget)

    budget_usage = price / max(budget, 1.0) if budget > 0 else 0.0
    budget_overrun = max(0.0, budget_usage - 1.0)
    llm_scores = product.get("llm_preference_scores") or {}
    purpose_fit = blend_llm_fit_score(
        preference_similarity_score(user.get("usage_purpose"), product),
        llm_scores.get("usage_purpose_fit"),
    )
    factor_fit = blend_llm_fit_score(
        preference_similarity_score(user.get("important_factors"), product),
        llm_scores.get("important_factor_fit"),
    )

    preferred_tokens = tokenize_preference_text(user.get("preferred_brands"))
    brand_tokens = tokenize_preference_text(product.get("brand"), product.get("name"))
    brand_fit = resolve_brand_fit_score(preferred_tokens, brand_tokens, llm_scores)

    target_price = safe_float(product.get("target_price"))
    alternative_price = safe_float(product.get("best_alternative_price"))
    if alternative_price > 0 and price > 0:
        alternative_risk = clamp_unit((price - alternative_price) / max(price, 1.0))
    elif budget > 0 and price > budget:
        alternative_risk = clamp_unit((price - budget) / max(price, 1.0))
    elif target_price > 0 and price > target_price:
        alternative_risk = clamp_unit((price - target_price) / max(price, 1.0))
    else:
        alternative_risk = 0.0

    market_low_price = safe_float(product.get("market_low_price") or product.get("lowest_price"))
    if market_low_price > 0 and price > market_low_price:
        price_reliability_risk = clamp_unit((price - market_low_price) / max(market_low_price, 1.0))
    else:
        price_reliability_risk = 0.35 if product.get("price_estimated") else 0.05

    negative_review_risk = clamp_unit(len(detect_negative_review_signals(product)) / 5.0)
    trust_score = product_trust_score(product)
    data_reliability = 0.35
    if not product.get("price_estimated") and price > 0:
        data_reliability += 0.25
    if product.get("source_url") or product.get("product_url"):
        data_reliability += 0.15
    if product.get("review_data_available"):
        data_reliability += 0.25
    necessity = purchase_necessity_score(user, product)

    return {
        "예산초과율": clamp_unit(budget_overrun),
        "예산사용율": min(budget_usage, 3.0),
        **profile_features,
        "사용목적적합도": purpose_fit,
        "중요요소일치도": factor_fit,
        "브랜드선호일치도": brand_fit,
        "대체상품우위위험": alternative_risk,
        "가격신뢰위험": price_reliability_risk,
        "리뷰부정위험": negative_review_risk,
        "상품신뢰도": trust_score,
        "데이터신뢰도": clamp_unit(data_reliability),
        "구매필요성점수": necessity,
    }


def build_training_row(
    user: Dict[str, Any],
    product: Dict[str, Any],
) -> Dict[str, Any]:
    price = safe_float(product.get("price"))
    review_count = safe_int(product.get("review_count"), 20)
    return_rate = safe_float(product.get("return_rate"), 5.0)
    price_estimated = bool(product.get("price_estimated"))
    rating = safe_float(product.get("rating"), 3.5)
    budget = safe_float(user.get("budget"))
    purchase_unit_price = safe_float(
        user.get("purchase_unit_price")
        or product.get("purchase_unit_price")
        or product.get("target_price")
        or price
    )
    category = normalize_category(product.get("category"))
    brand = normalize_brand(product.get("brand"), product.get("name"))
    decision_product = dict(product)
    decision_product["category"] = category
    decision_product["brand"] = brand
    decision_features = purchase_decision_features(user, decision_product)

    views = max(2, min(60, int(review_count / 20) + 10))
    comparison_count = max(0, min(60, int(return_rate / 2 + max(0.0, 4.0 - rating) * 3)))
    revisit_count = max(0, min(60, int(return_rate / 3 + max(0.0, 4.0 - rating) * 2)))
    installment = "Y" if price >= 500_000 else "N"

    return {
        "성별": user.get("gender") or "male",
        "나이": safe_int(user.get("age"), 40),
        "월수입": safe_float(user.get("monthly_income"), budget if budget > 0 else 3_790_000),
        "직업": user.get("job") or "사무직",
        "결혼여부": user.get("marital_status") or "미혼",
        "소비성향": user.get("consumption_type") or "보수적",
        "예산": budget,
        "구매단가": purchase_unit_price,
        "판매단가": price,
        "평점": rating,
        "리뷰수": review_count,
        "가격등급": product.get("price_grade") or price_grade(price),
        "카테고리": category,
        "브랜드": brand,
        "상품정보출처": product.get("product_info_source") or product.get("review_source") or product.get("source") or product.get("source_url") or "user_input",
        "상품조회수": views,
        "비교상품수": comparison_count,
        "구매시간대": safe_int(user.get("purchase_hour"), 15),
        "카드할부여부": product.get("card_installment") or installment,
        "재방문횟수": revisit_count,
        **decision_features,
    }


## 예산, 평점, 반품률, 리뷰 수 등을 바탕으로 후회 원인 목록을 생성합니다.

## Normalize a user input value into a readable list for cause messages.
def normalize_user_input_list(
    value: Any,
) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = re.split(r"[,/|\n\t]+", str(value))
    return [str(item).strip() for item in values if str(item).strip()]


## Detect explicit risk words in product name, category, and description.
def detect_product_risk_keywords(
    product: Dict[str, Any],
) -> List[str]:
    text = product_preference_text(product).lower()
    keyword_groups = {
        "중고": ["중고", "used", "secondhand"],
        "리퍼": ["리퍼", "리퍼비시", "refurb", "refurbished"],
        "호환품": ["호환", "compatible", "replacement"],
        "부품": ["부품", "parts", "part only"],
        "벌크": ["벌크", "bulk"],
        "해외직구": ["해외직구", "직구", "import", "구매대행"],
    }
    found: List[str] = []
    for label, keywords in keyword_groups.items():
        if any(keyword.lower() in text for keyword in keywords):
            found.append(label)
    return found


## Detect negative signals from collected review text samples.
def detect_negative_review_signals(
    product: Dict[str, Any],
) -> List[str]:
    review_texts = product.get("review_texts") or []
    if not isinstance(review_texts, list):
        return []
    negative_keywords = {
        "불량", "고장", "파손", "반품", "환불", "실망", "별로", "최악", "느림", "소음", "냄새",
        "defect", "broken", "refund", "return", "disappointed", "bad", "worst", "noise",
    }
    joined = " ".join(str(text).lower() for text in review_texts)
    found = [keyword for keyword in negative_keywords if keyword.lower() in joined]
    return sorted(found)[:5]

def make_regret_causes(
    model_score: float,
    user: Dict[str, Any],
    product: Dict[str, Any],
    alignment: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    causes: List[Dict[str, Any]] = []
    budget = safe_float(user.get("budget"))
    price = safe_float(product.get("price"))
    price_estimated = bool(product.get("price_estimated"))
    rating = safe_float(product.get("rating"), 3.5)
    rating_missing = bool(product.get("rating_missing"))
    return_rate = safe_float(product.get("return_rate"), 5.0)
    review_count = safe_int(product.get("review_count"))
    review_data_available = bool(product.get("review_data_available"))
    alignment = alignment or {}

    if product.get("price_estimated"):
        causes.append({
            "code": "PRICE_ESTIMATED",
            "title": "가격 정보 추정",
            "message": "이미지 분석에서 가격이 확인되지 않아 상품명과 카테고리 기준으로 보수적 추정가를 적용했습니다.",
            "severity": "medium",
            "impact_score": 0.12,
        })

    if budget > 0 and price > budget:
        over_ratio = (price - budget) / budget
        causes.append({
            "code": "PRICE_OVER_BUDGET",
            "title": "예산 초과",
            "message": f"상품 가격이 예산보다 {over_ratio * 100:.1f}% 높아 구매 후 부담을 느낄 가능성이 있습니다.",
            "severity": "high" if over_ratio >= 0.3 else "medium",
            "impact_score": round(min(over_ratio, 1.0), 4),
        })
    if not rating_missing and rating < 3.8:
        causes.append({
            "code": "LOW_RATING",
            "title": "낮은 사용자 평점",
            "message": "평점이 낮아 실제 사용 만족도가 기대보다 낮을 수 있습니다.",
            "severity": "high" if rating < 3.5 else "medium",
            "impact_score": round((3.8 - rating) / 3.8, 4),
        })
    if return_rate >= 10:
        causes.append({
            "code": "HIGH_RETURN_RATE",
            "title": "높은 반품율",
            "message": "반품율이 높아 같은 상품을 구매한 사용자의 불만 가능성이 상대적으로 큽니다.",
            "severity": "high" if return_rate >= 15 else "medium",
            "impact_score": round(min(return_rate / 30, 1.0), 4),
        })
    if not review_data_available:
        causes.append({
            "code": "REVIEW_DATA_MISSING",
            "title": "평가정보 부족",
            "message": "평점과 후기개수를 확인하지 못해 실제 만족도 검증이 부족합니다.",
            "severity": "low",
            "impact_score": 0.16,
        })
    elif review_count < 20:
        causes.append({
            "code": "LOW_REVIEW_COUNT",
            "title": "리뷰 수 부족",
            "message": "후기 수가 적어 실제 사용자 만족도를 판단하기에 근거가 부족합니다.",
            "severity": "low",
            "impact_score": 0.18,
        })
    preferred_brands = normalize_user_input_list(user.get("preferred_brands"))
    if preferred_brands and not alignment.get("matched_preferred_brand"):
        causes.append({
            "code": "PREFERRED_BRAND_MISMATCH",
            "title": "선호브랜드 불일치",
            "message": f"입력한 선호브랜드({', '.join(preferred_brands[:3])})와 대상 상품 브랜드가 충분히 일치하지 않습니다.",
            "severity": "medium",
            "impact_score": 0.24,
        })

    important_factors = normalize_user_input_list(user.get("important_factors"))
    purpose = str(user.get("usage_purpose") or "").strip()
    condition_similarity = safe_float(alignment.get("condition_similarity"))
    if important_factors and condition_similarity < 0.18:
        causes.append({
            "code": "IMPORTANT_FACTOR_MISMATCH",
            "title": "중요요소 불일치",
            "message": f"중요하게 입력한 요소({', '.join(important_factors[:3])})가 상품명/브랜드/카테고리/설명에서 충분히 확인되지 않습니다.",
            "severity": "medium",
            "impact_score": 0.26,
        })
    if purpose and condition_similarity < 0.18:
        causes.append({
            "code": "PURPOSE_MISMATCH",
            "title": "사용목적 적합도 낮음",
            "message": f"사용목적({purpose[:40]})과 상품 정보의 연관성이 낮아 실제 사용 후 불만이 생길 수 있습니다.",
            "severity": "medium",
            "impact_score": 0.25,
        })

    decision_features = purchase_decision_features(user, product)
    income_price_burden = safe_float(decision_features.get("월수입대비가격부담율"))
    consumption_type = str(user.get("consumption_type") or "").strip()
    job = str(user.get("job") or "").strip()
    if income_price_burden >= 0.5:
        causes.append({
            "code": "INCOME_PRICE_BURDEN",
            "title": "소득 대비 가격 부담",
            "message": f"상품 가격이 월수입의 {income_price_burden * 100:.1f}% 수준이라 구매 후 지출 부담을 느낄 가능성이 있습니다.",
            "severity": "high" if income_price_burden >= 0.8 else "medium",
            "impact_score": round(min(income_price_burden, 1.0), 4),
        })
    if consumption_type == "충동형" and price >= 500_000:
        causes.append({
            "code": "IMPULSE_HIGH_PRICE",
            "title": "충동구매 위험",
            "message": "소비성향이 충동형이고 상품 가격이 높아 구매 직후 후회 가능성이 커질 수 있습니다.",
            "severity": "medium",
            "impact_score": 0.34,
        })
    if consumption_type == "보수적" and budget > 0 and price > budget:
        causes.append({
            "code": "CONSERVATIVE_OVER_BUDGET",
            "title": "보수적 소비성향과 예산 초과",
            "message": "보수적 소비성향에서는 예산을 넘는 구매가 심리적 부담과 후회로 이어질 가능성이 큽니다.",
            "severity": "medium",
            "impact_score": 0.31,
        })
    if job in {"학생", "무직"} and price >= 500_000:
        causes.append({
            "code": "JOB_INCOME_STABILITY_RISK",
            "title": "소득 안정성 대비 고가 구매",
            "message": f"직업 정보({job})를 기준으로 볼 때 고가 상품 구매 부담이 상대적으로 클 수 있습니다.",
            "severity": "medium",
            "impact_score": 0.30,
        })

    risk_keywords = detect_product_risk_keywords(product)
    if risk_keywords:
        causes.append({
            "code": "RISK_KEYWORD_USED",
            "title": "상품명 위험 키워드",
            "message": f"상품 정보에 {', '.join(risk_keywords)} 표현이 포함되어 구매 전 상태와 조건을 더 확인해야 합니다.",
            "severity": "high" if any(keyword in risk_keywords for keyword in ["중고", "리퍼", "부품"]) else "medium",
            "impact_score": 0.42,
        })

    negative_reviews = detect_negative_review_signals(product)
    if negative_reviews:
        causes.append({
            "code": "REVIEW_NEGATIVE_SIGNAL",
            "title": "리뷰 부정 신호",
            "message": f"수집된 리뷰 샘플에서 {', '.join(negative_reviews)} 관련 표현이 감지되었습니다.",
            "severity": "medium",
            "impact_score": 0.32,
        })

    if model_score >= 0.4 and not causes:
        causes.append({
            "code": "MODEL_HIGH_RISK",
            "title": "모델 예측 위험",
            "message": "학습 모델이 유사 구매 패턴에서 후회 가능성이 높다고 판단했습니다.",
            "severity": "medium",
            "impact_score": round(model_score, 4),
        })
    if not causes:
        causes.append({
            "code": "NO_MAJOR_RISK",
            "title": "주요 후회 요인 없음",
            "message": "현재 입력 조건 기준으로 두드러진 구매 후회 위험 요인은 보이지 않습니다.",
            "severity": "low",
            "impact_score": 0.0,
        })
    return sorted(causes, key=lambda item: item["impact_score"], reverse=True)[:5]


## 모델이 과소평가할 수 있는 명시적 위험요인을 규칙 기반 점수로 계산합니다.

## Tokenize user preference and product text for lightweight similarity scoring.
def tokenize_preference_text(
    *values: Any,
) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            tokens.update(tokenize_preference_text(*value))
            continue
        text = str(value).lower()
        for separator in [",", "/", "|", "&", "?", "-", "\\", "\n", "\t"]:
            text = text.replace(separator, " ")
        raw_tokens = [token.strip() for token in text.split() if token.strip()]
        tokens.update(raw_tokens)
        compact = "".join(raw_tokens)
        if compact:
            tokens.add(compact)
    aliases = {
        "apple": ["iphone", "airpods", "macbook", "애플"],
        "samsung": ["galaxy", "buds", "삼성"],
        "sony": ["wh-1000xm", "headphone", "소니"],
        "price": ["budget", "cheap", "value", "가격", "예산", "가성비", "저렴"],
        "quality": ["rating", "performance", "premium", "품질", "평점", "성능", "완성도"],
        "portable": ["light", "battery", "mobile", "휴대", "가벼움", "배터리"],
        "work": ["office", "business", "productivity", "job", "업무", "사무", "문서", "사업", "회사"],
        "server": ["server", "workstation", "microserver", "proliant", "nas", "서버", "워크스테이션", "백업", "스토리지", "홈서버"],
        "storage": ["storage", "backup", "raid", "disk", "스토리지", "백업", "저장", "디스크"],
        "game": ["gaming", "performance", "display", "게임", "고성능"],
        "study": ["education", "lecture", "reading", "공부", "강의", "학습"],
        "travel": ["trip", "portable", "battery", "여행", "휴대"],
        "stable": ["return", "durable", "reliable", "안정", "내구", "반품"],
    }
    expanded = set(tokens)
    for canonical, words in aliases.items():
        if canonical in tokens or any(word in tokens for word in words):
            expanded.add(canonical)
            expanded.update(words)
    return expanded


## Build searchable text from product fields for preference similarity scoring.
def product_preference_text(
    product: Dict[str, Any],
) -> str:
    base_text = " ".join(
        str(product.get(key) or "")
        for key in ["name", "brand", "category", "description"]
    )
    return f"{base_text} {normalize_category(product.get('category'))}"


## 두 선호 토큰이 완전 일치 또는 의미 있는 부분 일치 관계인지 판단합니다.
def preference_tokens_match(
    left: str,
    right: str,
) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) < 2 or len(right) < 2:
        return False
    return left in right or right in left


## 입력 토큰 중 상품 토큰과 일치하거나 부분 일치하는 토큰을 찾습니다.
def matched_preference_tokens(
    input_tokens: set[str],
    product_tokens: set[str],
) -> set[str]:
    return {
        input_token
        for input_token in input_tokens
        if any(preference_tokens_match(input_token, product_token) for product_token in product_tokens)
    }


## 별칭 확장으로 분모가 커지는 문제를 줄여 선호 토큰 매칭 점수를 계산합니다.
def preference_token_match_score(
    input_tokens: set[str],
    product_tokens: set[str],
) -> float:
    if not input_tokens:
        return 0.5
    if not product_tokens:
        return 0.0
    matched = matched_preference_tokens(input_tokens, product_tokens)
    if not matched:
        return 0.0
    compact_denominator = max(min(len(input_tokens), 4), 1)
    score = len(matched) / compact_denominator
    return clamp_unit(max(score, 0.35))


## LLM 응답 문자열에서 JSON 객체를 추출합니다.
def extract_json_object(
    text: str,
) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


## LLM 별칭을 실제 제공자와 모델 ID로 변환합니다.
def resolve_llm_model_config(
    model_name: Optional[str] = None,
) -> Dict[str, str]:
    requested = (model_name or LLM_MODEL_NAME or "gpt").strip()
    lowered = requested.lower()
    alias = LLM_MODEL_ALIASES.get(lowered)
    if alias:
        provider = str(alias["provider"])
        model_id = os.getenv("LLM_BASE_MODEL_ID") or str(alias["model_id"])
    elif lowered.startswith("hf:"):
        provider = "local_hf"
        model_id = requested[3:]
    elif lowered.startswith("gemini"):
        provider = "gemini"
        model_id = os.getenv("LLM_BASE_MODEL_ID") or requested
    else:
        provider = "openai"
        model_id = os.getenv("LLM_BASE_MODEL_ID") or requested
    provider = os.getenv("LLM_PROVIDER", provider).strip().lower()
    return {
        "requested": requested,
        "provider": provider,
        "model_id": model_id,
    }


## OpenAI Chat Completions API로 JSON 응답을 생성합니다.
def generate_openai_json(
    prompt: str,
    model_id: str,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 필요합니다.")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=LLM_PREFERENCE_TIMEOUT_SECONDS)
    response = client.chat.completions.create(
        model=model_id,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "당신은 상품 구매조건과 상품정보의 의미적 적합도를 보수적으로 평가하는 분석가입니다.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


## Gemini API로 JSON 응답을 생성합니다.
def generate_gemini_json(
    prompt: str,
    model_id: str,
) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 또는 GOOGLE_API_KEY가 필요합니다.")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    return response.text or ""


## Hugging Face transformers 로컬 모델을 로딩하고 프로세스 캐시에 저장합니다.
def load_local_hf_generator(
    model_id: str,
) -> Any:
    cache_key = model_id
    generator = _LOCAL_TEXT_GENERATOR.get(cache_key)
    if generator is not None:
        return generator

    with _LOCAL_TEXT_GENERATOR_LOCK:
        generator = _LOCAL_TEXT_GENERATOR.get(cache_key)
        if generator is not None:
            return generator

        from transformers import pipeline

        trust_remote_code = os.getenv("LOCAL_LLM_TRUST_REMOTE_CODE", "true").lower() in {"1", "true", "yes", "y"}
        logger.info("local LLM loading model: model_id=%s", model_id)
        generator = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        _LOCAL_TEXT_GENERATOR[cache_key] = generator
        logger.info("local LLM model loaded: model_id=%s", model_id)
        return generator


## Hugging Face transformers 로컬 모델로 JSON 응답을 생성합니다.
def generate_local_hf_json(
    prompt: str,
    model_id: str,
) -> str:
    generator = load_local_hf_generator(model_id)
    messages = [
        {
            "role": "system",
            "content": "당신은 상품 구매조건과 상품정보의 의미적 적합도를 보수적으로 평가하는 분석가입니다. 반드시 JSON 객체만 출력하세요.",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        tokenizer = generator.tokenizer
        model_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        model_input = (
            "시스템: 당신은 상품 구매조건과 상품정보의 의미적 적합도를 보수적으로 평가하는 분석가입니다. 반드시 JSON 객체만 출력하세요.\n"
            f"사용자: {prompt}\n"
            "응답:"
        )
    outputs = generator(
        model_input,
        max_new_tokens=LOCAL_LLM_MAX_NEW_TOKENS,
        do_sample=False,
        return_full_text=False,
    )
    if isinstance(outputs, list) and outputs:
        return str(outputs[0].get("generated_text") or "")
    return str(outputs or "")


## 설정에 따라 agent 시작 시점에 로컬 LLM 모델을 미리 로딩합니다.
def warm_up_preference_llm() -> Dict[str, Any]:
    config = resolve_llm_model_config()
    provider = config["provider"]
    model_id = config["model_id"]
    if not LLM_PREFERENCE_ENABLED:
        logger.info("preference LLM warm-up skipped: disabled")
        return {**config, "warmed": False, "reason": "disabled"}
    if not PREFERENCE_LLM_WARMUP_ENABLED:
        logger.info("preference LLM warm-up skipped: warm-up disabled")
        return {**config, "warmed": False, "reason": "warmup_disabled"}
    if provider not in {"local", "local_hf", "hf", "transformers"}:
        logger.info(
            "preference LLM warm-up skipped: requested=%s provider=%s model_id=%s",
            config["requested"],
            provider,
            model_id,
        )
        return {**config, "warmed": False, "reason": "not_local_llm"}

    logger.info(
        "preference LLM warm-up start: requested=%s provider=%s model_id=%s",
        config["requested"],
        provider,
        model_id,
    )
    load_local_hf_generator(model_id)
    logger.info(
        "preference LLM warm-up completed: requested=%s provider=%s model_id=%s",
        config["requested"],
        provider,
        model_id,
    )
    return {**config, "warmed": True}


## 설정된 LLM 제공자에 따라 JSON 응답을 생성합니다.
def generate_llm_json(
    prompt: str,
) -> Dict[str, Any]:
    config = resolve_llm_model_config()
    provider = config["provider"]
    model_id = config["model_id"]
    logger.info(
        "preference LLM start: requested=%s provider=%s model_id=%s",
        config["requested"],
        provider,
        model_id,
    )
    try:
        if provider == "openai":
            content = generate_openai_json(prompt, model_id)
        elif provider == "gemini":
            content = generate_gemini_json(prompt, model_id)
        elif provider in {"local", "local_hf", "hf", "transformers"}:
            content = generate_local_hf_json(prompt, model_id)
        else:
            raise RuntimeError(f"지원하지 않는 LLM 제공자입니다: {provider}")
    except Exception:
        logger.exception(
            "preference LLM failed: requested=%s provider=%s model_id=%s",
            config["requested"],
            provider,
            model_id,
        )
        raise
    logger.info(
        "preference LLM completed: requested=%s provider=%s model_id=%s response_length=%s",
        config["requested"],
        provider,
        model_id,
        len(content or ""),
    )
    parsed = extract_json_object(content)
    if not parsed:
        logger.warning(
            "preference LLM returned non-json output: requested=%s provider=%s model_id=%s raw=%s",
            config["requested"],
            provider,
            model_id,
            str(content or "")[:800],
        )
    parsed["_llm_config"] = config
    return parsed


## LLM을 이용해 사용목적과 중요요소가 상품 정보와 의미적으로 맞는지 평가합니다.
def evaluate_preference_fit_with_llm(
    user: Dict[str, Any],
    product: Dict[str, Any],
) -> Dict[str, Any]:
    if not LLM_PREFERENCE_ENABLED:
        return {"used_llm": False, "reason": "disabled"}

    usage_purpose = str(user.get("usage_purpose") or "").strip()
    important_factors = normalize_user_input_list(user.get("important_factors"))
    preferred_brands = normalize_user_input_list(user.get("preferred_brands"))
    if not usage_purpose and not important_factors and not preferred_brands:
        return {"used_llm": False, "reason": "empty_preference_input"}

    product_payload = {
        "상품명": product.get("name"),
        "브랜드": product.get("brand"),
        "카테고리": product.get("category"),
        "정규화카테고리": normalize_category(product.get("category")),
        "상품설명": product.get("description"),
        "가격": product.get("price"),
        "평점": product.get("rating"),
        "리뷰수": product.get("review_count"),
    }
    user_payload = {
        "사용목적": usage_purpose,
        "중요요소": important_factors,
        "선호브랜드": preferred_brands,
        "예산": user.get("budget"),
    }
    prompt = (
        "사용자의 구매조건과 상품정보를 비교해 의미적 적합도를 평가하세요. "
        "가격이나 예산 초과 여부는 판단하지 마세요. "
        "사용목적과 중요요소는 상품명, 브랜드, 카테고리, 상품설명과 얼마나 맞는지 평가하세요. "
        "선호브랜드는 상품 브랜드/상품명이 같은 브랜드, 같은 회사, 공식 영문명/한글명, 제품라인 관계로 볼 수 있는지 판별하세요. "
        "점수와 신뢰도는 0.0부터 1.0 사이 숫자로 작성하고, 반드시 JSON 객체만 반환하세요. "
        "evidence는 짧은 문장 1개만 작성하고 JSON을 반드시 닫으세요.\n\n"
        f"사용자 입력:\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n\n"
        f"상품 정보:\n{json.dumps(product_payload, ensure_ascii=False, indent=2)}\n\n"
        "반환 형식:\n"
        "{\n"
        '  "usage_purpose_fit": 0.0,\n'
        '  "important_factor_fit": 0.0,\n'
        '  "same_brand": false,\n'
        '  "brand_confidence": 0.0,\n'
        '  "brand_fit": 0.0,\n'
        '  "brand_relationship": "same_company|same_brand|product_line|different|unknown",\n'
        '  "evidence": ["근거 1", "근거 2"]\n'
        "}"
    )

    try:
        parsed = generate_llm_json(prompt)
        llm_config = parsed.pop("_llm_config", resolve_llm_model_config())
        if "usage_purpose_fit" not in parsed and "important_factor_fit" not in parsed and "same_brand" not in parsed:
            return {"used_llm": False, "reason": "empty_llm_scores"}
        same_brand = parsed.get("same_brand")
        if isinstance(same_brand, str):
            same_brand = same_brand.strip().lower() in {"true", "yes", "y", "1", "same", "동일"}
        return {
            "used_llm": True,
            "usage_purpose_fit": coerce_llm_unit_score(parsed.get("usage_purpose_fit"), 0.5),
            "important_factor_fit": coerce_llm_unit_score(parsed.get("important_factor_fit"), 0.5),
            "same_brand": bool(same_brand),
            "brand_confidence": coerce_llm_unit_score(parsed.get("brand_confidence"), 0.0),
            "brand_fit": coerce_llm_unit_score(parsed.get("brand_fit"), 0.5),
            "brand_relationship": str(parsed.get("brand_relationship") or "unknown"),
            "evidence": parsed.get("evidence") if isinstance(parsed.get("evidence"), list) else ([str(parsed.get("evidence"))] if parsed.get("evidence") else []),
            "model": llm_config["model_id"],
            "model_name": llm_config["requested"],
            "provider": llm_config["provider"],
        }
    except Exception as exc:
        return {"used_llm": False, "reason": f"llm_error: {exc}"}


## Calculate how well a product matches user brand, factor, and purpose preferences.
def preference_alignment(
    user: Dict[str, Any],
    product: Dict[str, Any],
) -> Dict[str, Any]:
    product_text = product_preference_text(product)
    product_tokens = tokenize_preference_text(product_text)
    brand_tokens = tokenize_preference_text(product.get("brand"), product.get("name"))
    preferred_tokens = tokenize_preference_text(user.get("preferred_brands"))
    factor_tokens = tokenize_preference_text(user.get("important_factors"))
    purpose_tokens = tokenize_preference_text(user.get("usage_purpose"))
    llm_scores = product.get("llm_preference_scores") or {}
    budget_factor_tokens = {"price", "budget", "cheap", "value", "가격", "예산", "가성비", "저렴"}
    condition_tokens = (factor_tokens - budget_factor_tokens) | purpose_tokens

    brand_fit = resolve_brand_fit_score(preferred_tokens, brand_tokens, llm_scores)
    matched_preferred = bool(preferred_tokens) and brand_fit >= 0.8
    condition_matches = matched_preference_tokens(condition_tokens, product_tokens)
    condition_similarity = preference_token_match_score(condition_tokens, product_tokens) if condition_tokens else 0.0

    price = safe_float(product.get("price"))
    budget = safe_float(user.get("budget"))
    rating = safe_float(product.get("rating"), 3.5)
    return_rate = safe_float(product.get("return_rate"), 5.0)

    adjustment = 0.0
    reasons: List[str] = []
    if preferred_tokens:
        if matched_preferred:
            adjustment -= 0.04
            reasons.append("preferred_brand_match")
        else:
            adjustment += 0.02
            reasons.append("preferred_brand_mismatch")

    if condition_similarity >= 0.18:
        adjustment -= min(condition_similarity * 0.08, 0.06)
        reasons.append("condition_text_match")
    elif condition_tokens:
        adjustment += 0.03
        reasons.append("condition_text_mismatch")

    if factor_tokens & budget_factor_tokens:
        if budget > 0 and price > budget:
            over_ratio = (price - budget) / budget
            adjustment += min(0.04 + math.log1p(over_ratio) * 0.035, 0.24)
            reasons.append("price_factor_over_budget")
        elif budget > 0 and price <= budget:
            under_ratio = (budget - price) / max(budget, 1.0)
            adjustment -= min(0.04 + under_ratio * 0.04, 0.08)
            reasons.append("price_factor_in_budget")

    if factor_tokens & {"quality", "rating", "performance", "품질", "평점", "성능", "완성도"}:
        if rating >= 4.3:
            adjustment -= 0.035
            reasons.append("quality_factor_good_rating")
        elif rating < 4.0:
            adjustment += 0.05
            reasons.append("quality_factor_low_rating")

    if factor_tokens & {"stable", "return", "durable", "안정", "내구", "반품"}:
        if return_rate <= 5:
            adjustment -= 0.03
            reasons.append("stability_factor_low_return")
        elif return_rate >= 10:
            adjustment += 0.06
            reasons.append("stability_factor_high_return")

    adjustment = max(-0.14, min(0.24, adjustment))
    alignment_score = max(0.0, min(1.0, 0.5 - adjustment * 2.5))
    return {
        "adjustment": adjustment,
        "alignment_score": alignment_score,
        "condition_similarity": round(condition_similarity, 4),
        "matched_tokens": sorted(condition_matches),
        "matched_preferred_brand": matched_preferred,
        "brand_fit": round(brand_fit, 4),
        "brand_match_source": "llm" if llm_scores.get("used_llm") and safe_float(llm_scores.get("brand_confidence")) >= 0.6 else "rule",
        "reasons": reasons,
    }


def explicit_risk_score(
    user: Dict[str, Any],
    product: Dict[str, Any],
) -> float:
    budget = safe_float(user.get("budget"))
    price = safe_float(product.get("price"))
    price_estimated = bool(product.get("price_estimated"))
    rating = safe_float(product.get("rating"), 3.5)
    rating_missing = bool(product.get("rating_missing"))
    return_rate = safe_float(product.get("return_rate"), 5.0)
    review_count = safe_int(product.get("review_count"))
    review_data_available = bool(product.get("review_data_available"))
    decision_features = purchase_decision_features(user, product)

    score = 0.0
    if budget > 0 and price > budget:
        over_ratio = (price - budget) / budget
        score += min(0.16 + math.log1p(over_ratio) * 0.12, 0.82)
    elif budget > 0 and price > budget * 0.85:
        near_ratio = (price - budget * 0.85) / max(budget * 0.15, 1.0)
        score += 0.06 + near_ratio * 0.08
    elif budget > 0 and price <= budget:
        under_ratio = (budget - price) / max(budget, 1.0)
        score -= min(under_ratio * 0.08, 0.08)

    score += max(0.0, 0.5 - decision_features["사용목적적합도"]) * 0.28
    score += max(0.0, 0.5 - decision_features["중요요소일치도"]) * 0.20
    score += max(0.0, 0.5 - decision_features["브랜드선호일치도"]) * 0.08
    score += decision_features["대체상품우위위험"] * 0.12
    score += decision_features["가격신뢰위험"] * 0.07
    score += decision_features["리뷰부정위험"] * 0.07
    score += max(0.0, 0.5 - decision_features["구매필요성점수"]) * 0.06
    score += max(0.0, 0.55 - decision_features["상품신뢰도"]) * 0.05
    score += max(0.0, 0.5 - decision_features["데이터신뢰도"]) * 0.04
    score += max(0.0, decision_features["월수입대비가격부담율"] - 0.25) * 0.16
    score += max(0.0, decision_features["월수입대비예산부담율"] - 0.12) * 0.08
    score += decision_features["소비성향위험도"] * 0.10
    score += decision_features["충동구매위험도"] * 0.08
    score += decision_features["연령대위험도"] * 0.06
    score += max(0.0, 0.55 - decision_features["직업소득안정성"]) * 0.14
    score += max(0.0, decision_features["가구부담보정값"] - 0.45) * 0.12

    consumption_type = str(user.get("consumption_type") or "").strip()
    job = str(user.get("job") or "").strip()
    marital_status = str(user.get("marital_status") or "").strip()
    age = safe_int(user.get("age"), 40)
    monthly_income = safe_float(user.get("monthly_income"))
    if consumption_type == "충동형" and price >= 500_000:
        score += 0.09
    if consumption_type == "보수적" and budget > 0 and price > budget:
        score += 0.09
    if consumption_type == "가성비형" and rating < 4.2:
        score += 0.06
    if consumption_type == "프리미엄형" and (rating_missing or rating < 4.4):
        score += 0.06
    if job in {"학생", "무직"} and price >= 500_000:
        score += 0.11
    if age <= 24 and price >= 500_000:
        score += 0.05
    elif age >= 70 and price >= 500_000:
        score += 0.04
    if marital_status == "기혼" and monthly_income > 0 and price > monthly_income * 0.35:
        score += 0.04
    if monthly_income > 0 and price > monthly_income * 0.5:
        score += 0.10

    if not rating_missing and rating < 4.2:
        score += min((4.2 - rating) / 2.2, 1.0) * 0.08
    if return_rate > 5:
        score += min((return_rate - 5) / 25.0, 1.0) * 0.22
    if not review_data_available:
        score += 0.025
    elif review_count < 30:
        score += ((30 - max(review_count, 0)) / 30.0) * 0.08

    if price >= 1_000_000 and (return_rate >= 10 or (not rating_missing and rating < 3.8)):
        score += 0.1
    if price_estimated:
        score += 0.18
    return clamp_score(score)


## 딥러닝 예측값과 규칙 기반 위험점수를 결합해 최종 후회점수를 보정합니다.
def calibrate_regret_score(
    model_score: float,
    rule_score: float,
) -> float:
    if rule_score >= 0.65:
        blended = model_score * 0.35 + rule_score * 0.65
    elif rule_score >= 0.4:
        blended = model_score * 0.45 + rule_score * 0.55
    elif model_score >= 0.75:
        blended = model_score * 0.55 + rule_score * 0.45
    else:
        blended = model_score * 0.65 + rule_score * 0.35
    return clamp_score(blended)


## 대체상품 추천에 사용하는 간단한 내장 상품 카탈로그입니다.
@dataclass
class ProductCatalog:

    products: List[Dict[str, Any]]

    ## 추천 테스트와 fallback에 사용할 샘플 상품 목록을 반환합니다.
    @classmethod
    def sample(
        cls,
    ) -> "ProductCatalog":
        return cls([
            {"product_id": 101, "name": "Galaxy S24 FE", "brand": "Samsung", "category": "스마트폰", "price": 699000, "rating": 4.4, "review_count": 1250, "return_rate": 3.2, "days_since_release": 120, "description": "합리적인 가격의 갤럭시 스마트폰"},
            {"product_id": 102, "name": "iPhone 15", "brand": "Apple", "category": "스마트폰", "price": 1250000, "rating": 4.6, "review_count": 3200, "return_rate": 2.1, "days_since_release": 210, "description": "안정적인 사용자 경험의 프리미엄 스마트폰"},
            {"product_id": 103, "name": "Pixel 8a", "brand": "Google", "category": "스마트폰", "price": 649000, "rating": 4.4, "review_count": 890, "return_rate": 2.8, "days_since_release": 180, "description": "카메라와 순정 안드로이드 경험이 강점"},
            {"product_id": 104, "name": "Xiaomi 14T", "brand": "Xiaomi", "category": "스마트폰", "price": 599000, "rating": 4.2, "review_count": 980, "return_rate": 3.4, "days_since_release": 90, "description": "가격 대비 성능이 좋은 스마트폰"},
            {"product_id": 105, "name": "OnePlus 12R", "brand": "OnePlus", "category": "스마트폰", "price": 749000, "rating": 4.3, "review_count": 740, "return_rate": 3.0, "days_since_release": 140, "description": "성능과 배터리가 강점인 스마트폰"},
            {"product_id": 106, "name": "Nothing Phone 2a", "brand": "Nothing", "category": "스마트폰", "price": 499000, "rating": 4.1, "review_count": 620, "return_rate": 3.6, "days_since_release": 110, "description": "디자인과 가격이 강점인 스마트폰"},
            {"product_id": 201, "name": "LG Gram 16", "brand": "LG", "category": "노트북", "price": 1650000, "rating": 4.5, "review_count": 780, "return_rate": 3.5, "days_since_release": 200, "description": "가벼운 고성능 업무용 노트북"},
            {"product_id": 202, "name": "MacBook Air M3", "brand": "Apple", "category": "노트북", "price": 1590000, "rating": 4.8, "review_count": 2100, "return_rate": 1.8, "days_since_release": 150, "description": "긴 배터리와 높은 만족도의 노트북"},
            {"product_id": 203, "name": "Galaxy Book Pro", "brand": "Samsung", "category": "노트북", "price": 1890000, "rating": 4.4, "review_count": 560, "return_rate": 4.1, "days_since_release": 190, "description": "갤럭시 생태계 연동이 좋은 노트북"},
            {"product_id": 204, "name": "Dell XPS 13", "brand": "Dell", "category": "노트북", "price": 1490000, "rating": 4.5, "review_count": 640, "return_rate": 3.1, "days_since_release": 160, "description": "휴대성과 완성도가 좋은 울트라북"},
            {"product_id": 205, "name": "Lenovo Yoga Slim 7", "brand": "Lenovo", "category": "노트북", "price": 1290000, "rating": 4.3, "review_count": 520, "return_rate": 3.8, "days_since_release": 130, "description": "가격과 성능 균형이 좋은 노트북"},
            {"product_id": 206, "name": "ASUS Zenbook 14", "brand": "ASUS", "category": "노트북", "price": 1390000, "rating": 4.4, "review_count": 610, "return_rate": 3.3, "days_since_release": 175, "description": "OLED 디스플레이와 휴대성이 좋은 노트북"},
            {"product_id": 301, "name": "Sony WH-1000XM5", "brand": "Sony", "category": "이어폰/헤드폰", "price": 379000, "rating": 4.7, "review_count": 4500, "return_rate": 2.3, "days_since_release": 365, "description": "노이즈 캔슬링 헤드폰"},
            {"product_id": 302, "name": "AirPods Pro 2", "brand": "Apple", "category": "이어폰/헤드폰", "price": 329000, "rating": 4.6, "review_count": 6700, "return_rate": 2.0, "days_since_release": 400, "description": "애플 기기와 궁합이 좋은 무선 이어폰"},
            {"product_id": 303, "name": "Bose QuietComfort Ultra", "brand": "Bose", "category": "이어폰/헤드폰", "price": 429000, "rating": 4.6, "review_count": 2600, "return_rate": 2.4, "days_since_release": 220, "description": "착용감과 소음 차단이 좋은 헤드폰"},
            {"product_id": 304, "name": "Galaxy Buds3 Pro", "brand": "Samsung", "category": "이어폰/헤드폰", "price": 299000, "rating": 4.4, "review_count": 3100, "return_rate": 2.7, "days_since_release": 120, "description": "갤럭시 기기와 연동성이 좋은 이어폰"},
            {"product_id": 305, "name": "JBL Tour Pro 3", "brand": "JBL", "category": "이어폰/헤드폰", "price": 279000, "rating": 4.3, "review_count": 1800, "return_rate": 3.1, "days_since_release": 100, "description": "기능과 가격 균형이 좋은 무선 이어폰"},
        ])


## xlsx/csv 학습 데이터를 읽고 필수 컬럼이 있는지 검증합니다.

## 원천 학습 데이터 행을 온라인 예측과 같은 방식의 모델 입력 피처로 변환합니다.
def source_row_to_training_row(
    row: Dict[str, Any],
) -> Dict[str, Any]:
    user = {
        "gender": row.get("성별"),
        "age": row.get("나이"),
        "monthly_income": row.get("월수입"),
        "job": row.get("직업"),
        "marital_status": row.get("결혼여부"),
        "consumption_type": row.get("소비성향"),
        "budget": row.get("예산"),
        "purchase_unit_price": row.get("구매단가"),
        "usage_purpose": row.get("사용목적"),
        "important_factors": row.get("중요요소"),
        "preferred_brands": row.get("선호브랜드"),
        "purchase_hour": row.get("구매시간대"),
    }
    product = {
        "name": row.get("상품명") or row.get("name"),
        "brand": row.get("브랜드"),
        "category": row.get("카테고리"),
        "price": row.get("판매단가") if row.get("판매단가") not in (None, "") else row.get("가격"),
        "purchase_unit_price": row.get("구매단가"),
        "target_price": row.get("구매단가"),
        "rating": row.get("평점") if row.get("평점") not in (None, "") else row.get("rating"),
        "review_count": row.get("리뷰수") if row.get("리뷰수") not in (None, "") else row.get("review_count"),
        "description": row.get("상품설명") or row.get("description"),
        "product_info_source": row.get("상품정보출처"),
        "card_installment": row.get("카드할부여부"),
        "price_grade": row.get("가격등급"),
        "market_low_price": row.get("최저가") or row.get("market_low_price") or row.get("lowest_price"),
        "best_alternative_price": row.get("대체상품최저가") or row.get("best_alternative_price"),
        "review_data_available": row.get("평점") not in (None, "") or row.get("리뷰수") not in (None, ""),
    }
    training_row = build_training_row(user, product)
    if row.get("상품조회수") not in (None, ""):
        training_row["상품조회수"] = safe_int(row.get("상품조회수"))
    if row.get("비교상품수") not in (None, ""):
        training_row["비교상품수"] = safe_int(row.get("비교상품수"))
    if row.get("재방문횟수") not in (None, ""):
        training_row["재방문횟수"] = safe_int(row.get("재방문횟수"))
    if row.get("구매시간대") not in (None, ""):
        training_row["구매시간대"] = safe_int(row.get("구매시간대"), 15)
    if row.get("카드할부여부") not in (None, ""):
        training_row["카드할부여부"] = str(row.get("카드할부여부"))
    return training_row


## 원천 컬럼 중심 학습 데이터를 모델 입력 피처로 준비합니다.
def prepare_training_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    df = frame.copy()
    if "판매단가" not in df.columns and "가격" in df.columns:
        df["판매단가"] = df["가격"]
    if "구매단가" not in df.columns:
        df["구매단가"] = df.get("판매단가", df.get("가격", 0))
    if "예산" not in df.columns:
        income = pd.to_numeric(df.get("월수입"), errors="coerce").fillna(3_790_000.0)
        df["예산"] = income * 0.08
    if "평점" not in df.columns and "rating" in df.columns:
        df["평점"] = df["rating"]
    if "리뷰수" not in df.columns and "review_count" in df.columns:
        df["리뷰수"] = df["review_count"]
    if "상품명" not in df.columns:
        df["상품명"] = df.get("name", "분석 대상 상품")
    if "상품설명" not in df.columns:
        df["상품설명"] = df.get("description", "")
    if "상품정보출처" not in df.columns:
        df["상품정보출처"] = "unknown"
    if "사용목적" not in df.columns:
        df["사용목적"] = ""
    if "중요요소" not in df.columns:
        df["중요요소"] = ""
    if "선호브랜드" not in df.columns:
        df["선호브랜드"] = ""

    rows = []
    for _, row in df.iterrows():
        rows.append(source_row_to_training_row(row.to_dict()))
    prepared = pd.DataFrame(rows)
    if TARGET_COLUMN in df.columns:
        prepared[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0.0).astype(float)
    missing = [column for column in TRAINING_COLUMNS if column not in prepared.columns]
    if missing:
        raise ValueError(f"Prepared training frame is missing columns: {missing}")
    columns = TRAINING_COLUMNS + ([TARGET_COLUMN] if TARGET_COLUMN in prepared.columns else [])
    return prepared[columns].copy()


def load_dataset(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = prepare_training_frame(df)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset is missing required target column: {TARGET_COLUMN}")
    return df[TRAINING_COLUMNS + [TARGET_COLUMN]].copy()


## 숫자 표준화 통계와 범주형 원핫 인코딩 사전을 순수 dict로 학습합니다.
def fit_preprocessor(
    frame: pd.DataFrame,
) -> Dict[str, Any]:
    numeric_stats = {}
    for column in MODEL_NUMERIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)
        mean = float(values.mean())
        scale = float(values.std(ddof=0))
        numeric_stats[column] = {
            "mean": mean,
            "scale": scale if scale > 0 else 1.0,
        }

    categorical_values = {}
    for column in CATEGORICAL_COLUMNS:
        values = frame[column].fillna("").astype(str)
        categorical_values[column] = sorted(value for value in values.unique().tolist() if value != "")

    return {
        "type": "manual_v1",
        "numeric_stats": numeric_stats,
        "categorical_values": categorical_values,
        "input_dim": len(MODEL_NUMERIC_COLUMNS) + sum(len(values) for values in categorical_values.values()),
    }


## dict 전처리 정보를 사용해 DataFrame을 신경망 입력 배열로 변환합니다.
def transform_features(
    frame: pd.DataFrame,
    preprocessor: Dict[str, Any],
) -> np.ndarray:
    if preprocessor.get("type") != "manual_v1":
        raise ValueError("Unsupported preprocessor format. Retrain the model.")

    parts = []
    for column in MODEL_NUMERIC_COLUMNS:
        stats = preprocessor["numeric_stats"][column]
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)
        parts.append(((values - stats["mean"]) / stats["scale"]).to_numpy(dtype=np.float32).reshape(-1, 1))

    for column in CATEGORICAL_COLUMNS:
        values = frame[column].fillna("").astype(str)
        categories = preprocessor["categorical_values"][column]
        encoded = np.zeros((len(frame), len(categories)), dtype=np.float32)
        category_index = {value: index for index, value in enumerate(categories)}
        for row_index, value in enumerate(values):
            index = category_index.get(value)
            if index is not None:
                encoded[row_index, index] = 1.0
        parts.append(encoded)

    return np.concatenate(parts, axis=1).astype(np.float32)


## CUDA 사용 가능 여부에 따라 학습/추론 장치를 선택합니다.
def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


## 전처리된 학습/검증 배열로 PyTorch 회귀 모델을 학습합니다.
def train_torch_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    log_interval: int,
    device: torch.device,
) -> tuple[RegretTorchRegressor, int, float]:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    model = RegretTorchRegressor(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0005)
    loss_fn = nn.MSELoss()

    train_x = torch.tensor(x_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32)
    valid_x = torch.tensor(x_valid, dtype=torch.float32, device=device)
    valid_y = torch.tensor(y_valid, dtype=torch.float32, device=device)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = 60
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(train_x.shape[0])
        train_loss_sum = 0.0
        train_count = 0
        for start in range(0, train_x.shape[0], batch_size):
            batch_index = indices[start:start + batch_size]
            batch_x = train_x[batch_index].to(device)
            batch_y = train_y[batch_index].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(batch_index)
            train_count += len(batch_index)

        model.eval()
        with torch.no_grad():
            valid_pred = model(valid_x)
            valid_loss = float(loss_fn(valid_pred, valid_y).item())
            valid_accuracy = float((torch.abs(valid_pred - valid_y) <= 0.05).float().mean().item())
        train_loss = train_loss_sum / max(train_count, 1)
        if valid_loss + 1e-7 < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if log_interval > 0 and (epoch == 1 or epoch % log_interval == 0 or epoch == epochs):
            print(
                f"epoch={epoch}/{epochs} "
                f"loss={train_loss:.6f} "
                f"valid_loss={valid_loss:.6f} "
                f"accuracy={valid_accuracy:.4f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if stale_epochs >= patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_valid_loss={best_loss:.6f}", flush=True)
            break

    if best_state:
        model.load_state_dict(best_state)
    return model.cpu(), best_epoch, best_loss


## 데이터셋을 읽어 전처리, 모델 학습, 평가, artifact 저장까지 수행합니다.
def train_model(
    dataset_path: Path,
    model_path: Path,
    model_type: str = "torch",
    epochs: int = 800,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    log_interval: int = 1,
) -> Dict[str, Any]:
    if model_type != "torch":
        raise ValueError("Only model_type='torch' is supported.")

    df = load_dataset(dataset_path)
    x = df[TRAINING_COLUMNS]
    y = df[TARGET_COLUMN].astype(float).clip(0.0, 1.0).to_numpy(dtype=np.float32)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    preprocessor = fit_preprocessor(x_train)
    x_train_array = transform_features(x_train, preprocessor)
    x_test_array = transform_features(x_test, preprocessor)
    device = select_device()
    model, best_epoch, best_loss = train_torch_regressor(
        x_train_array,
        y_train,
        x_test_array,
        y_test,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        log_interval=log_interval,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x_test_array, dtype=torch.float32)).numpy()
    pred = np.clip(pred, 0.0, 1.0)
    metrics = {
        "rows": int(len(df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "mae": round(float(mean_absolute_error(y_test, pred)), 6),
        "r2": round(float(r2_score(y_test, pred)), 6),
        "target_mean": round(float(y.mean()), 6),
        "model_type": "torch_regression",
        "framework": "pytorch",
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_valid_loss": round(float(best_loss), 8),
        "input_dim": int(x_train_array.shape[1]),
        "features": TRAINING_COLUMNS,
    }

    artifact = {
        "model_state": model.state_dict(),
        "input_dim": int(x_train_array.shape[1]),
        "preprocessor": preprocessor,
        "metrics": metrics,
        "training_columns": TRAINING_COLUMNS,
        "target_column": TARGET_COLUMN,
        "model_type": "torch_regression",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    return metrics


## 저장된 후회예측 모델 artifact를 로드하고 단건 예측을 제공합니다.
class PurchaseRegretModel:

    ## 모델 경로와 데이터셋 경로를 받아 모델을 로드하거나 필요 시 학습합니다.
    def __init__(
        self,
        model_path: Path,
        dataset_path: Optional[Path] = None,
    ):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.artifact = self._load_or_train()
        self.model = RegretTorchRegressor(int(self.artifact["input_dim"]))
        self.model.load_state_dict(self.artifact["model_state"])
        self.model.eval()

    ## 유효한 모델 artifact를 로드하고, 없거나 구버전이면 다시 학습합니다.
    def _load_or_train(
        self,
    ) -> Dict[str, Any]:
        if self.model_path.exists():
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", InconsistentVersionWarning)
                loaded = joblib.load(self.model_path)
            has_version_warning = any(issubclass(item.category, InconsistentVersionWarning) for item in caught_warnings)
            has_manual_preprocessor = isinstance(loaded, dict) and isinstance(loaded.get("preprocessor"), dict) and loaded["preprocessor"].get("type") == "manual_v1"
            preprocessor = loaded.get("preprocessor") if isinstance(loaded, dict) else {}
            has_required_features = (
                has_manual_preprocessor
                and all(column in preprocessor.get("numeric_stats", {}) for column in MODEL_NUMERIC_COLUMNS)
                and all(column in preprocessor.get("categorical_values", {}) for column in CATEGORICAL_COLUMNS)
                and int(loaded.get("input_dim", 0)) == int(preprocessor.get("input_dim", -1))
            )
            if (
                isinstance(loaded, dict)
                and "model_state" in loaded
                and "preprocessor" in loaded
                and has_required_features
                and not has_version_warning
            ):
                return loaded
        dataset_path = self.dataset_path or DEFAULT_DATASET_PATH
        train_model(dataset_path, self.model_path, log_interval=0)
        loaded = joblib.load(self.model_path)
        if not isinstance(loaded, dict) or "model_state" not in loaded:
            raise ValueError(f"Invalid model artifact: {self.model_path}")
        return loaded

    ## 학습 피처 한 행을 받아 구매후회 점수를 예측합니다.
    def predict_row(
        self,
        row: Dict[str, Any],
    ) -> float:
        frame = pd.DataFrame([{column: row.get(column) for column in TRAINING_COLUMNS}])
        transformed = transform_features(frame, self.artifact["preprocessor"])
        with torch.no_grad():
            score = self.model(torch.tensor(transformed, dtype=torch.float32))[0].item()
        return clamp_score(score)

    ## 모델 artifact에 저장된 학습/평가 지표를 반환합니다.
    @property
    def metrics(
        self,
    ) -> Dict[str, Any]:
        return self.artifact.get("metrics", {})


## 온라인 요청을 받아 후회예측, 원인 분석, 대체상품 추천을 수행합니다.
class RegretPredictor:

    ## 예측 모델, 데이터셋, 추천 임계값을 초기화합니다.
    def __init__(
        self,
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        threshold: float = 0.4,
        options_path: Optional[str] = None,
    ):
        self.options_path = Path(
            options_path or os.getenv("AGENT_OPTIONS_PATH") or DEFAULT_OPTIONS_PATH
        )
        self.options = load_agent_options(self.options_path)
        self.threshold = normalize_option_threshold(
            self.options.get("alternative_regret_score_threshold"),
            threshold,
        )
        self.max_alternative_products = normalize_option_count(
            self.options.get("max_alternative_products"),
            DEFAULT_AGENT_OPTIONS["max_alternative_products"],
        )
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.dataset_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
        self.model = PurchaseRegretModel(self.model_path, self.dataset_path)
        self.catalog = ProductCatalog.sample()

    ## 사용자/상품 입력으로 모델 점수, 규칙 점수, 최종 점수와 원인을 계산합니다.
    def _predict_score(
        self,
        user: Dict[str, Any],
        product: Dict[str, Any],
        use_llm_preference: bool = True,
    ) -> Dict[str, Any]:
        scoring_product = dict(product)
        llm_preference = (
            evaluate_preference_fit_with_llm(user, scoring_product)
            if use_llm_preference
            else {"used_llm": False, "reason": "skipped"}
        )
        if llm_preference.get("used_llm"):
            scoring_product["llm_preference_scores"] = llm_preference

        row = build_training_row(user, scoring_product)
        model_score = self.model.predict_row(row)
        rule_score = explicit_risk_score(user, scoring_product)
        base_regret_score = calibrate_regret_score(model_score, rule_score)
        alignment = preference_alignment(user, scoring_product)
        preference_adjustment = alignment["adjustment"]
        if rule_score >= 0.65 and preference_adjustment < 0:
            preference_adjustment = max(preference_adjustment, -0.04)
        elif rule_score >= 0.4 and preference_adjustment < 0:
            preference_adjustment = max(preference_adjustment, -0.05)
        regret_score = clamp_score(base_regret_score + preference_adjustment)
        causes = make_regret_causes(regret_score, user, scoring_product, alignment)
        return {
            "feature": row,
            "model_regret_score": model_score,
            "cause_score": rule_score,
            "base_regret_score": base_regret_score,
            "preference_adjustment": preference_adjustment,
            "preference_alignment": alignment,
            "llm_preference_evaluation": llm_preference,
            "decision_features": purchase_decision_features(user, scoring_product),
            "regret_score": regret_score,
            "regret_causes": causes,
        }

    ## 단일 상품에 대한 최종 후회예측 응답 payload를 생성합니다.
    def predict(
        self,
        user: Dict[str, Any],
        product: Dict[str, Any],
        alternative_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        raw_price = safe_float(product.get("price"))
        price_estimated = raw_price <= 0
        normalized_price = estimate_missing_price(product) if price_estimated else raw_price
        display_price = None if price_estimated else raw_price
        estimated_price = normalized_price if price_estimated else None
        rating_missing = product.get("rating") in (None, "")
        review_count_missing = product.get("review_count") in (None, "")
        review_data_available = bool(product.get("review_data_available"))
        normalized_product = {
            "name": product.get("name") or "분석 대상 상품",
            "brand": product.get("brand"),
            "category": product.get("category"),            "price": normalized_price,
            "display_price": display_price,
            "estimated_price": estimated_price,
            "price_estimated": price_estimated,
            "price_missing": bool(product.get("price_missing")) or price_estimated,
            "rating": safe_float(product.get("rating"), 3.5),
            "rating_missing": rating_missing,
            "review_count": safe_int(product.get("review_count")),
            "review_count_missing": review_count_missing,
            "review_data_available": review_data_available,
            "review_texts": product.get("review_texts") or [],
            "review_source": product.get("review_source"),
            "return_rate": safe_float(product.get("return_rate"), 5.0),
            "days_since_release": safe_int(product.get("days_since_release"), 180),
            "description": product.get("description"),
            "image_url": product.get("image_url"),
            "source_url": product.get("source_url"),
            "product_url": product.get("product_url") or product.get("source_url"),
            "shop": product.get("shop"),
            "product_info_source": product.get("product_info_source"),
            "product_info_missing": bool(product.get("product_info_missing")),
            "search_query": product.get("search_query"),
        }
        score_result = self._predict_score(user, normalized_product)
        regret_score = score_result["regret_score"]
        alternatives = []
        if regret_score >= self.threshold:
            alternatives = self.recommend_alternatives(
                user,
                normalized_product,
                regret_score,
                alternative_candidates=alternative_candidates,
            )

        return {
            "product": normalized_product,
            "product_name": normalized_product["name"],
            "regret_score": round(regret_score, 4),
            "regret_level": regret_level(regret_score),
            "model_regret_score": round(score_result["model_regret_score"], 4),
            "cause_score": round(score_result["cause_score"], 4),
            "base_regret_score": round(score_result["base_regret_score"], 4),
            "preference_adjustment": round(score_result["preference_adjustment"], 4),
            "preference_alignment": score_result["preference_alignment"],
            "llm_preference_evaluation": score_result["llm_preference_evaluation"],
            "decision_features": score_result["decision_features"],
            "threshold": self.threshold,
            "max_alternative_products": self.max_alternative_products,
            "agent_options": self.options,
            "should_reconsider": regret_score >= self.threshold,
            "regret_causes": score_result["regret_causes"],
            "regret_reasons": [cause["message"] for cause in score_result["regret_causes"]],
            "alternatives": alternatives,
            "llm_analysis": self._llm_analysis(
                normalized_product,
                regret_score,
                score_result["regret_causes"],
                alternatives,
            ),
            "model_metrics": self.model.metrics,
            "prediction_features": score_result["feature"],
            "summary": self._summary(normalized_product, regret_score, alternatives),
        }

    ## 현재 상품보다 후회점수가 낮은 대체상품 후보를 최대 5개 추천합니다.
    def recommend_alternatives(
        self,
        user: Dict[str, Any],
        product: Dict[str, Any],
        target_score: float,
        alternative_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        target_category = normalize_category(product.get("category"))
        target_price = safe_float(product.get("price"))
        budget = safe_float(user.get("budget"))
        use_external_candidates = alternative_candidates is not None
        external_candidates = [
            self._normalize_alternative_candidate(candidate)
            for candidate in (alternative_candidates or [])
            if isinstance(candidate, dict)
        ]
        external_candidates = [candidate for candidate in external_candidates if candidate.get("name")]
        catalog_candidates = self.catalog.products
        candidates = external_candidates if use_external_candidates else catalog_candidates
        if target_category:
            same_category = [item for item in candidates if normalize_category(item.get("category")) == target_category]
            if len(same_category) >= self.max_alternative_products:
                candidates = same_category

        results = []
        seen_keys = set()
        for candidate in candidates:
            dedupe_key = str(candidate.get("source_url") or candidate.get("product_url") or candidate.get("product_id") or candidate.get("name") or "").lower()
            if dedupe_key and dedupe_key in seen_keys:
                continue
            if dedupe_key:
                seen_keys.add(dedupe_key)
            if candidate.get("name") == product.get("name"):
                continue
            base_score = self._predict_score(user, candidate, use_llm_preference=False)["regret_score"]
            condition_adjustment = self._alternative_condition_adjustment(user, candidate, product)
            score = clamp_score(base_score + condition_adjustment["adjustment"])
            if score > target_score:
                continue
            match_score = self._match_score(user, candidate, target_price)
            improvement = max(target_score - score, 0)
            price_score = self._price_advantage(candidate, target_price, budget)
            alignment = preference_alignment(user, candidate)
            preference_score = safe_float(alignment.get("alignment_score"), 0.5)
            source_bonus = 0.12 if candidate.get("alternative_source") == "naver_shopping" else 0.0
            relevance_bonus = min(safe_float(candidate.get("alternative_relevance_score")) * 0.25, 0.10)
            final_score = match_score * 0.40 + improvement * 0.25 + price_score * 0.10 + preference_score * 0.13 + source_bonus + relevance_bonus
            results.append({
                "product_id": candidate.get("product_id"),
                "name": candidate.get("name"),
                "brand": candidate.get("brand"),
                "category": candidate.get("category"),
                "price": candidate.get("price"),
                "rating": candidate.get("rating"),
                "review_count": candidate.get("review_count"),
                "return_rate": candidate.get("return_rate"),
                "image_url": candidate.get("image_url") or PRODUCT_IMAGE_URLS.get(str(candidate.get("name", "")).lower()),
                "source_url": candidate.get("source_url"),
                "product_url": candidate.get("product_url"),
                "mall_name": candidate.get("mall_name"),
                "alternative_source": candidate.get("alternative_source") or "sample_catalog",
                "alternative_search_query": candidate.get("alternative_search_query"),
                "alternative_relevance_score": candidate.get("alternative_relevance_score"),
                "regret_score": round(score, 4),
                "base_regret_score": round(base_score, 4),
                "alternative_condition_adjustment": condition_adjustment,
                "match_score": round(match_score, 4),
                "improvement_score": round(improvement, 4),
                "final_score": round(final_score, 4),
                "preference_alignment": alignment,
                "recommendation_reason": self._recommendation_reason(user, candidate, product, score, target_score, condition_adjustment),
            })
        results.sort(key=lambda item: (item["regret_score"], -item["final_score"], item["price"]))
        return results[:self.max_alternative_products]

    ## 네이버 쇼핑 후보를 구매후회예측과 화면 출력에 맞는 대체상품 구조로 정규화합니다.
    def _normalize_alternative_candidate(
        self,
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        price = safe_float(candidate.get("price"))
        rating_missing = candidate.get("rating") in (None, "")
        review_count_missing = candidate.get("review_count") in (None, "")
        return {
            "product_id": candidate.get("product_id"),
            "name": candidate.get("name"),
            "brand": candidate.get("brand"),
            "category": candidate.get("category"),
            "price": price,
            "rating": safe_float(candidate.get("rating"), 3.5),
            "rating_missing": rating_missing,
            "review_count": safe_int(candidate.get("review_count")),
            "review_count_missing": review_count_missing,
            "review_data_available": bool(candidate.get("review_data_available")),
            "review_texts": candidate.get("review_texts") or [],
            "review_source": candidate.get("review_source"),
            "return_rate": safe_float(candidate.get("return_rate"), 5.0),
            "days_since_release": safe_int(candidate.get("days_since_release"), 180),
            "description": candidate.get("description"),
            "image_url": candidate.get("image_url"),
            "source_url": candidate.get("source_url"),
            "product_url": candidate.get("product_url"),
            "mall_name": candidate.get("mall_name"),
            "alternative_source": candidate.get("alternative_source"),
            "alternative_search_query": candidate.get("alternative_search_query"),
            "alternative_relevance_score": candidate.get("alternative_relevance_score"),
            "price_estimated": price <= 0,
        }

    ## 예산, 선호 브랜드, 평점, 반품률, 가격 이점을 종합한 추천 적합도를 계산합니다.
    def _match_score(
        self,
        user: Dict[str, Any],
        product: Dict[str, Any],
        target_price: float,
    ) -> float:
        score = 0.0
        price = safe_float(product.get("price"))
        budget = safe_float(user.get("budget"))
        rating = safe_float(product.get("rating"))
        return_rate = safe_float(product.get("return_rate"))
        if budget > 0 and price <= budget:
            score += 0.3
        if product.get("brand") in (user.get("preferred_brands") or []):
            score += 0.2
        if rating >= 4.3:
            score += 0.2
        elif rating >= 4.0:
            score += 0.1
        if return_rate <= 5:
            score += 0.15
        elif return_rate <= 10:
            score += 0.08
        if target_price > 0 and price < target_price:
            score += 0.15
        return min(score, 1.0)

    ## 대체상품이 예산과 현재 상품 가격 대비 얼마나 유리한지 계산합니다.
    def _price_advantage(
        self,
        product: Dict[str, Any],
        target_price: float,
        budget: float,
    ) -> float:
        price = safe_float(product.get("price"))
        score = 0.0
        if budget > 0 and price <= budget:
            score += 0.5
        if target_price > 0 and price < target_price:
            score += min((target_price - price) / target_price, 0.5)
        return min(score, 1.0)

    ## 대체상품 후보가 사용자의 예산, 선호브랜드, 중요요소, 사용목적에 맞는지 별도 보정값으로 계산합니다.
    def _alternative_condition_adjustment(
        self,
        user: Dict[str, Any],
        candidate: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Dict[str, Any]:
        price = safe_float(candidate.get("price"))
        budget = safe_float(user.get("budget"))
        target_price = safe_float(target.get("price"))
        preferred_tokens = tokenize_preference_text(user.get("preferred_brands"))
        brand_tokens = tokenize_preference_text(candidate.get("brand"), candidate.get("name"))
        factor_tokens = tokenize_preference_text(user.get("important_factors"))
        purpose_tokens = tokenize_preference_text(user.get("usage_purpose"))
        budget_tokens = {"price", "budget", "cheap", "value", "가격", "예산", "가성비", "저렴"}

        adjustment = 0.0
        reasons: List[str] = []

        if budget > 0 and price > 0:
            if price > budget:
                over_ratio = (price - budget) / max(budget, 1.0)
                penalty = min(0.08 + math.log1p(over_ratio) * 0.11, 0.34)
                if factor_tokens & budget_tokens:
                    penalty += min(0.04 + over_ratio * 0.04, 0.12)
                adjustment += penalty
                reasons.append("예산 초과")
            else:
                under_ratio = (budget - price) / max(budget, 1.0)
                bonus = min(0.05 + under_ratio * 0.10, 0.16)
                if factor_tokens & budget_tokens:
                    bonus += min(0.03 + under_ratio * 0.05, 0.08)
                adjustment -= bonus
                reasons.append("예산 적합")

        if target_price > 0 and price > 0:
            if price < target_price:
                cheaper_ratio = (target_price - price) / max(target_price, 1.0)
                adjustment -= min(cheaper_ratio * 0.08, 0.08)
                reasons.append("대상상품보다 저렴")
            elif price > target_price:
                expensive_ratio = (price - target_price) / max(target_price, 1.0)
                adjustment += min(expensive_ratio * 0.06, 0.10)
                reasons.append("대상상품보다 비쌈")

        brand_fit = resolve_brand_fit_score(preferred_tokens, brand_tokens, candidate.get("llm_preference_scores"))
        if preferred_tokens:
            if brand_fit >= 0.8:
                adjustment -= 0.10
                reasons.append("선호브랜드 일치")
            elif brand_fit >= 0.45:
                adjustment -= 0.03
                reasons.append("선호브랜드 부분 일치")
            else:
                adjustment += 0.07
                reasons.append("선호브랜드 불일치")

        important_fit = preference_similarity_score(user.get("important_factors"), candidate)
        if factor_tokens:
            if important_fit >= 0.35:
                adjustment -= min(0.04 + important_fit * 0.10, 0.13)
                reasons.append("중요요소 적합")
            elif important_fit < 0.12:
                adjustment += 0.06
                reasons.append("중요요소 부족")

        purpose_fit = preference_similarity_score(user.get("usage_purpose"), candidate)
        if purpose_tokens:
            if purpose_fit >= 0.30:
                adjustment -= min(0.03 + purpose_fit * 0.09, 0.11)
                reasons.append("사용목적 적합")
            elif purpose_fit < 0.12:
                adjustment += 0.05
                reasons.append("사용목적 부족")

        relevance_score = clamp_unit(candidate.get("alternative_relevance_score"))
        if relevance_score >= 0.6:
            adjustment -= min(relevance_score * 0.04, 0.04)
            reasons.append("검색 관련도 높음")

        adjustment = max(-0.35, min(0.42, adjustment))
        return {
            "adjustment": round(adjustment, 4),
            "budget": budget,
            "price": price,
            "brand_fit": round(brand_fit, 4),
            "important_factor_fit": round(important_fit, 4),
            "usage_purpose_fit": round(purpose_fit, 4),
            "reasons": reasons,
        }

    ## 대체상품 추천 사유를 사용자에게 보여줄 문장으로 조합합니다.
    def _recommendation_reason(
        self,
        user: Dict[str, Any],
        candidate: Dict[str, Any],
        target: Dict[str, Any],
        score: float,
        target_score: float,
        condition_adjustment: Optional[Dict[str, Any]] = None,
    ) -> str:
        reasons = []
        if safe_float(user.get("budget")) and safe_float(candidate.get("price")) <= safe_float(user.get("budget")):
            reasons.append("예산 범위에 맞습니다")
        if safe_float(target.get("price")) and safe_float(candidate.get("price")) < safe_float(target.get("price")):
            reasons.append("현재 후보보다 가격 부담이 적습니다")
        if safe_float(candidate.get("rating")) >= 4.3:
            reasons.append("사용자 평점이 높습니다")
        if safe_float(candidate.get("return_rate")) <= 5:
            reasons.append("반품율이 낮습니다")
        if score < target_score:
            reasons.append("예측 후회 점수가 더 낮습니다")
        if candidate.get("brand") in (user.get("preferred_brands") or []):
            reasons.append("선호 브랜드와 일치합니다")
        for reason in (condition_adjustment or {}).get("reasons") or []:
            if reason in {"중요요소 적합", "사용목적 적합", "선호브랜드 일치"}:
                reasons.append(reason)
        return ", ".join(reasons) if reasons else "현재 상품보다 종합 위험도가 낮은 대체 후보입니다"

    ## Build the analysis object consumed by the frontend AI analysis tab.
    def _llm_analysis(
        self,
        product: Dict[str, Any],
        regret_score: float,
        causes: List[Dict[str, Any]],
        alternatives: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        percent = round(regret_score * 100)
        major_causes = [cause for cause in causes if cause.get("code") != "NO_MAJOR_RISK"]
        top_cause = major_causes[0] if major_causes else None
        top_alternative = alternatives[0] if alternatives else None
        product_name = product.get("name") or "분석 대상 상품"

        if regret_score >= 0.7:
            tone = "구매 전 재검토가 강하게 필요한 상태입니다"
        elif regret_score >= 0.4:
            tone = "몇 가지 후회 요인이 있어 신중한 비교가 필요한 상태입니다"
        else:
            tone = "현재 입력 조건에서는 비교적 부담이 낮은 상태입니다"

        risk_explanation = (
            f"가장 큰 위험 요인은 '{top_cause.get('title')}'입니다. {top_cause.get('message')}"
            if top_cause
            else "현재 입력 조건에서는 뚜렷한 고위험 요인이 발견되지 않았습니다."
        )
        purchase_advice = (
            "예산, 리뷰 수, 평점, 반품률을 다시 확인한 뒤 구매를 결정하는 것이 좋습니다. "
            "특히 가격이 추정된 경우 실제 판매가를 확인해 점수를 다시 계산해보세요."
        )
        alternative_strategy = (
            f"대체상품 중 '{top_alternative.get('name')}'을 먼저 비교해보세요. "
            f"예측 후회 점수는 {round(safe_float(top_alternative.get('regret_score')) * 100)}점입니다."
            if top_alternative
            else "현재 조건에서는 추천할 대체상품이 충분하지 않습니다. 예산이나 선호 브랜드를 입력하면 추천 품질이 좋아집니다."
        )

        return {
            "used_llm": False,
            "summary": f"{product_name}의 구매 후회 가능성은 {percent}점으로 평가됩니다. {tone}.",
            "risk_explanation": risk_explanation,
            "purchase_advice": purchase_advice,
            "alternative_strategy": alternative_strategy,
        }

    def _summary(
        self,
        product: Dict[str, Any],
        regret_score: float,
        alternatives: List[Dict[str, Any]],
    ) -> str:
        percent = round(regret_score * 100)
        if alternatives:
            return f"{product.get('name')}의 구매 후회 가능성은 {percent}%입니다. 대체상품 {len(alternatives)}개를 함께 검토하는 것이 좋습니다."
        return f"{product.get('name')}의 구매 후회 가능성은 {percent}%입니다. 현재 조건에서는 별도 대체상품 추천이 필요하지 않습니다."


## CLI predict 입력을 JSON payload 또는 테이블 파일 지시자로 로드합니다.
def load_prediction_payload(
    input_json: Optional[str],
    input_file: Optional[str],
) -> Dict[str, Any]:
    if input_file:
        path = Path(input_file)
        if path.suffix.lower() in {".xlsx", ".xls", ".csv"}:
            return {"__input_kind__": "table", "__input_file__": str(path)}
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    if input_json:
        return json.loads(input_json)
    raise ValueError("Either --input-json or --input-file is required for predict.")


## xlsx/csv 파일 전체에 대해 행별 후회예측과 평가 지표를 생성합니다.
def predict_table_file(
    model_path: Path,
    dataset_path: Path,
    input_file: Path,
    limit: int,
) -> Dict[str, Any]:
    df = load_dataset(input_file)
    model = PurchaseRegretModel(model_path, dataset_path)
    predictions = []
    actual_values = []
    predicted_values = []

    for index, row in df.iterrows():
        row_dict = {column: row.get(column) for column in TRAINING_COLUMNS}
        predicted = model.predict_row(row_dict)
        item = {
            "row": int(index) + 1,
            "predicted_regret_score": round(predicted, 4),
        }
        if TARGET_COLUMN in row:
            actual = clamp_score(row.get(TARGET_COLUMN))
            item["actual_regret_score"] = round(actual, 4)
            item["absolute_error"] = round(abs(predicted - actual), 4)
            actual_values.append(actual)
            predicted_values.append(predicted)
        if len(predictions) < limit:
            predictions.append(item)

    result = {
        "input_file": str(input_file),
        "rows": int(len(df)),
        "returned_rows": int(len(predictions)),
        "predictions": predictions,
        "model_metrics": model.metrics,
    }
    if actual_values:
        actual_array = np.array(actual_values, dtype=np.float32)
        predicted_array = np.array(predicted_values, dtype=np.float32)
        result["evaluation"] = {
            "mae": round(float(mean_absolute_error(actual_array, predicted_array)), 6),
            "r2": round(float(r2_score(actual_array, predicted_array)), 6),
            "accuracy": round(float((np.abs(predicted_array - actual_array) <= 0.05).mean()), 6),
            "accuracy_tolerance": 0.05,
        }
    return result


## CLI 인자를 해석해 학습(train) 또는 예측(predict)을 실행합니다.
def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run the purchase regret prediction model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model from purchase_regret_training_dataset_source.xlsx")
    train_parser.add_argument("--data", default=str(DEFAULT_DATASET_PATH), help="Path to xlsx/csv training dataset.")
    train_parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Output model path.")
    train_parser.add_argument("--model-type", choices=["torch"], default="torch", help="Estimator type.")
    train_parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs.")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    train_parser.add_argument("--learning-rate", type=float, default=0.001, help="AdamW learning rate.")
    train_parser.add_argument("--log-interval", type=int, default=1, help="Print training metrics every N epochs.")

    predict_parser = subparsers.add_parser("predict", help="Predict purchase regret from a JSON, xlsx, or csv payload.")
    predict_parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Model path.")
    predict_parser.add_argument("--data", default=str(DEFAULT_DATASET_PATH), help="Dataset path used if model is missing.")
    predict_parser.add_argument("--input-json", help="JSON string containing user and product.")
    predict_parser.add_argument("--input-file", help="JSON, xlsx, or csv file containing prediction input.")
    predict_parser.add_argument("--threshold", type=float, default=0.4, help="Alternative recommendation threshold.")
    predict_parser.add_argument("--limit", type=int, default=20, help="Maximum rows to print for xlsx/csv prediction.")

    args = parser.parse_args()
    if args.command == "train":
        metrics = train_model(
            Path(args.data),
            Path(args.model),
            args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            log_interval=args.log_interval,
        )
        print(json.dumps({"model": args.model, "metrics": metrics}, ensure_ascii=False, indent=2))
    elif args.command == "predict":
        payload = load_prediction_payload(args.input_json, args.input_file)
        if payload.get("__input_kind__") == "table":
            result = predict_table_file(
                Path(args.model),
                Path(args.data),
                Path(payload["__input_file__"]),
                max(args.limit, 0),
            )
        else:
            predictor = RegretPredictor(model_path=args.model, dataset_path=args.data, threshold=args.threshold)
            result = predictor.predict(user=payload.get("user") or {}, product=payload.get("product") or payload)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

