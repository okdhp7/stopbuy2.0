import argparse
import json
import math
import os
import random
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


TARGET_COLUMN = "구매후회값"
NUMERIC_COLUMNS = [
    "나이",
    "월수입",
    "가격",
    "상품조회수",
    "비교상품수",
    "구매시간대",
    "재방문횟수",
]
CATEGORICAL_COLUMNS = [
    "성별",
    "직업",
    "결혼여부",
    "소비성향",
    "가격등급",
    "카테고리",
    "브랜드",
    "카드할부여부",
]
TRAINING_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "regret_model.pkl"
DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "datas" / "purchase_regret_training_dataset_1500.xlsx"

PRODUCT_IMAGE_URLS = {
    "galaxy s24 fe": "https://fdn2.gsmarena.com/vv/bigpic/samsung-galaxy-s24-fe.jpg",
    "iphone 15": "https://fdn2.gsmarena.com/vv/pics/apple/apple-iphone-15-1.jpg",
    "pixel 8a": "https://fdn2.gsmarena.com/vv/bigpic/google-pixel-8a.jpg",
    "xiaomi 14t": "https://fdn2.gsmarena.com/vv/bigpic/xiaomi-14t.jpg",
    "oneplus 12r": "https://fdn2.gsmarena.com/vv/bigpic/oneplus-12r.jpg",
    "nothing phone 2a": "https://fdn2.gsmarena.com/vv/bigpic/nothing-phone-2a.jpg",
}


## 구매후회값을 0~1 범위로 예측하는 PyTorch 회귀 신경망입니다.
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


## 입력 카테고리 문자열을 학습 데이터와 가까운 표준 카테고리로 정규화합니다.
def normalize_category(
    category: Optional[str],
) -> str:
    text = str(category or "").lower()
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
    }
    for keyword, value in brand_map.items():
        if keyword in source:
            return value
    return brand or "기타"


## 온라인 입력의 사용자/상품 정보를 모델 학습 피처 한 행으로 변환합니다.
def build_training_row(
    user: Dict[str, Any],
    product: Dict[str, Any],
) -> Dict[str, Any]:
    price = safe_float(product.get("price"))
    review_count = safe_int(product.get("review_count"), 20)
    return_rate = safe_float(product.get("return_rate"), 5.0)
    rating = safe_float(product.get("rating"), 3.5)
    budget = safe_float(user.get("budget"))
    category = normalize_category(product.get("category"))
    brand = normalize_brand(product.get("brand"), product.get("name"))

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
        "가격": price,
        "가격등급": product.get("price_grade") or price_grade(price),
        "카테고리": category,
        "브랜드": brand,
        "상품조회수": views,
        "비교상품수": comparison_count,
        "구매시간대": safe_int(user.get("purchase_hour"), 15),
        "카드할부여부": product.get("card_installment") or installment,
        "재방문횟수": revisit_count,
    }


## 예산, 평점, 반품률, 리뷰 수 등을 바탕으로 후회 원인 목록을 생성합니다.
def make_regret_causes(
    model_score: float,
    user: Dict[str, Any],
    product: Dict[str, Any],
) -> List[Dict[str, Any]]:
    causes: List[Dict[str, Any]] = []
    budget = safe_float(user.get("budget"))
    price = safe_float(product.get("price"))
    rating = safe_float(product.get("rating"), 3.5)
    return_rate = safe_float(product.get("return_rate"), 5.0)
    review_count = safe_int(product.get("review_count"))

    if budget > 0 and price > budget:
        over_ratio = (price - budget) / budget
        causes.append({
            "code": "PRICE_OVER_BUDGET",
            "title": "예산 초과",
            "message": f"상품 가격이 예산보다 {over_ratio * 100:.1f}% 높아 구매 후 부담을 느낄 가능성이 있습니다.",
            "severity": "high" if over_ratio >= 0.3 else "medium",
            "impact_score": round(min(over_ratio, 1.0), 4),
        })
    if rating < 3.8:
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
    if review_count < 20:
        causes.append({
            "code": "LOW_REVIEW_COUNT",
            "title": "검증 부족",
            "message": "리뷰 수가 적어 신뢰도와 만족도에 대한 검증이 충분하지 않습니다.",
            "severity": "medium",
            "impact_score": 0.35,
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
def explicit_risk_score(
    user: Dict[str, Any],
    product: Dict[str, Any],
) -> float:
    budget = safe_float(user.get("budget"))
    price = safe_float(product.get("price"))
    rating = safe_float(product.get("rating"), 3.5)
    return_rate = safe_float(product.get("return_rate"), 5.0)
    review_count = safe_int(product.get("review_count"))

    score = 0.0
    if budget > 0 and price > budget:
        over_ratio = (price - budget) / budget
        score += min(over_ratio / 2.0, 1.0) * 0.42
    elif budget > 0 and price > budget * 0.85:
        score += 0.12

    if rating < 4.2:
        score += min((4.2 - rating) / 2.2, 1.0) * 0.2
    if return_rate > 5:
        score += min((return_rate - 5) / 25.0, 1.0) * 0.22
    if review_count < 30:
        score += ((30 - max(review_count, 0)) / 30.0) * 0.16

    if price >= 1_000_000 and (return_rate >= 10 or rating < 3.8):
        score += 0.1
    return clamp_score(score)


## 딥러닝 예측값과 규칙 기반 위험점수를 결합해 최종 후회점수를 보정합니다.
def calibrate_regret_score(
    model_score: float,
    rule_score: float,
) -> float:
    if rule_score >= 0.65:
        blended = model_score * 0.35 + rule_score * 0.65
    elif rule_score >= 0.4:
        blended = model_score * 0.5 + rule_score * 0.5
    else:
        blended = model_score * 0.75 + rule_score * 0.25
    return clamp_score(max(model_score, blended))


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
def load_dataset(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    missing = [column for column in TRAINING_COLUMNS + [TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df[TRAINING_COLUMNS + [TARGET_COLUMN]].copy()


## 숫자 표준화 통계와 범주형 원핫 인코딩 사전을 순수 dict로 학습합니다.
def fit_preprocessor(
    frame: pd.DataFrame,
) -> Dict[str, Any]:
    numeric_stats = {}
    for column in NUMERIC_COLUMNS:
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
        "input_dim": len(NUMERIC_COLUMNS) + sum(len(values) for values in categorical_values.values()),
    }


## dict 전처리 정보를 사용해 DataFrame을 신경망 입력 배열로 변환합니다.
def transform_features(
    frame: pd.DataFrame,
    preprocessor: Dict[str, Any],
) -> np.ndarray:
    if preprocessor.get("type") != "manual_v1":
        raise ValueError("Unsupported preprocessor format. Retrain the model.")

    parts = []
    for column in NUMERIC_COLUMNS:
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
            if (
                isinstance(loaded, dict)
                and "model_state" in loaded
                and "preprocessor" in loaded
                and has_manual_preprocessor
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
    ):
        self.threshold = threshold
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.dataset_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
        self.model = PurchaseRegretModel(self.model_path, self.dataset_path)
        self.catalog = ProductCatalog.sample()

    ## 사용자/상품 입력으로 모델 점수, 규칙 점수, 최종 점수와 원인을 계산합니다.
    def _predict_score(
        self,
        user: Dict[str, Any],
        product: Dict[str, Any],
    ) -> Dict[str, Any]:
        row = build_training_row(user, product)
        model_score = self.model.predict_row(row)
        rule_score = explicit_risk_score(user, product)
        regret_score = calibrate_regret_score(model_score, rule_score)
        causes = make_regret_causes(regret_score, user, product)
        return {
            "feature": row,
            "model_regret_score": model_score,
            "cause_score": rule_score,
            "regret_score": regret_score,
            "regret_causes": causes,
        }

    ## 단일 상품에 대한 최종 후회예측 응답 payload를 생성합니다.
    def predict(
        self,
        user: Dict[str, Any],
        product: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized_product = {
            "name": product.get("name") or "분석 대상 상품",
            "brand": product.get("brand"),
            "category": product.get("category"),
            "price": safe_float(product.get("price")),
            "rating": safe_float(product.get("rating"), 3.5),
            "review_count": safe_int(product.get("review_count")),
            "return_rate": safe_float(product.get("return_rate"), 5.0),
            "days_since_release": safe_int(product.get("days_since_release"), 180),
            "description": product.get("description"),
            "image_url": product.get("image_url"),
            "source_url": product.get("source_url"),
        }
        score_result = self._predict_score(user, normalized_product)
        regret_score = score_result["regret_score"]
        alternatives = []
        if regret_score >= self.threshold:
            alternatives = self.recommend_alternatives(user, normalized_product, regret_score)

        return {
            "product": normalized_product,
            "product_name": normalized_product["name"],
            "regret_score": round(regret_score, 4),
            "regret_level": regret_level(regret_score),
            "model_regret_score": round(score_result["model_regret_score"], 4),
            "cause_score": round(score_result["cause_score"], 4),
            "threshold": self.threshold,
            "should_reconsider": regret_score >= self.threshold,
            "regret_causes": score_result["regret_causes"],
            "regret_reasons": [cause["message"] for cause in score_result["regret_causes"]],
            "alternatives": alternatives,
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
    ) -> List[Dict[str, Any]]:
        target_category = normalize_category(product.get("category"))
        target_price = safe_float(product.get("price"))
        budget = safe_float(user.get("budget"))
        candidates = self.catalog.products
        if target_category:
            same_category = [item for item in candidates if normalize_category(item.get("category")) == target_category]
            if len(same_category) >= 5:
                candidates = same_category

        results = []
        for candidate in candidates:
            if candidate.get("name") == product.get("name"):
                continue
            score = self._predict_score(user, candidate)["regret_score"]
            if score > target_score:
                continue
            match_score = self._match_score(user, candidate, target_price)
            improvement = max(target_score - score, 0)
            price_score = self._price_advantage(candidate, target_price, budget)
            final_score = match_score * 0.55 + improvement * 0.35 + price_score * 0.10
            results.append({
                "product_id": candidate.get("product_id"),
                "name": candidate.get("name"),
                "brand": candidate.get("brand"),
                "category": candidate.get("category"),
                "price": candidate.get("price"),
                "rating": candidate.get("rating"),
                "return_rate": candidate.get("return_rate"),
                "image_url": candidate.get("image_url") or PRODUCT_IMAGE_URLS.get(str(candidate.get("name", "")).lower()),
                "regret_score": round(score, 4),
                "match_score": round(match_score, 4),
                "improvement_score": round(improvement, 4),
                "final_score": round(final_score, 4),
                "recommendation_reason": self._recommendation_reason(user, candidate, product, score, target_score),
            })
        results.sort(key=lambda item: (-item["final_score"], item["regret_score"], item["price"]))
        return results[:5]

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

    ## 대체상품 추천 사유를 사용자에게 보여줄 문장으로 조합합니다.
    def _recommendation_reason(
        self,
        user: Dict[str, Any],
        candidate: Dict[str, Any],
        target: Dict[str, Any],
        score: float,
        target_score: float,
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
        return ", ".join(reasons) if reasons else "현재 상품보다 종합 위험도가 낮은 대체 후보입니다"

    ## 후회예측 결과와 대체상품 여부를 짧은 요약 문장으로 만듭니다.
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

    train_parser = subparsers.add_parser("train", help="Train model from purchase_regret_training_dataset_1500.xlsx")
    train_parser.add_argument("--data", default=str(DEFAULT_DATASET_PATH), help="Path to xlsx/csv training dataset.")
    train_parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Output model path.")
    train_parser.add_argument("--model-type", choices=["torch"], default="torch", help="Estimator type.")
    train_parser.add_argument("--epochs", type=int, default=800, help="Maximum training epochs.")
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

