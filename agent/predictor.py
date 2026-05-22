import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency fallback
    LGBMRegressor = None

from sklearn.ensemble import GradientBoostingRegressor


FEATURES = [
    "budget",
    "price",
    "brand_match",
    "rating",
    "review_count",
    "return_rate",
    "important_factor_match_count",
    "days_since_release",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        parsed = float(str(value).replace(",", ""))
        if math.isnan(parsed):
            return default
        return parsed
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(str(value).replace(",", "")))
    except Exception:
        return default


def clamp_score(value: Any) -> float:
    return max(0.0, min(1.0, safe_float(value)))


def normalize_specs(specs: Any) -> Dict[str, Any]:
    if isinstance(specs, dict):
        return specs
    if isinstance(specs, str):
        try:
            parsed = json.loads(specs)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def regret_level(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def build_feature_row(user: Dict[str, Any], product: Dict[str, Any]) -> Dict[str, Any]:
    preferred_brands = user.get("preferred_brands") or []
    important_factors = user.get("important_factors") or []
    specs = normalize_specs(product.get("specs"))
    description = str(product.get("description") or "")

    factor_match_count = sum(
        1
        for factor in important_factors
        if factor and (factor in specs or factor.lower() in description.lower())
    )

    return {
        "budget": safe_float(user.get("budget")),
        "price": safe_float(product.get("price")),
        "brand_match": 1 if product.get("brand") in preferred_brands else 0,
        "rating": safe_float(product.get("rating"), 3.5),
        "review_count": safe_int(product.get("review_count")),
        "return_rate": safe_float(product.get("return_rate"), 5.0),
        "important_factor_match_count": factor_match_count,
        "days_since_release": safe_int(product.get("days_since_release"), 180),
    }


def make_regret_causes(feature: Dict[str, Any], user: Dict[str, Any]) -> List[Dict[str, Any]]:
    causes: List[Dict[str, Any]] = []
    budget = safe_float(feature.get("budget"))
    price = safe_float(feature.get("price"))
    rating = safe_float(feature.get("rating"))
    return_rate = safe_float(feature.get("return_rate"))
    review_count = safe_int(feature.get("review_count"))
    brand_match = safe_int(feature.get("brand_match"))
    factor_match_count = safe_int(feature.get("important_factor_match_count"))
    days_since_release = safe_int(feature.get("days_since_release"))

    if budget > 0 and price > budget:
        over_ratio = (price - budget) / budget
        causes.append({
            "code": "PRICE_OVER_BUDGET",
            "title": "예산 초과",
            "message": f"상품 가격이 예산보다 {over_ratio * 100:.1f}% 높아 구매 후 부담을 느낄 가능성이 있습니다.",
            "severity": "high" if over_ratio >= 0.3 else "medium",
            "impact_score": round(min(over_ratio, 1.0), 4),
        })

    if rating and rating < 3.8:
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
            "title": "높은 반품률",
            "message": "반품률이 높아 같은 상품을 구매한 사용자의 불만 가능성이 상대적으로 큽니다.",
            "severity": "high" if return_rate >= 15 else "medium",
            "impact_score": round(min(return_rate / 30, 1.0), 4),
        })

    if review_count < 20:
        causes.append({
            "code": "LOW_REVIEW_COUNT",
            "title": "검증 부족",
            "message": "리뷰 수가 적어 품질과 만족도에 대한 검증이 충분하지 않습니다.",
            "severity": "medium",
            "impact_score": 0.35,
        })

    if brand_match == 0 and user.get("preferred_brands"):
        causes.append({
            "code": "BRAND_MISMATCH",
            "title": "선호 브랜드 불일치",
            "message": "사용자가 선호하는 브랜드와 달라 만족감이 낮을 수 있습니다.",
            "severity": "low",
            "impact_score": 0.25,
        })

    if factor_match_count == 0 and user.get("important_factors"):
        causes.append({
            "code": "IMPORTANT_FACTOR_MISMATCH",
            "title": "중요 조건 미충족",
            "message": "사용자가 중요하게 보는 조건이 상품 정보와 충분히 맞지 않습니다.",
            "severity": "high",
            "impact_score": 0.7,
        })

    if days_since_release > 900:
        causes.append({
            "code": "OLD_PRODUCT",
            "title": "출시 후 장기간 경과",
            "message": "출시된 지 오래되어 최신 대체상품 대비 경쟁력이 낮을 수 있습니다.",
            "severity": "medium",
            "impact_score": 0.4,
        })

    if not causes:
        causes.append({
            "code": "NO_MAJOR_RISK",
            "title": "주요 후회 요인 없음",
            "message": "현재 입력 조건 기준으로 뚜렷한 구매 후회 위험 요인은 보이지 않습니다.",
            "severity": "low",
            "impact_score": 0.0,
        })

    return sorted(causes, key=lambda item: item["impact_score"], reverse=True)[:5]


def calculate_cause_score(causes: List[Dict[str, Any]]) -> float:
    scores = [clamp_score(c.get("impact_score")) for c in causes if c.get("code") != "NO_MAJOR_RISK"]
    if not scores:
        return 0.0
    return clamp_score(max(scores) * 0.7 + (sum(scores) / len(scores)) * 0.3)


@dataclass
class ProductCatalog:
    products: List[Dict[str, Any]]

    @classmethod
    def sample(cls) -> "ProductCatalog":
        return cls([
            {"product_id": 101, "name": "Galaxy S24 FE", "brand": "Samsung", "category": "스마트폰", "price": 699000, "rating": 4.4, "review_count": 1250, "return_rate": 3.2, "days_since_release": 120, "description": "합리적인 가격의 갤럭시 스마트폰"},
            {"product_id": 102, "name": "iPhone 15", "brand": "Apple", "category": "스마트폰", "price": 1250000, "rating": 4.6, "review_count": 3200, "return_rate": 2.1, "days_since_release": 210, "description": "안정적인 사용자 경험의 프리미엄 스마트폰"},
            {"product_id": 103, "name": "Pixel 8a", "brand": "Google", "category": "스마트폰", "price": 649000, "rating": 4.4, "review_count": 890, "return_rate": 2.8, "days_since_release": 180, "description": "카메라와 순정 안드로이드 경험이 강점"},
            {"product_id": 201, "name": "LG Gram 16", "brand": "LG", "category": "노트북", "price": 1650000, "rating": 4.5, "review_count": 780, "return_rate": 3.5, "days_since_release": 200, "description": "가벼운 고성능 업무용 노트북"},
            {"product_id": 202, "name": "MacBook Air M3", "brand": "Apple", "category": "노트북", "price": 1590000, "rating": 4.8, "review_count": 2100, "return_rate": 1.8, "days_since_release": 150, "description": "긴 배터리와 높은 만족도의 노트북"},
            {"product_id": 203, "name": "Galaxy Book Pro", "brand": "Samsung", "category": "노트북", "price": 1890000, "rating": 4.4, "review_count": 560, "return_rate": 4.1, "days_since_release": 190, "description": "갤럭시 생태계 연동이 좋은 노트북"},
            {"product_id": 301, "name": "Sony WH-1000XM5", "brand": "Sony", "category": "이어폰/헤드폰", "price": 379000, "rating": 4.7, "review_count": 4500, "return_rate": 2.3, "days_since_release": 365, "description": "노이즈 캔슬링 헤드폰"},
            {"product_id": 302, "name": "AirPods Pro 2", "brand": "Apple", "category": "이어폰/헤드폰", "price": 329000, "rating": 4.6, "review_count": 6700, "return_rate": 2.0, "days_since_release": 400, "description": "애플 기기와 궁합이 좋은 무선 이어폰"},
        ])


class RegretPredictor:
    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.4):
        self.threshold = threshold
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), "models", "regret_model.pkl")
        self.model = self._load_or_create_model()
        self.catalog = ProductCatalog.sample()

    def _load_or_create_model(self):
        if os.path.exists(self.model_path):
            try:
                return joblib.load(self.model_path)
            except Exception:
                # Model pickles are not portable across every sklearn version.
                # Rebuild a local synthetic model when an old artifact cannot load.
                try:
                    os.remove(self.model_path)
                except OSError:
                    pass

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        rng = np.random.default_rng(42)
        rows = 700
        budget = rng.uniform(100000, 3000000, rows)
        price = rng.uniform(50000, 3500000, rows)
        brand_match = rng.integers(0, 2, rows)
        rating = rng.uniform(2.0, 5.0, rows)
        review_count = rng.integers(0, 3000, rows)
        return_rate = rng.uniform(0, 25, rows)
        factor_match = rng.integers(0, 5, rows)
        days = rng.integers(0, 1500, rows)

        x = np.column_stack([budget, price, brand_match, rating, review_count, return_rate, factor_match, days])
        y = np.zeros(rows)
        y += np.clip((price - budget) / np.maximum(budget, 1), 0, 1) * 0.3
        y += np.clip((3.8 - rating) / 3.8, 0, 1) * 0.25
        y += np.clip(return_rate / 30, 0, 1) * 0.2
        y += (1 - brand_match) * 0.1
        y += (1 - np.clip(factor_match / 3, 0, 1)) * 0.1
        y += np.clip((days - 900) / 600, 0, 1) * 0.05
        y = np.clip(y + rng.normal(0, 0.04, rows), 0, 1)

        if LGBMRegressor:
            model = LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
        else:
            model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=42)
        model.fit(x, y)
        joblib.dump(model, self.model_path)
        return model

    def _predict_score(self, user: Dict[str, Any], product: Dict[str, Any]) -> Dict[str, Any]:
        feature = build_feature_row(user, product)
        x = np.array([[feature[name] for name in FEATURES]], dtype=np.float64)
        model_score = clamp_score(self.model.predict(x)[0])
        causes = make_regret_causes(feature, user)
        cause_score = calculate_cause_score(causes)
        regret_score = clamp_score(model_score * 0.75 + cause_score * 0.25)
        return {
            "feature": feature,
            "model_regret_score": model_score,
            "cause_score": cause_score,
            "regret_score": regret_score,
            "regret_causes": causes,
        }

    def predict(self, user: Dict[str, Any], product: Dict[str, Any]) -> Dict[str, Any]:
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
            "summary": self._summary(normalized_product, regret_score, alternatives),
        }

    def recommend_alternatives(self, user: Dict[str, Any], product: Dict[str, Any], target_score: float) -> List[Dict[str, Any]]:
        target_category = product.get("category")
        target_price = safe_float(product.get("price"))
        budget = safe_float(user.get("budget"))
        candidates = self.catalog.products
        if target_category:
            same_category = [item for item in candidates if item.get("category") == target_category]
            if len(same_category) >= 2:
                candidates = same_category

        results = []
        for candidate in candidates:
            if candidate.get("name") == product.get("name"):
                continue
            score = self._predict_score(user, candidate)["regret_score"]
            if score > target_score:
                continue
            match_score = self._match_score(user, candidate, target_price)
            improvement = target_score - score
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
                "regret_score": round(score, 4),
                "match_score": round(match_score, 4),
                "improvement_score": round(improvement, 4),
                "final_score": round(final_score, 4),
                "recommendation_reason": self._recommendation_reason(user, candidate, product, score, target_score),
            })

        results.sort(key=lambda item: (-item["final_score"], item["regret_score"], item["price"]))
        return results[:5]

    def _match_score(self, user: Dict[str, Any], product: Dict[str, Any], target_price: float) -> float:
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

    def _price_advantage(self, product: Dict[str, Any], target_price: float, budget: float) -> float:
        price = safe_float(product.get("price"))
        score = 0.0
        if budget > 0 and price <= budget:
            score += 0.5
        if target_price > 0 and price < target_price:
            score += min((target_price - price) / target_price, 0.5)
        return min(score, 1.0)

    def _recommendation_reason(self, user: Dict[str, Any], candidate: Dict[str, Any], target: Dict[str, Any], score: float, target_score: float) -> str:
        reasons = []
        if safe_float(user.get("budget")) and safe_float(candidate.get("price")) <= safe_float(user.get("budget")):
            reasons.append("예산 범위에 맞습니다")
        if safe_float(target.get("price")) and safe_float(candidate.get("price")) < safe_float(target.get("price")):
            reasons.append("현재 후보보다 가격 부담이 낮습니다")
        if safe_float(candidate.get("rating")) >= 4.3:
            reasons.append("사용자 평점이 높습니다")
        if safe_float(candidate.get("return_rate")) <= 5:
            reasons.append("반품률이 낮습니다")
        if score < target_score:
            reasons.append("예측 후회 점수가 더 낮습니다")
        if candidate.get("brand") in (user.get("preferred_brands") or []):
            reasons.append("선호 브랜드와 일치합니다")
        return ", ".join(reasons) if reasons else "현재 상품보다 종합 위험이 낮은 대체 후보입니다"

    def _summary(self, product: Dict[str, Any], regret_score: float, alternatives: List[Dict[str, Any]]) -> str:
        percent = round(regret_score * 100)
        if alternatives:
            return f"{product.get('name')}의 구매 후회 가능성은 {percent}%입니다. 대체상품 {len(alternatives)}개를 함께 검토하는 것이 좋습니다."
        return f"{product.get('name')}의 구매 후회 가능성은 {percent}%입니다. 현재 조건에서는 별도 대체상품 추천이 필요하지 않습니다."
