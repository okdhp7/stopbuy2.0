/*
 * StopBuy - 대체상품 카드 컴포넌트
 * Design: "Analytical Dual Theme" CSS 변수(--sb-*) 기반 색상 처리
 */
import { ExternalLink, Star, TrendingDown, ShieldCheck, Tag } from "lucide-react";
import type { AlternativeProduct } from "@/hooks/useStopBuyWS";
import { getAlternativeProductImage, PRODUCT_PLACEHOLDER_IMAGE } from "@/lib/productImages";

interface AlternativeCardProps {
  product: AlternativeProduct;
  rank: number;
  targetRegretScore?: number;
  style?: React.CSSProperties;
}

export function AlternativeCard({ product, rank, targetRegretScore, style }: AlternativeCardProps) {
  const regretScore = product.regret_score ?? 0;
  const productImageUrl = getAlternativeProductImage(product);
  const productUrl = product.product_url || product.source_url;
  const sourceLabel = product.alternative_source === "naver_shopping"
    ? product.mall_name || "네이버 쇼핑"
    : product.mall_name || "추천 카탈로그";
  const improvement = targetRegretScore != null
    ? Math.round((targetRegretScore - regretScore) * 100)
    : null;

  const rankColors = ["#F59E0B", "#94A3B8", "#CD7F32"];
  const rankColor = rankColors[rank - 1] || "#64748B";

  const openProductDetail = (event: React.MouseEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (!productUrl) return;
    window.open(productUrl, "_blank", "noopener,noreferrer");
  };

  return (
    <div
      className="glass-card rounded-xl overflow-hidden transition-all duration-200 group animate-slide-up"
      style={{
        border: "1px solid var(--sb-card-border)",
        ...style,
      }}
    >
      <div className="relative">
        <div
          className="absolute top-2 left-2 z-10 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold"
          style={{
            background: rankColor,
            color: rank === 1 ? "#0A0E1A" : "#fff",
            fontFamily: "'Space Grotesk', monospace",
          }}
        >
          {rank}
        </div>

        {improvement !== null && improvement > 0 && (
          <div
            className="absolute top-2 right-2 z-10 flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold"
            style={{
              background: "color-mix(in oklch, var(--sb-green) 12%, transparent)",
              color: "var(--sb-green)",
              border: "1px solid color-mix(in oklch, var(--sb-green) 30%, transparent)",
              fontFamily: "'Space Grotesk', monospace",
            }}
          >
            <TrendingDown size={11} />
            -{improvement}%
          </div>
        )}

        <button
          type="button"
          onClick={openProductDetail}
          disabled={!productUrl}
          aria-label={productUrl ? `${product.name || "상품"} 상세보기` : undefined}
          className="w-full h-36 overflow-hidden block p-0 border-0 cursor-pointer disabled:cursor-default"
          style={{ background: "var(--sb-input-bg)" }}
        >
          <img
            src={productImageUrl}
            alt={product.name || "상품 이미지"}
            className="w-full h-full object-contain p-3 transition-transform duration-300 group-hover:scale-105"
            onError={(e) => {
              (e.target as HTMLImageElement).src = PRODUCT_PLACEHOLDER_IMAGE;
            }}
          />
        </button>
      </div>

      <div className="p-3 flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-medium truncate" style={{ color: "var(--sb-blue-light)" }}>
            {product.brand || sourceLabel}
          </span>
          {product.alternative_source === "naver_shopping" && (
            <span
              className="text-[11px] px-1.5 py-0.5 rounded-full shrink-0"
              style={{
                background: "color-mix(in oklch, var(--sb-green) 10%, transparent)",
                color: "var(--sb-green)",
                border: "1px solid color-mix(in oklch, var(--sb-green) 25%, transparent)",
              }}
            >
              네이버
            </span>
          )}
        </div>

        <h3
          className="text-sm font-semibold leading-snug line-clamp-2"
          style={{ color: "var(--sb-text-primary)" }}
        >
          {product.name || "상품명 없음"}
        </h3>

        <div className="flex items-center justify-between gap-2">
          <span className="text-xs truncate" style={{ color: "var(--sb-text-dim)" }}>
            {sourceLabel}
          </span>
          {productUrl && (
            <button
              type="button"
              onClick={openProductDetail}
              className="inline-flex items-center gap-1 text-xs shrink-0 transition-colors"
              style={{ color: "var(--sb-blue-light)" }}
            >
              상품보기
              <ExternalLink size={11} />
            </button>
          )}
        </div>

        {product.price != null && (
          <div className="flex items-center gap-1.5">
            <Tag size={13} style={{ color: "var(--sb-text-secondary)" }} />
            <span
              className="text-base font-bold"
              style={{ color: "var(--sb-text-primary)", fontFamily: "'Space Grotesk', monospace" }}
            >
              {product.price.toLocaleString("ko-KR")}원
            </span>
          </div>
        )}

        <div className="flex items-center gap-3">
          {product.rating != null && (
            <div className="flex items-center gap-1">
              <Star size={12} fill="var(--sb-amber)" stroke="none" />
              <span
                className="text-xs font-medium"
                style={{ color: "var(--sb-amber)", fontFamily: "'Space Grotesk', monospace" }}
              >
                {product.rating.toFixed(1)}
              </span>
            </div>
          )}
          {product.return_rate != null && (
            <div className="flex items-center gap-1">
              <ShieldCheck size={12} style={{ color: "var(--sb-green)" }} />
              <span
                className="text-xs"
                style={{ color: "var(--sb-text-secondary)", fontFamily: "'Space Grotesk', monospace" }}
              >
                반품률 {product.return_rate.toFixed(1)}%
              </span>
            </div>
          )}
        </div>

        <div
          className="flex items-center justify-between px-2.5 py-1.5 rounded-lg transition-colors duration-300"
          style={{
            background: "color-mix(in oklch, var(--sb-green) 8%, transparent)",
            border: "1px solid color-mix(in oklch, var(--sb-green) 20%, transparent)",
          }}
        >
          <span className="text-xs" style={{ color: "var(--sb-text-secondary)" }}>예측 후회 점수</span>
          <span
            className="text-sm font-bold"
            style={{ color: "var(--sb-green)", fontFamily: "'Space Grotesk', monospace" }}
          >
            {Math.round(regretScore * 100)}
          </span>
        </div>

        {product.recommendation_reason && (
          <p
            className="text-xs leading-relaxed line-clamp-2"
            style={{ color: "var(--sb-text-muted)" }}
          >
            {product.recommendation_reason}
          </p>
        )}
      </div>
    </div>
  );
}
