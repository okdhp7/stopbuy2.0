/*
 * StopBuy - 메인 홈 페이지
 * Design: "Analytical Dual Theme" — Black (Dark Navy) + White (Clean Light)
 * Layout: 비대칭 분할 — 왼쪽 입력 패널(40%) + 오른쪽 결과 패널(60%)
 * Typography: Space Grotesk (data) + Noto Sans KR (body)
 * Theme: CSS 변수(--sb-*) 기반으로 모든 색상 처리
 */
import { useState } from "react";
import { Link } from "wouter";
import { ShoppingCart, RotateCcw, Wifi, WifiOff, ChevronRight, Sun, Moon, History, CheckCircle2, UserRound } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useStopBuyWS, type UserProfile } from "@/hooks/useStopBuyWS";
import { ProductInputForm } from "@/components/ProductInputForm";
import { StopBuyLogo } from "@/components/StopBuyLogo";
import { AnalysisProgress } from "@/components/AnalysisProgress";
import { RegretGauge } from "@/components/RegretGauge";
import { RegretCauseList } from "@/components/RegretCauseList";
import { AlternativeCard } from "@/components/AlternativeCard";
import { LLMAnalysisPanel } from "@/components/LLMAnalysisPanel";
import { UserProfileSettings, loadSavedUserProfile } from "@/components/UserProfileSettings";
import { useTheme } from "@/contexts/ThemeContext";

const HERO_BG = "https://d2xsxph8kpxj0f.cloudfront.net/310519663632451254/8SbY3RBNCHrKBA7jPh2CJe/stopbuy_hero-KnzLRbMywdJiHKwExzwAxg.webp";

export default function Home() {
  const {
    status,
    progress,
    progressMessage,
    result,
    error,
    isConnected,
    productCandidates,
    candidateQuery,
    extractedProduct,
    analyze,
    selectProductCandidate,
    reset,
  } = useStopBuyWS();
  const { theme, toggleTheme } = useTheme();

  const [activeResultTab, setActiveResultTab] = useState<"causes" | "alternatives" | "analysis">(
    "causes"
  );
  const [isUserProfileOpen, setIsUserProfileOpen] = useState(false);
  const [savedUserProfile, setSavedUserProfile] = useState<UserProfile>(() => loadSavedUserProfile());

  const isAnalyzing = status === "connecting" || status === "analyzing";
  const isSelectingProduct = status === "selecting_product";
  const hasResult = status === "completed" && result;
  const hasError = status === "error";
  const targetProduct = result?.product;
  const targetProductName = result?.product_name || targetProduct?.name || "분석 상품";
  const targetProductImageUrl = targetProduct?.image_url;
  const targetProductUrl = targetProduct?.product_url || targetProduct?.source_url;
  const targetDisplayPrice = targetProduct?.display_price ?? (!targetProduct?.price_estimated ? targetProduct?.price : null);
  const isTargetPriceEstimated = Boolean(targetProduct?.price_estimated || targetProduct?.price_missing);
  const isTargetProductInfoMissing = Boolean(targetProduct?.product_info_missing);
  const isDark = theme === "dark";

  const handleAnalyze = (params: Parameters<typeof analyze>[0]) => {
    setActiveResultTab("causes");
    analyze({
      ...params,
      user: {
        ...savedUserProfile,
        ...(params.user || {}),
      },
    });
  };

  return (
    <div
      className="min-h-screen flex flex-col transition-colors duration-300"
      style={{ background: "var(--sb-page-bg)" }}
    >
      {/* ── 헤더 ── */}
      <header
        className="sticky top-0 z-50 flex items-center justify-between px-4 sm:px-6 h-14 transition-colors duration-300"
        style={{
          background: "var(--sb-header-bg)",
          backdropFilter: "blur(12px)",
          borderBottom: "1px solid var(--sb-header-border)",
        }}
      >
        {/* 로고 */}
        <div className="flex items-center gap-2.5">
          <StopBuyLogo size={32} cartSize={16} isDark={isDark} />
          <div>
            <span
              className="font-bold text-lg leading-none"
              style={{ fontFamily: "'Space Grotesk', monospace", color: "var(--sb-text-primary)" }}
            >
              StopBuy
            </span>
            <span
              className="hidden sm:block text-xs leading-none mt-0.5"
              style={{ color: "var(--sb-text-muted)" }}
            >
              구매 후회 예측 AI
            </span>
          </div>
        </div>

        {/* 우측 컨트롤 */}
        <div className="flex items-center gap-2">
          {/* 연결 상태 */}
          <div className="flex items-center gap-1.5 mr-1">
            {isConnected ? (
              <Wifi size={14} style={{ color: "var(--sb-green)" }} />
            ) : (
              <WifiOff size={14} style={{ color: "var(--sb-text-dim)" }} />
            )}
            <span className="text-xs hidden sm:block" style={{ color: isConnected ? "var(--sb-green)" : "var(--sb-text-dim)" }}>
              {isConnected ? "연결됨" : "대기 중"}
            </span>
          </div>

          {/* 이력 페이지 링크 */}
          <button
            type="button"
            onClick={() => setIsUserProfileOpen(true)}
            className="flex items-center gap-1.5 text-xs h-8 px-2 rounded-lg transition-colors"
            style={{ color: "var(--sb-text-muted)", background: "transparent" }}
            onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"; }}
            onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = "transparent"; }}
            title="내정보설정"
          >
            <UserRound size={14} />
            <span className="hidden sm:inline">내정보설정</span>
          </button>

          <Link href="/history">
            <button
              className="flex items-center gap-1.5 text-xs h-8 px-2 rounded-lg transition-colors"
              style={{ color: "var(--sb-text-muted)", background: "transparent" }}
              onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"; }}
              onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = "transparent"; }}
              title="분석 이력 보기"
            >
              <History size={14} />
              <span className="hidden sm:inline">이력</span>
            </button>
          </Link>

          {/* 다시 분석 버튼 */}
          {hasResult && (
            <Button
              variant="ghost"
              size="sm"
              onClick={reset}
              className="flex items-center gap-1.5 text-xs h-8"
              style={{ color: "var(--sb-text-muted)" }}
            >
              <RotateCcw size={13} />
              <span className="hidden sm:inline">다시 분석</span>
            </Button>
          )}

          {/* 테마 토글 버튼 */}
          <button
            onClick={toggleTheme}
            title={isDark ? "화이트 테마로 전환" : "블랙 테마로 전환"}
            className="w-9 h-9 rounded-lg flex items-center justify-center transition-all duration-200"
            style={{
              background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)",
              border: `1px solid ${isDark ? "rgba(255,255,255,0.10)" : "rgba(0,0,0,0.10)"}`,
              color: "var(--sb-text-secondary)",
            }}
            onMouseEnter={e => {
              (e.currentTarget as HTMLButtonElement).style.background = isDark
                ? "rgba(255,255,255,0.12)"
                : "rgba(0,0,0,0.10)";
            }}
            onMouseLeave={e => {
              (e.currentTarget as HTMLButtonElement).style.background = isDark
                ? "rgba(255,255,255,0.06)"
                : "rgba(0,0,0,0.06)";
            }}
          >
            {isDark ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </header>

      <UserProfileSettings
        open={isUserProfileOpen}
        profile={savedUserProfile}
        onOpenChange={setIsUserProfileOpen}
        onSave={setSavedUserProfile}
      />

      {/* ── 히어로 배너 (분석 전만 표시) ── */}
      {status === "idle" && (
        <div
          className="relative overflow-hidden"
          style={{ height: "clamp(200px, 30vw, 320px)" }}
        >
          <img
            src={HERO_BG}
            alt="StopBuy 히어로"
            className="w-full h-full object-cover"
            style={{ opacity: isDark ? 0.6 : 0.35 }}
          />
          <div
            className="absolute inset-0 flex flex-col items-center justify-center text-center px-4"
            style={{ background: "var(--sb-hero-overlay)" }}
          >
            <h1
              className="font-bold mb-3"
              style={{
                fontFamily: "'Space Grotesk', monospace",
                fontSize: "clamp(1.5rem, 4vw, 2.5rem)",
                color: isDark ? "#E2E8F0" : "#1E293B",
                textShadow: isDark ? "0 2px 20px rgba(0,0,0,0.5)" : "0 2px 12px rgba(255,255,255,0.8)",
              }}
            >
              구매 전에{" "}
              <span style={{ color: "var(--sb-blue-light)" }}>후회할지</span> 먼저 알아보세요
            </h1>
            <p
              className="max-w-lg text-sm sm:text-base"
              style={{ color: "var(--sb-text-secondary)" }}
            >
              AI가 상품 정보를 분석하여 구매 후회 가능성을 예측하고,
              더 나은 대체상품을 추천합니다.
            </p>
          </div>
        </div>
      )}

      {/* ── 메인 콘텐츠 ── */}
      <main className="flex-1 container py-6">
        <div className={`flex gap-6 ${hasResult || isAnalyzing || isSelectingProduct || hasError ? "flex-col lg:flex-row" : "flex-col max-w-xl mx-auto"}`}>

          {/* 왼쪽: 입력 패널 */}
          <div className={hasResult || isAnalyzing || isSelectingProduct || hasError ? "lg:w-[38%] shrink-0" : "w-full"}>
            <div
              className="glass-card rounded-2xl p-5"
            >
              <div className="flex items-center gap-2 mb-5">
                <ShoppingCart size={18} style={{ color: "var(--sb-blue-light)" }} />
                <h2
                  className="font-semibold text-base"
                  style={{ color: "var(--sb-text-primary)", fontFamily: "'Space Grotesk', monospace" }}
                >
                  상품 분석 요청
                </h2>
              </div>
              <ProductInputForm onSubmit={handleAnalyze} isAnalyzing={isAnalyzing} />
            </div>

            {/* 사용 안내 */}
            {status === "idle" && (
              <div
                className="mt-4 p-4 rounded-xl transition-colors duration-300"
                style={{
                  background: "var(--sb-hint-bg)",
                  border: "1px solid var(--sb-hint-border)",
                }}
              >
                <p className="text-xs font-semibold mb-2" style={{ color: "var(--sb-text-muted)" }}>
                  이렇게 사용하세요
                </p>
                {[
                  "쇼핑몰 상품 URL을 붙여넣기 하거나",
                  "상품 이미지를 업로드하거나",
                  "상품 정보를 직접 입력하세요",
                  "내 예산과 선호 조건을 설정하면 더 정확합니다",
                ].map((tip, i) => (
                  <div key={i} className="flex items-center gap-2 mt-1.5">
                    <ChevronRight size={12} style={{ color: "var(--sb-blue-light)", flexShrink: 0 }} />
                    <span className="text-xs" style={{ color: "var(--sb-text-dim)" }}>{tip}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* 오른쪽: 결과 패널 */}
          {(isAnalyzing || isSelectingProduct || hasResult || hasError) && (
            <div className="flex-1 min-w-0">

              {/* 분석 중 */}
              {isAnalyzing && (
                <div className="glass-card rounded-2xl">
                  <AnalysisProgress progress={progress} message={progressMessage} />
                </div>
              )}

              {/* 오류 */}


              {/* product candidate selection */}
              {isSelectingProduct && (
                <div className="glass-card rounded-2xl p-5 flex flex-col gap-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h3
                        className="font-semibold text-base"
                        style={{ color: "var(--sb-text-primary)", fontFamily: "'Space Grotesk', monospace" }}
                      >
                        네이버 쇼핑 상품 후보
                      </h3>
                      <p className="text-xs mt-1" style={{ color: "var(--sb-text-muted)" }}>
                        이미지에서 추정한 상품과 가장 가까운 항목을 선택하면 해당 상품으로 후회 예측을 진행합니다.
                      </p>
                    </div>
                    {candidateQuery && (
                      <span
                        className="shrink-0 text-xs px-2.5 py-1 rounded-lg"
                        style={{ background: "var(--sb-input-bg)", color: "var(--sb-text-dim)", border: "1px solid var(--sb-border)" }}
                      >
                        {candidateQuery}
                      </span>
                    )}
                  </div>

                  {extractedProduct?.name && (
                    <div
                      className="rounded-xl p-3 text-xs"
                      style={{ background: "var(--sb-input-bg)", border: "1px solid var(--sb-input-border)", color: "var(--sb-text-muted)" }}
                    >
                      이미지 분석 결과: <span style={{ color: "var(--sb-text-primary)" }}>{extractedProduct.name}</span>
                    </div>
                  )}

                  {productCandidates.length > 0 ? (
                    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
                      {productCandidates.map((candidate, index) => (
                        <button
                          key={candidate.product_id ?? `${candidate.name}-${index}`}
                          type="button"
                          onClick={() => selectProductCandidate(candidate, savedUserProfile)}
                          className="group text-left rounded-xl overflow-hidden transition-all duration-150"
                          style={{ background: "var(--sb-input-bg)", border: "1px solid var(--sb-input-border)" }}
                          onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--sb-blue)"; }}
                          onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--sb-input-border)"; }}
                        >
                          <div className="aspect-square w-full overflow-hidden" style={{ background: "var(--sb-card-bg)" }}>
                            {candidate.image_url ? (
                              <img src={candidate.image_url} alt={candidate.name || "상품 후보"} className="w-full h-full object-cover" />
                            ) : (
                              <div className="w-full h-full flex items-center justify-center" style={{ color: "var(--sb-text-dim)" }}>
                                <ShoppingCart size={28} />
                              </div>
                            )}
                          </div>
                          <div className="p-3 flex flex-col gap-2">
                            <div className="min-h-10">
                              <p className="text-sm font-semibold line-clamp-2" style={{ color: "var(--sb-text-primary)" }}>
                                {candidate.name || "상품명 미확인"}
                              </p>
                              <p className="text-xs mt-1 truncate" style={{ color: "var(--sb-text-dim)" }}>
                                {[candidate.mall_name, candidate.brand, candidate.category].filter(Boolean).join(" · ") || "네이버 쇼핑"}
                              </p>
                            </div>
                            <div className="flex items-center justify-between gap-2">
                              <span className="font-bold text-sm" style={{ color: "var(--sb-blue-light)", fontFamily: "'Space Grotesk', monospace" }}>
                                {candidate.price ? `${candidate.price.toLocaleString()}원` : "가격 미확인"}
                              </span>
                              <span className="inline-flex items-center gap-1 text-xs font-medium" style={{ color: "var(--sb-green)" }}>
                                <CheckCircle2 size={13} /> 선택
                              </span>
                            </div>
                          </div>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div
                      className="rounded-xl p-6 text-center text-sm"
                      style={{ background: "var(--sb-input-bg)", border: "1px solid var(--sb-input-border)", color: "var(--sb-text-muted)" }}
                    >
                      네이버 쇼핑에서 유사 상품을 찾지 못했습니다. 이미지가 선명한지 확인한 뒤 다시 시도해주세요.
                    </div>
                  )}
                </div>
              )}
              {hasError && (
                <div
                  className="glass-card rounded-2xl p-6 flex flex-col items-center gap-4 text-center"
                  style={{ borderColor: "rgba(239,68,68,0.25)" }}
                >
                  <div
                    className="w-12 h-12 rounded-full flex items-center justify-center"
                    style={{ background: "rgba(239,68,68,0.1)" }}
                  >
                    <span style={{ fontSize: 24 }}>⚠️</span>
                  </div>
                  <div>
                    <p className="font-semibold mb-1" style={{ color: "var(--sb-red)" }}>분석 오류</p>
                    <p className="text-sm" style={{ color: "var(--sb-text-secondary)" }}>{error}</p>
                  </div>
                  <Button
                    onClick={reset}
                    variant="outline"
                    size="sm"
                    style={{ borderColor: "rgba(239,68,68,0.3)", color: "var(--sb-red)" }}
                  >
                    다시 시도
                  </Button>
                </div>
              )}

              {/* 분석 결과 */}
              {hasResult && (
                <div className="flex flex-col gap-4">

                  {/* 결과 헤더 카드 */}
                  <div className="glass-card rounded-2xl p-5">
                    <div className="flex flex-col sm:flex-row items-center gap-6">
                      {/* 게이지 */}
                      <div className="shrink-0">
                        <RegretGauge
                          score={result.regret_score ?? 0}
                          level={result.regret_level ?? "low"}
                          size={180}
                        />
                      </div>

                      {/* 상품 정보 + 스코어 요약 */}
                      <div className="flex-1 min-w-0">

                        <div className="mb-4 flex items-start gap-3">
                          {targetProductImageUrl && (
                            targetProductUrl ? (
                              <a
                                href={targetProductUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                title="상품 상세 보기"
                                className="shrink-0 w-16 h-16 sm:w-20 sm:h-20 rounded-xl overflow-hidden transition-transform hover:scale-[1.03]"
                                style={{ background: "var(--sb-input-bg)", border: "1px solid var(--sb-input-border)" }}
                              >
                                <img src={targetProductImageUrl} alt={targetProductName || "분석 상품 이미지"} className="w-full h-full object-cover" />
                              </a>
                            ) : (
                              <div
                                className="shrink-0 w-16 h-16 sm:w-20 sm:h-20 rounded-xl overflow-hidden"
                                style={{ background: "var(--sb-input-bg)", border: "1px solid var(--sb-input-border)" }}
                              >
                                <img src={targetProductImageUrl} alt={targetProductName || "분석 상품 이미지"} className="w-full h-full object-cover" />
                              </div>
                            )
                          )}
                          <div className="min-w-0 flex-1">
                            <h2
                              className="font-bold text-lg leading-snug"
                              style={{ color: "var(--sb-text-primary)", fontFamily: "'Space Grotesk', monospace" }}
                            >
                              {targetProductName}
                            </h2>
                            {(result.product?.brand || result.product?.category || targetDisplayPrice != null || isTargetPriceEstimated || isTargetProductInfoMissing) && (
                              <div className="flex flex-wrap items-center gap-2 mt-2 text-xs" style={{ color: "var(--sb-text-muted)" }}>
                                {result.product?.brand && (
                                  <span
                                    className="px-2 py-1 rounded-lg"
                                    style={{ background: "var(--sb-input-bg)", border: "1px solid var(--sb-input-border)" }}
                                  >
                                    브랜드: <strong style={{ color: "var(--sb-text-secondary)" }}>{result.product.brand}</strong>
                                  </span>
                                )}
                                {result.product?.category && (
                                  <span
                                    className="px-2 py-1 rounded-lg"
                                    style={{ background: "var(--sb-input-bg)", border: "1px solid var(--sb-input-border)" }}
                                  >
                                    카테고리: <strong style={{ color: "var(--sb-text-secondary)" }}>{result.product.category}</strong>
                                  </span>
                                )}
                                {isTargetProductInfoMissing && (
                                  <span
                                    className="px-2 py-1 rounded-lg"
                                    style={{ background: "var(--sb-input-bg)", border: "1px solid var(--sb-input-border)" }}
                                  >
                                    상품정보 미확인
                                  </span>
                                )}
                                <span
                                  className="px-2 py-1 rounded-lg"
                                  style={{ background: "var(--sb-input-bg)", border: "1px solid var(--sb-input-border)" }}
                                >
                                  가격: <strong style={{ color: "var(--sb-blue-light)" }}>
                                    {targetDisplayPrice != null ? `${targetDisplayPrice.toLocaleString("ko-KR")}원` : "가격 미확인"}
                                  </strong>
                                </span>
                              </div>
                            )}
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                          {[
                            {
                              label: "종합 후회 점수",
                              value: Math.round((result.regret_score ?? 0) * 100),
                              color: result.regret_level === "high" ? "var(--sb-red)" : result.regret_level === "medium" ? "var(--sb-amber)" : "var(--sb-green)",
                            },
                            { label: "모델 예측 점수", value: Math.round((result.model_regret_score ?? 0) * 100), color: "var(--sb-blue-light)" },
                            { label: "원인 분석 점수", value: Math.round((result.cause_score ?? 0) * 100), color: "#A78BFA" },
                            { label: "대체상품 수", value: result.alternatives?.length ?? 0, color: "var(--sb-green)", suffix: "개" },
                          ].map((item) => (
                            <div
                              key={item.label}
                              className="rounded-xl p-3 transition-colors duration-300"
                              style={{
                                background: "var(--sb-input-bg)",
                                border: "1px solid var(--sb-input-border)",
                              }}
                            >
                              <p className="text-xs mb-1" style={{ color: "var(--sb-text-muted)" }}>{item.label}</p>
                              <p
                                className="text-2xl font-bold"
                                style={{ color: item.color, fontFamily: "'Space Grotesk', monospace" }}
                              >
                                {item.value}{item.suffix || ""}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* 결과 탭 */}
                  <div
                    className="flex rounded-xl p-1 gap-1 transition-colors duration-300"
                    style={{ background: "var(--sb-input-bg)" }}
                  >
                    {[
                      {
                        id: "causes" as const,
                        label: "후회 원인",
                        count: result.regret_causes?.filter(c => c.code !== "NO_MAJOR_RISK").length ?? 0,
                      },
                      {
                        id: "alternatives" as const,
                        label: "대체상품",
                        count: result.alternatives?.length ?? 0,
                      },
                      {
                        id: "analysis" as const,
                        label: "AI 분석",
                        showCount: false,
                      },
                    ].map((tab) => (
                      <button
                        key={tab.id}
                        onClick={() => setActiveResultTab(tab.id)}
                        className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg text-sm font-medium transition-all duration-200"
                        style={{
                          background: activeResultTab === tab.id ? "var(--sb-tab-active-bg)" : "transparent",
                          color: activeResultTab === tab.id ? "var(--sb-tab-active-color)" : "var(--sb-tab-inactive-color)",
                          border: activeResultTab === tab.id ? "1px solid var(--sb-tab-active-border)" : "1px solid transparent",
                        }}
                      >
                        {tab.label}
                        {(tab as { showCount?: boolean }).showCount !== false && (tab as { count?: number }).count !== undefined && (
                          <span
                            className="text-xs px-1.5 py-0.5 rounded-full"
                            style={{
                              background: (tab as { count?: number }).count! > 0
                                ? (activeResultTab === tab.id ? "var(--sb-badge-active-bg)" : "color-mix(in oklch, var(--sb-blue) 12%, transparent)")
                                : "var(--sb-badge-bg)",
                              color: (tab as { count?: number }).count! > 0
                                ? (activeResultTab === tab.id ? "var(--sb-badge-active-color)" : "var(--sb-blue-light)")
                                : "var(--sb-text-dim)",
                              fontFamily: "'Space Grotesk', monospace",
                            }}
                          >
                            {(tab as { count?: number }).count}
                          </span>
                        )}
                      </button>
                    ))}
                  </div>

                  {/* 탭 콘텐츠 */}
                  <div className="glass-card rounded-2xl p-5">

                    {/* 후회 원인 탭 */}
                    {activeResultTab === "causes" && result.regret_causes && (
                      <div className="animate-fade-in">
                        <h3
                          className="font-semibold mb-4 text-sm"
                          style={{
                            color: "var(--sb-text-secondary)",
                            fontFamily: "'Space Grotesk', monospace",
                            textTransform: "uppercase",
                            letterSpacing: "0.05em",
                          }}
                        >
                          후회 원인 분석
                        </h3>
                        <RegretCauseList causes={result.regret_causes} />
                      </div>
                    )}

                    {/* 대체상품 탭 */}
                    {activeResultTab === "alternatives" && (
                      <div className="animate-fade-in">
                        <h3
                          className="font-semibold mb-4 text-sm"
                          style={{
                            color: "var(--sb-text-secondary)",
                            fontFamily: "'Space Grotesk', monospace",
                            textTransform: "uppercase",
                            letterSpacing: "0.05em",
                          }}
                        >
                          추천 대체상품
                        </h3>
                        {result.alternatives && result.alternatives.length > 0 ? (
                          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                            {result.alternatives.map((alt, idx) => (
                              <AlternativeCard
                                key={alt.product_id ?? idx}
                                product={alt}
                                rank={idx + 1}
                                targetRegretScore={result.regret_score}
                                style={{ animationDelay: `${idx * 80}ms`, animationFillMode: "both" }}
                              />
                            ))}
                          </div>
                        ) : (
                          <div
                            className="flex flex-col items-center gap-3 py-8 text-center"
                            style={{ color: "var(--sb-text-dim)" }}
                          >
                            <span style={{ fontSize: 32 }}>✓</span>
                            <p className="text-sm">
                              후회 가능성이 낮아 대체상품 추천이 필요하지 않습니다.
                            </p>
                          </div>
                        )}
                      </div>
                    )}

                    {/* AI 분석 탭 */}
                    {activeResultTab === "analysis" && (
                      <div className="animate-fade-in">
                        {result.llm_analysis ? (
                          <LLMAnalysisPanel analysis={result.llm_analysis} />
                        ) : (
                          <div
                            className="flex flex-col items-center gap-3 py-8 text-center"
                            style={{ color: "var(--sb-text-dim)" }}
                          >
                            <span style={{ fontSize: 32 }}>🧠</span>
                            <p className="text-sm">AI 분석 결과가 없습니다.</p>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      {/* ── 푸터 ── */}
      <footer
        className="py-4 px-6 text-center transition-colors duration-300"
        style={{ borderTop: "1px solid var(--sb-card-border)" }}
      >
        <p className="text-xs" style={{ color: "var(--sb-text-dim)" }}>
          StopBuy — AI 기반 구매 후회 예측 서비스 | FastAPI + React + LightGBM
        </p>
      </footer>
    </div>
  );
}
