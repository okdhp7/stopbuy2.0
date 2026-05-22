import React, { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const HERO_BG =
  "https://d2xsxph8kpxj0f.cloudfront.net/310519663632451254/8SbY3RBNCHrKBA7jPh2CJe/stopbuy_hero-KnzLRbMywdJiHKwExzwAxg.webp";

const idleProgress = {
  label: "대기 중",
  message: "상품 URL, 사진, 직접 입력 중 하나를 선택해 분석을 시작하세요.",
  value: 0,
};

function makeSessionId() {
  return Math.random().toString(36).slice(2, 14);
}

function splitList(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function numberOrUndefined(value) {
  return value === "" || value == null ? undefined : Number(value);
}

function money(value) {
  const parsed = Number(value || 0);
  return parsed > 0 ? `${parsed.toLocaleString()}원` : "가격 미확인";
}

function levelTone(level) {
  if (level === "high") return "high";
  if (level === "medium") return "medium";
  return "low";
}

function App() {
  const [theme, setTheme] = useState("dark");
  const [mode, setMode] = useState("url");
  const [status, setStatus] = useState("idle");
  const [progress, setProgress] = useState(idleProgress);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("causes");
  const [preview, setPreview] = useState(null);
  const [imageBase64, setImageBase64] = useState(null);
  const wsRef = useRef(null);

  const [form, setForm] = useState({
    productUrl: "",
    budget: "800000",
    preferredBrands: "Samsung, Apple, LG",
    importantFactors: "성능, 배터리, 가격",
    usagePurpose: "일상 사용",
    name: "",
    brand: "",
    category: "",
    price: "",
    rating: "",
    reviewCount: "",
    returnRate: "",
  });

  const isAnalyzing = status === "connecting" || status === "analyzing";
  const hasResult = status === "completed" && result;
  const isConnected = isAnalyzing || hasResult;

  const productMeta = useMemo(() => {
    const product = result?.product || {};
    return [product.brand, product.category, money(product.price), product.rating ? `평점 ${product.rating}` : null]
      .filter(Boolean)
      .join(" · ");
  }, [result]);

  function updateForm(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function reset() {
    wsRef.current?.close();
    setStatus("idle");
    setResult(null);
    setError("");
    setActiveTab("causes");
    setProgress(idleProgress);
  }

  function buildPayload() {
    const user = {
      budget: numberOrUndefined(form.budget),
      preferred_brands: splitList(form.preferredBrands),
      important_factors: splitList(form.importantFactors),
      usage_purpose: form.usagePurpose || undefined,
    };

    if (mode === "url") {
      if (!form.productUrl.trim()) throw new Error("상품 URL을 입력하세요.");
      return { input_type: "url", product_url: form.productUrl.trim(), user };
    }

    if (mode === "image") {
      if (!imageBase64) throw new Error("상품 이미지를 업로드하세요.");
      return { input_type: "image", image_base64: imageBase64, user };
    }

    if (!form.name.trim()) throw new Error("상품명을 입력하세요.");
    return {
      input_type: "manual",
      user,
      product: {
        name: form.name.trim(),
        brand: form.brand.trim() || undefined,
        category: form.category.trim() || undefined,
        price: numberOrUndefined(form.price),
        rating: numberOrUndefined(form.rating),
        review_count: numberOrUndefined(form.reviewCount),
        return_rate: numberOrUndefined(form.returnRate),
      },
    };
  }

  function startAnalysis(event) {
    event.preventDefault();

    let payload;
    try {
      payload = buildPayload();
    } catch (validationError) {
      setStatus("error");
      setError(validationError.message);
      setProgress({ label: "입력 확인", message: validationError.message, value: 0 });
      return;
    }

    const sessionId = makeSessionId();
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${window.location.host}/ws/${sessionId}`);

    wsRef.current?.close();
    wsRef.current = socket;
    setStatus("connecting");
    setResult(null);
    setError("");
    setActiveTab("causes");
    setProgress({ label: "연결 중", message: "Backend WebSocket에 연결하고 있습니다.", value: 8 });

    socket.onopen = () => {
      setStatus("analyzing");
      setProgress({ label: "요청 전송", message: "AI Agent로 분석 요청을 전달합니다.", value: 14 });
      socket.send(JSON.stringify({ type: "request", session_id: sessionId, data: payload }));
    };

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "progress") {
        setStatus("analyzing");
        setProgress({
          label: "분석 중",
          message: message.message || "상품 정보를 분석하고 있습니다.",
          value: message.progress || 0,
        });
        return;
      }

      if (message.type === "result") {
        setStatus("completed");
        setProgress({ label: "분석 완료", message: "AI Agent 응답을 수신했습니다.", value: 100 });
        setResult(message.data);
        socket.close();
        return;
      }

      if (message.type === "error") {
        setStatus("error");
        setError(message.message || "분석 중 오류가 발생했습니다.");
        setProgress({ label: "오류", message: message.message || "분석 중 오류가 발생했습니다.", value: 100 });
        socket.close();
      }
    };

    socket.onerror = () => {
      setStatus("error");
      setError("Backend 또는 Agent 연결을 확인하세요.");
      setProgress({ label: "연결 실패", message: "Backend 또는 Agent 연결을 확인하세요.", value: 100 });
    };
  }

  function handleImage(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setImageBase64(reader.result);
      setPreview(reader.result);
    };
    reader.readAsDataURL(file);
  }

  return (
    <div className={`app-shell ${theme}`}>
      <Header
        isConnected={isConnected}
        hasResult={hasResult}
        theme={theme}
        onReset={reset}
        onToggleTheme={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
      />

      {status === "idle" && <Hero />}

      <main className={`workspace ${status === "idle" ? "centered" : ""}`}>
        <section className="glass-card input-card">
          <div className="section-title">
            <span className="cart-icon">SB</span>
            <div>
              <p>Product Request</p>
              <h2>상품 분석 요청</h2>
            </div>
          </div>

          <form onSubmit={startAnalysis}>
            <div className="mode-tabs">
              {[
                ["url", "상품 URL"],
                ["image", "상품 사진"],
                ["manual", "직접 입력"],
              ].map(([id, label]) => (
                <button key={id} type="button" className={mode === id ? "active" : ""} onClick={() => setMode(id)}>
                  {label}
                </button>
              ))}
            </div>

            {mode === "url" && (
              <Field label="상품 URL">
                <input
                  value={form.productUrl}
                  onChange={(event) => updateForm("productUrl", event.target.value)}
                  placeholder="https://www.example.com/product/..."
                  type="url"
                />
              </Field>
            )}

            {mode === "image" && (
              <label className="drop-zone">
                <input type="file" accept="image/*" onChange={(event) => handleImage(event.target.files?.[0])} />
                {preview ? <img src={preview} alt="업로드된 상품" /> : <span>상품 이미지를 업로드하세요</span>}
              </label>
            )}

            {mode === "manual" && (
              <div className="grid two">
                <Field label="상품명"><input value={form.name} onChange={(event) => updateForm("name", event.target.value)} /></Field>
                <Field label="브랜드"><input value={form.brand} onChange={(event) => updateForm("brand", event.target.value)} /></Field>
                <Field label="카테고리"><input value={form.category} onChange={(event) => updateForm("category", event.target.value)} /></Field>
                <Field label="가격"><input type="number" value={form.price} onChange={(event) => updateForm("price", event.target.value)} /></Field>
                <Field label="평점"><input type="number" min="0" max="5" step="0.1" value={form.rating} onChange={(event) => updateForm("rating", event.target.value)} /></Field>
                <Field label="리뷰 수"><input type="number" value={form.reviewCount} onChange={(event) => updateForm("reviewCount", event.target.value)} /></Field>
                <Field label="반품률"><input type="number" min="0" max="100" step="0.1" value={form.returnRate} onChange={(event) => updateForm("returnRate", event.target.value)} /></Field>
              </div>
            )}

            <div className="divider" />

            <div className="grid two">
              <Field label="예산"><input type="number" value={form.budget} onChange={(event) => updateForm("budget", event.target.value)} /></Field>
              <Field label="사용 목적"><input value={form.usagePurpose} onChange={(event) => updateForm("usagePurpose", event.target.value)} /></Field>
              <Field label="선호 브랜드"><input value={form.preferredBrands} onChange={(event) => updateForm("preferredBrands", event.target.value)} /></Field>
              <Field label="중요 조건"><input value={form.importantFactors} onChange={(event) => updateForm("importantFactors", event.target.value)} /></Field>
            </div>

            <button className="primary-button" type="submit" disabled={isAnalyzing}>
              {isAnalyzing ? "분석 중..." : "후회 가능성 분석"}
            </button>
          </form>

          {status === "idle" && (
            <div className="hint-box">
              <strong>이렇게 사용하세요</strong>
              <p>상품 URL을 붙여 넣거나, 사진을 업로드하거나, 상품 정보를 직접 입력하세요. 예산과 중요 조건을 넣으면 추천 품질이 좋아집니다.</p>
            </div>
          )}
        </section>

        {status !== "idle" && (
          <section className="result-column">
            {isAnalyzing && (
              <div className="glass-card">
                <AnalysisProgress progress={progress} />
              </div>
            )}

            {status === "error" && (
              <div className="glass-card error-card">
                <div className="error-icon">!</div>
                <h2>분석 오류</h2>
                <p>{error}</p>
                <button type="button" onClick={reset}>다시 시도</button>
              </div>
            )}

            {hasResult && (
              <>
                <div className="glass-card result-summary">
                  <RegretGauge score={result.regret_score || 0} level={result.regret_level || "low"} />
                  <div className="result-copy">
                    {result._demo && <span className="demo-badge">Demo fallback</span>}
                    <h2>{result.product_name || result.product?.name || "분석 상품"}</h2>
                    <p>{productMeta}</p>
                    <div className="score-grid">
                      <ScoreBox label="종합 후회 점수" value={Math.round((result.regret_score || 0) * 100)} tone={levelTone(result.regret_level)} />
                      <ScoreBox label="모델 예측 점수" value={Math.round((result.model_regret_score || 0) * 100)} />
                      <ScoreBox label="원인 분석 점수" value={Math.round((result.cause_score || 0) * 100)} tone="purple" />
                      <ScoreBox label="대체상품 수" value={result.alternatives?.length || 0} suffix="개" tone="low" />
                    </div>
                  </div>
                </div>

                <div className="result-tabs">
                  {[
                    ["causes", "후회 원인", result.regret_causes?.filter((cause) => cause.code !== "NO_MAJOR_RISK").length || 0],
                    ["alternatives", "대체상품", result.alternatives?.length || 0],
                    ["analysis", "AI 분석", null],
                  ].map(([id, label, count]) => (
                    <button key={id} className={activeTab === id ? "active" : ""} onClick={() => setActiveTab(id)}>
                      {label}
                      {count !== null && <span>{count}</span>}
                    </button>
                  ))}
                </div>

                <div className="glass-card tab-content">
                  {activeTab === "causes" && (
                    <section>
                      <h3>후회 원인 분석</h3>
                      {(result.regret_causes || []).length ? (
                        <div className="cause-list">
                          {result.regret_causes.map((cause) => <CauseCard key={cause.code} cause={cause} />)}
                        </div>
                      ) : (
                        <EmptyMessage text="주요 후회 위험 요인이 발견되지 않았습니다." />
                      )}
                    </section>
                  )}

                  {activeTab === "alternatives" && (
                    <section>
                      <h3>추천 대체상품</h3>
                      {(result.alternatives || []).length ? (
                        <div className="alternative-grid">
                          {result.alternatives.map((item, index) => (
                            <AlternativeCard key={item.product_id || item.name || index} item={item} rank={index + 1} />
                          ))}
                        </div>
                      ) : (
                        <EmptyMessage text="현재 점수에서는 추천 대체상품이 필요하지 않습니다." />
                      )}
                    </section>
                  )}

                  {activeTab === "analysis" && (
                    <section>
                      <h3>AI 분석 요약</h3>
                      <div className="analysis-panel">
                        <strong>{result.should_reconsider ? "구매 재검토 권장" : "구매 가능성 양호"}</strong>
                        <p>{result.summary || "AI Agent가 상품의 후회 가능성과 추천 근거를 분석했습니다."}</p>
                        {(result.regret_reasons || []).map((reason, index) => (
                          <span key={index}>{reason}</span>
                        ))}
                      </div>
                    </section>
                  )}
                </div>
              </>
            )}
          </section>
        )}
      </main>

      <footer className="app-footer">StopBuy2.0 · React + FastAPI + AI Agent · WebSocket purchase regret prediction</footer>
    </div>
  );
}

function Header({ isConnected, hasResult, theme, onReset, onToggleTheme }) {
  return (
    <header className="topbar">
      <div className="logo-group">
        <div className="logo-mark">SB</div>
        <div>
          <strong>StopBuy</strong>
          <span>구매 후회 예측 AI</span>
        </div>
      </div>
      <div className="top-actions">
        <span className={`connection ${isConnected ? "on" : ""}`}>{isConnected ? "연결됨" : "대기 중"}</span>
        {hasResult && <button type="button" onClick={onReset}>다시 분석</button>}
        <button type="button" onClick={onToggleTheme}>{theme === "dark" ? "Light" : "Dark"}</button>
      </div>
    </header>
  );
}

function Hero() {
  return (
    <section className="hero">
      <img src={HERO_BG} alt="StopBuy hero" />
      <div className="hero-overlay">
        <h1>구매 전에 후회할지 먼저 알아보세요</h1>
        <p>AI가 상품 정보를 분석하여 구매 후회 가능성을 예측하고, 더 나은 대체상품을 추천합니다.</p>
      </div>
    </section>
  );
}

function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function AnalysisProgress({ progress }) {
  return (
    <div className="progress-card">
      <div className="progress-header">
        <span>{progress.label}</span>
        <strong>{progress.value}%</strong>
      </div>
      <div className="progress-bar"><div style={{ width: `${progress.value}%` }} /></div>
      <p>{progress.message}</p>
    </div>
  );
}

function RegretGauge({ score, level }) {
  const percent = Math.round(score * 100);
  const tone = levelTone(level);
  return (
    <div className={`gauge ${tone}`}>
      <svg viewBox="0 0 120 120" role="img" aria-label={`후회 가능성 ${percent}%`}>
        <circle cx="60" cy="60" r="48" />
        <circle cx="60" cy="60" r="48" className="gauge-value" style={{ strokeDashoffset: 302 - (302 * percent) / 100 }} />
      </svg>
      <div>
        <strong>{percent}%</strong>
        <span>{level || "low"}</span>
      </div>
    </div>
  );
}

function ScoreBox({ label, value, suffix = "%", tone = "blue" }) {
  return (
    <div className={`score-box ${tone}`}>
      <span>{label}</span>
      <strong>{value}{suffix}</strong>
    </div>
  );
}

function CauseCard({ cause }) {
  return (
    <article className={`cause-card ${cause.severity || "medium"}`}>
      <div>
        <strong>{cause.title || cause.code}</strong>
        <p>{cause.message}</p>
      </div>
      <span>{Math.round((cause.impact_score || 0) * 100)}%</span>
    </article>
  );
}

function AlternativeCard({ item, rank }) {
  return (
    <article className="alternative-card">
      <div className="rank">#{rank}</div>
      <strong>{item.name}</strong>
      <p>{[item.brand, item.category, item.rating ? `평점 ${item.rating}` : null].filter(Boolean).join(" · ")}</p>
      <p>{item.recommendation_reason || "후회 가능성을 낮출 수 있는 대체상품입니다."}</p>
      <div className="alternative-meta">
        <span>{money(item.price)}</span>
        <span>매칭 {Math.round((item.match_score || 0) * 100)}%</span>
      </div>
    </article>
  );
}

function EmptyMessage({ text }) {
  return (
    <div className="empty-message">
      <span>—</span>
      <p>{text}</p>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
