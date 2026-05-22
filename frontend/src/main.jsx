import React, { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const initialProgress = {
  label: "대기 중",
  message: "상품 URL 또는 이미지를 입력하면 분석을 시작할 수 있습니다.",
  value: 0,
  live: false,
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

function scoreTone(level) {
  if (level === "high") return "risk";
  if (level === "medium") return "watch";
  return "good";
}

function money(value) {
  const parsed = Number(value || 0);
  return parsed > 0 ? `${parsed.toLocaleString()}원` : "가격 미확인";
}

function App() {
  const [mode, setMode] = useState("url");
  const [progress, setProgress] = useState(initialProgress);
  const [result, setResult] = useState(null);
  const [preview, setPreview] = useState(null);
  const [imageBase64, setImageBase64] = useState(null);
  const [busy, setBusy] = useState(false);
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

  const productMeta = useMemo(() => {
    const product = result?.product || {};
    return [product.brand, product.category, money(product.price), product.rating ? `평점 ${product.rating}` : null]
      .filter(Boolean)
      .join(" · ");
  }, [result]);

  function updateForm(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function buildPayload() {
    const user = {
      budget: numberOrUndefined(form.budget),
      preferred_brands: splitList(form.preferredBrands),
      important_factors: splitList(form.importantFactors),
      usage_purpose: form.usagePurpose || undefined,
    };

    if (mode === "url") {
      if (!form.productUrl.trim()) throw new Error("상품 URL을 입력해주세요.");
      return { input_type: "url", product_url: form.productUrl.trim(), user };
    }

    if (mode === "image") {
      if (!imageBase64) throw new Error("상품 이미지를 업로드해주세요.");
      return { input_type: "image", image_base64: imageBase64, user };
    }

    if (!form.name.trim()) throw new Error("상품명을 입력해주세요.");
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
    } catch (error) {
      setProgress({ label: "입력 확인", message: error.message, value: 0, live: false });
      return;
    }

    const sessionId = makeSessionId();
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${window.location.host}/ws/${sessionId}`);
    wsRef.current?.close();
    wsRef.current = socket;
    setBusy(true);
    setResult(null);
    setProgress({ label: "연결 중", message: "Backend WebSocket에 연결하고 있습니다.", value: 8, live: true });

    socket.onopen = () => {
      setProgress({ label: "요청 전송", message: "AI Agent로 분석 요청을 전달합니다.", value: 14, live: true });
      socket.send(JSON.stringify({ type: "request", session_id: sessionId, data: payload }));
    };

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "progress") {
        setProgress({
          label: "분석 중",
          message: message.message || "분석 중입니다.",
          value: message.progress || 0,
          live: true,
        });
        return;
      }
      if (message.type === "result") {
        setProgress({ label: "분석 완료", message: "AI Agent 응답을 수신했습니다.", value: 100, live: false });
        setResult(message.data);
        setBusy(false);
        socket.close();
        return;
      }
      if (message.type === "error") {
        setProgress({ label: "오류", message: message.message || "분석 중 오류가 발생했습니다.", value: 100, live: false });
        setBusy(false);
        socket.close();
      }
    };

    socket.onerror = () => {
      setProgress({ label: "연결 실패", message: "Backend 또는 Agent 연결을 확인해주세요.", value: 100, live: false });
      setBusy(false);
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
    <main className="app">
      <section className="hero">
        <div className="hero-copy">
          <div className="brand-row">
            <span className="brand-mark">SB</span>
            <span>StopBuy2.0</span>
          </div>
          <h1>구매 전, 후회 가능성을 먼저 계산합니다.</h1>
          <p>
            상품 URL 또는 사진을 AI Agent로 보내 후회 점수를 예측하고, 기준값을 넘으면 더 나은 대체상품을 추천합니다.
          </p>
        </div>
        <div className="hero-metrics">
          <Metric label="Agent" value="WebSocket" />
          <Metric label="Threshold" value="40%" />
          <Metric label="Output" value="Alternatives" />
        </div>
      </section>

      <section className="workspace">
        <form className="panel input-panel" onSubmit={startAnalysis}>
          <div className="panel-head">
            <div>
              <p className="kicker">Input</p>
              <h2>상품 분석 요청</h2>
            </div>
            <span className={`live-badge ${progress.live ? "on" : ""}`}>{progress.label}</span>
          </div>

          <div className="segmented">
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

          <button className="primary" type="submit" disabled={busy}>
            {busy ? "분석 중..." : "후회 가능성 분석"}
          </button>
        </form>

        <section className="panel result-panel">
          <div className="progress-card">
            <div className="progress-top">
              <span>{progress.label}</span>
              <strong>{progress.value}%</strong>
            </div>
            <div className="progress-track"><div style={{ width: `${progress.value}%` }} /></div>
            <p>{progress.message}</p>
          </div>

          {!result ? (
            <div className="empty-state">
              <span className="empty-orbit" />
              <h2>분석 결과가 여기에 표시됩니다.</h2>
              <p>AI Agent가 후회 점수, 주요 원인, 대체상품 리스트를 실시간으로 반환합니다.</p>
            </div>
          ) : (
            <div className="result-stack">
              <div className="result-header">
                <div>
                  <p className="kicker">Result</p>
                  <h2>{result.product_name}</h2>
                  <p>{productMeta}</p>
                </div>
                <ScoreCard score={result.regret_score || 0} level={result.regret_level} />
              </div>

              <div className={`summary-card ${scoreTone(result.regret_level)}`}>
                <strong>{result.should_reconsider ? "구매 재검토 권장" : "구매 가능성 양호"}</strong>
                <p>{result.summary}</p>
              </div>

              <Section title="후회 요인">
                {(result.regret_causes || []).map((cause) => (
                  <Cause key={cause.code} cause={cause} />
                ))}
              </Section>

              <Section title="추천 대체상품">
                {(result.alternatives || []).length ? (
                  <div className="alternative-grid">
                    {result.alternatives.map((item) => <Alternative key={item.product_id || item.name} item={item} />)}
                  </div>
                ) : (
                  <p className="muted">기준값보다 후회 가능성이 낮거나 적절한 대체상품이 없습니다.</p>
                )}
              </Section>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}

function Metric({ label, value }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
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

function ScoreCard({ score, level }) {
  const pct = Math.round(score * 100);
  return (
    <div className={`score-card ${scoreTone(level)}`}>
      <div className="score-ring" style={{ "--score": `${pct * 3.6}deg` }}>
        <span>{pct}%</span>
      </div>
      <strong>{level || "low"}</strong>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <section className="result-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function Cause({ cause }) {
  return (
    <article className="cause-card">
      <div>
        <strong>{cause.title || cause.code}</strong>
        <p>{cause.message}</p>
      </div>
      <span>{Math.round((cause.impact_score || 0) * 100)}%</span>
    </article>
  );
}

function Alternative({ item }) {
  return (
    <article className="alternative-card">
      <div className="alternative-top">
        <strong>{item.name}</strong>
        <span>{Math.round((item.regret_score || 0) * 100)}%</span>
      </div>
      <p>{[item.brand, item.category, item.rating ? `평점 ${item.rating}` : null].filter(Boolean).join(" · ")}</p>
      <p>{item.recommendation_reason}</p>
      <div className="alternative-bottom">
        <span>{money(item.price)}</span>
        <span>매칭 {Math.round((item.match_score || 0) * 100)}%</span>
      </div>
    </article>
  );
}

createRoot(document.getElementById("root")).render(<App />);
