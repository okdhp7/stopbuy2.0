/**
 * StopBuy - WebSocket 훅
 * Design: Analytical Dark — 실시간 분석 상태 관리
 * 백엔드 FastAPI WebSocket과 연결하여 분석 요청/결과 처리
 * 백엔드 미연결 시 데모 모드로 동작
 *
 * 수정 이력:
 * - 연결 타임아웃 3초 → 12초로 증가 (Agent 초기화 시간 고려)
 * - 재연결 로직 강화: 연결 실패 시 최대 2회 재시도
 * - 메시지 전송 전 연결 상태 재확인
 */
import { useCallback, useEffect, useRef, useState } from "react";

export type RegretLevel = "low" | "medium" | "high";
export type AnalysisStatus = "idle" | "connecting" | "analyzing" | "selecting_product" | "completed" | "error";

export interface RegretCause {
  code: string;
  title: string;
  message: string;
  severity: "low" | "medium" | "high";
  impact_score: number;
}

export interface PreferenceAlignment {
  adjustment?: number;
  alignment_score?: number;
  condition_similarity?: number;
  matched_tokens?: string[];
  matched_preferred_brand?: boolean;
  reasons?: string[];
}

export interface AlternativeProduct {
  product_id?: number;
  name?: string;
  brand?: string;
  category?: string;
  price?: number;
  rating?: number;
  return_rate?: number;
  regret_score?: number;
  match_score?: number;
  improvement_score?: number;
  final_score?: number;
  recommendation_reason?: string;
  image_url?: string;
  source_url?: string;
  product_url?: string;
  mall_name?: string;
  alternative_source?: string;
  alternative_search_query?: string;
  alternative_relevance_score?: number;
  preference_alignment?: PreferenceAlignment;
}

export interface LLMAnalysis {
  used_llm: boolean;
  summary?: string;
  risk_explanation?: string;
  purchase_advice?: string;
  alternative_strategy?: string;
}

export interface AnalysisResult {
  product?: ProductInfo;
  product_name?: string;
  regret_score?: number;
  regret_level?: RegretLevel;
  model_regret_score?: number;
  cause_score?: number;
  base_regret_score?: number;
  preference_adjustment?: number;
  preference_alignment?: PreferenceAlignment;
  threshold?: number;
  should_reconsider?: boolean;
  regret_causes?: RegretCause[];
  regret_reasons?: string[];
  alternatives?: AlternativeProduct[];
  llm_analysis?: LLMAnalysis;
}

export interface UserProfile {
  gender?: string;
  age?: number;
  monthly_income?: number;
  job?: string;
  marital_status?: string;
  consumption_type?: string;
  budget?: number;
  preferred_brands?: string[];
  important_factors?: string[];
  usage_purpose?: string;
}

export interface ProductInfo {
  name?: string;
  brand?: string;
  category?: string;
  price?: number;
  display_price?: number | null;
  estimated_price?: number | null;
  price_estimated?: boolean;
  price_missing?: boolean;
  rating?: number;
  review_count?: number;
  return_rate?: number;
  days_since_release?: number;
  description?: string;
  image_url?: string;
  source_url?: string;
  product_url?: string;
  mall_name?: string;
  naver_product_id?: string;
  search_query?: string;
  review_data_available?: boolean;
  review_source?: string;
  review_texts?: string[];
  product_info_missing?: boolean;
  product_info_source?: string;
  shop?: string;
}

export interface ProductCandidate extends ProductInfo {
  product_id?: number | string;
}

interface UseStopBuyWSOptions {
  backendUrl?: string;
}

interface UseStopBuyWSReturn {
  status: AnalysisStatus;
  progress: number;
  progressMessage: string;
  result: AnalysisResult | null;
  error: string | null;
  sessionId: string;
  isConnected: boolean;
  productCandidates: ProductCandidate[];
  candidateQuery: string;
  extractedProduct: ProductInfo | null;
  analyze: (params: {
    inputType: "url" | "image" | "manual";
    productUrl?: string;
    imageBase64?: string;
    user?: UserProfile;
    product?: ProductInfo;
  }) => void;
  selectProductCandidate: (candidate: ProductCandidate, userOverride?: UserProfile) => void;
  reset: () => void;
}

function generateSessionId(): string {
  return Math.random().toString(36).substring(2, 18);
}

function getBackendWsUrl(): string {
  // 환경변수로 명시적 지정 시 우선 사용
  const envUrl = import.meta.env.VITE_BACKEND_WS_URL;
  if (envUrl) return envUrl;

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";

  // 항상 상대 경로 /ws 사용:
  // - 개발 서버(port 3000/5173): Vite proxy가 ws://localhost:8000으로 전달
  // - Docker/프로덕션: Nginx가 http://backend:8000으로 전달
  // 직접 포트 지정 시 CORS 및 프록시 우회 문제 발생하므로 사용하지 않음
  return `${protocol}//${window.location.host}/ws`;
}

function shouldShowAlternatives(result: AnalysisResult): boolean {
  if (typeof result.should_reconsider === "boolean") {
    return result.should_reconsider;
  }

  const threshold = result.threshold ?? 0.4;
  return (result.regret_score ?? 0) >= threshold;
}

function normalizeAgentResult(result: AnalysisResult): AnalysisResult {
  return {
    ...result,
    alternatives: shouldShowAlternatives(result) ? result.alternatives ?? [] : [],
  };
}

export function useStopBuyWS(options: UseStopBuyWSOptions = {}): UseStopBuyWSReturn {
  const [status, setStatus] = useState<AnalysisStatus>("idle");
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [productCandidates, setProductCandidates] = useState<ProductCandidate[]>([]);
  const [candidateQuery, setCandidateQuery] = useState("");
  const [extractedProduct, setExtractedProduct] = useState<ProductInfo | null>(null);
  const [sessionId] = useState(() => generateSessionId());

  const wsRef = useRef<WebSocket | null>(null);
  const lastUserRef = useRef<UserProfile | undefined>(undefined);
  const pendingRequestRef = useRef<{ type: string; session_id: string; data: object } | null>(null);
  const completionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryCountRef = useRef(0);

  const backendWsUrl = options.backendUrl || getBackendWsUrl();
  const wsUrl = `${backendWsUrl}/${sessionId}`;

  const clearCompletionTimer = useCallback(() => {
    if (completionTimerRef.current) {
      clearTimeout(completionTimerRef.current);
      completionTimerRef.current = null;
    }
  }, []);

  const handleMessage = useCallback((message: {
    type: string;
    data?: any;
    progress?: number;
    message?: string;
    error?: string;
  }) => {
    const { type } = message;
    if (type === "progress") {
      setProgress(message.progress || 0);
      setProgressMessage(message.message || "분석 중...");
    } else if (type === "product_candidates") {
      const payload = message.data || {};
      setProductCandidates(Array.isArray(payload.candidates) ? payload.candidates : []);
      setCandidateQuery(payload.query || "");
      setExtractedProduct(payload.extracted_product || null);
      setResult(null);
      setProgress(100);
      setProgressMessage("상품 후보를 선택하세요.");
      setStatus("selecting_product");
    } else if (type === "result") {
      setProductCandidates([]);
      setCandidateQuery("");
      setExtractedProduct(null);
      setResult(message.data ? normalizeAgentResult(message.data) : null);
      setProgress(100);
      setProgressMessage("분석 완료!");
      clearCompletionTimer();
      completionTimerRef.current = setTimeout(() => {
        setStatus("completed");
        completionTimerRef.current = null;
      }, 200);
    } else if (type === "error") {
      pendingRequestRef.current = null;
      retryCountRef.current = 0;
      clearCompletionTimer();
      setProgress(100);
      setProgressMessage(message.message || "요청을 완료하지 못했습니다.");
      setError(message.message || "요청을 완료하지 못했습니다.");
      setStatus("error");
    }
  }, [clearCompletionTimer]);

  const failAgentRequest = useCallback(() => {
    clearCompletionTimer();
    setError("AI Agent에 분석을 요청하지 못했습니다. 백엔드와 Agent 컨테이너 상태를 확인해 주세요.");
    setStatus("error");
  }, [clearCompletionTimer]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    if (wsRef.current?.readyState === WebSocket.CONNECTING) return;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      // 타임아웃 12초 (Agent 초기화 시간 고려)
      const connectionTimeout = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          ws.close();
          wsRef.current = null;
          // 재시도 또는 데모 모드 전환
          if (pendingRequestRef.current) {
            if (retryCountRef.current < 2) {
              retryCountRef.current += 1;
              setTimeout(() => connect(), 2000);
            } else {
              pendingRequestRef.current = null;
              retryCountRef.current = 0;
              failAgentRequest();
            }
          }
        }
      }, 12000);

      ws.onopen = () => {
        clearTimeout(connectionTimeout);
        setIsConnected(true);
        retryCountRef.current = 0;
        setStatus("analyzing");
        if (pendingRequestRef.current) {
          ws.send(JSON.stringify(pendingRequestRef.current));
          pendingRequestRef.current = null;
        }
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          handleMessage(message);
        } catch (e) {
          console.error("WebSocket 메시지 파싱 오류:", e);
        }
      };

      ws.onclose = () => {
        clearTimeout(connectionTimeout);
        setIsConnected(false);
        wsRef.current = null;
      };

      ws.onerror = (err) => {
        clearTimeout(connectionTimeout);
        console.error("WebSocket 연결 오류:", err);
        setIsConnected(false);
        wsRef.current = null;
        if (pendingRequestRef.current) {
          if (retryCountRef.current < 2) {
            retryCountRef.current += 1;
            setTimeout(() => connect(), 2000);
          } else {
            pendingRequestRef.current = null;
            retryCountRef.current = 0;
            failAgentRequest();
          }
        }
      };
    } catch (err) {
      console.error("WebSocket 생성 실패:", err);
      if (pendingRequestRef.current) {
        pendingRequestRef.current = null;
        retryCountRef.current = 0;
        failAgentRequest();
      }
    }
  }, [wsUrl, handleMessage, failAgentRequest]);

  const analyze = useCallback(
    (params: {
      inputType: "url" | "image" | "manual";
      productUrl?: string;
      imageBase64?: string;
      user?: UserProfile;
      product?: ProductInfo;
    }) => {
      setStatus("connecting");
      clearCompletionTimer();
      setProgress(5);
      setProgressMessage("연결 중...");
      setResult(null);
      setError(null);
      setProductCandidates([]);
      setCandidateQuery("");
      setExtractedProduct(null);
      lastUserRef.current = params.user;
      retryCountRef.current = 0;

      const requestMessage = {
        type: "request",
        session_id: sessionId,
        data: {
          input_type: params.inputType,
          product_url: params.productUrl,
          image_base64: params.imageBase64,
          user: params.user || {},
          product: params.product || null,
        },
      };

      const activeSocket = wsRef.current;
      if (activeSocket?.readyState === WebSocket.OPEN) {
        try {
          activeSocket.send(JSON.stringify(requestMessage));
          setStatus("analyzing");
          setProgress(10);
          setProgressMessage("분석 요청 전송 완료...");
          return;
        } catch (err) {
          console.error("WebSocket send failed, reconnecting:", err);
          activeSocket.close();
          wsRef.current = null;
        }
      }

      pendingRequestRef.current = requestMessage;
      connect();
    },
    [sessionId, connect, clearCompletionTimer]
  );


  const selectProductCandidate = useCallback(
    (candidate: ProductCandidate, userOverride?: UserProfile) => {
      const selectedProduct: ProductInfo = {
        name: candidate.name,
        brand: candidate.brand,
        category: candidate.category,
        price: candidate.price,
        rating: candidate.rating,
        review_count: candidate.review_count,
        return_rate: candidate.return_rate,
        description: candidate.description,
        image_url: candidate.image_url,
        source_url: candidate.source_url || candidate.product_url,
        product_url: candidate.product_url || candidate.source_url,
        mall_name: candidate.mall_name,
        naver_product_id: candidate.naver_product_id,
        search_query: candidate.search_query,
        review_data_available: candidate.review_data_available,
        review_source: candidate.review_source,
        review_texts: candidate.review_texts,
      };

      setProductCandidates([]);
      setCandidateQuery("");
      setExtractedProduct(null);
      setResult(null);
      setError(null);
      setStatus("connecting");
      setProgress(5);
      setProgressMessage("선택한 상품으로 분석을 준비하고 있습니다.");
      const selectedUser = userOverride
        ? { ...(lastUserRef.current || {}), ...userOverride }
        : lastUserRef.current;

      analyze({
        inputType: "manual",
        user: selectedUser,
        product: selectedProduct,
      });
    },
    [analyze]
  );


  const reset = useCallback(() => {
    clearCompletionTimer();
    setStatus("idle");
    setProgress(0);
    setProgressMessage("");
    setResult(null);
    setError(null);
    setProductCandidates([]);
    setCandidateQuery("");
    setExtractedProduct(null);
    retryCountRef.current = 0;
  }, [clearCompletionTimer]);

  useEffect(() => {
    return () => {
      clearCompletionTimer();
      wsRef.current?.close();
    };
  }, [clearCompletionTimer]);

  return {
    status,
    progress,
    progressMessage,
    result,
    error,
    sessionId,
    isConnected,
    productCandidates,
    candidateQuery,
    extractedProduct,
    analyze,
    selectProductCandidate,
    reset,
  };
}
