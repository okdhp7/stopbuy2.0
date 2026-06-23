import { useEffect, useState } from "react";
import { UserRound } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { UserProfile } from "@/hooks/useStopBuyWS";

export const USER_PROFILE_STORAGE_KEY = "stopbuy-user-profile";

const whiteInputStyle = {
  background: "#FFFFFF",
  border: "1px solid #CBD5E1",
  color: "#0F172A",
};

const whiteLabelStyle = { color: "#334155" };

const whiteSelectStyle = {
  ...whiteInputStyle,
  height: 36,
  borderRadius: 8,
  padding: "0 10px",
};

const ageOptions = [
  { label: "10대", value: 15 },
  { label: "20대", value: 25 },
  { label: "30대", value: 35 },
  { label: "40대", value: 45 },
  { label: "50대", value: 55 },
  { label: "60대", value: 65 },
  { label: "70대 이상", value: 75 },
];

const jobOptions = [
  "사무직",
  "전문직",
  "기술직",
  "서비스직",
  "영업직",
  "자영업",
  "공무원",
  "학생",
  "주부",
  "프리랜서",
  "무직",
  "기타",
];

function parseNumber(value: string): number | undefined {
  const normalized = value.replace(/,/g, "").trim();
  if (!normalized) return undefined;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function normalizeAgeToDecadeValue(value: number | undefined): string {
  if (value == null || !Number.isFinite(value)) return "";
  if (value < 20) return "15";
  if (value < 30) return "25";
  if (value < 40) return "35";
  if (value < 50) return "45";
  if (value < 60) return "55";
  if (value < 70) return "65";
  return "75";
}

function wonToManwon(value: number | undefined): string {
  if (value == null || !Number.isFinite(value)) return "";
  return String(Math.round(value / 10_000));
}

function manwonToWon(value: string): number | undefined {
  const parsed = parseNumber(value);
  return parsed == null ? undefined : parsed * 10_000;
}

function cleanProfile(profile: UserProfile): UserProfile {
  const cleaned: UserProfile = {};
  if (profile.gender) cleaned.gender = profile.gender;
  if (profile.age != null) cleaned.age = profile.age;
  if (profile.monthly_income != null) cleaned.monthly_income = profile.monthly_income;
  if (profile.job) cleaned.job = profile.job;
  if (profile.marital_status) cleaned.marital_status = profile.marital_status;
  if (profile.consumption_type) cleaned.consumption_type = profile.consumption_type;
  return cleaned;
}

export function loadSavedUserProfile(): UserProfile {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(USER_PROFILE_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return {};
    return cleanProfile({
      gender: typeof parsed.gender === "string" ? parsed.gender : undefined,
      age: typeof parsed.age === "number" ? parsed.age : parseNumber(String(parsed.age ?? "")),
      monthly_income:
        typeof parsed.monthly_income === "number"
          ? parsed.monthly_income
          : parseNumber(String(parsed.monthly_income ?? "")),
      job: typeof parsed.job === "string" ? parsed.job : undefined,
      marital_status: typeof parsed.marital_status === "string" ? parsed.marital_status : undefined,
      consumption_type: typeof parsed.consumption_type === "string" ? parsed.consumption_type : undefined,
    });
  } catch {
    return {};
  }
}

function saveUserProfile(profile: UserProfile) {
  window.localStorage.setItem(USER_PROFILE_STORAGE_KEY, JSON.stringify(cleanProfile(profile)));
}

interface UserProfileSettingsProps {
  open: boolean;
  profile: UserProfile;
  onOpenChange: (open: boolean) => void;
  onSave: (profile: UserProfile) => void;
}

export function UserProfileSettings({
  open,
  profile,
  onOpenChange,
  onSave,
}: UserProfileSettingsProps) {
  const [gender, setGender] = useState(profile.gender || "");
  const [age, setAge] = useState(normalizeAgeToDecadeValue(profile.age));
  const [monthlyIncomeManwon, setMonthlyIncomeManwon] = useState(wonToManwon(profile.monthly_income));
  const [job, setJob] = useState(profile.job || "");
  const [maritalStatus, setMaritalStatus] = useState(profile.marital_status || "");
  const [consumptionType, setConsumptionType] = useState(profile.consumption_type || "");

  useEffect(() => {
    if (!open) return;
    setGender(profile.gender || "");
    setAge(normalizeAgeToDecadeValue(profile.age));
    setMonthlyIncomeManwon(wonToManwon(profile.monthly_income));
    setJob(profile.job || "");
    setMaritalStatus(profile.marital_status || "");
    setConsumptionType(profile.consumption_type || "");
  }, [open, profile]);

  const handleSave = () => {
    const nextProfile = cleanProfile({
      gender: gender || undefined,
      age: parseNumber(age),
      monthly_income: manwonToWon(monthlyIncomeManwon),
      job: job || undefined,
      marital_status: maritalStatus || undefined,
      consumption_type: consumptionType || undefined,
    });
    saveUserProfile(nextProfile);
    onSave(nextProfile);
    onOpenChange(false);
  };

  const handleClear = () => {
    window.localStorage.removeItem(USER_PROFILE_STORAGE_KEY);
    onSave({});
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="sm:max-w-xl"
        style={{
          background: "#FFFFFF",
          border: "1px solid #CBD5E1",
          color: "#0F172A",
          boxShadow: "0 24px 80px rgba(15, 23, 42, 0.22)",
        }}
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2" style={{ color: "#0F172A" }}>
            <UserRound size={18} />
            내정보설정
          </DialogTitle>
          <DialogDescription style={{ color: "#64748B" }}>
            입력하신 정보는 구매후회예측을 위한 입력정보로 사용됩니다.
          </DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <Label className="text-xs mb-1.5 block" style={whiteLabelStyle}>성별</Label>
            <select value={gender} onChange={(event) => setGender(event.target.value)} className="w-full text-sm" style={whiteSelectStyle}>
              <option value="">선택 안 함</option>
              <option value="male">남성</option>
              <option value="female">여성</option>
            </select>
          </div>
          <div>
            <Label className="text-xs mb-1.5 block" style={whiteLabelStyle}>나이대</Label>
            <select value={age} onChange={(event) => setAge(event.target.value)} className="w-full text-sm" style={whiteSelectStyle}>
              <option value="">선택 안 함</option>
              {ageOptions.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </div>
          <div>
            <Label className="text-xs mb-1.5 block" style={whiteLabelStyle}>월수입</Label>
            <div className="flex items-center gap-2">
              <Input value={monthlyIncomeManwon} onChange={(event) => setMonthlyIncomeManwon(event.target.value)} placeholder="예: 379" type="number" min="0" step={50} className="text-sm" style={whiteInputStyle} />
              <span className="text-sm shrink-0" style={{ color: "#475569" }}>만원</span>
            </div>
          </div>
          <div>
            <Label className="text-xs mb-1.5 block" style={whiteLabelStyle}>직업</Label>
            <select value={job} onChange={(event) => setJob(event.target.value)} className="w-full text-sm" style={whiteSelectStyle}>
              <option value="">선택 안 함</option>
              {jobOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
          <div>
            <Label className="text-xs mb-1.5 block" style={whiteLabelStyle}>결혼여부</Label>
            <select value={maritalStatus} onChange={(event) => setMaritalStatus(event.target.value)} className="w-full text-sm" style={whiteSelectStyle}>
              <option value="">선택 안 함</option>
              <option value="미혼">미혼</option>
              <option value="기혼">기혼</option>
            </select>
          </div>
          <div>
            <Label className="text-xs mb-1.5 block" style={whiteLabelStyle}>소비성향</Label>
            <select value={consumptionType} onChange={(event) => setConsumptionType(event.target.value)} className="w-full text-sm" style={whiteSelectStyle}>
              <option value="">선택 안 함</option>
              <option value="보수적">보수적</option>
              <option value="균형형">균형형</option>
              <option value="충동형">충동형</option>
              <option value="가성비형">가성비형</option>
              <option value="프리미엄형">프리미엄형</option>
            </select>
          </div>
        </div>

        <DialogFooter>
          <Button type="button" variant="ghost" onClick={handleClear} style={{ color: "#475569" }}>
            초기화
          </Button>
          <Button type="button" onClick={handleSave} style={{ background: "#2563EB", color: "#FFFFFF" }}>
            저장
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
