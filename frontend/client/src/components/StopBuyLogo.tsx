import { Ban, ShoppingCart } from "lucide-react";

interface StopBuyLogoProps {
  size?: number;
  cartSize?: number;
  isDark?: boolean;
}

export function StopBuyLogo({ size = 32, cartSize = 18, isDark = true }: StopBuyLogoProps) {
  return (
    <div
      className="relative rounded-lg flex items-center justify-center shrink-0"
      style={{
        width: size,
        height: size,
        background: "linear-gradient(135deg, var(--sb-blue), var(--sb-blue-dark))",
      }}
      aria-label="StopBuy"
    >
      <Ban
        size={Math.round(size * 0.84)}
        strokeWidth={2.15}
        className="absolute left-1/2 top-1/2"
        style={{
          color: "var(--sb-red)",
          opacity: 0.76,
          transform: "translate(-50%, -50%)",
        }}
      />
      <ShoppingCart
        size={cartSize}
        strokeWidth={2.6}
        className="relative z-10"
        style={{
          color: isDark ? "#0A0E1A" : "#fff",
          filter: "drop-shadow(0 1px 1px rgba(255,255,255,0.28))",
        }}
      />
    </div>
  );
}



