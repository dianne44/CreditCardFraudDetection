import React from "react";
import { TransactionInputs } from "../types";
import { Coffee, ShieldCheck, ShoppingBag, Terminal } from "lucide-react";

interface PresetProfile {
  name: string;
  description: string;
  icon: React.ReactNode;
  risk: "LOW" | "MEDIUM" | "HIGH";
  values: TransactionInputs;
}

export const PRESET_PROFILES: PresetProfile[] = [
  {
    name: "Legitimate Coffee Purchase",
    description: "Typical low-value retail checkout with standard normal PCA signals.",
    icon: <Coffee className="w-4 h-4 text-emerald-400" />,
    risk: "LOW",
    values: {
      Time: 52200, // 2:30 PM
      Amount: 8.50,
      V14: 0.35,
      V4: -0.22,
      V10: 0.18,
      V12: 0.29,
      V17: 0.12,
      V11: -0.45,
      V8: 0.05,
      V20: -0.08
    }
  },
  {
    name: " Skimmed Card Splurge (Critical Risk)",
    description: "High-amount, nocturnal checkout showing severe negative V14/V17 anomalies.",
    icon: <Terminal className="w-4 h-4 text-rose-400 animate-pulse" />,
    risk: "HIGH",
    values: {
      Time: 11700, // 3:15 AM
      Amount: 1850.00,
      V14: -8.50,
      V4: 5.80,
      V10: -6.20,
      V12: -7.10,
      V17: -9.40,
      V11: 4.20,
      V8: -3.50,
      V20: 1.80
    }
  },
  {
    name: "Suspicious ElectronicsSplurge",
    description: "Moderate anomalies across multiple core PCA vectors at late night.",
    icon: <ShoppingBag className="w-4 h-4 text-amber-400" />,
    risk: "MEDIUM",
    values: {
      Time: 84600, // 11:30 PM
      Amount: 820.00,
      V14: -3.80,
      V4: 2.10,
      V10: -2.90,
      V12: -3.10,
      V17: -3.40,
      V11: 1.80,
      V8: 0.50,
      V20: 0.45
    }
  },
  {
    name: "Stable Corporate Whitelist",
    description: "Extremely high value order with robust normal-range PCA signatures.",
    icon: <ShieldCheck className="w-4 h-4 text-blue-400" />,
    risk: "LOW",
    values: {
      Time: 36900, // 10:15 AM
      Amount: 4950.00,
      V14: 0.15,
      V4: -0.10,
      V10: 0.05,
      V12: 0.12,
      V17: 0.08,
      V11: -0.15,
      V8: -0.05,
      V20: -0.02
    }
  }
];

interface TransactionPresetsProps {
  onSelect: (values: TransactionInputs) => void;
  selectedName?: string;
  theme?: "dark" | "light";
}

export const TransactionPresets: React.FC<TransactionPresetsProps> = ({ onSelect, selectedName, theme = "light" }) => {
  const isDark = theme !== "light";
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3 w-full mb-6">
      {PRESET_PROFILES.map((profile) => {
        const isSelected = selectedName === profile.name;
        const riskColors = {
          LOW: isDark 
            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" 
            : "bg-emerald-50 text-emerald-700 border-emerald-200",
          MEDIUM: isDark 
            ? "bg-amber-500/10 text-amber-400 border-amber-500/20" 
            : "bg-amber-50 text-amber-700 border-amber-200",
          HIGH: isDark 
            ? "bg-rose-500/10 text-rose-400 border-rose-500/20 animate-pulse" 
            : "bg-rose-50 text-rose-700 border-rose-200 animate-pulse"
        };

        return (
          <button
            key={profile.name}
            type="button"
            onClick={() => onSelect(profile.values)}
            className={`flex flex-col items-start p-4 rounded-xl border text-left transition-all duration-300 ${
              isSelected 
                ? isDark
                  ? "bg-[#1c1c20] border-cyan-500/40 ring-1 ring-cyan-500/20 shadow-[0_0_15px_rgba(6,182,212,0.15)]" 
                  : "bg-cyan-50/50 border-cyan-500/40 ring-1 ring-cyan-500/20 shadow-md shadow-cyan-100"
                : isDark
                  ? "bg-[#111114] border-white/5 hover:border-white/10 hover:bg-[#111114]/80"
                  : "bg-white border-gray-200 hover:border-gray-300 hover:bg-gray-50/50"
            }`}
            id={`preset-btn-${profile.name.replace(/\s+/g, "-").toLowerCase()}`}
          >
            <div className="flex w-full justify-between items-start mb-2">
              <div className={`p-1.5 rounded-lg border shadow-inner ${
                isDark ? "bg-[#1c1c20] border-white/5" : "bg-gray-50 border-gray-200"
              }`}>
                {profile.icon}
              </div>
              <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${riskColors[profile.risk]}`}>
                {profile.risk} Risk
              </span>
            </div>
            <h4 className={`text-xs font-semibold font-sans tracking-wide mb-1 leading-tight ${
              isDark ? "text-zinc-100" : "text-gray-900"
            }`}>
              {profile.name}
            </h4>
            <p className={`text-[10.5px] font-sans leading-relaxed ${
              isDark ? "text-zinc-500" : "text-gray-500"
            }`}>
              {profile.description}
            </p>
          </button>
        );
      })}
    </div>
  );
};
