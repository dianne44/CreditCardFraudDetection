import React from "react";
import { Shield, Sparkles, Zap, Award } from "lucide-react";

interface MetricsPanelProps {
  theme?: "dark" | "light";
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ theme = "light" }) => {
  const isDark = theme !== "light";
  // Metrics from Table 17 of the paper (Page 21)
  const models = [
    { name: "Logistic Regression", type: "Classical", accuracy: 95.00, precision: 96.55, recall: 93.33, f1: 94.92, specificity: 96.67, status: "Baseline" },
    { name: "Quantum SVM (XGBoost)", type: "Quantum-Classical", accuracy: 91.67, precision: 87.88, recall: 96.67, f1: 92.06, specificity: 86.67, status: "Very High Recall" },
    { name: "Quantum KNN (QPCA)", type: "Quantum-Classical", accuracy: 99.99, precision: 0.00, recall: 0.00, f1: 0.00, specificity: 100.00, status: "Biased" },
    { name: "XGBoost-QNN (Our Model)", type: "Quantum Neural Net", accuracy: 89.98, precision: 97.00, recall: 82.00, f1: 89.00, specificity: 97.00, status: "Best Balanced QML" },
    { name: "Quantum Deep Q-Network", type: "Reinforced QML", accuracy: 49.98, precision: 49.98, recall: 49.93, f1: 49.95, specificity: 50.02, status: "Theoretical" },
    { name: "Quantum Q-Learning", type: "Reinforced QML", accuracy: 50.03, precision: 50.03, recall: 50.13, f1: 50.08, specificity: 49.93, status: "Theoretical" }
  ];

  return (
    <div className="space-y-6 animate-fade-in" id="metrics-panel-root">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        {[
          { label: "Testing Accuracy", value: "89.83%", color: isDark ? "text-white" : "text-gray-900", icon: <Award className="w-4 h-4 text-cyan-500" />, desc: "Overall correct rate" },
          { label: "Precision Score", value: "0.97", color: "text-emerald-500", icon: <Shield className="w-4 h-4 text-emerald-500" />, desc: "When flagged, true fraud" },
          { label: "Recall Rate", value: "0.82", color: "text-cyan-500", icon: <Zap className="w-4 h-4 text-cyan-500" />, desc: "Fraud instances caught" },
          { label: "F1 Optimality", value: "0.89", color: "text-cyan-600", icon: <Sparkles className="w-4 h-4 text-cyan-600" />, desc: "Harmonic balance of scores" },
          { label: "Specificity", value: "0.97", color: "text-cyan-700", icon: <Shield className="w-4 h-4 text-cyan-700" />, desc: "Legitimate clearance rate" }
        ].map((stat, i) => (
          <div key={stat.label} className={`p-5 rounded-xl border flex flex-col justify-between shadow-sm transition-all duration-300 ${
            isDark ? "bg-[#111114] border-white/5 hover:border-white/10" : "bg-white border-gray-200 hover:border-gray-350 hover:shadow-md"
          }`} id={`stat-box-${i}`}>
            <div className="flex w-full justify-between items-center mb-1">
              <span className={`text-[11px] font-mono tracking-wider font-bold uppercase ${
                isDark ? "text-gray-500" : "text-gray-400"
              }`}>{stat.label}</span>
              {stat.icon}
            </div>
            <div className="my-2">
              <span className={`text-3xl font-light font-mono tracking-tight ${stat.color}`}>{stat.value}</span>
            </div>
            <p className={`text-[10px] font-sans leading-tight italic ${
              isDark ? "text-gray-400" : "text-gray-500"
            }`}>
              {stat.desc}
            </p>
          </div>
        ))}
      </div>

      {/* Main Content Layout */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        
        {/* Table representation */}
        <div className={`p-5 rounded-xl border shadow-sm ${
          isDark ? "border-white/5 bg-[#111114]" : "border-gray-200 bg-white"
        }`} id="comparison-table-card">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h3 className={`text-xs font-bold uppercase tracking-widest ${
                isDark ? "text-gray-400" : "text-gray-700"
              }`}>
                Literature Performance Matrix
              </h3>
              <p className={`text-[11px] font-sans mt-0.5 ${
                isDark ? "text-gray-500" : "text-gray-450"
              }`}>
                Comparative analysis of different architectures trained on credit_card_fraud (Hugging Face).
              </p>
            </div>
          </div>

          <div className={`overflow-x-auto border rounded-xl ${
            isDark ? "border-white/5 bg-[#0c0c0e]" : "border-gray-200 bg-gray-50/50"
          }`}>
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className={`border-b font-mono text-[10px] ${
                  isDark ? "border-white/5 bg-[#111114] text-gray-400" : "border-gray-200 bg-gray-100 text-gray-600"
                }`}>
                  <th className="py-3 px-4 font-bold uppercase tracking-wider">Model Architecture</th>
                  <th className="py-3 px-4 font-bold uppercase tracking-wider">Type</th>
                  <th className="py-3 px-4 font-bold uppercase tracking-wider text-right">Accuracy</th>
                  <th className="py-3 px-4 font-bold uppercase tracking-wider text-right">Precision</th>
                  <th className="py-3 px-4 font-bold uppercase tracking-wider text-right">Recall</th>
                  <th className="py-3 px-4 font-bold uppercase tracking-wider text-right">F1-Score</th>
                  <th className="py-3 px-4 font-bold uppercase tracking-wider text-right">Specificity</th>
                </tr>
              </thead>
              <tbody className={`divide-y font-mono text-[11px] ${
                isDark ? "divide-white/5" : "divide-gray-200"
              }`}>
                {models.map((m, index) => {
                  const isOurModel = m.name.includes("XGBoost-QNN");
                  return (
                    <tr 
                      key={m.name} 
                      className={`transition-colors duration-200 ${
                        isOurModel 
                          ? isDark 
                            ? "bg-[#1c1c20] text-cyan-400 font-semibold" 
                            : "bg-cyan-50/50 text-cyan-700 font-semibold"
                          : isDark
                            ? "text-gray-400 hover:bg-white/5"
                            : "text-gray-700 hover:bg-gray-50"
                      }`}
                      id={`model-row-${index}`}
                    >
                      <td className={`py-3 px-4 font-sans font-medium flex items-center gap-1.5 ${
                        isOurModel ? (isDark ? "text-white" : "text-cyan-800") : (isDark ? "text-white" : "text-gray-900")
                      }`}>
                        {isOurModel && <Sparkles className="w-3.5 h-3.5 text-cyan-500 animate-pulse" />}
                        {m.name}
                      </td>
                      <td className={`py-3 px-4 font-sans text-[10px] ${
                        isDark ? "text-gray-500" : "text-gray-500"
                      }`}>{m.type}</td>
                      <td className={`py-3 px-4 text-right ${isDark ? "text-gray-200" : "text-gray-900"}`}>{(m.accuracy * 100).toFixed(2)}%</td>
                      <td className="py-3 px-4 text-right">{m.precision.toFixed(3)}</td>
                      <td className="py-3 px-4 text-right">{m.recall.toFixed(3)}</td>
                      <td className={`py-3 px-4 text-right font-medium ${isDark ? "text-white" : "text-gray-900"}`}>{m.f1.toFixed(3)}</td>
                      <td className="py-3 px-4 text-right">{m.specificity.toFixed(3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          
          <div className={`mt-4 p-4 border rounded-lg text-[11px] leading-relaxed font-sans ${
            isDark ? "bg-cyan-900/10 border-cyan-500/20 text-cyan-200/90" : "bg-cyan-50 border-cyan-200 text-cyan-900"
          }`}>
            <strong className={`font-bold font-mono uppercase tracking-wider text-[10px] block mb-1 ${
              isDark ? "text-cyan-400" : "text-cyan-700"
            }`}>Architecture Synergy:</strong>
            While Classical Logistic Regression achieves slightly higher accuracy, it fails to capture complex non-linear entanglements. The hybrid <strong className={isDark ? "text-white" : "text-cyan-950 font-bold"}>XGBoost-Quantum Neural Network (XGB-QNN)</strong> achieves excellent <strong className={isDark ? "text-white" : "text-cyan-950 font-bold"}>97% Precision</strong> and high <strong className={isDark ? "text-white" : "text-cyan-950 font-bold"}>97% Specificity</strong>, proving that PQC (Parameterized Quantum Circuits) with Angle Embedding can significantly reduce expensive False Positive rates in enterprise fintech pipelines.
          </div>
        </div>

        {/* Confusion Matrix Card */}
        <div className={`p-5 rounded-xl border flex flex-col justify-between shadow-sm ${
          isDark ? "border-white/5 bg-[#111114]" : "border-gray-200 bg-white"
        }`} id="confusion-matrix-card">
          <div>
            <h3 className={`text-xs font-bold uppercase tracking-widest ${
              isDark ? "text-gray-400" : "text-gray-700"
            }`}>
              QNN-XGB Confusion Matrix
            </h3>
            <p className={`text-[11px] font-sans mt-0.5 mb-4 ${
              isDark ? "text-gray-500" : "text-gray-450"
            }`}>
              Actual validation sample partition (Figure 23, Page 17).
            </p>
          </div>

          {/* Matrix Grid representation */}
          <div className={`relative flex flex-col items-center justify-center p-4 rounded-xl border ${
            isDark ? "border-white/5 bg-[#0c0c0e]" : "border-gray-200 bg-gray-50"
          }`}>
            {/* Top Row: Labels */}
            <div className={`grid grid-cols-3 w-full text-center text-[10px] font-mono mb-2 ${
              isDark ? "text-gray-500" : "text-gray-450"
            }`}>
              <div />
              <div className="uppercase tracking-widest font-bold">Predicted Legit</div>
              <div className="uppercase tracking-widest font-bold">Predicted Fraud</div>
            </div>

            {/* Matrix Cells */}
            <div className="grid grid-cols-12 w-full gap-2 items-center">
              {/* Row 1 Header */}
              <div className={`col-span-2 text-[10px] font-mono uppercase tracking-widest font-bold text-right pr-2 ${
                isDark ? "text-gray-500" : "text-gray-600"
              }`}>
                Actual Legit
              </div>

              {/* True Negative Cell */}
              <div className={`col-span-5 p-4 rounded-lg border flex flex-col items-center justify-center transition-all group ${
                isDark ? "border-white/5 bg-[#111114] hover:border-emerald-500/30" : "border-gray-250 bg-white hover:border-emerald-500"
              }`} id="cell-tn">
                <span className="text-3xl font-light font-mono text-emerald-500 group-hover:scale-105 transition-transform">4</span>
                <span className={`text-[9px] font-sans uppercase mt-1 italic ${isDark ? "text-gray-400" : "text-gray-600"}`}>True Negative</span>
                <span className="text-[8px] font-mono text-emerald-500/60 mt-0.5">80% Accuracy segment</span>
              </div>

              {/* False Positive Cell */}
              <div className={`col-span-5 p-4 rounded-lg border flex flex-col items-center justify-center transition-all group ${
                isDark ? "border-white/5 bg-[#111114] hover:border-rose-500/30" : "border-gray-250 bg-white hover:border-rose-450"
              }`} id="cell-fp">
                <span className="text-3xl font-light font-mono text-rose-500 group-hover:scale-105 transition-transform">1</span>
                <span className={`text-[9px] font-sans uppercase mt-1 italic ${isDark ? "text-gray-400" : "text-gray-600"}`}>False Positive</span>
                <span className="text-[8px] font-mono text-rose-500/50 mt-0.5">10% Misclass segment</span>
              </div>

              {/* Row 2 Header */}
              <div className={`col-span-2 text-[10px] font-mono uppercase tracking-widest font-bold text-right pr-2 mt-2 ${
                isDark ? "text-gray-500" : "text-gray-650"
              }`}>
                Actual Fraud
              </div>

              {/* False Negative Cell */}
              <div className={`col-span-5 p-4 rounded-lg border flex flex-col items-center justify-center transition-all group ${
                isDark ? "border-white/5 bg-[#111114] hover:border-rose-500/30" : "border-gray-250 bg-white hover:border-rose-450"
              }`} id="cell-fn">
                <span className="text-3xl font-light font-mono text-rose-500 group-hover:scale-105 transition-transform">1</span>
                <span className={`text-[9px] font-sans uppercase mt-1 italic ${isDark ? "text-gray-400" : "text-gray-600"}`}>False Negative</span>
                <span className="text-[8px] font-mono text-rose-500/50 mt-0.5">10% Misclass segment</span>
              </div>

              {/* True Positive Cell */}
              <div className={`col-span-5 p-4 rounded-lg border flex flex-col items-center justify-center transition-all group ${
                isDark ? "border-white/5 bg-[#111114] hover:border-emerald-500/30" : "border-gray-250 bg-white hover:border-emerald-500"
              }`} id="cell-tp">
                <span className="text-3xl font-light font-mono text-emerald-500 group-hover:scale-105 transition-transform">4</span>
                <span className={`text-[9px] font-sans uppercase mt-1 italic ${isDark ? "text-gray-400" : "text-gray-600"}`}>True Positive</span>
                <span className="text-[8px] font-mono text-emerald-500/60 mt-0.5">80% Accuracy segment</span>
              </div>
            </div>
          </div>

          <div className={`mt-4 flex gap-1.5 justify-center items-center text-[10px] font-mono ${
            isDark ? "text-gray-500" : "text-gray-500"
          }`}>
            <span className="inline-block w-2.5 h-2.5 rounded bg-emerald-500" />
            <span>80.00% Correct Classification</span>
            <span className="inline-block w-2.5 h-2.5 rounded bg-gray-450 ml-3" />
            <span>20.00% Error Segment</span>
          </div>
        </div>

      </div>
    </div>
  );
};
