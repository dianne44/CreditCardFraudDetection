import React from "react";
import { QubitState } from "../types";

interface BlochSphereProps {
  qubit: QubitState;
  theme?: "dark" | "light";
}

export const BlochSphere: React.FC<BlochSphereProps> = ({ qubit, theme = "light" }) => {
  const isDark = theme !== "light";
  // Map x, y, z to SVG coordinates
  // Center is (50, 50), radius is 32
  const center = 50;
  const r = 32;

  // We project X onto the horizontal axis, and Z onto the vertical axis (inverted for screen Y)
  const tipX = center + qubit.x * r;
  const tipY = center - qubit.z * r; // Negative Z points upwards in physics, which is smaller Y in SVG

  // Color coordinate based on state
  // If z is positive (closer to |0>), it's green-slate. If z is negative (closer to |1>), it's magenta-red.
  const isHighRisk = qubit.z < -0.3;
  const colorClass = isHighRisk 
    ? "stroke-rose-500 fill-rose-500/10 shadow-rose-900" 
    : "stroke-emerald-500 fill-emerald-500/5 shadow-emerald-900";

  return (
    <div className={`relative flex flex-col items-center justify-between p-3.5 rounded-xl border transition-all duration-300 shadow-sm ${
      isDark 
        ? "border-white/5 bg-[#111114] hover:bg-[#111114]/80 hover:border-white/10" 
        : "border-gray-200 bg-white hover:bg-gray-50/50 hover:border-gray-300"
    }`} id={`qubit-card-${qubit.id}`}>
      <div className={`flex w-full justify-between items-center text-[10px] font-mono mb-1 ${
        isDark ? "text-gray-400" : "text-gray-500"
      }`}>
        <span>Q<sub>{qubit.id}</sub>: {qubit.feature}</span>
        <span className={isHighRisk ? "text-rose-400 font-bold" : "text-emerald-400"}>
          {isHighRisk ? "Risk" : "Stable"}
        </span>
      </div>

      {/* SVG Bloch Sphere Visualizer */}
      <div className="relative w-24 h-24 my-2">
        <svg className="w-full h-full" viewBox="0 0 100 100">
          {/* Background Sphere Ring */}
          <circle 
            cx={center} 
            cy={center} 
            r={r} 
            className={`${isDark ? "stroke-white/10" : "stroke-gray-200"} stroke-1 fill-none`} 
            strokeDasharray="2 2"
          />
          {/* Ellipse representing 3D perspective of the equator */}
          <ellipse 
            cx={center} 
            cy={center} 
            rx={r} 
            ry={r * 0.3} 
            className={`${isDark ? "stroke-white/5" : "stroke-gray-150"} stroke-1 fill-none`} 
            strokeDasharray="4 4"
          />
          {/* Y-axis representation (diagonal angle) */}
          <line 
            x1={center - r * 0.7} 
            y1={center + r * 0.2} 
            x2={center + r * 0.7} 
            y2={center - r * 0.2} 
            className={`${isDark ? "stroke-white/5" : "stroke-gray-150"} stroke-1 stroke-dash`}
            strokeDasharray="2 4"
          />

          {/* Core coordinate lines */}
          <line x1={center - r} y1={center} x2={center + r} y2={center} className={`${isDark ? "stroke-white/10" : "stroke-gray-200"} stroke-1`} />
          <line x1={center} y1={center - r} x2={center} y2={center + r} className={`${isDark ? "stroke-white/10" : "stroke-gray-200"} stroke-1`} />

          {/* Labels for basis states */}
          <text x={center} y={center - r - 3} className={`${isDark ? "fill-gray-500" : "fill-gray-600"} font-mono text-[7px]`} textAnchor="middle">|0⟩</text>
          <text x={center} y={center + r + 8} className={`${isDark ? "fill-gray-500" : "fill-gray-600"} font-mono text-[7px]`} textAnchor="middle">|1⟩</text>
          <text x={center + r + 3} y={center + 2.5} className={`${isDark ? "fill-gray-650" : "fill-gray-500"} font-mono text-[6px]`} textAnchor="start">|+⟩</text>
          <text x={center - r - 8} y={center + 2.5} className={`${isDark ? "fill-gray-650" : "fill-gray-500"} font-mono text-[6px]`} textAnchor="start">|-⟩</text>

          {/* Glow effect filter */}
          <defs>
            <filter id={`glow-${qubit.id}`} x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* The State Vector arrow */}
          <line 
            x1={center} 
            y1={center} 
            x2={tipX} 
            y2={tipY} 
            className={`${isHighRisk ? "stroke-rose-400" : "stroke-emerald-400"} stroke-[2]`}
            markerEnd="url(#arrow)"
            filter={`url(#glow-${qubit.id})`}
          />

          {/* Vector Tip Point */}
          <circle 
            cx={tipX} 
            cy={tipY} 
            r="3" 
            className={isHighRisk ? "fill-rose-300" : "fill-emerald-300"} 
          />
          <circle 
            cx={center} 
            cy={center} 
            r="2" 
            className="fill-gray-600" 
          />
          
          {/* Arrowhead marker definition */}
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" className={isHighRisk ? "fill-rose-400" : "fill-emerald-400"} />
            </marker>
          </defs>
        </svg>
      </div>

      {/* Coordinate values */}
      <div className={`w-full mt-2 font-mono text-[9px] flex justify-between items-center border-t pt-1 ${
        isDark ? "text-gray-500 border-white/5" : "text-gray-650 border-gray-150"
      }`}>
        <div>
          <span>X: <strong className={isDark ? "text-gray-400" : "text-gray-900"}>{qubit.x >= 0 ? "+" : ""}{qubit.x.toFixed(2)}</strong></span>
        </div>
        <div>
          <span>Z: <strong className={isDark ? "text-gray-400" : "text-gray-900"}>{qubit.z >= 0 ? "+" : ""}{qubit.z.toFixed(2)}</strong></span>
        </div>
      </div>

      {/* Superposition split */}
      <div className={`w-full mt-1.5 flex gap-1 items-center h-2.5 rounded overflow-hidden border ${
        isDark ? "bg-[#0c0c0e] border-white/5" : "bg-gray-100 border-gray-200"
      }`}>
        <div 
          className="h-full bg-emerald-500/80 transition-all duration-300" 
          style={{ width: `${qubit.superposition0 * 100}%` }}
          title={`|0⟩ probability: ${(qubit.superposition0 * 100).toFixed(0)}%`}
        />
        <div 
          className="h-full bg-rose-500/80 transition-all duration-300" 
          style={{ width: `${qubit.superposition1 * 100}%` }}
          title={`|1⟩ probability: ${(qubit.superposition1 * 100).toFixed(0)}%`}
        />
      </div>
      <div className={`w-full flex justify-between text-[8px] font-mono mt-0.5 ${
        isDark ? "text-gray-500" : "text-gray-600"
      }`}>
        <span>P(|0⟩): {(qubit.superposition0 * 100).toFixed(0)}%</span>
        <span>P(|1⟩): {(qubit.superposition1 * 100).toFixed(0)}%</span>
      </div>
    </div>
  );
};

