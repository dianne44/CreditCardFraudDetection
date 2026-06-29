import React, { useState, useEffect } from "react";
import { TransactionInputs, PredictionResult } from "./types";
import { BlochSphere } from "./components/BlochSphere";
import { TransactionPresets, PRESET_PROFILES } from "./components/TransactionPresets";
import { MetricsPanel } from "./components/MetricsPanel";
import { 
  ShieldAlert, 
  ShieldCheck, 
  Sparkles, 
  TrendingUp, 
  Cpu, 
  Zap, 
  Sliders, 
  RefreshCw, 
  FileText, 
  CheckCircle,
  HelpCircle,
  LineChart,
  Sun,
  Moon,
  ChevronDown,
  ChevronUp,
  Info,
  AlertTriangle
} from "lucide-react";

export default function App() {
  const [theme, setTheme] = useState<"dark" | "light">("light");
  const [tab, setTab] = useState<"dashboard" | "metrics">("dashboard");
  const [inputs, setInputs] = useState<TransactionInputs>(PRESET_PROFILES[0].values);
  const [selectedPresetName, setSelectedPresetName] = useState<string>(PRESET_PROFILES[0].name);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [selectedQubit, setSelectedQubit] = useState<number | null>(null);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  // Stepper state for animated QNN execution
  const [pipelineStep, setPipelineStep] = useState<"idle" | "preprocessing" | "embedding" | "entangling" | "classification" | "done">("idle");

  // Field validation and direct string tracking state (for seamless user typing)
  const [errors, setErrors] = useState<{ [key: string]: string }>({});
  const [inputStrings, setInputStrings] = useState<{ [key: string]: string }>(() => {
    const initialStrings: { [key: string]: string } = {};
    Object.keys(PRESET_PROFILES[0].values).forEach(k => {
      initialStrings[k] = PRESET_PROFILES[0].values[k as keyof TransactionInputs].toString();
    });
    return initialStrings;
  });

  // Run default transaction on load
  useEffect(() => {
    handleRunAnalysis(PRESET_PROFILES[0].values);
  }, []);

  const validateField = (key: string, valStr: string): string => {
    const trimmed = valStr.trim();
    if (trimmed === "") return "Field cannot be empty";
    
    const num = parseFloat(trimmed);
    if (isNaN(num)) return "Must be a valid numeric value";

    if (key === "Amount") {
      if (num < 0.01) return "Amount must be at least $0.01";
      if (num > 5000.00) return "Amount cannot exceed $5,000.00 (model ceiling)";
    } else if (key === "Time") {
      const intNum = parseInt(trimmed, 10);
      if (isNaN(intNum) || intNum !== num) return "Time must be a whole integer";
      if (intNum < 0) return "Time cannot be negative";
      if (intNum > 172800) return "Time cannot exceed 172,800s (48 hours)";
    } else {
      if (num < -25.0 || num > 25.0) return `${key} score must reside in range [-25, 25]`;
    }
    return "";
  };

  const handlePresetSelect = (values: TransactionInputs) => {
    setInputs(values);
    
    // Sync text input boxes with the preset values
    const newStrings: { [key: string]: string } = {};
    Object.keys(values).forEach(k => {
      newStrings[k] = values[k as keyof TransactionInputs].toString();
    });
    setInputStrings(newStrings);
    setErrors({}); // reset validation state
    
    // Find matching preset name
    const preset = PRESET_PROFILES.find(p => 
      Object.keys(values).every(k => values[k as keyof TransactionInputs] === p.values[k as keyof TransactionInputs])
    );
    if (preset) {
      setSelectedPresetName(preset.name);
    } else {
      setSelectedPresetName("");
    }
  };

  const handleTextChange = (key: keyof TransactionInputs, strVal: string) => {
    setInputStrings(prev => ({
      ...prev,
      [key]: strVal
    }));
    setSelectedPresetName(""); // Clear preset tag

    const errMsg = validateField(key, strVal);
    setErrors(prev => ({
      ...prev,
      [key]: errMsg
    }));

    if (!errMsg) {
      const parsed = key === "Time" ? parseInt(strVal, 10) : parseFloat(strVal);
      setInputs(prev => ({
        ...prev,
        [key]: parsed
      }));
    }
  };

  const handleSliderChange = (key: keyof TransactionInputs, val: number) => {
    setInputs(prev => ({
      ...prev,
      [key]: val
    }));
    setInputStrings(prev => ({
      ...prev,
      [key]: val.toString()
    }));
    setSelectedPresetName("");

    // Clear error for this field since sliders are guaranteed within bounds
    setErrors(prev => ({
      ...prev,
      [key]: ""
    }));
  };

  const handleRandomize = () => {
    const randomVals: TransactionInputs = {
      Time: Math.floor(Math.random() * 172800), // up to 48 hours
      Amount: parseFloat((Math.random() * 2500).toFixed(2)),
      V14: parseFloat((Math.random() * 10 - 5).toFixed(2)),
      V4: parseFloat((Math.random() * 8 - 3).toFixed(2)),
      V10: parseFloat((Math.random() * 10 - 5).toFixed(2)),
      V12: parseFloat((Math.random() * 10 - 5).toFixed(2)),
      V17: parseFloat((Math.random() * 12 - 6).toFixed(2)),
      V11: parseFloat((Math.random() * 8 - 3).toFixed(2)),
      V8: parseFloat((Math.random() * 8 - 4).toFixed(2)),
      V20: parseFloat((Math.random() * 4 - 2).toFixed(2))
    };
    handlePresetSelect(randomVals);
    setSelectedPresetName("Custom Randomized");
  };

  const handleRunAnalysis = async (targetInputs = inputs) => {
    // Block action if there are unresolved validation errors
    const hasActiveErrors = Object.values(errors).some(err => err !== "");
    if (hasActiveErrors) {
      alert("Please resolve input format errors on the form before detecting fraud.");
      return;
    }

    setLoading(true);
    setResult(null);
    
    // Simulate pipeline state machine transitions for visual fidelity
    setPipelineStep("preprocessing");
    await new Promise(r => setTimeout(r, 400));

    setPipelineStep("embedding");
    await new Promise(r => setTimeout(r, 450));

    setPipelineStep("entangling");
    await new Promise(r => setTimeout(r, 450));

    setPipelineStep("classification");
    await new Promise(r => setTimeout(r, 300));

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(targetInputs)
      });
      const data = await response.json();
      if (data.success) {
        setResult(data);
      } else {
        console.error("API error:", data.error);
      }
    } catch (err) {
      console.error("Network error running prediction:", err);
    } finally {
      setPipelineStep("done");
      setLoading(false);
    }
  };

  const isDark = theme === "dark";
  const hasErrors = Object.values(errors).some(err => err !== "");

  return (
    <div className={`min-h-screen font-sans flex flex-col transition-colors duration-300 ${
      isDark 
        ? "bg-[#09090b] text-gray-200 selection:bg-cyan-950 selection:text-white" 
        : "bg-slate-50 text-slate-800 selection:bg-cyan-100 selection:text-cyan-900"
    }`} id="app-root">
      
      {/* HEADER BAR */}
      <header className={`border-b sticky top-0 z-50 px-4 md:px-8 py-3 flex flex-col sm:flex-row justify-between items-center gap-4 backdrop-blur-md transition-colors duration-300 ${
        isDark 
          ? "border-white/10 bg-[#0c0c0e]/95 text-white" 
          : "border-slate-200 bg-white/95 text-slate-900 shadow-sm"
      }`} id="app-header">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-cyan-500 rounded flex items-center justify-center shadow-[0_0_15px_rgba(6,182,212,0.4)]">
            <span className="text-black font-bold text-xs font-mono">QNN</span>
          </div>
          <div>
            <h1 className="text-sm font-bold font-display uppercase tracking-wider flex items-center gap-1.5 leading-tight">
              Credit Card Fraud Detector 
              <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded border ${
                isDark ? "bg-[#1c1c20] text-cyan-400 border-cyan-500/20" : "bg-cyan-50 text-cyan-700 border-cyan-200"
              }`}>
                XGB-QNN v1.0
              </span>
            </h1>
            <p className="text-[10px] uppercase tracking-widest text-cyan-500 font-bold mt-0.5">
              Credit Card Fraud Detection Node
            </p>
          </div>
        </div>

        {/* Status Indicators, Mode Selector + Tabs */}
        <div className="flex flex-wrap items-center gap-3 sm:gap-4" id="header-controls">
          
          {/* Light/Dark Toggle */}
          <button
            type="button"
            onClick={() => setTheme(isDark ? "light" : "dark")}
            className={`p-2 rounded-lg border transition-all duration-200 flex items-center justify-center ${
              isDark 
                ? "bg-[#111114] border-white/10 text-yellow-400 hover:bg-[#1c1c20]" 
                : "bg-slate-100 border-slate-200 text-slate-700 hover:bg-slate-200 shadow-sm"
            }`}
            title={`Toggle ${isDark ? "Light" : "Dark"} Mode`}
            id="theme-toggle-btn"
          >
            {isDark ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
          </button>

          <div className={`flex p-1 rounded-lg border gap-1 transition-all duration-200 ${
            isDark ? "bg-[#111114] border-white/5" : "bg-slate-100 border-slate-200 shadow-inner"
          }`} id="navigation-tabs">
            <button
              type="button"
              onClick={() => setTab("dashboard")}
              className={`flex items-center gap-1 px-3 py-1.5 rounded-md text-[10.5px] font-mono transition-all duration-150 ${
                tab === "dashboard"
                  ? isDark 
                    ? "bg-white/5 text-cyan-400 border border-white/10" 
                    : "bg-white text-cyan-600 border border-slate-200 shadow-sm"
                  : isDark 
                    ? "text-gray-400 hover:text-white" 
                    : "text-slate-500 hover:text-slate-800"
              }`}
              id="tab-btn-dashboard"
            >
              <Sliders className="w-3 h-3" />
              Terminal
            </button>
            <button
              type="button"
              onClick={() => setTab("metrics")}
              className={`flex items-center gap-1 px-3 py-1.5 rounded-md text-[10.5px] font-mono transition-all duration-150 ${
                tab === "metrics"
                  ? isDark 
                    ? "bg-white/5 text-cyan-400 border border-white/10" 
                    : "bg-white text-cyan-600 border border-slate-200 shadow-sm"
                  : isDark 
                    ? "text-gray-400 hover:text-white" 
                    : "text-slate-500 hover:text-slate-800"
              }`}
              id="tab-btn-metrics"
            >
              <LineChart className="w-3 h-3" />
              Metrics
            </button>
          </div>
        </div>
      </header>

      {/* CORE FRAMEWORK STAGE */}
      <main className="flex-1 max-w-[1700px] w-full mx-auto px-4 md:px-8 py-6 flex flex-col" id="app-main">
        
        {tab === "dashboard" ? (
          <div className="flex flex-col flex-1" id="tab-dashboard-view">
            
            {/* 1. Preset Profiles Select Section */}
            <div className="mb-2">
              <h2 className={`text-[11px] font-bold uppercase tracking-widest mb-2.5 flex items-center gap-1.5 ${
                isDark ? "text-gray-500" : "text-slate-400"
              }`}>
                <Sparkles className="w-3.5 h-3.5 text-cyan-500" />
                Select Transaction Profile Preset
              </h2>
              <TransactionPresets 
                onSelect={handlePresetSelect} 
                selectedName={selectedPresetName} 
                theme={theme}
              />
            </div>

            {/* 2. Main Terminal Panel Split */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
              
              {/* LEFT COLUMN: Input Form with High-Fidelity Validation */}
              <div className={`xl:col-span-5 p-6 rounded-xl border transition-all duration-300 space-y-6 shadow-sm ${
                isDark ? "border-white/5 bg-[#111114]" : "border-slate-200 bg-white"
              }`} id="parameter-sliders-card">
                <div>
                  <h3 className={`text-xs font-bold uppercase tracking-widest flex items-center gap-1.5 ${
                    isDark ? "text-white" : "text-slate-800"
                  }`}>
                    <Sliders className="w-3.5 h-3.5 text-cyan-500" />
                    Transaction Fraud Parameter Form
                  </h3>
                  <p className={`text-[11px] font-sans mt-1 italic ${
                    isDark ? "text-gray-500" : "text-slate-500"
                  }`}>
                    Provide financial parameters. Instantly formats, bounds, and performs schema validation before deploying onto the quantum classifier.
                  </p>
                </div>

                <div className={`space-y-5 border-t border-b py-5 ${isDark ? "border-white/5" : "border-slate-100"}`}>
                  
                  {/* TRANSACTION AMOUNT FIELD */}
                  <div className="flex flex-col gap-1.5" id="form-field-amount">
                    <div className="flex justify-between items-center font-mono text-[11px]">
                      <span className={`font-semibold flex items-center gap-1 ${isDark ? "text-gray-300" : "text-slate-700"}`}>
                        1. Transaction Amount
                        <span className={`text-[8.5px] px-1.5 py-0.5 rounded font-mono ${
                          isDark ? "bg-[#1c1c20] text-cyan-400" : "bg-slate-100 text-cyan-700"
                        }`}>USD</span>
                      </span>
                      {errors.Amount ? (
                        <span className="text-rose-500 flex items-center gap-1 text-[10px] font-bold">
                          <AlertTriangle className="w-3 h-3" /> {errors.Amount}
                        </span>
                      ) : (
                        <span className="text-cyan-500 font-bold">${inputs.Amount.toFixed(2)}</span>
                      )}
                    </div>
                    
                    <div className="flex gap-2">
                      <div className="relative flex-1">
                        <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[11px] font-mono text-gray-400">$</span>
                        <input
                          type="text"
                          value={inputStrings.Amount || ""}
                          onChange={(e) => handleTextChange("Amount", e.target.value)}
                          className={`w-full text-xs font-mono pl-6 pr-3 py-1.5 rounded-lg border transition-all duration-150 ${
                            errors.Amount
                              ? "border-rose-500 bg-rose-500/5 focus:ring-1 focus:ring-rose-500 focus:outline-none"
                              : isDark
                                ? "border-white/5 bg-[#1c1c20] focus:border-cyan-500/50 focus:outline-none text-white"
                                : "border-slate-200 bg-slate-50 focus:border-cyan-500/50 focus:outline-none text-slate-900"
                          }`}
                          placeholder="e.g. 150.00"
                        />
                      </div>
                      <span className={`text-[10px] font-mono px-3 py-1.5 rounded-lg border flex items-center ${
                        isDark ? "bg-[#1c1c20]/60 border-white/5 text-gray-500" : "bg-slate-50 border-slate-150 text-slate-500"
                      }`}>
                        Bounds: $0.01 - $5000
                      </span>
                    </div>

                    <input 
                      type="range" 
                      min="1.00" 
                      max="5000.00" 
                      step="5.00"
                      value={inputs.Amount}
                      onChange={(e) => handleSliderChange("Amount", parseFloat(e.target.value))}
                      className="w-full h-1 rounded-lg accent-cyan-500 appearance-none cursor-pointer bg-cyan-900/15"
                    />
                  </div>

                  {/* TRANSACTION TIME FIELD */}
                  <div className="flex flex-col gap-1.5" id="form-field-time">
                    <div className="flex justify-between items-center font-mono text-[11px]">
                      <span className={`font-semibold flex items-center gap-1 ${isDark ? "text-gray-300" : "text-slate-700"}`}>
                        2. Transaction Time
                        <span className={`text-[8.5px] px-1.5 py-0.5 rounded font-mono ${
                          isDark ? "bg-[#1c1c20] text-cyan-400" : "bg-slate-100 text-cyan-700"
                        }`}>SEC</span>
                      </span>
                      {errors.Time ? (
                        <span className="text-rose-500 flex items-center gap-1 text-[10px] font-bold">
                          <AlertTriangle className="w-3 h-3" /> {errors.Time}
                        </span>
                      ) : (
                        <span className="text-cyan-500 font-bold">
                          {inputs.Time}s ({Math.floor(inputs.Time / 3600)}h {Math.floor((inputs.Time % 3600) / 60)}m)
                        </span>
                      )}
                    </div>

                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={inputStrings.Time || ""}
                        onChange={(e) => handleTextChange("Time", e.target.value)}
                        className={`w-full text-xs font-mono px-3 py-1.5 rounded-lg border flex-1 transition-all duration-150 ${
                          errors.Time
                            ? "border-rose-500 bg-rose-500/5 focus:ring-1 focus:ring-rose-500 focus:outline-none"
                            : isDark
                              ? "border-white/5 bg-[#1c1c20] focus:border-cyan-500/50 focus:outline-none text-white"
                              : "border-slate-200 bg-slate-50 focus:border-cyan-500/50 focus:outline-none text-slate-900"
                        }`}
                        placeholder="e.g. 36000"
                      />
                      <span className={`text-[10px] font-mono px-3 py-1.5 rounded-lg border flex items-center ${
                        isDark ? "bg-[#1c1c20]/60 border-white/5 text-gray-500" : "bg-slate-50 border-slate-150 text-slate-500"
                      }`}>
                        Bounds: 0 - 172800s
                      </span>
                    </div>

                    <input 
                      type="range" 
                      min="0" 
                      max="172800" 
                      step="60"
                      value={inputs.Time}
                      onChange={(e) => handleSliderChange("Time", parseInt(e.target.value, 10))}
                      className="w-full h-1 rounded-lg accent-cyan-500 appearance-none cursor-pointer bg-cyan-900/15"
                    />
                  </div>

                  {/* LOCATION & USER HISTORY INDICATORS (EXPLICIT DEMAND) */}
                  <div className="space-y-4 pt-1">
                    <h4 className={`text-[10px] font-mono uppercase tracking-widest font-bold flex items-center gap-1 ${
                      isDark ? "text-gray-400" : "text-slate-600"
                    }`}>
                      <Info className="w-3 h-3 text-cyan-500" />
                      Location & History Security Vectors
                    </h4>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* V4 - Location Anomaly factor */}
                      <div className="flex flex-col gap-1.5" id="form-field-v4">
                        <div className="flex justify-between items-center text-[10.5px]">
                          <span className={`font-mono font-semibold ${isDark ? "text-gray-400" : "text-slate-650"}`} title="PCA component V4 representing location anomaly">
                            Location Anomaly (V4)
                          </span>
                          <span className={`font-mono text-[10px] font-bold ${inputs.V4 > 1.5 ? "text-rose-500" : "text-cyan-500"}`}>
                            {inputs.V4.toFixed(2)}
                          </span>
                        </div>
                        <input
                          type="text"
                          value={inputStrings.V4 || ""}
                          onChange={(e) => handleTextChange("V4", e.target.value)}
                          className={`w-full text-[11px] font-mono px-2.5 py-1 rounded-md border transition-all duration-150 ${
                            errors.V4
                              ? "border-rose-500 bg-rose-500/5 focus:ring-1 focus:ring-rose-500 focus:outline-none"
                              : isDark
                                ? "border-white/5 bg-[#1c1c20] text-white focus:outline-none"
                                : "border-slate-200 bg-slate-50 text-slate-900 focus:outline-none"
                          }`}
                        />
                        <input
                          type="range"
                          min="-5.0"
                          max="10.0"
                          step="0.1"
                          value={inputs.V4}
                          onChange={(e) => handleSliderChange("V4", parseFloat(e.target.value))}
                          className="w-full h-1 accent-cyan-500 cursor-pointer bg-cyan-900/10"
                        />
                      </div>

                      {/* V14 - Distance Anomaly factor */}
                      <div className="flex flex-col gap-1.5" id="form-field-v14">
                        <div className="flex justify-between items-center text-[10.5px]">
                          <span className={`font-mono font-semibold ${isDark ? "text-gray-400" : "text-slate-650"}`} title="PCA component V14 representing merchant IP distance">
                            IP Distance Score (V14)
                          </span>
                          <span className={`font-mono text-[10px] font-bold ${inputs.V14 < -1.5 ? "text-rose-500" : "text-cyan-500"}`}>
                            {inputs.V14.toFixed(2)}
                          </span>
                        </div>
                        <input
                          type="text"
                          value={inputStrings.V14 || ""}
                          onChange={(e) => handleTextChange("V14", e.target.value)}
                          className={`w-full text-[11px] font-mono px-2.5 py-1 rounded-md border transition-all duration-150 ${
                            errors.V14
                              ? "border-rose-500 bg-rose-500/5 focus:ring-1 focus:ring-rose-500 focus:outline-none"
                              : isDark
                                ? "border-white/5 bg-[#1c1c20] text-white focus:outline-none"
                                : "border-slate-200 bg-slate-50 text-slate-900 focus:outline-none"
                          }`}
                        />
                        <input
                          type="range"
                          min="-15.0"
                          max="5.0"
                          step="0.1"
                          value={inputs.V14}
                          onChange={(e) => handleSliderChange("V14", parseFloat(e.target.value))}
                          className="w-full h-1 accent-cyan-500 cursor-pointer bg-cyan-900/10"
                        />
                      </div>

                      {/* V12 - Historical frequency */}
                      <div className="flex flex-col gap-1.5" id="form-field-v12">
                        <div className="flex justify-between items-center text-[10.5px]">
                          <span className={`font-mono font-semibold ${isDark ? "text-gray-400" : "text-slate-650"}`} title="PCA component V12 representing frequency anomaly">
                            History Frequency (V12)
                          </span>
                          <span className={`font-mono text-[10px] font-bold ${inputs.V12 < -1.5 ? "text-rose-500" : "text-cyan-500"}`}>
                            {inputs.V12.toFixed(2)}
                          </span>
                        </div>
                        <input
                          type="text"
                          value={inputStrings.V12 || ""}
                          onChange={(e) => handleTextChange("V12", e.target.value)}
                          className={`w-full text-[11px] font-mono px-2.5 py-1 rounded-md border transition-all duration-150 ${
                            errors.V12
                              ? "border-rose-500 bg-rose-500/5 focus:ring-1 focus:ring-rose-500 focus:outline-none"
                              : isDark
                                ? "border-white/5 bg-[#1c1c20] text-white focus:outline-none"
                                : "border-slate-200 bg-slate-50 text-slate-900 focus:outline-none"
                          }`}
                        />
                        <input
                          type="range"
                          min="-15.0"
                          max="5.0"
                          step="0.1"
                          value={inputs.V12}
                          onChange={(e) => handleSliderChange("V12", parseFloat(e.target.value))}
                          className="w-full h-1 accent-cyan-500 cursor-pointer bg-cyan-900/10"
                        />
                      </div>

                      {/* V17 - Device velocity count */}
                      <div className="flex flex-col gap-1.5" id="form-field-v17">
                        <div className="flex justify-between items-center text-[10.5px]">
                          <span className={`font-mono font-semibold ${isDark ? "text-gray-400" : "text-slate-650"}`} title="PCA component V17 representing device count anomaly">
                            Device velocity (V17)
                          </span>
                          <span className={`font-mono text-[10px] font-bold ${inputs.V17 < -1.5 ? "text-rose-500" : "text-cyan-500"}`}>
                            {inputs.V17.toFixed(2)}
                          </span>
                        </div>
                        <input
                          type="text"
                          value={inputStrings.V17 || ""}
                          onChange={(e) => handleTextChange("V17", e.target.value)}
                          className={`w-full text-[11px] font-mono px-2.5 py-1 rounded-md border transition-all duration-150 ${
                            errors.V17
                              ? "border-rose-500 bg-rose-500/5 focus:ring-1 focus:ring-rose-500 focus:outline-none"
                              : isDark
                                ? "border-white/5 bg-[#1c1c20] text-white focus:outline-none"
                                : "border-slate-200 bg-slate-50 text-slate-900 focus:outline-none"
                          }`}
                        />
                        <input
                          type="range"
                          min="-15.0"
                          max="5.0"
                          step="0.1"
                          value={inputs.V17}
                          onChange={(e) => handleSliderChange("V17", parseFloat(e.target.value))}
                          className="w-full h-1 accent-cyan-500 cursor-pointer bg-cyan-900/10"
                        />
                      </div>
                    </div>
                  </div>

                  {/* UNCLUTTER PRINCIPLE: COLLAPSIBLE ACCORDION FOR ADVANCED PCA PARAMETERS */}
                  <div className="pt-1">
                    <button
                      type="button"
                      onClick={() => setShowAdvanced(!showAdvanced)}
                      className={`w-full flex justify-between items-center py-2 px-3 rounded-lg border font-mono text-[10.5px] font-bold transition-all ${
                        isDark 
                          ? "bg-[#1c1c20] border-white/5 text-gray-400 hover:text-white" 
                          : "bg-slate-50 border-slate-200 text-slate-600 hover:text-slate-800"
                      }`}
                    >
                      <span className="flex items-center gap-1.5">
                        <Cpu className="w-3.5 h-3.5" />
                        {showAdvanced ? "Hide" : "Show"} Advanced Latent PCA Scores (V10, V11, V8, V20)
                      </span>
                      {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                    </button>

                    {showAdvanced && (
                      <div className={`grid grid-cols-1 md:grid-cols-2 gap-3 mt-3 p-4 rounded-xl border animate-fade-in ${
                        isDark ? "border-white/5 bg-[#0c0c0e]" : "border-slate-150 bg-slate-100/50"
                      }`}>
                        {[
                          { key: "V10", min: -15, max: 5, label: "V10 (Anomaly Sig)" },
                          { key: "V11", min: -5, max: 10, label: "V11 (Booster PCA)" },
                          { key: "V8", min: -10, max: 10, label: "V8 (Latent Component)" },
                          { key: "V20", min: -10, max: 10, label: "V20 (Latent Component)" }
                        ].map((feature) => (
                          <div key={feature.key} className="flex flex-col gap-1.5" id={`advanced-field-${feature.key}`}>
                            <div className="flex justify-between items-center text-[10px] font-mono">
                              <span className="text-gray-500 font-bold">{feature.label}</span>
                              {errors[feature.key] ? (
                                <span className="text-rose-500 font-bold">Error</span>
                              ) : (
                                <span className="text-cyan-500 font-semibold">
                                  {inputs[feature.key as keyof TransactionInputs].toFixed(2)}
                                </span>
                              )}
                            </div>
                            <input
                              type="text"
                              value={inputStrings[feature.key] || ""}
                              onChange={(e) => handleTextChange(feature.key as keyof TransactionInputs, e.target.value)}
                              className={`w-full text-[10px] font-mono px-2 py-0.5 rounded border transition-all ${
                                errors[feature.key]
                                  ? "border-rose-500 bg-rose-500/5 focus:outline-none"
                                  : isDark
                                    ? "border-white/5 bg-[#1c1c20] text-white focus:outline-none"
                                    : "border-slate-200 bg-slate-50 text-slate-900 focus:outline-none"
                              }`}
                            />
                            <input 
                              type="range" 
                              min={feature.min} 
                              max={feature.max} 
                              step="0.1"
                              value={inputs[feature.key as keyof TransactionInputs]}
                              onChange={(e) => handleSliderChange(feature.key as keyof TransactionInputs, parseFloat(e.target.value))}
                              className="w-full h-1 accent-cyan-500 cursor-pointer bg-cyan-900/10"
                            />
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                </div>

                {/* Detect Fraud and Randomize CTA Controls */}
                <div className="flex flex-col sm:flex-row gap-3">
                  <button
                    type="button"
                    onClick={() => handleRunAnalysis()}
                    disabled={loading || hasErrors}
                    className={`flex-1 py-3 px-5 rounded-lg text-xs font-bold font-mono tracking-wider uppercase transition-all duration-300 flex items-center justify-center gap-2 ${
                      hasErrors 
                        ? "bg-slate-300 text-slate-500 cursor-not-allowed opacity-50 border border-slate-400/20"
                        : "bg-cyan-500 hover:bg-cyan-400 text-black hover:shadow-[0_0_20px_rgba(6,182,212,0.4)]"
                    }`}
                    id="submit-analysis-btn"
                  >
                    <Zap className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
                    Detect Fraud (XGB-QNN)
                  </button>
                  <button
                    type="button"
                    onClick={handleRandomize}
                    disabled={loading}
                    className={`p-3 rounded-lg border transition-all duration-200 flex items-center justify-center gap-1.5 text-xs font-mono font-bold ${
                      isDark 
                        ? "border-white/5 hover:border-white/10 bg-[#1c1c20] text-gray-400 hover:text-white" 
                        : "border-slate-200 bg-slate-50 text-slate-600 hover:text-slate-800 hover:bg-slate-100"
                    }`}
                    id="randomize-btn"
                    title="Randomize slider parameters"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    Randomize Input
                  </button>
                </div>
              </div>

              {/* RIGHT COLUMN: Results Section & Visual Explanations */}
              <div className="xl:col-span-7 flex flex-col gap-6" id="prediction-output-column">
                
                {/* 1. PIPELINE EXECUTION LOADER ANIMATION */}
                {loading && (
                  <div className={`p-8 rounded-xl border flex flex-col items-center justify-center min-h-[480px] shadow-sm space-y-6 animate-pulse ${
                    isDark ? "border-white/5 bg-[#111114]" : "border-slate-200 bg-white"
                  }`} id="pipeline-loader">
                    <div className="relative w-24 h-24 flex items-center justify-center">
                      <div className="absolute inset-0 border-t-2 border-r-2 border-cyan-500/80 rounded-full animate-spin" />
                      <div className="absolute inset-2 border-b-2 border-l-2 border-cyan-400/50 rounded-full animate-spin [animation-direction:reverse]" style={{ animationDuration: "1.2s" }} />
                      <div className="absolute inset-4 border-t border-l border-white/15 rounded-full animate-spin" style={{ animationDuration: "2.4s" }} />
                      <Cpu className="w-6 h-6 text-cyan-400 animate-pulse" />
                    </div>

                    <div className="text-center space-y-2 max-w-md">
                      <h3 className={`text-sm font-semibold font-mono uppercase tracking-widest ${
                        isDark ? "text-white" : "text-slate-800"
                      }`}>
                        XGB-QNN Pipeline Execution
                      </h3>
                      <p className={`text-xs leading-relaxed font-sans italic ${
                        isDark ? "text-gray-400" : "text-slate-500"
                      }`}>
                        Initializing classical Standard-Scaling... Mapping 10 features to AngleEmbedding Hilbert coordinates... Measuring expectation values across entangling registers...
                      </p>
                    </div>

                    {/* Step Timeline Indicator */}
                    <div className="grid grid-cols-4 gap-2 w-full max-w-lg border-t pt-5 font-mono text-[9px] text-gray-500 border-dashed border-gray-500/20">
                      {[
                        { step: "preprocessing", label: "Scaling" },
                        { step: "embedding", label: "AngleEmbed" },
                        { step: "entangling", label: "StrongLayers" },
                        { step: "classification", label: "Z0 Expectation" }
                      ].map((s, index) => {
                        const stepOrder = ["preprocessing", "embedding", "entangling", "classification"];
                        const currentIdx = stepOrder.indexOf(pipelineStep);
                        const selfIdx = stepOrder.indexOf(s.step);
                        const isCurrent = pipelineStep === s.step;
                        const isDone = selfIdx < currentIdx;

                        return (
                          <div key={s.step} className="flex flex-col items-center text-center space-y-1.5">
                            <div className={`w-2 h-2 rounded-full transition-all duration-300 ${
                              isDone ? "bg-emerald-500" : isCurrent ? "bg-cyan-500 animate-ping" : "bg-gray-300/25"
                            }`} />
                            <span className={isDone ? "text-emerald-500" : isCurrent ? "text-cyan-500 font-bold" : "text-gray-400"}>
                              {index + 1}. {s.label}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* 2. DYNAMIC RESULTS DISPLAY SECTION */}
                {result && !loading && (
                  <div className="space-y-6 animate-fade-in" id="prediction-result-card">
                    
                    {/* CONFIDENCE SCORE AND PREDICTION BANNER */}
                    <div className={`p-5 rounded-xl border flex flex-col md:flex-row items-center justify-between gap-4 shadow-md transition-all ${
                      result.decision === "FRAUD"
                        ? isDark 
                          ? "bg-rose-950/10 border-rose-500/20 text-rose-200" 
                          : "bg-rose-50 border-rose-200 text-rose-900"
                        : isDark
                          ? "bg-emerald-950/10 border-emerald-500/20 text-emerald-200"
                          : "bg-emerald-50 border-emerald-200 text-emerald-900"
                    }`} id="result-status-banner">
                      <div className="flex items-center gap-4 text-center md:text-left">
                        <div className={`p-3 rounded-lg border flex items-center justify-center ${
                          result.decision === "FRAUD"
                            ? isDark ? "bg-[#1c1c20] border-rose-500/25 text-rose-400" : "bg-white border-rose-300 text-rose-500 shadow-sm"
                            : isDark ? "bg-[#1c1c20] border-emerald-500/25 text-emerald-400" : "bg-white border-emerald-300 text-emerald-600 shadow-sm"
                        }`}>
                          {result.decision === "FRAUD" ? (
                            <ShieldAlert className="w-6 h-6 animate-bounce" />
                          ) : (
                            <ShieldCheck className="w-6 h-6" />
                          )}
                        </div>
                        <div>
                          <span className={`text-[9px] font-mono uppercase tracking-widest block mb-0.5 ${
                            isDark ? "text-gray-400" : "text-slate-500"
                          }`}>
                            Prediction Classification
                          </span>
                          <h2 className="text-base font-bold font-display uppercase tracking-wide leading-tight">
                            {result.decision === "FRAUD" ? (
                              "Likely Fraudulent (High Risk)"
                            ) : (
                              "Likely Legitimate (Cleared)"
                            )}
                          </h2>
                          <p className={`text-xs font-sans mt-0.5 leading-relaxed ${
                            isDark ? "text-gray-400" : "text-slate-650"
                          }`}>
                            {result.decision === "FRAUD"
                              ? "The XGB-QNN network isolated abnormal combinations of Location Angle deviations and Device History velocities."
                              : "The transaction parameters reside well within classical standard limits of safe customer purchasing habits."}
                          </p>
                        </div>
                      </div>

                      <div className={`flex flex-col items-center justify-center p-3.5 rounded-lg border min-w-[120px] font-mono shadow-inner ${
                        isDark ? "bg-[#1c1c20] border-white/5" : "bg-white border-slate-200"
                      }`}>
                        <span className="text-[9px] text-gray-500 uppercase tracking-widest block mb-0.5">Confidence</span>
                        <span className={`text-2.5xl font-bold tracking-tight ${
                          result.decision === "FRAUD" ? "text-rose-500" : "text-emerald-500"
                        }`}>
                          {(Math.max(result.combinedProbability, 1 - result.combinedProbability) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    {/* INDIVIDUAL PROBABILITY DRIVERS (XGB VS QNN CONSENSUS) */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* XGBoost classical risk */}
                      <div className={`p-5 rounded-xl border flex flex-col justify-between ${
                        isDark ? "border-white/5 bg-[#111114]" : "border-slate-200 bg-white shadow-sm"
                      }`} id="xgb-prob-card">
                        <div className="flex w-full justify-between items-center text-[10px] font-mono text-gray-500 font-bold uppercase tracking-wider">
                          <span>XGBoost Classification</span>
                          <TrendingUp className="w-3.5 h-3.5" />
                        </div>
                        <div className="my-3 flex items-baseline justify-between">
                          <span className={`text-2.5xl font-light font-mono tracking-tight ${isDark ? "text-gray-200" : "text-slate-800"}`}>
                            {(result.xgbProbability * 100).toFixed(1)}%
                          </span>
                          <span className="text-[10px] font-sans text-gray-500 italic">fraud risk</span>
                        </div>
                        <div className={`w-full h-1.5 rounded-full overflow-hidden border ${
                          isDark ? "bg-[#1c1c20] border-white/5" : "bg-slate-100 border-slate-200"
                        }`}>
                          <div 
                            className="bg-rose-450 h-full transition-all duration-500" 
                            style={{ width: `${result.xgbProbability * 100}%` }}
                          />
                        </div>
                      </div>

                      {/* QNN Quantum Probability */}
                      <div className={`p-5 rounded-xl border flex flex-col justify-between ${
                        isDark ? "border-white/5 bg-[#111114]" : "border-slate-200 bg-white shadow-sm"
                      }`} id="qnn-prob-card">
                        <div className="flex w-full justify-between items-center text-[10px] font-mono text-gray-500 font-bold uppercase tracking-wider">
                          <span>QNN Register Expectation</span>
                          <Cpu className="w-3.5 h-3.5 text-cyan-500 animate-pulse" />
                        </div>
                        <div className="my-3 flex items-baseline justify-between">
                          <span className="text-2.5xl font-light font-mono tracking-tight text-cyan-500">
                            {(result.qnnProbability * 100).toFixed(1)}%
                          </span>
                          <span className="text-[10px] font-sans text-gray-500 italic">
                            ⟨Z₀⟩ Expectation: {result.expectationValue.toFixed(3)}
                          </span>
                        </div>
                        <div className={`w-full h-1.5 rounded-full overflow-hidden border ${
                          isDark ? "bg-[#1c1c20] border-white/5" : "bg-slate-100 border-slate-200"
                        }`}>
                          <div 
                            className="bg-cyan-500 h-full transition-all duration-500" 
                            style={{ width: `${result.qnnProbability * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>

                    {/* SHAP FORCE GEOMETRY / VISUAL INFLUENCE GRAPH */}
                    <div className={`p-5 rounded-xl border ${
                      isDark ? "border-white/5 bg-[#111114]" : "border-slate-200 bg-white shadow-sm"
                    }`} id="shap-tornado-chart">
                      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 mb-4">
                        <div>
                          <h3 className={`text-xs font-bold uppercase tracking-widest ${
                            isDark ? "text-gray-400" : "text-slate-700"
                          }`}>
                            SHAP-Value Influence Analysis
                          </h3>
                          <p className="text-[11px] text-gray-500 font-sans mt-0.5">
                            Visual explanation of feature importance. Pushing red bars to the right drives fraud risk, and pushing green bars to the left guarantees approval.
                          </p>
                        </div>
                        <span className={`text-[10px] font-mono border px-2 py-0.5 rounded ${
                          isDark ? "text-gray-400 border-white/5 bg-[#1c1c20]" : "text-slate-650 border-slate-200 bg-slate-50"
                        }`}>
                          XGBoost Base Bias: -2.50
                        </span>
                      </div>

                      <div className="space-y-2">
                        {Object.entries(result.shapValues).map(([key, value]) => {
                          const val = value as number;
                          const maxShap = 7.0; // scaling denominator for visual cap
                          const percent = Math.min(100, Math.abs((val / maxShap) * 100));
                          const isPositive = val >= 0;

                          // Translate raw names to beautiful descriptive tags
                          const labelMap: { [key: string]: string } = {
                            Amount: "Amount",
                            Time: "Time",
                            V4: "V4 Location Anomaly",
                            V14: "V14 Merchant Distance",
                            V12: "V12 History Velocity",
                            V17: "V17 Device Velocity",
                            V10: "V10 Latent Wave",
                            V11: "V11 Booster Latent",
                            V8: "V8 Latent Factor",
                            V20: "V20 Latent Factor"
                          };

                          return (
                            <div key={key} className="grid grid-cols-12 gap-3 items-center text-xs" id={`shap-bar-${key}`}>
                              {/* Descriptive Feature Label */}
                              <div className={`col-span-3 font-mono text-[10px] font-bold truncate ${
                                isDark ? "text-gray-400" : "text-slate-600"
                              }`} title={labelMap[key] || key}>
                                {labelMap[key] || key}
                              </div>

                              {/* SHAP Bar Slider representation */}
                              <div className={`col-span-7 relative h-5 flex items-center justify-center rounded border shadow-inner ${
                                isDark ? "bg-[#0c0c0e]/80 border-white/5" : "bg-slate-50 border-slate-150"
                              }`}>
                                <div className={`absolute top-0 bottom-0 left-1/2 w-px z-10 ${isDark ? "bg-white/10" : "bg-slate-300"}`} />

                                {isPositive ? (
                                  <div className="absolute left-1/2 right-0 flex items-center h-full">
                                    <div 
                                      className="h-3 bg-rose-500/80 rounded-r border-r border-rose-400 hover:bg-rose-450 transition-all duration-300 shadow-sm"
                                      style={{ width: `${percent / 2}%` }}
                                      title={`Adds +${val.toFixed(3)} to fraud probability logit`}
                                    />
                                  </div>
                                ) : (
                                  <div className="absolute right-1/2 left-0 flex items-center justify-end h-full">
                                    <div 
                                      className="h-3 bg-emerald-500/80 rounded-l border-l border-emerald-450 hover:bg-emerald-400 transition-all duration-300 shadow-sm"
                                      style={{ width: `${percent / 2}%` }}
                                      title={`Substracts ${val.toFixed(3)} from fraud probability logit`}
                                    />
                                  </div>
                                )}
                              </div>

                              {/* Numeric precise float */}
                              <div className={`col-span-2 text-right font-mono text-[10.5px] ${
                                isPositive ? "text-rose-500 font-bold" : "text-emerald-500 font-bold"
                              }`}>
                                {isPositive ? "+" : ""}{val.toFixed(3)}
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      <div className={`mt-4 border-t pt-3 flex justify-between text-[9px] font-mono ${
                        isDark ? "text-gray-500 border-white/5" : "text-slate-500 border-slate-100"
                      }`}>
                        <span>← SUPPORTS LEGITIMACY</span>
                        <span>DRIVES FRAUD RISK →</span>
                      </div>
                    </div>

                    {/* INTERACTIVE QUANTUM REGISTERS (BLOCH SPHERE COMPONENT GRID) */}
                    <div className={`p-5 rounded-xl border ${
                      isDark ? "border-white/5 bg-[#111114]" : "border-slate-200 bg-white shadow-sm"
                    }`} id="bloch-spheres-grid-card">
                      <div>
                        <h3 className={`text-xs font-bold uppercase tracking-widest flex items-center gap-1.5 ${
                          isDark ? "text-gray-400" : "text-slate-700"
                        }`}>
                          <Cpu className="w-3.5 h-3.5 text-cyan-500 animate-pulse" />
                          Parameterized Qubit Bloch Vectors (10 Qubits)
                        </h3>
                        <p className="text-[11px] text-gray-500 font-sans mt-0.5 mb-4">
                          Qubit state representation on the Bloch sphere. Click on any Qubit card below to examine Angle embedding rotations and statevector projections.
                        </p>
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
                        {result.qubits.map((qubit) => (
                          <div 
                            key={qubit.id}
                            onClick={() => setSelectedQubit(selectedQubit === qubit.id ? null : qubit.id)}
                            className="cursor-pointer"
                          >
                            <BlochSphere qubit={qubit} theme={theme} />
                          </div>
                        ))}
                      </div>

                      {/* Selected Qubit Detail Inspector */}
                      {selectedQubit !== null && (
                        <div className={`mt-4 p-4 rounded-xl border font-mono text-xs animate-fade-in ${
                          isDark ? "border-white/10 bg-[#1c1c20] text-gray-300" : "border-slate-350 bg-slate-50 text-slate-700"
                        }`} id="qubit-detail-drawer">
                          <div className={`flex justify-between items-center border-b pb-2 mb-2.5 ${
                            isDark ? "border-white/5" : "border-slate-200"
                          }`}>
                            <span className="font-semibold text-cyan-600">Qubit Q<sub>{selectedQubit}</sub> ({result.qubits[selectedQubit].feature}) Statevector Analysis</span>
                            <button 
                              type="button" 
                              onClick={() => setSelectedQubit(null)}
                              className={`text-[10px] border px-2 py-0.5 rounded transition-all ${
                                isDark 
                                  ? "text-gray-400 border-white/10 bg-[#0c0c0e] hover:text-white" 
                                  : "text-slate-500 border-slate-200 bg-white hover:text-slate-800"
                              }`}
                            >
                              Close
                            </button>
                          </div>
                          <p className="text-[11px] leading-relaxed font-sans mb-3 text-slate-500">
                            This qubit encodes the standard-scaled feature value <strong className={isDark ? "text-white" : "text-slate-900"}>{result.qubits[selectedQubit].feature}</strong> (Z-Score amplitude: <strong className={isDark ? "text-cyan-400" : "text-cyan-600"}>{result.scaledFeatures[result.qubits[selectedQubit].feature]}</strong>). Under the hood, Angle Embedding maps the value directly into unitary rotation parameters about the Bloch Y-axis. The state vector is then entangled over 3 Strongly Entangling QNN layers.
                          </p>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 font-mono text-[10px]">
                            <div>
                              <span className="text-gray-500 uppercase text-[8.5px] block">X projection</span>
                              <span className={`font-bold ${isDark ? "text-gray-200" : "text-slate-900"}`}>{result.qubits[selectedQubit].x.toFixed(4)}</span>
                            </div>
                            <div>
                              <span className="text-gray-500 uppercase text-[8.5px] block">Y projection</span>
                              <span className={`font-bold ${isDark ? "text-gray-200" : "text-slate-900"}`}>{result.qubits[selectedQubit].y.toFixed(4)}</span>
                            </div>
                            <div>
                              <span className="text-gray-500 uppercase text-[8.5px] block">Z projection</span>
                              <span className={`font-bold ${result.qubits[selectedQubit].z < -0.3 ? "text-rose-500" : "text-emerald-600"}`}>
                                {result.qubits[selectedQubit].z.toFixed(4)}
                              </span>
                            </div>
                            <div>
                              <span className="text-gray-500 uppercase text-[8.5px] block">Basis probability</span>
                              <span className={`font-bold ${isDark ? "text-gray-200" : "text-slate-900"}`}>
                                {(result.qubits[selectedQubit].superposition0 * 100).toFixed(0)}% |0⟩ • {(result.qubits[selectedQubit].superposition1 * 100).toFixed(0)}% |1⟩
                              </span>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* AI COGNITIVE CONSENSUS REPORT VIA GEMINI */}
                    <div className={`p-5 rounded-xl border ${
                      isDark ? "border-white/5 bg-[#111114]" : "border-slate-200 bg-white shadow-sm"
                    }`} id="xai-gemini-report">
                      <div className={`flex items-center gap-2 mb-3 border-b pb-3 ${
                        isDark ? "border-white/5" : "border-slate-100"
                      }`}>
                        <div className={`p-1.5 rounded-lg border ${
                          isDark ? "bg-cyan-950/40 border-cyan-500/20 text-cyan-400" : "bg-cyan-50 border-cyan-100 text-cyan-600"
                        }`}>
                          <FileText className="w-4 h-4" />
                        </div>
                        <div>
                          <h3 className={`text-xs font-bold uppercase tracking-widest ${
                            isDark ? "text-cyan-400" : "text-cyan-600"
                          }`}>
                            Explainable AI (XAI) Consensus Report
                          </h3>
                          <span className="text-[9px] font-mono text-gray-500 uppercase block mt-0.5">
                            Secure Server-Side LLM Interpretation Call • Parameter Grounding
                          </span>
                        </div>
                      </div>

                      <div className={`font-sans text-xs leading-relaxed space-y-3 whitespace-pre-line ${
                        isDark ? "text-gray-300" : "text-slate-700"
                      }`} id="report-text-area">
                        {result.explanation}
                      </div>

                      <div className={`mt-4 pt-3 border-t flex items-center gap-2 text-[9px] font-mono text-gray-500 ${
                        isDark ? "border-white/5" : "border-slate-100"
                      }`}>
                        <CheckCircle className="w-3.5 h-3.5 text-cyan-500" />
                        <span>Analyst compliance signature logged. Node Makerere-CC-01 online.</span>
                      </div>
                    </div>

                  </div>
                )}

                {/* 3. AWAITING FIRST RUN STAGE */}
                {!result && !loading && (
                  <div className={`p-8 rounded-xl border flex flex-col items-center justify-center min-h-[480px] text-center space-y-4 shadow-sm ${
                    isDark ? "border-white/5 bg-[#111114]" : "border-slate-200 bg-white"
                  }`} id="empty-terminal-state">
                    <div className={`p-4 rounded-full shadow-inner ${
                      isDark ? "bg-[#1c1c20] border border-white/5 text-gray-500" : "bg-slate-100 border border-slate-200 text-slate-400"
                    }`}>
                      <HelpCircle className="w-8 h-8 animate-pulse" />
                    </div>
                    <div className="space-y-1.5">
                      <h3 className={`text-sm font-semibold font-mono uppercase tracking-widest ${
                        isDark ? "text-white" : "text-slate-800"
                      }`}>
                        Terminal Awaiting Payload
                      </h3>
                      <p className={`text-xs max-w-sm mx-auto font-sans italic ${
                        isDark ? "text-gray-500" : "text-slate-500"
                      }`}>
                        Fine-tune sliders or type parameters in the form on the left, then click 'Detect Fraud' to route transactions through the quantum simulator pipeline.
                      </p>
                    </div>
                  </div>
                )}

              </div>

            </div>

          </div>
        ) : (
          <div className="flex-1 animate-fade-in" id="tab-metrics-view">
            <MetricsPanel theme={theme} />
          </div>
        )}

      </main>

      {/* SYSTEM FOOTER */}
      <footer className={`border-t py-4 px-8 mt-12 flex justify-between items-center text-[10px] font-mono transition-colors duration-300 ${
        isDark ? "border-white/5 bg-[#0c0c0e] text-gray-500" : "border-slate-200 bg-white text-slate-500"
      }`} id="app-footer">
        <div>
          <span>Credit Card Fraud Detector © 2026</span>
        </div>
        <div>
          <span>Active Session ID: 0X9B7A6F (Node Active)</span>
        </div>
      </footer>

    </div>
  );
}
