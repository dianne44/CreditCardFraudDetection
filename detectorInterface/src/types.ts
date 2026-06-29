export interface TransactionInputs {
  Time: number;
  Amount: number;
  V14: number;
  V4: number;
  V10: number;
  V12: number;
  V17: number;
  V11: number;
  V8: number;
  V20: number;
}

export interface QubitState {
  id: number;
  feature: string;
  x: number;
  y: number;
  z: number;
  superposition0: number;
  superposition1: number;
}

export interface PredictionResult {
  success: boolean;
  decision: "FRAUD" | "LEGITIMATE";
  combinedProbability: number;
  xgbProbability: number;
  qnnProbability: number;
  expectationValue: number;
  features: TransactionInputs;
  scaledFeatures: { [key: string]: number };
  shapValues: { [key: string]: number };
  limeValues: { [key: string]: { weight: number; label: string } };
  qubits: QubitState[];
  explanation: string;
}

export interface MetricRow {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  specificity: number;
  time: number;
}

export interface ModelMetrics {
  logisticRegression: MetricRow;
  quantumSVM: MetricRow;
  quantumKNN: MetricRow;
  qnnXGBoost: MetricRow;
}
