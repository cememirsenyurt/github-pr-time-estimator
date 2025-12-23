// frontend/src/App.jsx
import { useState } from "react";
import { FaGithub, FaClock, FaChartLine, FaCode } from "react-icons/fa";
import { SiMeta } from "react-icons/si";
import GHInputForm from "./components/GHInputForm";
import ManualPRForm from "./components/ManualPRForm";
import PRResult from "./components/PRResult";
import ModelMetrics from "./components/ModelMetrics";
import "./App.css";

export default function App() {
  const [mode, setMode] = useState("existing"); // "existing" or "manual"
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const switchMode = (newMode) => {
    setMode(newMode);
    setResult(null);
    setError("");
  };

  const handlePredict = async (payload, endpoint) => {
    setLoading(true);
    setResult(null);
    setError("");
    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        let msg = `Error ${res.status}`;
        try {
          const err = await res.json();
          msg = err.detail || err.message || msg;
        } catch {}
        throw new Error(msg);
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <FaGithub className="app-icon" />
        <span className="app-title">PR Time Estimator</span>
        <FaClock className="app-icon" style={{ color: "var(--secondary)" }} />
      </header>

      {/* Model Metrics Dashboard */}
      <ModelMetrics />

      {/* Navigation Tabs */}
      <div className="tabs">
        <button
          className={`tab-button ${mode === "existing" ? "active" : ""}`}
          onClick={() => switchMode("existing")}
        >
          <FaGithub style={{ marginRight: "0.5rem" }} />
          Lookup PR
        </button>
        <button
          className={`tab-button ${mode === "manual" ? "active" : ""}`}
          onClick={() => switchMode("manual")}
        >
          <FaCode style={{ marginRight: "0.5rem" }} />
          Manual Entry
        </button>
      </div>

      {/* Form Section */}
      <div className="form-wrapper">
        {mode === "existing" ? (
          <GHInputForm onSubmit={(p) => handlePredict(p, "/predict")} />
        ) : (
          <ManualPRForm onSubmit={(p) => handlePredict(p, "/estimate")} />
        )}
      </div>

      {/* Loading State */}
      {loading && (
        <div className="loading">
          <span>⏳</span>
          <span>Analyzing PR and predicting merge time...</span>
        </div>
      )}

      {/* Results */}
      <PRResult result={result} error={error} />
    </div>
  );
}
