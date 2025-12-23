// frontend/src/App.jsx
import { useState } from "react";
import { FaGithub, FaClock, FaChartLine, FaCode } from "react-icons/fa";
import { SiMeta } from "react-icons/si";
import GHInputForm from "./components/GHInputForm";
import ManualPRForm from "./components/ManualPRForm";
import PRResult from "./components/PRResult";
import ModelMetrics from "./components/ModelMetrics";
import "./App.css";

// Example PR numbers from facebook/react that users can try
const EXAMPLE_PRS = [
  { number: 32812, description: "DevTools fix - Fast merge (~40 min)" },
  { number: 32808, description: "Performance track tweaks (~2.5 hrs)" },
  { number: 32807, description: "Delete changelog (~2.7 hrs)" },
  { number: 32803, description: "Error handling fix (~10 hrs)" },
  { number: 32797, description: "Large feature - TransitionTypes" },
];

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
        <div className="header-icons">
          <FaGithub className="app-icon" />
          <SiMeta className="app-icon meta-icon" />
        </div>
        <span className="app-title">PR Time Estimator</span>
        <FaClock className="app-icon" style={{ color: "var(--secondary)" }} />
      </header>

      {/* Model Metrics Dashboard */}
      <ModelMetrics />

      {/* Main Content with Sidebar */}
      <div className="main-layout">
        {/* Left side - Forms and Results */}
        <div className="main-content-area">
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

        {/* Right side - Example PRs Notepad */}
        {mode === "existing" && (
          <div className="example-prs-sidebar">
            <div className="notepad">
              <div className="notepad-header">
                <span className="notepad-icon">📝</span>
                <h4 className="notepad-title">Example PRs to Try</h4>
              </div>
              <p className="notepad-subtitle">
                Click to copy PR number from facebook/react:
              </p>
              <ul className="pr-list">
                {EXAMPLE_PRS.map((pr) => (
                  <li 
                    key={pr.number}
                    className="pr-item"
                    onClick={() => {
                      navigator.clipboard.writeText(pr.number.toString());
                    }}
                    title="Click to copy"
                  >
                    <span className="pr-number">#{pr.number}</span>
                    <span className="pr-description">{pr.description}</span>
                  </li>
                ))}
              </ul>
              <p className="notepad-footer">
                💡 These are real merged PRs from the React repository
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
