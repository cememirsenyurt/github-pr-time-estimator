// frontend/src/App.jsx
import { useState } from "react";
import { FaGithub } from "react-icons/fa";
import { SiMeta   } from "react-icons/si";
import GHInputForm  from "./components/GHInputForm";
import ManualPRForm from "./components/ManualPRForm";
import PRResult     from "./components/PRResult";
import "./App.css";

export default function App() {
  const [mode,    setMode]    = useState("existing"); // "existing" or "manual"
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);
  const [error,   setError]   = useState("");

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
      <header className="app-header">
        <FaGithub className="app-icon" />
        <span className="app-title">Manual PR Time Estimator</span>
        <SiMeta className="app-icon" />
      </header>

      <div className="tabs">
        <button
          className={`tab-button ${mode === "existing" ? "active" : ""}`}
          onClick={() => switchMode("existing")}
        >
          Existing PR
        </button>
        <button
          className={`tab-button ${mode === "manual" ? "active" : ""}`}
          onClick={() => switchMode("manual")}
        >
          Manual PR
        </button>
      </div>

      <div className="form-wrapper">
        {mode === "existing" ? (
          <GHInputForm onSubmit={(p) => handlePredict(p, "/predict")} />
        ) : (
          <ManualPRForm onSubmit={(p) => handlePredict(p, "/estimate")} />
        )}
      </div>

      {loading && <p className="loading">⏳ Fetching & predicting…</p>}
      <PRResult result={result} error={error} />
    </div>
  );
}
