// frontend/src/App.jsx
import { useState } from "react";
import GHInputForm from "./components/GHInputForm";
import ManualPRForm from "./components/ManualPRForm";
import PRResult from "./components/PRResult";
import "./App.css";

export default function App() {
  const [mode, setMode] = useState("existing"); // "existing" or "manual"
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

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
        } catch (_) {}
        throw new Error(msg);
      }
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App" style={{ maxWidth: 600, margin: '2rem auto', padding: '1rem' }}>
      <h1 style={{ textAlign: 'center', marginBottom: '1rem' }}>
        Manual PR Time Estimator
      </h1>

      {/* mode toggle */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginBottom: '1.5rem' }}>
        <button
          className={mode === 'existing' ? 'active' : ''}
          onClick={() => setMode('existing')}
        >
          Existing PR
        </button>
        <button
          className={mode === 'manual' ? 'active' : ''}
          onClick={() => setMode('manual')}
        >
          Manual PR
        </button>
      </div>

      {/* conditional form */}
      {mode === 'existing' ? (
        <GHInputForm onSubmit={(p) => handlePredict(p, '/predict')} />
      ) : (
        <ManualPRForm onSubmit={(p) => handlePredict(p, '/estimate')} />
      )}

      {loading && <p>⏳ Fetching & predicting…</p>}
      <PRResult result={result} error={error} />
    </div>
  );
}
