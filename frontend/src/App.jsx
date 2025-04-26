import { useState } from "react";
import PRInputForm from "./components/PRInputForm";
import PRResult from "./components/PRResult";
import "./App.css";

function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState(null);
  const [error, setError]     = useState("");

  async function handlePredict(payload) {
    setLoading(true);
    setResult(null);
    setError("");

    try {
      const res = await fetch("/estimate", {
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
      console.error("prediction error", e);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="App" style={{ maxWidth: 600, margin: "2rem auto" }}>
      <h1>Manual PR Time Estimator</h1>
      <PRInputForm onSubmit={handlePredict} />
      {loading && <p>⏳ Fetching & predicting…</p>}
      <PRResult result={result} error={error} />
    </div>
  );
}

export default App;
