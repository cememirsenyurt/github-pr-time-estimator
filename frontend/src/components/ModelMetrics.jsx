// frontend/src/components/ModelMetrics.jsx
import React, { useState, useEffect } from "react";

export default function ModelMetrics() {
  const [metrics, setMetrics] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showComparison, setShowComparison] = useState(false);

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const [metricsRes, comparisonRes] = await Promise.all([
        fetch("/metrics"),
        fetch("/comparison")
      ]);
      
      if (metricsRes.ok) {
        setMetrics(await metricsRes.json());
      }
      if (comparisonRes.ok) {
        setComparison(await comparisonRes.json());
      }
    } catch (e) {
      setError("Could not load model metrics");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="metrics-panel">
        <div className="loading">Loading model metrics...</div>
      </div>
    );
  }

  if (error || !metrics) {
    return null;
  }

  // Get top 8 features by importance
  const featureImportance = metrics.feature_importance || {};
  const sortedFeatures = Object.entries(featureImportance)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);
  const maxImportance = sortedFeatures.length > 0 ? sortedFeatures[0][1] : 1;

  return (
    <div className="metrics-panel">
      <div className="metrics-header">
        <span className="card-icon">🤖</span>
        <h3 className="metrics-title">
          Model Performance — {metrics.best_model}
        </h3>
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-value">{metrics.test_rmse?.toFixed(1)}h</div>
          <div className="metric-label">RMSE</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{metrics.test_mae?.toFixed(1)}h</div>
          <div className="metric-label">MAE</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{(metrics.test_r2 * 100)?.toFixed(1)}%</div>
          <div className="metric-label">R² Score</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{metrics.num_features}</div>
          <div className="metric-label">Features</div>
        </div>
      </div>

      {sortedFeatures.length > 0 && (
        <div style={{ marginTop: "1.5rem" }}>
          <h4 style={{ 
            fontSize: "0.9rem", 
            color: "var(--text-muted)", 
            marginBottom: "1rem",
            textTransform: "uppercase",
            letterSpacing: "0.05em"
          }}>
            Feature Importance
          </h4>
          {sortedFeatures.map(([name, value]) => (
            <div key={name} className="feature-bar">
              <span className="feature-name">{formatFeatureName(name)}</span>
              <div className="feature-bar-container">
                <div 
                  className="feature-bar-fill" 
                  style={{ width: `${(value / maxImportance) * 100}%` }}
                />
              </div>
              <span className="feature-value">{(value * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}

      {comparison && comparison.models && (
        <div style={{ marginTop: "1.5rem" }}>
          <button
            onClick={() => setShowComparison(!showComparison)}
            style={{
              background: "transparent",
              border: "1px solid var(--border)",
              borderRadius: "0.5rem",
              padding: "0.5rem 1rem",
              color: "var(--text-muted)",
              cursor: "pointer",
              fontSize: "0.85rem",
              display: "flex",
              alignItems: "center",
              gap: "0.5rem"
            }}
          >
            {showComparison ? "▼" : "▶"} Model Comparison ({comparison.models.length} models)
          </button>
          
          {showComparison && (
            <div style={{ marginTop: "1rem", overflowX: "auto" }}>
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>CV RMSE</th>
                    <th>Test RMSE</th>
                    <th>Test MAE</th>
                    <th>R²</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.models.map((model) => (
                    <tr 
                      key={model.model_name}
                      className={model.model_name === comparison.best_model ? "best-model" : ""}
                    >
                      <td>
                        {model.model_name}
                        {model.model_name === comparison.best_model && " ⭐"}
                      </td>
                      <td>{model.cv_rmse?.toFixed(2)}</td>
                      <td>{model.test_rmse?.toFixed(2)}</td>
                      <td>{model.test_mae?.toFixed(2)}</td>
                      <td>{(model.test_r2 * 100)?.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      <div style={{ 
        marginTop: "1rem", 
        fontSize: "0.8rem", 
        color: "var(--text-muted)",
        textAlign: "center"
      }}>
        Trained on {metrics.num_samples} PRs • {metrics.num_features} features
      </div>
    </div>
  );
}

function formatFeatureName(name) {
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace("Pr", "PR")
    .replace("Sin", "(sin)")
    .replace("Cos", "(cos)");
}
