// frontend/src/components/PRResult.jsx
import React from "react";
import { FaClock, FaRobot, FaCheckCircle, FaExclamationTriangle } from "react-icons/fa";

export default function PRResult({ result, error }) {
  if (error) {
    return (
      <div className="error" style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.5rem" }}>
        <FaExclamationTriangle />
        <span>{error}</span>
      </div>
    );
  }
  
  if (!result) return null;

  const {
    title = null,
    body = null,
    actual_hours = null,
    predicted_hours,
    features_used = null,
  } = result;

  const formatHours = (hours) => {
    if (hours < 1) {
      return `${Math.round(hours * 60)} min`;
    } else if (hours < 24) {
      return `${hours.toFixed(1)} hours`;
    } else {
      const days = hours / 24;
      return `${days.toFixed(1)} days`;
    }
  };

  const getAccuracyColor = () => {
    if (!actual_hours) return "var(--primary)";
    const diff = Math.abs(predicted_hours - actual_hours);
    const percentDiff = (diff / actual_hours) * 100;
    if (percentDiff < 20) return "var(--success)";
    if (percentDiff < 50) return "var(--warning)";
    return "var(--danger)";
  };

  return (
    <div style={{ marginTop: "2rem" }}>
      {/* Main Result Cards */}
      <div style={{ 
        display: "grid", 
        gridTemplateColumns: actual_hours != null ? "1fr 1fr" : "1fr",
        gap: "1.5rem",
        marginBottom: "1.5rem"
      }}>
        {/* Predicted Time Card */}
        <div className="card" style={{ 
          background: "linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(34, 211, 238, 0.1) 100%)",
          border: "1px solid rgba(99, 102, 241, 0.4)"
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
            <FaRobot style={{ fontSize: "1.5rem", color: "var(--primary-light)" }} />
            <h4 style={{ margin: 0, color: "var(--text)" }}>Predicted Merge Time</h4>
          </div>
          <div style={{ 
            fontSize: "3rem", 
            fontWeight: "700",
            background: "linear-gradient(135deg, var(--primary-light) 0%, var(--secondary) 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text"
          }}>
            {formatHours(predicted_hours)}
          </div>
          <div style={{ 
            fontSize: "0.85rem", 
            color: "var(--text-muted)",
            marginTop: "0.5rem"
          }}>
            {predicted_hours.toFixed(2)} hours total
          </div>
        </div>

        {/* Actual Time Card (only shown for existing PRs) */}
        {actual_hours != null && (
          <div className="card" style={{ 
            background: "linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(245, 158, 11, 0.1) 100%)",
            border: `1px solid ${getAccuracyColor()}40`
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
              <FaClock style={{ fontSize: "1.5rem", color: "var(--success)" }} />
              <h4 style={{ margin: 0, color: "var(--text)" }}>Actual Merge Time</h4>
            </div>
            <div style={{ 
              fontSize: "3rem", 
              fontWeight: "700",
              color: "var(--success)"
            }}>
              {formatHours(actual_hours)}
            </div>
            <div style={{ 
              fontSize: "0.85rem", 
              color: "var(--text-muted)",
              marginTop: "0.5rem"
            }}>
              {actual_hours.toFixed(2)} hours total
            </div>
          </div>
        )}
      </div>

      {/* Accuracy Badge (for existing PRs) */}
      {actual_hours != null && (
        <div className="card" style={{ 
          display: "flex", 
          alignItems: "center", 
          justifyContent: "center",
          gap: "1rem",
          padding: "1rem",
          background: `${getAccuracyColor()}15`,
          border: `1px solid ${getAccuracyColor()}40`
        }}>
          <FaCheckCircle style={{ color: getAccuracyColor(), fontSize: "1.25rem" }} />
          <span style={{ color: "var(--text)" }}>
            Prediction was <strong style={{ color: getAccuracyColor() }}>
              {Math.abs(((predicted_hours - actual_hours) / actual_hours) * 100).toFixed(1)}%
            </strong> {predicted_hours > actual_hours ? "over" : "under"} actual time
          </span>
        </div>
      )}

      {/* PR Details (for existing PRs) */}
      {title && (
        <div className="card" style={{ marginTop: "1.5rem" }}>
          <div className="card-header">
            <span className="card-icon">📋</span>
            <h4 className="card-title">PR Details</h4>
          </div>
          
          <div style={{ marginBottom: "1rem" }}>
            <div style={{ 
              fontSize: "0.8rem", 
              color: "var(--text-muted)", 
              marginBottom: "0.25rem",
              textTransform: "uppercase",
              letterSpacing: "0.05em"
            }}>
              Title
            </div>
            <div style={{ color: "var(--text)", fontWeight: "500" }}>{title}</div>
          </div>

          {body && (
            <div>
              <div style={{ 
                fontSize: "0.8rem", 
                color: "var(--text-muted)", 
                marginBottom: "0.25rem",
                textTransform: "uppercase",
                letterSpacing: "0.05em"
              }}>
                Description
              </div>
              <div style={{ 
                maxHeight: "200px",
                overflow: "auto",
                background: "rgba(10, 10, 26, 0.6)",
                padding: "1rem",
                borderRadius: "0.5rem",
                whiteSpace: "pre-wrap",
                fontSize: "0.9rem",
                color: "var(--text-muted)",
                border: "1px solid var(--border)"
              }}>
                {body}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Features Used (optional debug info) */}
      {features_used && Object.keys(features_used).length > 0 && (
        <details style={{ marginTop: "1rem" }}>
          <summary style={{ 
            cursor: "pointer",
            color: "var(--text-muted)",
            fontSize: "0.85rem",
            padding: "0.5rem"
          }}>
            View extracted features
          </summary>
          <div className="card" style={{ marginTop: "0.5rem" }}>
            <div style={{ 
              display: "grid", 
              gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
              gap: "0.75rem",
              fontSize: "0.85rem"
            }}>
              {Object.entries(features_used).map(([key, value]) => (
                <div key={key} style={{ 
                  display: "flex", 
                  justifyContent: "space-between",
                  padding: "0.5rem",
                  background: "rgba(10, 10, 26, 0.4)",
                  borderRadius: "0.25rem"
                }}>
                  <span style={{ color: "var(--text-muted)" }}>
                    {key.replace(/_/g, " ")}:
                  </span>
                  <span style={{ color: "var(--text)", fontWeight: "500" }}>
                    {typeof value === "number" ? value.toFixed(2) : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </details>
      )}
    </div>
  );
}
