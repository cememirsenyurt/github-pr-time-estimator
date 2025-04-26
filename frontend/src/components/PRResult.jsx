export default function PRResult({ result, error }) {
  if (error) {
    return <p className="error-text">❌ {error}</p>;
  }
  if (result) {
    return (
      <div className="result-container">
        <h2>Predicted Merge Time</h2>
        <p className="result-hours">
          ⏱ <strong>{result.predicted_hours.toFixed(2)}</strong> hours
        </p>
      </div>
    );
  }
  return null;
}