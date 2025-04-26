export default function PRResult({ result, error }) {
  if (error) {
    return <p style={{ color: "salmon" }}>❌ {error}</p>;
  }

  if (result) {
    return (
      <div
        style={{
          marginTop: "1rem",
          padding: "1rem",
          background: "#222",
          borderRadius: 4,
        }}
      >
        <h2>Predicted Merge Time</h2>
        <p style={{ fontSize: "1.5rem" }}>
          ⏱ <strong>{result.predicted_hours.toFixed(2)}</strong> hours
        </p>
      </div>
    );
  }

  return null;
}
