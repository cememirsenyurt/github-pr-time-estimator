import { useState } from "react";

export default function ManualPRForm({ onSubmit }) {
  const [title, setTitle]         = useState("");
  const [body, setBody]           = useState("");
  const [numLabels, setNumLabels] = useState(0);
  const [isClosed, setIsClosed]   = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({ title, body, num_labels: numLabels, is_closed: isClosed });
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: "grid", gap: "1rem" }}>
      <h2 style={{ color: "#5ac8fa" }}>Manual PR Entry</h2>
      <label>
        Title
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          required
        />
      </label>

      <label>
        Body
        <textarea
          rows={4}
          value={body}
          onChange={(e) => setBody(e.target.value)}
        />
      </label>

      <label>
        # Labels
        <input
          type="number"
          min="0"
          value={numLabels}
          onChange={(e) => setNumLabels(Number(e.target.value))}
        />
      </label>

      <label style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <input
          type="checkbox"
          checked={isClosed}
          onChange={(e) => setIsClosed(e.target.checked)}
        />
        Already Closed?
      </label>

      <button type="submit" className="btn">
        Estimate Merge Time
      </button>
    </form>
  );
}
