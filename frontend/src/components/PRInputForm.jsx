import { useState } from "react";

export default function PRInputForm({ onSubmit }) {
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [numLabels, setNumLabels] = useState(0);
  const [isClosed, setIsClosed] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({ title, body, num_labels: numLabels, is_closed: isClosed });
  };

  return (
    <form onSubmit={handleSubmit} className="input-form">
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
          value={body}
          onChange={(e) => setBody(e.target.value)}
          rows={4}
        />
      </label>

      <div className="row">
        <label>
          # Labels
          <input
            type="number"
            value={numLabels}
            onChange={(e) => setNumLabels(+e.target.value)}
            min={0}
          />
        </label>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={isClosed}
            onChange={(e) => setIsClosed(e.target.checked)}
          />
          Already Closed?
        </label>
      </div>

      <button type="submit" className="neon-button">
        Estimate Merge Time
      </button>
    </form>
  );
}