// frontend/src/components/ManualPRForm.jsx
import { useState, useRef, useEffect } from "react";
import { FaEdit, FaRocket } from "react-icons/fa";
import LabelDropdown from "./LabelDropdown";

const MIN_TEXTAREA_HEIGHT = 100;

export default function ManualPRForm({ onSubmit }) {
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [labels, setLabels] = useState([]);
  const textareaRef = useRef(null);

  // Auto-expand textarea as content grows
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.max(MIN_TEXTAREA_HEIGHT, ta.scrollHeight) + "px";
  }, [body]);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      title,
      body,
      num_labels: labels.length,
      is_closed: false,
    });
  };

  return (
    <div className="card">
      <div className="card-header">
        <FaEdit className="card-icon" style={{ color: "var(--secondary)" }} />
        <h3 className="card-title">Manual PR Entry</h3>
      </div>

      <form onSubmit={handleSubmit} className="input-form">
        <div className="form-group">
          <label className="form-label">PR Title</label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            required
            placeholder="e.g., [Feature] Add new authentication flow"
            className="form-input"
          />
        </div>

        <div className="form-group">
          <label className="form-label">Description</label>
          <textarea
            ref={textareaRef}
            value={body}
            onChange={(e) => setBody(e.target.value)}
            placeholder="Describe what your PR does, include any relevant context..."
            className="form-input form-textarea"
          />
        </div>

        <div className="form-group">
          <label className="form-label">Labels</label>
          <LabelDropdown
            selected={labels}
            onChange={setLabels}
            placeholder="Select labels..."
          />
        </div>

        <button type="submit" className="btn btn-primary" style={{ marginTop: "0.5rem" }}>
          <FaRocket />
          <span>Estimate Merge Time</span>
        </button>
      </form>

      <p style={{ 
        marginTop: "1rem", 
        fontSize: "0.8rem", 
        color: "var(--text-muted)",
        textAlign: "center"
      }}>
        Enter PR details to get an estimated merge time prediction
      </p>
    </div>
  );
}
