// frontend/src/components/GHInputForm.jsx
import { useState } from "react";
import { FaGithub, FaSearch } from "react-icons/fa";

export default function GHInputForm({ onSubmit }) {
  const [prNumber, setPrNumber] = useState(32812);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      owner: "facebook",
      repo: "react",
      pr_number: Number(prNumber),
    });
  };

  return (
    <div className="card">
      <div className="card-header">
        <FaGithub className="card-icon" style={{ color: "var(--primary)" }} />
        <h3 className="card-title">Lookup Existing PR</h3>
      </div>
      
      <form onSubmit={handleSubmit} className="input-form">
        <div className="form-group">
          <label className="form-label">Repository Owner</label>
          <input
            type="text"
            value="facebook"
            readOnly
            className="form-input"
          />
        </div>

        <div className="form-group">
          <label className="form-label">Repository Name</label>
          <input
            type="text"
            value="react"
            readOnly
            className="form-input"
          />
        </div>

        <div className="form-group">
          <label className="form-label">Pull Request Number</label>
          <input
            type="number"
            min="1"
            value={prNumber}
            onChange={(e) => setPrNumber(e.target.value)}
            required
            className="form-input"
            placeholder="Enter PR number..."
          />
        </div>

        <button type="submit" className="btn btn-primary" style={{ marginTop: "0.5rem" }}>
          <FaSearch />
          <span>Fetch & Predict</span>
        </button>
      </form>
      
      <p style={{ 
        marginTop: "1rem", 
        fontSize: "0.8rem", 
        color: "var(--text-muted)",
        textAlign: "center"
      }}>
        Enter a PR number from facebook/react to analyze its merge time
      </p>
    </div>
  );
}
