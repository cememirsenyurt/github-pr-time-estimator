// frontend/src/components/PRInputForm.jsx
import { useState } from "react";

export default function PRInputForm({ onSubmit }) {
  const [owner, setOwner] = useState("facebook");
  const [repo, setRepo] = useState("react");
  const [prNumber, setPrNumber] = useState("32812");

  const handleSubmit = (e) => {
    e.preventDefault();
    // make sure pr_number is an integer
    onSubmit({
      owner: owner.trim(),
      repo: repo.trim(),
      pr_number: parseInt(prNumber, 10),
    });
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: "grid", gap: "1rem" }}>
      <label>
        Owner
        <input
          type="text"
          value={owner}
          onChange={(e) => setOwner(e.target.value)}
          required
        />
      </label>

      <label>
        Repo
        <input
          type="text"
          value={repo}
          onChange={(e) => setRepo(e.target.value)}
          required
        />
      </label>

      <label>
        PR #
        <input
          type="number"
          value={prNumber}
          onChange={(e) => setPrNumber(e.target.value)}
          required
        />
      </label>

      <button type="submit" style={{ padding: "0.75rem", fontSize: "1rem" }}>
        Estimate Merge Time
      </button>
    </form>
  );
}
