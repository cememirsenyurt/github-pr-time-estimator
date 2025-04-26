import { useState } from "react";

export default function GHInputForm({ onSubmit }) {
  const [owner, setOwner]     = useState("facebook");
  const [repo, setRepo]       = useState("react");
  const [prNumber, setPrNumber] = useState(1);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      owner: owner.trim(),
      repo: repo.trim(),
      pr_number: Number(prNumber),
    });
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: "grid", gap: "1rem" }}>
      <h2 style={{ color: "#5ac8fa" }}>Lookup Existing PR</h2>
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
          min="1"
          value={prNumber}
          onChange={(e) => setPrNumber(e.target.value)}
          required
        />
      </label>

      <button type="submit" className="btn">
        Fetch &amp; Predict
      </button>
    </form>
  );
}
