// frontend/src/components/GHInputForm.jsx
import { useState } from "react";

export default function GHInputForm({ onSubmit }) {
  const [prNumber, setPrNumber] = useState(1);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      owner: "facebook",
      repo: "react",
      pr_number: Number(prNumber),
    });
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: "grid", gap: "1rem" }}>
      <h2 style={{ color: "#5ac8fa" }}>Lookup Existing PR</h2>
      
      <div>
        <label className="block text-white font-semibold">Owner</label>
        <input
          type="text"
          value="facebook"
          readOnly
          className="p-2 rounded bg-gray-700 text-white cursor-not-allowed"
        />
      </div>

      <div>
        <label className="block text-white font-semibold">Repo</label>
        <input
          type="text"
          value="react"
          readOnly
          className="p-2 rounded bg-gray-700 text-white cursor-not-allowed"
        />
      </div>

      <div>
        <label className="block text-white font-semibold">PR #</label>
        <input
          type="number"
          min="1"
          value={prNumber}
          onChange={(e) => setPrNumber(e.target.value)}
          required
          className="p-2 rounded bg-gray-800 text-white focus:outline-none"
        />
      </div>

      <button
        type="submit"
        className="mt-4 py-2 bg-cyan-500 hover:bg-cyan-600 rounded text-white font-bold transition"
      >
        Fetch &amp; Predict
      </button>
    </form>
  );
}
