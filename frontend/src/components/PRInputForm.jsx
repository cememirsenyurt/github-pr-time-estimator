import { useState } from "react";

export default function PRInputForm({ onSubmit }) {
  const [title,     setTitle]     = useState("");
  const [body,      setBody]      = useState("");
  const [numLabels, setNumLabels] = useState(0);
  const [isClosed,  setIsClosed]  = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      title,
      body,
      num_labels: numLabels,
      is_closed: isClosed,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4 max-w-lg">
      <div>
        <label className="block mb-1">Title</label>
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          className="w-full border rounded p-2"
          required
        />
      </div>
      <div>
        <label className="block mb-1">Body</label>
        <textarea
          value={body}
          onChange={(e) => setBody(e.target.value)}
          className="w-full border rounded p-2"
          rows={4}
        />
      </div>
      <div className="flex space-x-4">
        <div>
          <label className="block mb-1"># Labels</label>
          <input
            type="number"
            value={numLabels}
            onChange={(e) => setNumLabels(+e.target.value)}
            className="border rounded p-1 w-20"
            min={0}
          />
        </div>
        <div className="flex items-center">
          <input
            type="checkbox"
            checked={isClosed}
            onChange={(e) => setIsClosed(e.target.checked)}
            className="mr-2"
          />
          <label>Already Closed?</label>
        </div>
      </div>
      <button
        type="submit"
        className="px-4 py-2 bg-blue-600 text-white rounded"
      >
        Estimate Merge Time
      </button>
    </form>
  );
}
