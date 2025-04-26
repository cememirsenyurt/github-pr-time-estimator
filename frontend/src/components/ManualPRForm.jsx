// frontend/src/components/ManualPRForm.jsx
import { useState, useRef, useEffect } from "react";
import LabelDropdown from "./LabelDropdown";

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
    ta.style.height = ta.scrollHeight + "px";
  }, [body]);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      title,
      body,
      num_labels: labels.length,
      is_closed: false,        // always false for a new/manual PR
    });
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="max-w-2xl mx-auto p-6 flex flex-col space-y-10 bg-gradient-to-br from-gray-900 to-black rounded-3xl shadow-[0_0_30px_rgba(0,255,255,0.5)]"
    >
      <h2 className="text-4xl font-extrabold text-cyan-300 text-center">
        Manual PR Entry
      </h2>

      <div className="flex flex-col">
        <label className="mb-2 text-lg font-semibold text-white">Title</label>
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          required
          placeholder="Enter PR title…"
          className="p-3 rounded-2xl bg-white/10 backdrop-blur-sm border border-cyan-400/50 text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-cyan-400 transition-all"
        />
      </div>

      <div className="flex flex-col">
        <label className="mb-2 text-lg font-semibold text-white">Description</label>
        <textarea
          ref={textareaRef}
          rows={1}
          value={body}
          onChange={(e) => setBody(e.target.value)}
          placeholder="Describe what your PR does…"
          className="p-3 rounded-2xl bg-white/10 backdrop-blur-sm border border-cyan-400/50 text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-cyan-400 transition-all overflow-hidden resize-none min-h-[3rem]"
        />
      </div>

      <div className="flex flex-col">
        <label className="mb-2 text-lg font-semibold text-white">Labels</label>
        <LabelDropdown
          selected={labels}
          onChange={setLabels}
          placeholder="Select labels…"
        />
      </div>

      <button
        type="submit"
        className="mt-6 py-4 bg-cyan-500 hover:bg-cyan-600 rounded-2xl text-xl font-bold text-black uppercase shadow-[0_0_20px_rgba(0,255,255,0.7)] transition-all"
      >
        Estimate Merge Time
      </button>
    </form>
  );
}
