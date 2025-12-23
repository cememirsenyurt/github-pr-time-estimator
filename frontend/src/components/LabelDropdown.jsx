// frontend/src/components/LabelDropdown.jsx
import React from "react";
import Select from "react-select";

const options = [
  { value: "bug", label: "🐞 Bug" },
  { value: "documentation", label: "📘 Documentation" },
  { value: "duplicate", label: "🔁 Duplicate" },
  { value: "enhancement", label: "🚀 Enhancement" },
  { value: "good first issue", label: "🌱 Good First Issue" },
  { value: "help wanted", label: "🆘 Help Wanted" },
  { value: "invalid", label: "🚫 Invalid" },
  { value: "question", label: "❓ Question" },
  { value: "wontfix", label: "🙅‍♂️ Won't Fix" },
  { value: "feature", label: "✨ Feature" },
  { value: "performance", label: "⚡ Performance" },
  { value: "security", label: "🔒 Security" },
  { value: "testing", label: "🧪 Testing" },
  { value: "refactor", label: "♻️ Refactor" },
];

export default function LabelDropdown({ selected, onChange, placeholder }) {
  const customStyles = {
    control: (provided, state) => ({
      ...provided,
      background: "rgba(15, 15, 35, 0.8)",
      borderColor: state.isFocused ? "var(--primary)" : "var(--border)",
      boxShadow: state.isFocused ? "0 0 0 3px var(--glow)" : "none",
      borderRadius: "0.75rem",
      padding: "0.25rem",
      "&:hover": {
        borderColor: "var(--primary)",
      },
    }),
    menu: (provided) => ({
      ...provided,
      background: "rgba(20, 20, 40, 0.98)",
      border: "1px solid var(--border)",
      borderRadius: "0.75rem",
      boxShadow: "0 10px 40px rgba(0, 0, 0, 0.5)",
      overflow: "hidden",
    }),
    menuList: (provided) => ({
      ...provided,
      padding: "0.5rem",
    }),
    option: (provided, state) => ({
      ...provided,
      background: state.isFocused ? "rgba(99, 102, 241, 0.2)" : "transparent",
      color: state.isSelected ? "var(--primary-light)" : "var(--text)",
      cursor: "pointer",
      borderRadius: "0.5rem",
      padding: "0.75rem 1rem",
      marginBottom: "0.25rem",
      "&:active": {
        background: "rgba(99, 102, 241, 0.3)",
      },
    }),
    multiValue: (provided) => ({
      ...provided,
      background: "rgba(99, 102, 241, 0.3)",
      borderRadius: "0.5rem",
    }),
    multiValueLabel: (provided) => ({
      ...provided,
      color: "var(--text)",
      fontWeight: "500",
      padding: "0.25rem 0.5rem",
    }),
    multiValueRemove: (provided) => ({
      ...provided,
      color: "var(--text-muted)",
      "&:hover": {
        background: "rgba(239, 68, 68, 0.3)",
        color: "var(--danger)",
      },
    }),
    placeholder: (provided) => ({
      ...provided,
      color: "var(--text-muted)",
      opacity: 0.6,
    }),
    input: (provided) => ({
      ...provided,
      color: "var(--text)",
    }),
    singleValue: (provided) => ({
      ...provided,
      color: "var(--text)",
    }),
  };

  return (
    <Select
      isMulti
      options={options}
      value={options.filter((o) => selected.includes(o.value))}
      onChange={(vals) => onChange(vals ? vals.map((v) => v.value) : [])}
      placeholder={placeholder}
      styles={customStyles}
      theme={(theme) => ({
        ...theme,
        borderRadius: 8,
        colors: {
          ...theme.colors,
          primary25: "rgba(99, 102, 241, 0.2)",
          primary: "#6366f1",
          neutral0: "rgba(15, 15, 35, 0.8)",
          neutral80: "#f1f5f9",
        },
      })}
    />
  );
}
