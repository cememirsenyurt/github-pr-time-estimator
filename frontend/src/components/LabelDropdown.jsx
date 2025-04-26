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
];

export default function LabelDropdown({ selected, onChange, placeholder }) {
  const customStyles = {
    control: (provided) => ({
      ...provided,
      background: "rgba(255,255,255,0.1)",
      borderColor: "rgba(255,255,255,0.3)",
      boxShadow: "0 0 10px rgba(0,255,255,0.5)",
      backdropFilter: "blur(10px)",
    }),
    menu: (provided) => ({
      ...provided,
      background: "#1a1a1a",
      boxShadow: "0 0 20px rgba(0,255,255,0.7)",
    }),
    option: (provided, state) => ({
      ...provided,
      background: state.isFocused ? "rgba(0,255,255,0.2)" : "transparent",
      color: "#fff",
      cursor: "pointer",
    }),
    multiValue: (provided) => ({
      ...provided,
      background: "rgba(0,255,255,0.3)",
      color: "#000",
    }),
    multiValueLabel: (provided) => ({
      ...provided,
      color: "#000",
      fontWeight: "600",
    }),
  };

  return (
    <Select
      isMulti
      options={options}
      value={options.filter((o) => selected.includes(o.value))}
      onChange={(vals) => onChange(vals.map((v) => v.value))}
      placeholder={placeholder}
      styles={customStyles}
      theme={(theme) => ({
        ...theme,
        borderRadius: 8,
        colors: {
          ...theme.colors,
          primary25: "rgba(0,255,255,0.2)",
          primary: "#00ffff",
        },
      })}
    />
  );
}
