// frontend/src/components/PRResult.jsx

import React from "react";

export default function PRResult({ result, error }) {
  if (error) {
    return <p className="error">❌ {error}</p>;
  }
  if (!result) return null;

  const {
    title = null,
    body = null,
    actual_hours = null,
    predicted_hours,
  } = result;

  return (
    <div className="mt-12 space-y-12">
      {/* Only show PR Details & Actual vs Predicted when we have actual_hours */}
      {actual_hours != null ? (
        <>
          {/* PR Details Card */}
          <div className="bg-gradient-to-r from-purple-800 to-indigo-800 p-8 rounded-2xl shadow-2xl ring-2 ring-offset-4 ring-indigo-600">
            <h3 className="text-3xl font-extrabold text-white mb-6 flex items-center space-x-3">
              <span className="text-4xl">📋</span>
              <span>PR Details</span>
            </h3>
            <div className="space-y-4 text-white">
              {title && (
                <p>
                  <span className="font-semibold">Title:</span> {title}
                </p>
              )}
              {body && (
                <div>
                  <span className="font-semibold block mb-2">Body:</span>
                  <div className="max-h-60 overflow-auto bg-black/40 p-4 rounded-lg ring-1 ring-white/20 whitespace-pre-wrap">
                    {body}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Actual vs Predicted */}
          <div className="flex flex-col sm:flex-row gap-12">
            <div className="flex-1 bg-gradient-to-br from-yellow-400 to-pink-400 p-6 rounded-xl shadow-lg ring-1 ring-yellow-300">
              <h4 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                ⏱ Actual Merge Time
              </h4>
              <p className="text-5xl font-extrabold text-white">
                {actual_hours.toFixed(2)}h
              </p>
            </div>
            <div className="flex-1 bg-gradient-to-br from-green-400 to-blue-500 p-6 rounded-xl shadow-lg ring-1 ring-green-300">
              <h4 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                🔮 Predicted Merge Time
              </h4>
              <p className="text-5xl font-extrabold text-white">
                {predicted_hours.toFixed(2)}h
              </p>
            </div>
          </div>
        </>
      ) : (
        /* Manual‐entry only predicted time */
        <div className="flex justify-center">
          <div className="w-full max-w-md bg-gradient-to-br from-green-400 to-blue-500 p-6 rounded-xl shadow-lg ring-1 ring-green-300 text-center">
            <h4 className="text-2xl font-bold text-white mb-4 flex items-center justify-center gap-2">
              🔮 Predicted Merge Time
            </h4>
            <p className="text-5xl font-extrabold text-white">
              {predicted_hours.toFixed(2)}h
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
