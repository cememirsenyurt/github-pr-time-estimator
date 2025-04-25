import React from 'react';

export default function PRResult({ pr, predictedHours }) {
  if (!pr) return null;

  return (
    <div style={{
      background: '#fff',
      padding: '1rem',
      borderRadius: '4px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
    }}>
      <h2>PR #{pr.number}: {pr.title}</h2>
      <p><strong>State:</strong> {pr.state}</p>
      <p><strong>Created:</strong> {new Date(pr.created_at).toLocaleString()}</p>
      {predictedHours != null && (
        <p style={{ marginTop: '1rem', fontSize: '1.2rem' }}>
          Estimated merge time: <strong>{predictedHours.toFixed(1)} hours</strong>
        </p>
      )}
    </div>
  );
}
