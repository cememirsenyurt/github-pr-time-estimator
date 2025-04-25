import React, { useState } from 'react';
import PRInputForm from './components/PRInputForm';
import PRResult from './components/PRResult';
import { fetchPR, predictMergeTime } from './api/backend';

export default function App() {
  const [pr, setPr] = useState(null);
  const [pred, setPred] = useState(null);
  const [error, setError] = useState('');

  const handleSubmit = async ({ owner, repo, number }) => {
    setError('');
    setPr(null);
    setPred(null);

    try {
      const prData = await fetchPR(owner, repo, number);
      setPr(prData);
      const hours = await predictMergeTime({
        title: prData.title,
        body: prData.body || '',
        labels: prData.labels || [],
        state: prData.state
      });
      setPred(hours);
    } catch (e) {
      setError(e.response?.data || e.message);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: '0 auto' }}>
      <h1>PR Merge Time Estimator</h1>
      <PRInputForm onSubmit={handleSubmit} />
      {error && <p style={{ color: 'red' }}>{error}</p>}
      <PRResult pr={pr} predictedHours={pred} />
    </div>
  );
}
