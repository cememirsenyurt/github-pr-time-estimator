import React, { useState } from 'react';

export default function PRInputForm({ onSubmit }) {
  const [owner, setOwner] = useState('');
  const [repo, setRepo] = useState('');
  const [number, setNumber] = useState('');

  const handle = e => {
    e.preventDefault();
    onSubmit({ owner, repo, number: Number(number) });
  };

  return (
    <form onSubmit={handle} style={{ marginBottom: '1rem' }}>
      <input
        type="text"
        placeholder="Owner (e.g. facebook)"
        value={owner}
        onChange={e => setOwner(e.target.value)}
        required
      />
      <input
        type="text"
        placeholder="Repo (e.g. react)"
        value={repo}
        onChange={e => setRepo(e.target.value)}
        required
      />
      <input
        type="number"
        placeholder="PR #"
        value={number}
        onChange={e => setNumber(e.target.value)}
        required
      />
      <button type="submit">Go!</button>
    </form>
  );
}
