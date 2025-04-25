import axios from "axios";

const BASE = process.env.REACT_APP_API_URL || "http://localhost:3001";

export async function fetchPR(owner, repo, number) {
  const { data } = await axios.get(
    `${BASE}/api/github/${owner}/${repo}/${number}`
  );
  return data;
}

export async function predictMergeTime(prData) {
  const { data } = await axios.post(`${BASE}/api/predict`, prData);
  return data.predicted_hours;
}
