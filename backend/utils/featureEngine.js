// Convert a GitHub PR JSON into the numeric features your model expects
function featurize(pr) {
  return {
    title_length: pr.title?.length || 0,
    body_length: pr.body?.length || 0,
    num_labels: Array.isArray(pr.labels) ? pr.labels.length : 0,
    is_closed: pr.state?.toLowerCase() === "closed" ? 1 : 0,
  };
}

module.exports = { featurize };
