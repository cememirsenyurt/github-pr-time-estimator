// If you want to augment PR data with OpenAI, stub it here
const axios = require("axios");

async function summarize(body) {
  const resp = await axios.post(`${process.env.PYTHON_API_URL}/summarize`, {
    text: body,
  });
  return resp.data.summary;
}

module.exports = { summarize };
