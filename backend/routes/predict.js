const express = require("express");
const router = express.Router();
const axios = require("axios");

const PY_API = process.env.PYTHON_API_URL;

router.post("/", async (req, res, next) => {
  try {
    const resp = await axios.post(`${PY_API}/predict`, {
      owner: req.body.owner,
      repo: req.body.repo,
      pr_number: req.body.pr_number,
    });
    res.json(resp.data);
  } catch (err) {
    next(err);
  }
});

module.exports = router;
