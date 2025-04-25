const express = require("express");
const router = express.Router();
const axios = require("axios");
const { featurize } = require("../utils/featureEngine");

const PY_API = process.env.PYTHON_API_URL;

router.post("/", async (req, res, next) => {
  try {
    // 1. Featurize locally (optional—could also send raw PR to Python)
    const features = featurize(req.body);
    // 2. Send to Python microservice
    const resp = await axios.post(`${PY_API}/predict`, features);
    res.json(resp.data);
  } catch (err) {
    next(err);
  }
});

module.exports = router;
