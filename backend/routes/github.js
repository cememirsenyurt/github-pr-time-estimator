const express = require("express");
const router = express.Router();
const { fetchPR } = require("../utils/github");

router.get("/:owner/:repo/:number", async (req, res, next) => {
  try {
    const pr = await fetchPR(
      req.params.owner,
      req.params.repo,
      req.params.number
    );
    res.json(pr);
  } catch (err) {
    next(err);
  }
});

module.exports = router;
