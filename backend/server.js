// backend/server.js
const path = require("path");
require("dotenv").config({ path: path.resolve(__dirname, "..", ".env") });

const express = require("express");
const cors = require("cors");

const githubRoute = require("./routes/github");
const predictRoute = require("./routes/predict");

const app = express();
app.use(cors());
app.use(express.json());

app.use("/api/github", githubRoute);
app.use("/api/predict", predictRoute);

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`🚀 Node API listening on port ${PORT}`));
