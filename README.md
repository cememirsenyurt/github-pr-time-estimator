# GitHub PR Time Estimator

An advanced ML-powered tool that predicts how long a GitHub Pull Request will take to merge. This project combines sophisticated feature engineering, multiple ML model comparison, and a modern full-stack application to provide accurate merge time predictions.

## 🌟 Features

- **Advanced ML Models**: Compares 7 different ML models (XGBoost, LightGBM, Random Forest, Gradient Boosting, Extra Trees, Ridge, Elastic Net) and automatically selects the best performer
- **Comprehensive Feature Engineering**: Extracts 16+ features from PR data including text analysis, temporal patterns, and author history
- **Modern UI**: Clean, responsive interface with real-time model metrics visualization
- **Dual Modes**: 
  - **Lookup Mode**: Analyze existing GitHub PRs by number
  - **Manual Mode**: Estimate merge time for new PRs before submission

## 📊 Model Performance

The system evaluates multiple regression models using cross-validation and test set metrics:

| Model | CV RMSE | Test RMSE | Test MAE | R² Score |
|-------|---------|-----------|----------|----------|
| XGBoost ⭐ | 37.93 | 16.02 | 10.79 | 59.4% |
| Extra Trees | 37.80 | 16.05 | 12.12 | 59.3% |
| Random Forest | 36.10 | 18.55 | 13.42 | 45.6% |
| LightGBM | 36.78 | 21.00 | 18.87 | 30.3% |
| Ridge | 37.86 | 21.32 | 17.05 | 28.2% |
| Elastic Net | 37.10 | 22.59 | 17.46 | 19.4% |
| Gradient Boosting | 40.96 | 27.31 | 16.14 | -17.8% |

**Best Model**: XGBoost with ~10.8 hours MAE on test set

## 🔧 Feature Engineering

The model uses 16 engineered features extracted from PR data:

### Text Features
- `title_length`: Character count of PR title
- `title_word_count`: Word count of PR title
- `body_length`: Character count of PR body/description
- `body_word_count`: Word count of PR body
- `body_length_log`: Log-transformed body length (handles long-tailed distribution)

### Binary Indicators
- `has_body`: Whether PR has a description
- `title_has_brackets`: Contains categorization brackets like [Feature], [Fix]
- `body_has_code`: Contains code blocks (```)
- `body_has_links`: Contains URLs
- `is_closed`: PR state indicator

### Metadata Features
- `num_labels`: Number of labels attached to PR
- `author_pr_count`: Historical PR count by author

### Temporal Features (Cyclic Encoding)
- `hour_sin`, `hour_cos`: Time of day when PR was created
- `day_sin`, `day_cos`: Day of week when PR was created

## 🏗️ Architecture

```
github-pr-time-estimator/
├── frontend/                 # React + Vite frontend
│   ├── src/
│   │   ├── components/       # React components
│   │   │   ├── App.jsx       # Main application
│   │   │   ├── GHInputForm.jsx    # GitHub PR lookup form
│   │   │   ├── ManualPRForm.jsx   # Manual PR entry form
│   │   │   ├── PRResult.jsx       # Results display
│   │   │   ├── ModelMetrics.jsx   # ML metrics dashboard
│   │   │   └── LabelDropdown.jsx  # Multi-select labels
│   │   └── ...
│   └── package.json
├── backend/
│   └── python-service/       # FastAPI backend
│       ├── app.py            # API endpoints
│       ├── pr_time_model.joblib  # Trained model
│       └── requirements.txt
├── ml/
│   ├── model/
│   │   ├── train_advanced_models.py  # Model training pipeline
│   │   ├── model_metrics.json        # Best model metrics
│   │   └── model_comparison.json     # All models comparison
│   ├── data/
│   │   ├── processed_pr_data.csv     # Training dataset
│   │   └── github_prs_raw.json       # Raw PR data
│   └── api/
│       └── github_fetch.py           # GitHub data fetching
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 18+
- GitHub Personal Access Token

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/github-pr-time-estimator.git
cd github-pr-time-estimator
```

2. **Set up environment variables**
```bash
# Create .env file in root directory
echo "GITHUB_TOKEN=your_github_token_here" > .env
```

3. **Install Python dependencies**
```bash
cd backend/python-service
pip install -r requirements.txt
```

4. **Install Node dependencies**
```bash
cd frontend
npm install
```

5. **Train the model (optional - pre-trained model included)**
```bash
cd ml/model
python train_advanced_models.py
```

### Running the Application

**Option 1: Run both services together**
```bash
npm start
```

**Option 2: Run services separately**

Backend:
```bash
cd backend/python-service
uvicorn app:app --reload --port 8080
```

Frontend:
```bash
cd frontend
npm run dev
```

Access the application at `http://localhost:5173`

## 📡 API Endpoints

### Health Check
```
GET /
```
Returns API status and available endpoints.

### Predict (Existing PR)
```
POST /predict
Content-Type: application/json

{
  "owner": "facebook",
  "repo": "react",
  "pr_number": 32812
}
```

### Estimate (Manual PR)
```
POST /estimate
Content-Type: application/json

{
  "title": "Your PR title",
  "body": "PR description...",
  "num_labels": 2,
  "is_closed": false
}
```

### Get Model Metrics
```
GET /metrics
```
Returns current model performance metrics.

### Get Model Comparison
```
GET /comparison
```
Returns comparison of all evaluated models.

## 🔬 Technical Details

### ML Pipeline
1. **Data Collection**: PRs fetched from GitHub API (facebook/react repository)
2. **Feature Engineering**: Extract text, temporal, and metadata features
3. **Model Training**: Train multiple regression models with cross-validation
4. **Model Selection**: Automatically select best model based on test RMSE
5. **Deployment**: Export model with feature columns for inference

### Model Selection Rationale
- **XGBoost** selected for best test performance and generalization
- Ensemble methods (XGBoost, Random Forest) outperform linear models
- Feature importance shows author history and temporal patterns are most predictive

### Training Data
- Source: facebook/react repository (closed & merged PRs)
- Sample size: 91 PRs
- Target variable: Time to merge (hours)
- Data characteristics:
  - Mean merge time: 18.07 hours
  - Median merge time: 0.86 hours
  - Range: 0.08 - 169.94 hours

## 🎨 UI Features

- **Dark theme** with modern glassmorphism design
- **Real-time metrics** dashboard showing model performance
- **Feature importance** visualization
- **Model comparison** table (expandable)
- **Responsive design** for mobile and desktop
- **Extracted features** display for transparency

## 📈 Future Improvements

- [ ] Add more training data from multiple repositories
- [ ] Implement deep learning models (LSTM for sequential patterns)
- [ ] Add PR complexity analysis (diff size, file types)
- [ ] Integrate with GitHub Actions for automated predictions
- [ ] Add confidence intervals to predictions
- [ ] Support for custom repository selection

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines before submitting PRs.

---

Built with ❤️ using React, FastAPI, XGBoost, and modern ML techniques.
