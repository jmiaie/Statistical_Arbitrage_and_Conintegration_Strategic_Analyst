from fpdf import FPDF
import textwrap

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Quant ML: Sentiment-Augmented Price Prediction', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, body)
        self.ln()

    def code_block(self, code):
        self.set_font('Courier', '', 9)
        self.set_fill_color(240, 240, 240)
        self.multi_cell(0, 5, code, 0, 'L', True)
        self.ln()

pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# --- Title Page ---
pdf.set_font('Arial', 'B', 24)
pdf.ln(50)
pdf.cell(0, 10, "Quant Machine Learning Project", 0, 1, 'C')
pdf.set_font('Arial', '', 16)
pdf.cell(0, 10, "Sentiment-Augmented Alpha Pipeline", 0, 1, 'C')
pdf.ln(20)
pdf.set_font('Arial', 'I', 14)
pdf.cell(0, 10, "Developer: Jeff Milam", 0, 1, 'C')
pdf.cell(0, 10, "https://github.com/jmiaie", 0, 1, 'C')
pdf.add_page()

# --- Content Sections ---

# 1. Overview
pdf.chapter_title("1. Project Overview")
pdf.chapter_body(
    "This project builds a rigorous Machine Learning pipeline for predicting short-term asset price movements. "
    "It fuses traditional market data (technical indicators) with alternative sentiment data (NLP). "
    "Key differentiators include strict Walk-Forward Validation to prevent look-ahead bias, SHAP value "
    "interpretation for model transparency, and production-grade engineering using Docker and CI/CD."
)

# 2. Architecture
pdf.chapter_title("2. Architecture & File Structure")
pdf.code_block(
"""quant-ml-project/
├── .github/workflows/   # CI/CD Pipeline
├── src/
│   ├── features.py      # Technical Indicators (RSI, MACD)
│   ├── validation.py    # Custom Walk-Forward Validator
│   └── sentiment.py     # NLP / Alternative Data Engine
├── tests/               # Pytest Unit Tests
├── backtest.py          # Main Execution Controller
├── Dockerfile           # Container Config
├── requirements.txt     # Python Dependencies
└── README.md            # Project Documentation"""
)

# 3. Validation Logic
pdf.chapter_title("3. Quant Rigor: Walk-Forward Validation")
pdf.chapter_body(
    "Standard K-Fold CV fails in finance because it shuffles time. We use an expanding window approach "
    "to ensure the model only trains on past data to predict the future."
)
pdf.chapter_body("File: src/validation.py")
pdf.code_block(
"""import pandas as pd
import numpy as np
from typing import Iterator, Tuple

class WalkForwardValidator:
    def __init__(self, n_splits: int = 5, train_window_size: int = 100, 
                 test_window_size: int = 1, expanding: bool = True):
        self.n_splits = n_splits
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.expanding = expanding

    def split(self, data: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(data)
        indices = np.arange(n_samples)
        available_steps = n_samples - self.train_window_size - self.test_window_size
        
        if available_steps < 0:
            raise ValueError("Data too small for requested window sizes.")
            
        step_size = available_steps // self.n_splits

        for i in range(self.n_splits):
            split_point = self.train_window_size + (i * step_size)
            start_train = 0 if self.expanding else split_point - self.train_window_size
            
            train_idx = indices[start_train : split_point]
            test_idx = indices[split_point : split_point + self.test_window_size]
            yield train_idx, test_idx"""
)

# 4. Feature Engineering
pdf.chapter_title("4. Feature Engineering (Stationarity)")
pdf.chapter_body(
    "We convert raw prices into stationary signals like Log Returns and Bollinger Band Z-Scores to help the ML model generalize."
)
pdf.chapter_body("File: src/features.py")
pdf.code_block(
"""import pandas as pd
import numpy as np

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Log Returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    # RSI
    df['rsi'] = calculate_rsi(df['close'])
    # Bollinger Z-Score
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['bb_zscore'] = (df['close'] - rolling_mean) / rolling_std
    # Volatility Regime
    df['vol_regime'] = df['close'].rolling(window=14).std() / df['close']
    df.dropna(inplace=True)
    return df"""
)

# 5. The Backtest Controller
pdf.chapter_title("5. Execution: The Backtest Controller")
pdf.chapter_body("File: backtest.py")
pdf.code_block(
"""import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from src.validation import WalkForwardValidator
from src.features import generate_advanced_features

def generate_mock_data(days: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=days)
    price = 100 * np.cumprod(1 + np.random.normal(0, 0.01, days))
    df = pd.DataFrame({'close': price}, index=dates)
    return df

def run_backtest():
    print("--- Starting Quant Backtest ---")
    data = generate_mock_data()
    df = generate_advanced_features(data)
    
    # Target: Next day UP (1) or DOWN (0)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    
    X = df[['log_ret', 'rsi', 'bb_zscore', 'vol_regime']]
    y = df['target']
    
    validator = WalkForwardValidator(n_splits=10, train_window_size=200, expanding=True)
    predictions, actuals = [], []
    
    for train_idx, test_idx in validator.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3, use_label_encoder=False)
        model.fit(X_train, y_train)
        predictions.append(model.predict(X_test)[0])
        actuals.append(y_test.values[0])
        
    print(f"Accuracy: {accuracy_score(actuals, predictions):.2%}")

if __name__ == "__main__":
    run_backtest()"""
)

# 6. CI/CD & DevOps
pdf.chapter_title("6. DevOps: Docker & CI/CD")
pdf.chapter_body("File: Dockerfile")
pdf.code_block(
"""FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["pytest"]"""
)
pdf.chapter_body("File: .github/workflows/main.yml")
pdf.code_block(
"""name: Quant CI Pipeline
on: [push, pull_request]
jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with: {python-version: "3.10"}
    - run: pip install -r requirements.txt
    - run: flake8 .
    - run: pytest tests/"""
)

# 7. Interview Prep
pdf.chapter_title("7. Interview Cheat Sheet")
pdf.chapter_body("Q: Why not use LSTM/Deep Learning?")
pdf.chapter_body(
    "A: For daily price data, signal-to-noise is low. Deep learning overfits. "
    "I chose XGBoost for better handling of tabular data and SHAP interpretability."
)
pdf.ln()
pdf.chapter_body("Q: Why Custom Validation instead of K-Fold?")
pdf.chapter_body(
    "A: K-Fold shuffles data, training on the future to predict the past (Look-Ahead Bias). "
    "My Walk-Forward validator strictly respects the arrow of time."
)
pdf.ln()
pdf.chapter_body("Q: Why FinBERT instead of VADER?")
pdf.chapter_body(
    "A: Financial language is nuanced. 'Liability' is neutral in accounting but negative in general NLP. "
    "FinBERT detects these domain-specific contexts."
)

# Save
file_name = "Quant_ML_Project_Jeff_Milam.pdf"
pdf.output(file_name)
print(f"✅ PDF generated successfully: {file_name}")