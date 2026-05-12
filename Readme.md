# 📈 Quant Risk Regime Dashboard

## 🚀 Overview

This project combines advanced methods from **econometrics**, **machine learning**, and **quantitative finance** into a single interactive dashboard.

It models market volatility using **Student-t GARCH**, detects hidden market regimes using **Hidden Markov Models (HMM)**, estimates **Value at Risk (VaR)** and **Expected Shortfall (CVaR)**, performs **walk-forward validation**, and implements **cointegration-based pairs trading**.

The application is built with **Streamlit** and **Plotly**, allowing users to explore professional-grade analytics through an intuitive web interface.

---

## ✨ Key Features

### 📊 Volatility Forecasting

* Historical and rolling volatility
* ARCH / GARCH models
* EGARCH and GJR-GARCH
* Student-t innovations for fat tails
* Multi-step volatility forecasts

### 🧠 Market Regime Detection

* Hidden Markov Models (HMM)
* Bull, Sideways, and Bear state classification
* Regime transition probabilities
* Regime duration analysis

### 🛡️ Risk Management

* Historical VaR
* Parametric VaR
* Dynamic GARCH-t VaR
* CVaR (Expected Shortfall)
* Kupiec VaR backtesting
* Portfolio loss estimates in INR

### 📈 Strategy Backtesting

* Regime-based trading strategy
* Transaction cost adjustment
* Equity curve and drawdown analysis
* Sharpe, Sortino, and Calmar ratios

### 🔄 Walk-Forward Validation

* Rolling train/test splits
* Honest out-of-sample evaluation
* Performance stability analysis

### 🔗 Statistical Arbitrage

* Engle-Granger cointegration
* Johansen cointegration
* Error Correction Model (ECM)
* Z-score pairs trading

### 🌍 Multi-Asset Analysis

* Nifty 50
* Nifty Bank
* Nifty IT
* Return and volatility correlation heatmaps

### 🌐 Interactive Dashboard

* Streamlit web app
* Plotly visualizations
* Cached model loading with pickle files
* Fast deployment to Streamlit Community Cloud

---

## 🧮 Mathematical Foundations

### Log Returns

```math
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) \times 100
```

### GARCH(1,1)

```math
\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
```

### Value at Risk (VaR)

```math
VaR_t = -(\mu_t + z_\alpha \sigma_t)
```

### Expected Shortfall (CVaR)

```math
CVaR = E[L \mid L > VaR]
```

### Cointegration Spread

```math
Spread_t = \log(P_A) - \beta \log(P_B)
```

---

## 📁 Project Structure

```text
quant-risk-regime-dashboard/
│
├── app.py
├── create_model_pickle.py
├── requirements.txt
├── README.md
│
├── main_df.pkl
├── params.pkl
├── hmm_model.pkl
├── garch_model.pkl
├── scaler.pkl
├── labels_map.pkl
├── color_map.pkl
├── returns.pkl
│
├── notebook.ipynb
└── assets/
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/quant-risk-regime-dashboard.git
cd quant-risk-regime-dashboard
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\\Scripts\\activate       # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Step 1: Generate Pickle Files

```bash
python create_model_pickle.py
```

### Step 2: Run the Streamlit App

```bash
streamlit run app.py
```

### Step 3: Open the Dashboard

```text
http://localhost:8501
```

---

## 📊 Dashboard Sections

* Executive Summary
* Volatility Forecasting
* Regime Detection
* VaR & CVaR Analysis
* Strategy Backtesting
* Walk-Forward Validation
* Pairs Trading
* Correlation Heatmaps

---

## 📈 Performance Metrics

The project computes:

* Total Return
* Annual Return
* Annual Volatility
* Sharpe Ratio
* Sortino Ratio
* Calmar Ratio
* Maximum Drawdown
* Win Rate
* VaR Exceptions

---

## 📚 Statistical Tests Used

* Jarque-Bera Normality Test
* Augmented Dickey-Fuller (ADF) Test
* Engle-Granger Cointegration Test
* Johansen Cointegration Test
* Kupiec Proportion of Failures Test

---

## 🛠️ Technology Stack

### Languages & Libraries

* Python
* pandas
* NumPy
* SciPy
* statsmodels
* arch
* hmmlearn
* scikit-learn
* yfinance
* Plotly
* Streamlit

### Quantitative Finance Concepts

* GARCH and volatility forecasting
* Hidden Markov Models
* VaR and Expected Shortfall
* Cointegration and ECM
* Walk-forward analysis
* Statistical arbitrage

---

## 📌 Example Insights

### Volatility Forecasting

The model captures volatility clustering and produces dynamic one-step-ahead forecasts.

### Regime Detection

The HMM identifies hidden market states and estimates transition probabilities.

### Risk Management

Dynamic VaR and CVaR adjust to current market conditions and quantify tail risk.

### Trading Strategy

Regime-based allocation demonstrates how state detection can improve tactical positioning.

---

## 💼 Real-World Applications

This project demonstrates techniques used in:

* Quantitative Research
* Market Risk Management
* Hedge Funds
* Algorithmic Trading
* Portfolio Management
* Financial Engineering

---

## 🎯 Why This Project Stands Out

* Uses Student-t innovations to model fat tails
* Includes rigorous out-of-sample testing
* Adjusts for transaction costs
* Implements advanced econometric methods
* Combines multiple independent strategies
* Deployable as a professional web application

---

## 🌐 Live Demo

Add your Streamlit deployment link here.

---

## 👨‍💻 Author

**Amitesh Srivastava**

Economic Science graduate with strong interests in:

* Quantitative Finance
* Econometrics
* Machine Learning
* Risk Management
* Financial Data Science

---



---

## 🏁 Final Summary

This project integrates advanced methods from econometrics and quantitative finance into a unified platform:

* 📉 Student-t GARCH volatility forecasting
* 🧠 Hidden Markov regime detection
* 🛡️ Dynamic VaR and CVaR estimation
* 🔄 Walk-forward validation
* 🔗 Cointegration-based pairs trading
* 🌐 Interactive Streamlit dashboard

It is a strong portfolio project for roles in:

* Quantitative Research
* Data Science
* Risk Analytics
* Algorithmic Trading
* Financial Engineering
