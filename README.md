# 💳 Aave Credit Score ML — DeFi Wallet Risk Ranking

> 🔍 Assigning credit scores (0–1000) to Aave V2 wallets using on-chain transaction behavior and machine learning.

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Model](https://img.shields.io/badge/model-XGBoost-green)
![Status](https://img.shields.io/badge/status-Completed-success)

---

## 📁 Overview

This project uses raw Aave V2 transaction data to generate **interpretable credit scores** for DeFi wallets. These scores (ranging from **0 to 1000**) quantify a wallet’s behavior — high scores indicate trustworthiness, while low scores reveal risk, bots, or exploitative patterns.

---

## 🚀 Project Highlights

- 🔄 Flattened and cleaned 100K+ transaction-level JSON
- 🏗️ Engineered 15+ behavioral features per wallet
- 📊 Pseudo-labeling using domain-driven rules
- 🤖 Trained a gradient-boosted model (XGBoost)
- 📉 Explained feature impact with SHAP values
- 📈 Exported wallet scores and generated score distribution plots

---

## 🧠 Key Features

| Feature | Description |
|--------|-------------|
| `tx_count` | Total transactions per wallet |
| `net_deposit_usd` | Deposits - Withdrawals (in USD) |
| `borrow_repay_ratio` | Proxy for repayment reliability |
| `liquidationcall_count` | Number of liquidations triggered |
| `entropy_of_actions` | Behavioral diversity measure |
| `days_since_last_tx` | Recency of wallet activity |

---

## 🧪 Pseudo-Label Strategy

| Label | Criteria |
|-------|----------|
| 800 (Safe) | ≥ 100 txs, 0 liquidations, net deposit > 0 |
| 500 (Neutral) | Default |
| 200 (Risky) | ≥ 3 liquidations or borrow/repay ratio > 2 |

These pseudo-labels were used to train a supervised model, which outputs a final credit score ∈ [0, 1000].

---
## 🧱 Project Structure

aave-credit-score-ml/
├── score_wallets.py # Main pipeline script
├── aave_txs.json # Sample input data (100K txs)
├── scores.csv # Output: wallet features + scores
├── shap_summary.png # Global feature importances
├── score_hist.png # Score distribution chart
├── README.md # Project summary
└── analysis.md # Deeper insights & visual stats

---

## 🛠️ How to Run

> Requirements: `pandas`, `numpy`, `xgboost`, `shap`, `matplotlib`
📂 Output files:
> 
scores.csv: Credit scores and wallet features
shap_summary.png: Feature importance visualization
score_hist.png: Score range histogram

| wallet   | tx\_count | net\_deposit\_usd | liquidationcall\_count | score |
| -------- | --------- | ----------------- | ---------------------- | ----- |
| 0xabc... | 145       | 1200.50           | 0                      | 875   |
| 0xdef... | 15        | -230.10           | 2                      | 320   |

📈 Future Improvements
Time-weighted scoring for behavior decay
Integrate cross-protocol wallet activity
Deploy real-time API using FastAPI
Enhance labeling using community oracles

