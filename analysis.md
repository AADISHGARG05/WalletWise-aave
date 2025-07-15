# 📊 Aave Credit Score ML — Wallet Score Analysis

This analysis highlights the behavioral trends of wallets across various credit score ranges. Using engineered features and model explainability (SHAP), we provide insights into how DeFi wallets on the Aave V2 protocol are distributed, and what defines risky vs. trustworthy behavior.

---

### 📉 Distribution Chart

- Most wallets cluster in the **mid-range (400–600)**.
- **High-scoring wallets (800+)** form a small elite segment.
- **Low-scoring wallets (0–200)** suggest risky or bot-like behavior.

---

## 🚩 Behavioral Patterns: Low-Scoring Wallets (0–200)

These wallets generally show:

- 🔥 **Frequent Liquidations:** `liquidationcall_count` ≥ 3  
- 📉 **High Borrow-to-Repay Ratio:** Often > 2.5  
- ⏳ **Short Activity Span:** Active for < 7 days  
- 🪙 **Minimal Deposits:** Low or negative `net_deposit_usd`  
- 🔁 **Repetitive Behavior:** Low `entropy_of_actions`

**Interpretation:** Likely risky actors, liquidation-prone users, bots, or flashloan abusers.

---

## 🧠 Behavioral Patterns: High-Scoring Wallets (800–1000)

These wallets consistently show:

- ✅ **Zero Liquidations:** Highly reliable usage
- 📊 **Large and Net-Positive Deposits:** `net_deposit_usd` is significantly positive
- 🧩 **Diversified Activity:** High `entropy_of_actions` with varied interaction types
- 📆 **Long-Term Engagement:** Active for 90+ days
- 🔄 **Consistent Repayments:** `borrow_repay_ratio` close to 1

**Interpretation:** Likely long-term users, LPs, DAOs, or highly responsible borrowers.

---

## 📊 Top 5 Features (SHAP Analysis)

The SHAP values indicate that the following features were most influential in score prediction:

1. **liquidationcall_count** — 🚨 Risk indicator
2. **net_deposit_usd** — 💰 Capital trustworthiness
3. **borrow_repay_ratio** — 📉 Debt reliability
4. **txn_per_day** — 📈 Engagement frequency
5. **entropy_of_actions** — 🔀 Behavioral diversity

---

## 🧪 Example Wallets

| Wallet | tx_count | net_deposit_usd | liquidations | entropy | Score |
|--------|----------|------------------|--------------|---------|--------|
| `0xA1...` | 240      | 10,250.80        | 0            | 2.1     | 920    |
| `0xB2...` | 40       | -560.00          | 3            | 0.5     | 180    |

*Note: High scores align with long-term, varied usage; low scores correlate with exploitative behavior.*

---

## 🧭 Insights & Applications

- 📌 **Protocol Insights:** Helps identify high-value users vs. risky borrowers.
- 📊 **Risk Profiling:** Adds a scoring layer for underwriting or protocol governance.
- 🔗 **Cross-Chain Use:** This model can be extended to other lending protocols.
- ⚙️ **APIs / Dashboards:** Ready for integration with live wallet dashboards or risk oracles.

---

## 🔮 Future Enhancements

- 📉 **Time-decayed features** to emphasize recent behavior
- 🌐 **Cross-protocol analysis** using wallet linkages
- 🧠 **Clustering + segmentation** of wallets beyond scoring
- 🚀 **Real-time scoring API** using FastAPI for protocol plug-in

---

## 🏁 Conclusion

This wallet scoring system offers a scalable, interpretable way to analyze user behavior on DeFi lending protocols like Aave V2. With well-defined features and explainable ML, it brings the rigor of credit scoring to the decentralized finance world.

> 🧠 **Smart wallets deserve smart scoring.**
