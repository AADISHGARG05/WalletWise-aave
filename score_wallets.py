
"""
score_wallets.py
Usage:
    python score_wallets.py path/to/aave_txs.json

Input: Aave V2 transaction-level JSON (as described).
Output:
    - scores.csv         (wallet, features..., score)
    - shap_summary.png   (global feature importances)
    - score_hist.png     (score distribution)
"""

import json, sys
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from datetime import datetime
import matplotlib.pyplot as plt

def load_data(json_path):
    print(f"Loading data from {json_path}...")
    with open(json_path) as f:
        raw = json.load(f)

    # Flatten each entry
    records = []
    for tx in raw:
        record = {
            'wallet': tx.get('userWallet'),
            'timestamp': tx.get('timestamp'),
            'action': tx.get('action')
        }
        action_data = tx.get('actionData', {})
        record['amount'] = float(action_data.get('amount', 0)) / 1e6  # Assuming USDC 6 decimals
        record['usd_price'] = float(action_data.get('assetPriceUSD', 1.0))
        records.append(record)

    df = pd.DataFrame(records)
    return df

def engineer_features(df):
    print("Engineering features...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['usd_amount'] = df['amount'] * df['usd_price']
    by_w = df.groupby('wallet')
    F = by_w['usd_amount'].agg([
        ('tx_count','count'),
        ('total_usd','sum'),
        ('avg_tx_usd','mean'),
        ('std_tx_usd','std')]).fillna(0)

    minmax = by_w['timestamp'].agg(['min','max'])
    F['active_days'] = (minmax['max'] - minmax['min']).dt.days + 1
    F['txn_per_day'] = F['tx_count'] / F['active_days'].replace(0,1.0)

    for action in ['deposit','borrow','repay','redeemunderlying','liquidationcall']:
        F[f'{action}_count'] = by_w.apply(lambda x,a=action: (x['action']==a).sum())

    F['net_deposit_usd'] = by_w.apply(lambda x:
        x.loc[x.action=='deposit','usd_amount'].sum() -
        x.loc[x.action=='redeemunderlying','usd_amount'].sum())

    F['net_borrow_usd'] = by_w.apply(lambda x:
        x.loc[x.action=='borrow','usd_amount'].sum() -
        x.loc[x.action=='repay','usd_amount'].sum())

    F['borrow_repay_ratio'] = F['borrow_count'] / (F['repay_count'] + 1.0)
    F['liquidation_flag'] = (F['liquidationcall_count'] > 0).astype(int)
    F['liquidation_freq'] = F['liquidationcall_count'] / F['active_days'].replace(0,1.0)
    # Convert to timezone-naive timestamp
    now_naive = pd.Timestamp.utcnow().tz_localize(None)
    F['days_since_last_tx'] = (now_naive - by_w['timestamp'].max()).dt.days


    def entropy(x):
        p = x.value_counts(normalize=True)
        return -(p * np.log2(p + 1e-9)).sum()
    F['entropy_of_actions'] = by_w['action'].apply(entropy)

    return F.fillna(0)

def label_wallets(F):
    print("Assigning pseudo‑labels...")
    risky = (F['liquidationcall_count'] >= 3) | (F['borrow_repay_ratio'] > 2)
    safe = (F['tx_count'] >= 100) & (F['liquidationcall_count'] == 0) & (F['net_deposit_usd'] > 0)
    F['label'] = np.where(risky, 200, np.where(safe, 800, 500))
    return F

def train_model(X, y):
    print("Training XGBoost regressor...")
    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X, y)
    return model

def create_visuals(F):
    print("Creating visualizations...")
    plt.figure(figsize=(8,5))
    F['score'].hist(bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribution of Wallet Scores")
    plt.xlabel("Score")
    plt.ylabel("Number of Wallets")
    plt.savefig("score_hist.png", bbox_inches='tight')
    plt.close()

def explain_shap(model, X):
    print("Computing SHAP explainability...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(10,7))
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("shap_summary.png", bbox_inches='tight')
    plt.close()

def main():
    if len(sys.argv) != 2:
        print("Usage: python score_wallets.py path/to/aave_txs.json")
        sys.exit(1)

    df = load_data(sys.argv[1])
    F = engineer_features(df)
    F = label_wallets(F)

    X = F.drop(columns=['label'])
    y = F['label']
    model = train_model(X, y)

    F['score'] = np.clip(model.predict(X), 0, 1000).round().astype(int)
    F.reset_index().to_csv('scores.csv', index=False)

    create_visuals(F)
    explain_shap(model, X)

    print("✅ Done! Outputs:")
    print("   - scores.csv")
    print("   - score_hist.png")
    print("   - shap_summary.png")

if __name__ == "__main__":
    main()
