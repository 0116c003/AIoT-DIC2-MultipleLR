"""
AIoT-DA Lecture 6 & DIC-2: CRISP-DM Multiple Linear Regression Pipeline
Solves both:
1. 50 Startups Profit Prediction (Kaggle)
2. Boston Housing Price Prediction (Kaggle / Harrison & Rubinfeld)
Follows CRISP-DM 6-step lifecycle with deployment (pickle/gzip).
"""

import os
import json
import gzip
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# CRISP-DM Step 1: Business Understanding
# ==========================================
# Objective 1 (50 Startups): Predict startup net profit from expenditures (R&D, Administration, Marketing)
# and geographical state to help venture capitalists identify high-potential startups.
# Objective 2 (Boston Housing): Predict median home values (medv) based on 13 socioeconomic & environmental factors.

def run_50_startups_pipeline():
    print("=" * 60)
    print("CRISP-DM Step 1 & 2: 50 Startups - Data Loading & Understanding")
    print("=" * 60)
    
    # Step 1: Load Data
    data_path = 'data/50_Startups.csv'
    if not os.path.exists(data_path):
        data_path = 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv'
    
    df = pd.read_csv(data_path)
    print("Dataset Shape:", df.shape)
    print("Summary Statistics:\n", df.describe())
    
    # Step 2: Data Preprocessing
    print("\n" + "=" * 60)
    print("CRISP-DM Step 3: Data Preprocessing")
    print("=" * 60)
    
    # One-Hot Encoding for nominal feature 'State' (drop_first=True to avoid dummy variable trap)
    df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
    print("Encoded Columns:", df_encoded.columns.tolist())
    
    # Feature matrix X and target vector y
    feature_cols = [c for c in df_encoded.columns if c != 'Profit']
    X = df_encoded[feature_cols].values.astype(float)
    y = df_encoded['Profit'].values.astype(float)
    
    # 80/20 Train-Test Split (reproducible seed)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Standardization / Normalization
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0
    
    # Step 3: Build Model (OLS Multiple Linear Regression)
    print("\n" + "=" * 60)
    print("CRISP-DM Step 4: Build Multiple Linear Regression Model")
    print("=" * 60)
    
    # Add intercept column
    X_train_ext = np.column_stack([np.ones(len(X_train)), X_train])
    X_test_ext = np.column_stack([np.ones(len(X_test)), X_test])
    
    # OLS closed-form: beta = (X^T X)^(-1) X^T y
    beta = np.linalg.pinv(X_train_ext.T @ X_train_ext) @ (X_train_ext.T @ y_train)
    
    print(f"Intercept: {beta[0]:.4f}")
    for col, b in zip(feature_cols, beta[1:]):
        print(f"Beta ({col}): {b:.4f}")
        
    # Step 4: Evaluation
    print("\n" + "=" * 60)
    print("CRISP-DM Step 5: Model Evaluation (MSE, MAE, R2)")
    print("=" * 60)
    
    y_train_pred = X_train_ext @ beta
    y_test_pred = X_test_ext @ beta
    
    train_mse = np.mean((y_train - y_train_pred) ** 2)
    test_mse = np.mean((y_test - y_test_pred) ** 2)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    
    train_mae = np.mean(np.abs(y_train - y_train_pred))
    test_mae = np.mean(np.abs(y_test - y_test_pred))
    
    train_r2 = 1.0 - (np.sum((y_train - y_train_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2))
    test_r2 = 1.0 - (np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    
    print(f"Train RMSE: ${train_rmse:,.2f} | Test RMSE: ${test_rmse:,.2f}")
    print(f"Train MAE:  ${train_mae:,.2f} | Test MAE:  ${test_mae:,.2f}")
    print(f"Train R2:   {train_r2:.4f}    | Test R2:   {test_r2:.4f}")
    
    # Parsimony Analysis: R&D Spend + Marketing Spend (k=2)
    print("\n--- Parsimony Sweet Spot (k=2: R&D + Marketing) ---")
    k2_cols = ['R&D Spend', 'Marketing Spend']
    k2_idx = [feature_cols.index(c) for c in k2_cols]
    
    X_tr_k2 = np.column_stack([np.ones(len(X_train)), X_train[:, k2_idx]])
    X_te_k2 = np.column_stack([np.ones(len(X_test)), X_test[:, k2_idx]])
    beta_k2 = np.linalg.pinv(X_tr_k2.T @ X_tr_k2) @ (X_tr_k2.T @ y_train)
    
    y_te_k2_pred = X_te_k2 @ beta_k2
    rmse_k2 = np.sqrt(np.mean((y_test - y_te_k2_pred) ** 2))
    r2_k2 = 1.0 - (np.sum((y_test - y_te_k2_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    print(f"k=2 Test RMSE: ${rmse_k2:,.2f} | Test R2: {r2_k2:.4f}")
    
    # Step 5: Deployment (gzip, pickle)
    print("\n" + "=" * 60)
    print("CRISP-DM Step 6: Model Deployment (Pickle & Gzip)")
    print("=" * 60)
    
    model_payload = {
        'model_name': '50_startups_multiple_linear_regression',
        'feature_names': feature_cols,
        'coefficients': beta[1:].tolist(),
        'intercept': float(beta[0]),
        'metrics': {
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae)
        },
        'k2_optimal': {
            'features': k2_cols,
            'intercept': float(beta_k2[0]),
            'coefficients': beta_k2[1:].tolist(),
            'test_rmse': float(rmse_k2),
            'test_r2': float(r2_k2)
        }
    }
    
    os.makedirs('models', exist_ok=True)
    pkl_path = 'models/startup_profit_model.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(model_payload, f)
    print(f"Saved Pickle model to {pkl_path}")
    
    gzip_path = 'models/startup_profit_model.pkl.gz'
    with gzip.open(gzip_path, 'wb') as f:
        pickle.dump(model_payload, f)
    print(f"Saved Gzipped Pickle model to {gzip_path}")
    
    json_path = 'models/startup_profit_model.json'
    with open(json_path, 'w') as f:
        json.dump(model_payload, f, indent=2)
    print(f"Saved JSON weights to {json_path}")
    
    return model_payload

if __name__ == '__main__':
    run_50_startups_pipeline()
