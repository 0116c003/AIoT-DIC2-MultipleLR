"""
Script to generate crisp_dm_multiple_regression.ipynb notebook
"""
import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# AIoT-DA Lecture 6 & DIC-2: Multiple Linear Regression\n",
    "## 50 Startups & Boston Housing (CRISP-DM Methodology)\n",
    "\n",
    "**Course**: AIoT Data Analytics\n",
    "**Assignment**: DIC-2 Kaggle 50 Startups & Boston Housing Multiple Linear Regression\n",
    "**Methodology**: CRISP-DM (Cross-Industry Standard Process for Data Mining)\n",
    "\n",
    "---\n",
    "\n",
    "### CRISP-DM 6 Steps Overview:\n",
    "1. **Business Understanding**: Define predictive objective & ROI for stakeholders.\n",
    "2. **Data Understanding**: EDA, distributions, correlation matrices, scatter plots.\n",
    "3. **Data Preparation**: One-Hot Encoding, train/test split, feature scaling, reshape.\n",
    "4. **Modeling**: Multiple Linear Regression (OLS), Feature Selection (Lasso L1, SFS, RFE).\n",
    "5. **Evaluation**: Metrics (MSE, MAE, R²), overfitting vs underfitting analysis.\n",
    "6. **Deployment**: Model serialization (`.pkl`, `.pkl.gz`, REST API, Web App on GitHub Pages)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 1: 50 Startups Profit Prediction\n",
    "### Step 1: Business Understanding\n",
    "- **Goal**: Predict startup profit based on R&D spend, Administration expenses, Marketing spend, and State.\n",
    "- **Decision Value**: Venture Capitalists allocate investment capital to startups maximizing net profit."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import pickle\n",
    "import gzip\n",
    "\n",
    "# Step 1 & 2: Load Data\n",
    "url = 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv'\n",
    "try:\n",
    "    df = pd.read_csv('data/50_Startups.csv')\n",
    "except Exception:\n",
    "    df = pd.read_csv(url)\n",
    "\n",
    "print('Dataset Shape:', df.shape)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# EDA: Summary Statistics & Correlation\n",
    "print(df.describe())\n",
    "\n",
    "# EDA: Scatter Plot (R&D Spend vs Profit)\n",
    "plt.figure(figsize=(10, 4))\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.scatter(df['R&D Spend'], df['Profit'], color='#4361ee', alpha=0.8)\n",
    "plt.title('R&D Spend vs Profit (r ~ 0.97)')\n",
    "plt.xlabel('R&D Spend ($)')\n",
    "plt.ylabel('Profit ($)')\n",
    "plt.grid(True, linestyle='--', alpha=0.5)\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.scatter(df['Marketing Spend'], df['Profit'], color='#2ec4b6', alpha=0.8)\n",
    "plt.title('Marketing Spend vs Profit (r ~ 0.75)')\n",
    "plt.xlabel('Marketing Spend ($)')\n",
    "plt.ylabel('Profit ($)')\n",
    "plt.grid(True, linestyle='--', alpha=0.5)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 3: Data Preparation\n",
    "- One-hot encode the categorical column `State` (California, Florida, New York).\n",
    "- Drop first dummy column (`drop_first=True`) to avoid multicollinearity / dummy variable trap.\n",
    "- 80/20 Train-Test Split."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# One-Hot Encoding for categorical features\n",
    "df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)\n",
    "features = [col for col in df_encoded.columns if col != 'Profit']\n",
    "\n",
    "X = df_encoded[features].values.astype(float)\n",
    "y = df_encoded['Profit'].values.astype(float)\n",
    "\n",
    "# 80/20 Train/Test split\n",
    "np.random.seed(42)\n",
    "indices = np.random.permutation(len(X))\n",
    "split = int(0.8 * len(X))\n",
    "train_idx, test_idx = indices[:split], indices[split:]\n",
    "\n",
    "X_train, X_test = X[train_idx], X[test_idx]\n",
    "y_train, y_test = y[train_idx], y[test_idx]\n",
    "print(f'Train samples: {len(X_train)}, Test samples: {len(X_test)}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 4 & 5: Modeling & Evaluation\n",
    "- Fit OLS Multiple Linear Regression: $\\hat{\\beta} = (X^T X)^{-1} X^T y$\n",
    "- Evaluate MSE, MAE, R², and analyze Parsimony (Occam's razor: 2 features yield minimum test error)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Add intercept column\n",
    "X_tr_ext = np.column_stack([np.ones(len(X_train)), X_train])\n",
    "X_te_ext = np.column_stack([np.ones(len(X_test)), X_test])\n",
    "\n",
    "beta = np.linalg.pinv(X_tr_ext.T @ X_tr_ext) @ (X_tr_ext.T @ y_train)\n",
    "\n",
    "y_train_pred = X_tr_ext @ beta\n",
    "y_test_pred = X_te_ext @ beta\n",
    "\n",
    "# Metrics\n",
    "def calc_metrics(y_true, y_pred):\n",
    "    mse = np.mean((y_true - y_pred)**2)\n",
    "    rmse = np.sqrt(mse)\n",
    "    mae = np.mean(np.abs(y_true - y_pred))\n",
    "    r2 = 1.0 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))\n",
    "    return mse, rmse, mae, r2\n",
    "\n",
    "tr_mse, tr_rmse, tr_mae, tr_r2 = calc_metrics(y_train, y_train_pred)\n",
    "te_mse, te_rmse, te_mae, te_r2 = calc_metrics(y_test, y_test_pred)\n",
    "\n",
    "print(f'Full Model (5 features):')\n",
    "print(f'Train RMSE: ${tr_rmse:,.2f} | Test RMSE: ${te_rmse:,.2f}')\n",
    "print(f'Train R2:   {tr_r2:.4f}     | Test R2:   {te_r2:.4f}')\n",
    "\n",
    "# Optimal 2-feature model (R&D Spend + Marketing Spend)\n",
    "k2_indices = [features.index('R&D Spend'), features.index('Marketing Spend')]\n",
    "X_tr_k2 = np.column_stack([np.ones(len(X_train)), X_train[:, k2_indices]])\n",
    "X_te_k2 = np.column_stack([np.ones(len(X_test)), X_test[:, k2_indices]])\n",
    "beta_k2 = np.linalg.pinv(X_tr_k2.T @ X_tr_k2) @ (X_tr_k2.T @ y_train)\n",
    "\n",
    "y_te_k2_pred = X_te_k2 @ beta_k2\n",
    "_, k2_rmse, k2_mae, k2_r2 = calc_metrics(y_test, y_te_k2_pred)\n",
    "print('\\nOptimal Parsimonious Model (k=2: R&D + Marketing Spend):')\n",
    "print(f'Test RMSE: ${k2_rmse:,.2f} | Test R2: {k2_r2:.4f}')\n",
    "print(f'Equation: Profit = {beta_k2[0]:.2f} + {beta_k2[1]:.4f}*(R&D) + {beta_k2[2]:.4f}*(Marketing)')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 6: Deployment\n",
    "- Save trained pipeline weights as `.pkl` and `.pkl.gz` for production serving."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "os.makedirs('models', exist_ok=True)\n",
    "\n",
    "deployment_bundle = {\n",
    "    'model': 'MultipleLinearRegression',\n",
    "    'features': ['R&D Spend', 'Marketing Spend'],\n",
    "    'intercept': float(beta_k2[0]),\n",
    "    'coefficients': beta_k2[1:].tolist(),\n",
    "    'rmse': float(k2_rmse),\n",
    "    'r2': float(k2_r2)\n",
    "}\n",
    "\n",
    "with open('models/startup_profit_model.pkl', 'wb') as f:\n",
    "    pickle.dump(deployment_bundle, f)\n",
    "\n",
    "with gzip.open('models/startup_profit_model.pkl.gz', 'wb') as f:\n",
    "    pickle.dump(deployment_bundle, f)\n",
    "\n",
    "print('Saved model successfully to models/startup_profit_model.pkl and .pkl.gz!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 2: Boston Housing Price Prediction\n",
    "### Applying the identical CRISP-DM Workflow to Problem 2\n",
    "- Target: `medv` (Median value of owner-occupied homes in $1000s)\n",
    "- 13 Features: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Boston Housing Dataset\n",
    "boston_df = pd.read_csv('data/boston_housing.csv')\n",
    "print('Boston Housing Shape:', boston_df.shape)\n",
    "b_features = [c for c in boston_df.columns if c != 'medv']\n",
    "\n",
    "X_b = boston_df[b_features].values.astype(float)\n",
    "y_b = boston_df['medv'].values.astype(float)\n",
    "\n",
    "np.random.seed(42)\n",
    "b_indices = np.random.permutation(len(X_b))\n",
    "b_split = int(0.8 * len(X_b))\n",
    "b_train_idx, b_test_idx = b_indices[:b_split], b_indices[b_split:]\n",
    "\n",
    "X_b_train, X_b_test = X_b[b_train_idx], X_b[b_test_idx]\n",
    "y_b_train, y_b_test = y_b[b_train_idx], y_b[b_test_idx]\n",
    "\n",
    "# Fit OLS for Boston Housing\n",
    "X_b_tr_ext = np.column_stack([np.ones(len(X_b_train)), X_b_train])\n",
    "X_b_te_ext = np.column_stack([np.ones(len(X_b_test)), X_b_test])\n",
    "beta_b = np.linalg.pinv(X_b_tr_ext.T @ X_b_tr_ext) @ (X_b_tr_ext.T @ y_b_train)\n",
    "\n",
    "y_b_test_pred = X_b_te_ext @ beta_b\n",
    "_, b_rmse, b_mae, b_r2 = calc_metrics(y_b_test, y_b_test_pred)\n",
    "\n",
    "print('Boston Housing OLS Results:')\n",
    "print(f'Test RMSE: {b_rmse:.3f} ($k) | Test MAE: {b_mae:.3f} ($k) | Test R2: {b_r2:.4f}')\n",
    "\n",
    "# Save Boston model\n",
    "boston_bundle = {\n",
    "    'model': 'BostonHousingMultipleRegression',\n",
    "    'features': b_features,\n",
    "    'intercept': float(beta_b[0]),\n",
    "    'coefficients': beta_b[1:].tolist(),\n",
    "    'test_rmse': float(b_rmse),\n",
    "    'test_r2': float(b_r2)\n",
    "}\n",
    "with gzip.open('models/boston_housing_model.pkl.gz', 'wb') as f:\n",
    "    pickle.dump(boston_bundle, f)\n",
    "print('Boston Housing Model serialized successfully!')"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open('crisp_dm_multiple_regression.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print('crisp_dm_multiple_regression.ipynb generated successfully!')
