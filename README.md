# AIoT-DA Lecture 6 & DIC-2: Multiple Linear Regression Explorer (CRISP-DM)

![CRISP-DM Machine Learning Project](https://img.shields.io/badge/Methodology-CRISP--DM-6366f1)
![GitHub Pages Ready](https://img.shields.io/badge/Deployment-GitHub%20Pages-10b981)
![Interactive ML](https://img.shields.io/badge/Engine-Pure%20JavaScript%20OLS-3b82f6)

本專案為 **AIoT-DA Lecture 6 (DIC-2 作業)** 的完整開源專案實作，嚴格遵循 **CRISP-DM (Cross-Industry Standard Process for Data Mining)** 資料採掘與機器學習生命週期標準，並實現高度視覺化、交互式的 **多元線性回歸探索與模型簡約性分析系統 (Multiple Linear Regression Explorer)**。

---

## 🏆 作業計分與加分項目完全對應清單

| 評分項目 | 項目要求 | 專案實作成效與位置 |
| :--- | :--- | :--- |
| **基本要求** | 完成兩個主題（50 Startups 與 Boston Housing），並將第二個主題套用相同 skill 或 workflow | 完整實作於單一系統中，兩者均遵循相同的 6-Step CRISP-DM 工作流程與多元回歸引擎。 |
| **加總分 1 分** ⭐ | **有下拉視窗呈現兩種 Problem** | 頁面上方設有 **「選擇問題 (Select Problem)」** 下拉選單，可無縫即時切換 `Problem 1: 50 Startups Profit Prediction` 與 `Problem 2: Boston Housing Price Prediction`。 |
| **加總分 2 分** ⭐⭐ | **可任意輸入 Dataset** (例如 `https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv`) | 設有專用 **Dataset CSV 網址輸入框**，支援載入任意 Raw CSV URL（GitHub 等），同時支援 **本機 CSV 檔案上傳**，具備自動欄位對應與 One-Hot Encoding 解析。 |
| **部署發布** | GitHub Pages 官方自動部署工作流 | 已內建 `.github/workflows/pages.yml`，推送到 GitHub 即可透過 GitHub Actions 自動發布上線！ |

---

## 🌟 核心功能與視覺化亮點 (對應 StatCrunch / ML Dashboard)

1. **特徵選擇演算法切換 (B:1)**：
   - 支援 **Lasso (L1 Ranking)**、**Forward SFS (貪婪前向選擇)**、**RFE (遞迴特徵消除)**、**ElasticNet (彈性網絡)**、**Exhaustive Search (窮舉最佳子集)**。
2. **模型簡約性與特徵數探索 (B:2)**：
   - 即時切換特徵數量 $k=1 \dots k=5$（波士頓房價可達 $k=13$）。
   - 自動標記 **Global Sweet Spot 最優平衡點**（例如 50 Startups 之 $k=2$，測試集 RMSE 最低、R² 最高）與 **Overfitting 崖壁警告**（全變數過度擬合）。
3. **5 大關鍵統計 KPI 卡片**：
   - **Test RMSE** (均方根誤差，含最佳點標籤)
   - **R² Score** (判定係數，解釋變異量百分比)
   - **Adjusted R²** (調整後 R²，懲罰冗餘參數)
   - **Mean Absolute Error (MAE)** (平均絕對誤差)
   - **F-Statistic** (F 檢定統計量與 p-value 顯著性檢定)
4. **奧卡姆剃刀 (Occam's Razor) 原則標籤**：
   - 清楚闡述「如無必要，勿增實體」：2 個關鍵特徵即可達到最佳泛化性能。
5. **雙 Chart.js 動態交互圖表**：
   - **RMSE by Number of Features**：折線圖高亮最佳谷底。
   - **R-squared by Number of Features**：折線圖高亮頂峰。
6. **動態擬合 OLS 回歸方程式**：
   - 以高可讀性數學排版呈現截距與各特徵之回歸係數 $\hat{\beta}$。
7. **詳細係數與統計檢定表格**：
   - 包含 Predictor 名稱、Estimate ($\beta$)、Std Error、t-stat、p-value (綠色打勾標註顯著性)、95% 信賴區間、VIF (共線性診斷指標)。
8. **即時預測模擬器 (Live Outcome Predictor)**：
   - 提供各預測變數之即時滑桿 (Slider) 與數字輸入框。
   - 支援即時運算並顯示動態預測結果（如預測 Startup Profit 或房價 MEDV）。
9. **測試集殘差與離群值診斷表 (Residuals & Outliers Diagnostic)**：
   - 列出測試集樣本之實際值、預測值、殘差 ($e = y - \hat{y}$)、絕對誤差、百分比誤差，並標記診斷狀態（High Residual 或 In Range），支援排序切換。
10. **完整 CRISP-DM 專案 Log 彈出式視窗 (Matching Lecture 6)**：
    - 點擊按鈕即可查看完整 6 大階段生命週期圖解與詳細文字說明，並提供模型權重與 Python 代碼直接下載。

---

## 🔄 CRISP-DM 6 大步驟實作對應

- **Step 1: 商業理解 (Business Understanding)**
  - 50 Startups：預測新創公司淨利潤，協助創投機構 (VC) 依據研發 (R&D)、管理 (Administration)、行銷 (Marketing) 支出進行精準投資決策。
  - Boston Housing：預測波士頓地區房屋中位價 (MEDV)，評估周遭環境與犯罪率等社會經濟指標之影響。
- **Step 2: 資料理解 (Data Understanding / EDA)**
  - 檢視數據維度、無缺失值與重複值。
  - 關鍵發現：R&D Spend 與 Profit 呈極高正相關 ($r \approx 0.97$)，Marketing Spend 為次要影響因素 ($r \approx 0.75$)，Administration 相關性偏低 ($r \approx 0.20$)。
- **Step 3: 資料準備 (Data Preparation)**
  - 針對名義類別特徵（如 `State`：California、Florida、New York）執行 One-Hot Encoding，並設定 `drop_first=True` 避免虛擬變數陷阱 (Dummy Variable Trap)。
  - 80/20 訓練集與測試集切換，保證模型泛化能力評估。
- **Step 4: 模型建立 (Modeling)**
  - 採用最小平方法 (OLS) 多元線性回歸：$\hat{\beta} = (X^T X)^{-1} X^T y$。
  - 進行多特徵選擇方法之簡約性分析。
- **Step 5: 模型評估 (Evaluation)**
  - 綜合評估 MSE、RMSE、MAE、R²、Adjusted R²、t 檢定、p-value 與殘差分佈，驗證過度擬合 (Overfitting) 與擬合不足 (Underfitting)。
- **Step 6: 部署應用 (Deployment)**
  - 前端以純客戶端即時引擎進行預測與互動。
  - 後端模型保存為 Pickle (`startup_profit_model.pkl`) 與 Gzip 壓縮檔 (`startup_profit_model.pkl.gz`)。

---

## 🚀 GitHub Pages 1 步快速啟用部署指南

1. **建立並推送至 GitHub 倉庫**：
   ```bash
   git remote set-url origin https://github.com/0116c003/AIoT-DIC2-MultipleLR.git
   git branch -M main
   git push -u origin main
   ```
2. **在 GitHub 啟用 GitHub Pages**：
   - 進入倉庫頁面：`https://github.com/0116c003/AIoT-DIC2-MultipleLR/settings/pages`
   - 在 **Build and deployment** 區塊：
     - **Source** 下拉選單選擇 **`GitHub Actions`**。系統將自動執行 `.github/workflows/pages.yml`，約 30 秒內自動完成部署上線！
3. **完成發布**：
   - 稍等約 1 分鐘後重新整理，即可在頁面看到綠色勾勾與發布成功的公開網址：
     `https://0116c003.github.io/AIoT-DIC2-MultipleLR/`

---

## 📂 專案檔案架構

```text
AIoT-DIC2-MultipleLR/
├── .github/
│   └── workflows/
│       └── pages.yml                # GitHub Actions 自動部署工作流
├── data/
│   ├── 50_Startups.csv              # 基準資料集 1
│   └── boston_housing.csv           # 基準資料集 2
├── models/
│   ├── startup_profit_model.pkl     # Python 序列化 Pickle 模型
│   ├── startup_profit_model.pkl.gz  # Gzip 壓縮模型
│   └── startup_profit_model.json    # 模型權重與結構 JSON
├── index.html                       # 現代化響應式探索器介面 (加分功能完全整合)
├── style.css                        # StatCrunch 風格現代深色主題樣式表
├── data_builtin.js                  # 內建零 CORS 離線資料集
├── app.js                           # 純 JavaScript 矩陣運算與回歸統計核心
├── crisp_dm_pipeline.py             # 完整 CRISP-DM Python 訓練與導出腳本
├── crisp_dm_multiple_regression.ipynb # 完整 Jupyter Notebook 講義與範例代碼
└── README.md                        # 專案詳細說明與評分指引
```
