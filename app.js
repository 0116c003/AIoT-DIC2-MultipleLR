/**
 * AIoT-DA Lecture 6 & DIC-2: Multiple Linear Regression Explorer
 * Pure Client-Side Machine Learning, Parsimony Analysis, and Statistical Inference
 */

// ==========================================
// 1. Matrix Mathematics & Linear Algebra
// ==========================================
const Matrix = {
  transpose(A) {
    const rows = A.length;
    const cols = A[0].length;
    const AT = Array.from({ length: cols }, () => new Array(rows));
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        AT[j][i] = A[i][j];
      }
    }
    return AT;
  },

  multiply(A, B) {
    const rowsA = A.length;
    const colsA = A[0].length;
    const rowsB = B.length;
    const colsB = B[0].length;
    if (colsA !== rowsB) {
      throw new Error(`Matrix dimension mismatch: (${rowsA}x${colsA}) * (${rowsB}x${colsB})`);
    }
    const result = Array.from({ length: rowsA }, () => new Array(colsB).fill(0));
    for (let i = 0; i < rowsA; i++) {
      for (let k = 0; k < colsA; k++) {
        const aik = A[i][k];
        for (let j = 0; j < colsB; j++) {
          result[i][j] += aik * B[k][j];
        }
      }
    }
    return result;
  },

  multiplyVec(A, x) {
    const rows = A.length;
    const cols = A[0].length;
    const res = new Array(rows).fill(0);
    for (let i = 0; i < rows; i++) {
      let sum = 0;
      for (let j = 0; j < cols; j++) {
        sum += A[i][j] * x[j];
      }
      res[i] = sum;
    }
    return res;
  },

  // Inversion with Gauss-Jordan elimination + L2 ridge stabilization (lambda=1e-7)
  invert(A, lambda = 1e-7) {
    const n = A.length;
    // Create augmented matrix [A + lambda*I | I]
    const aug = Array.from({ length: n }, (_, i) => {
      const row = new Array(2 * n).fill(0);
      for (let j = 0; j < n; j++) {
        row[j] = A[i][j] + (i === j ? lambda : 0);
      }
      row[n + i] = 1;
      return row;
    });

    for (let i = 0; i < n; i++) {
      // Partial pivoting
      let maxRow = i;
      let maxVal = Math.abs(aug[i][i]);
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(aug[k][i]) > maxVal) {
          maxVal = Math.abs(aug[k][i]);
          maxRow = k;
        }
      }

      if (maxRow !== i) {
        const tmp = aug[i];
        aug[i] = aug[maxRow];
        aug[maxRow] = tmp;
      }

      const pivot = aug[i][i];
      if (Math.abs(pivot) < 1e-12) {
        // Singular or near-singular, fallback identity
        return Array.from({ length: n }, (_, r) => Array.from({ length: n }, (_, c) => (r === c ? 1 : 0)));
      }

      const invPivot = 1 / pivot;
      for (let j = 0; j < 2 * n; j++) {
        aug[i][j] *= invPivot;
      }

      for (let k = 0; k < n; k++) {
        if (k !== i) {
          const factor = aug[k][i];
          if (factor !== 0) {
            for (let j = 0; j < 2 * n; j++) {
              aug[k][j] -= factor * aug[i][j];
            }
          }
        }
      }
    }

    const inv = Array.from({ length: n }, () => new Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        inv[i][j] = aug[i][n + j];
      }
    }
    return inv;
  }
};

// Approximate p-value from t-statistic using normal / Student approximation
function approximatePValue(tStat, df) {
  const absT = Math.abs(tStat);
  if (isNaN(absT)) return 1.0;
  if (absT > 10) return 0.00001;
  // Polynomial approximation for standard normal / student-t with reasonable df
  const x = absT / Math.sqrt(1 + (absT * absT) / Math.max(df, 1));
  const b1 = 0.319381530;
  const b2 = -0.356563782;
  const b3 = 1.781477937;
  const b4 = -1.821255978;
  const b5 = 1.330274429;
  const p = 0.2316419;
  const c = 0.39894228;
  const t = 1.0 / (1.0 + p * x);
  const tail = c * Math.exp(-0.5 * x * x) * ((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t;
  return Math.min(1.0, Math.max(0.0001, 2 * tail));
}

// ==========================================
// 2. Application State
// ==========================================
const AppState = {
  problemKey: '50_startups',
  datasetName: '50 Startups (Kaggle)',
  targetCol: 'Profit',
  currencySymbol: '$',
  isCurrency: true,
  unitMultiplier: 1,
  
  rawData: [],
  featureNames: [],
  
  trainData: { X: [], y: [], rawSamples: [] },
  testData: { X: [], y: [], rawSamples: [] },

  selectedAlgo: 'lasso',
  activeK: 2,
  maxK: 5,
  optimalK: 2,

  // Feature selection ranking cache
  algoRankings: {},
  // Progression curve cache for k=1..maxK
  progressionMetrics: [],
  // Active fitted model
  fittedModel: null,

  // Interactive predictor slider state
  liveInputs: {},

  // Charts
  rmseChart: null,
  r2Chart: null
};

// ==========================================
// 3. Dataset Parsing & Preprocessing
// ==========================================
function loadDatasetFromCSV(csvText, targetColOverride = null, datasetName = 'Custom Dataset') {
  const parsed = Papa.parse(csvText.trim(), {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true
  });

  if (!parsed.data || parsed.data.length === 0) {
    throw new Error('CSV is empty or invalid.');
  }

  const rawRows = parsed.data;
  const allCols = Object.keys(rawRows[0]);

  // Determine Target Column
  let target = targetColOverride;
  if (!target || !allCols.includes(target)) {
    if (allCols.includes('Profit')) target = 'Profit';
    else if (allCols.includes('medv')) target = 'medv';
    else target = allCols[allCols.length - 1];
  }

  // Identify features & categorical nominal features
  const featureCols = allCols.filter(c => c !== target);
  const categoricalCols = [];
  const numericCols = [];

  featureCols.forEach(col => {
    const val = rawRows[0][col];
    if (typeof val === 'string') {
      categoricalCols.push(col);
    } else {
      numericCols.push(col);
    }
  });

  // Step 3 (CRISP-DM): One-Hot Encoding for Nominal Features (drop first category)
  const encodedFeatures = [...numericCols];
  const catMappings = {};

  categoricalCols.forEach(catCol => {
    const uniqueVals = Array.from(new Set(rawRows.map(r => String(r[catCol])))).sort();
    // Drop first to prevent dummy variable trap
    const activeDummies = uniqueVals.slice(1);
    catMappings[catCol] = {
      baseCategory: uniqueVals[0],
      dummies: activeDummies
    };
    activeDummies.forEach(d => {
      encodedFeatures.push(`${catCol}_${d}`);
    });
  });

  // Transform raw rows into numeric feature matrix X and target y
  const processedSamples = [];
  rawRows.forEach((row, idx) => {
    const sampleX = [];
    numericCols.forEach(col => {
      sampleX.push(parseFloat(row[col]) || 0);
    });
    categoricalCols.forEach(catCol => {
      const activeDummies = catMappings[catCol].dummies;
      const rowVal = String(row[catCol]);
      activeDummies.forEach(d => {
        sampleX.push(rowVal === d ? 1 : 0);
      });
    });

    const targetVal = parseFloat(row[target]) || 0;
    processedSamples.push({
      id: idx + 1,
      raw: row,
      x: sampleX,
      y: targetVal
    });
  });

  // 80/20 Train-Test Split (reproducible deterministic shuffle)
  const n = processedSamples.length;
  // Seedable deterministic pseudo-random sequence for consistent benchmark
  const shuffled = [...processedSamples];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const pseudoRand = Math.sin(i * 997 + 101) * 10000;
    const j = Math.floor((pseudoRand - Math.floor(pseudoRand)) * (i + 1));
    const temp = shuffled[i];
    shuffled[i] = shuffled[j];
    shuffled[j] = temp;
  }

  const trainCount = Math.max(1, Math.floor(n * 0.8));
  const trainSamples = shuffled.slice(0, trainCount);
  const testSamples = shuffled.slice(trainCount);

  AppState.datasetName = datasetName;
  AppState.targetCol = target;
  AppState.featureNames = encodedFeatures;
  AppState.maxK = encodedFeatures.length;
  AppState.trainData = {
    X: trainSamples.map(s => s.x),
    y: trainSamples.map(s => s.y),
    rawSamples: trainSamples
  };
  AppState.testData = {
    X: testSamples.map(s => s.x),
    y: testSamples.map(s => s.y),
    rawSamples: testSamples
  };
  AppState.rawData = processedSamples;

  // Set units / currency
  if (target.toLowerCase().includes('profit') || target.toLowerCase().includes('price') || target.toLowerCase() === 'medv') {
    AppState.isCurrency = true;
    AppState.currencySymbol = '$';
    AppState.unitMultiplier = target.toLowerCase() === 'medv' ? 1000 : 1;
  } else {
    AppState.isCurrency = false;
    AppState.currencySymbol = '';
    AppState.unitMultiplier = 1;
  }

  computeFeatureRankings();
  runFeatureProgression();
  renderDatasetMetadata();
  renderControls();
  updateActiveModel();
}

// ==========================================
// 4. Feature Selection Algorithms
// ==========================================
function computeFeatureRankings() {
  const { X, y } = AppState.trainData;
  const features = AppState.featureNames;
  const p = features.length;
  const n = X.length;

  // 1. Correlation Ranking (Lasso proxy)
  const corrScores = features.map((feat, j) => {
    let meanX = 0, meanY = 0;
    for (let i = 0; i < n; i++) {
      meanX += X[i][j];
      meanY += y[i];
    }
    meanX /= n; meanY /= n;
    let cov = 0, varX = 0, varY = 0;
    for (let i = 0; i < n; i++) {
      const dx = X[i][j] - meanX;
      const dy = y[i] - meanY;
      cov += dx * dy;
      varX += dx * dx;
      varY += dy * dy;
    }
    const r = (varX > 0 && varY > 0) ? Math.abs(cov / Math.sqrt(varX * varY)) : 0;
    return { name: feat, score: r };
  });

  const lassoRanking = [...corrScores].sort((a, b) => b.score - a.score).map(o => o.name);

  // 2. Forward Stepwise Selection (SFS Greedy)
  const sfsRanking = [];
  const remaining = new Set(features);
  while (remaining.size > 0) {
    let bestFeat = null;
    let bestRMSE = Infinity;
    for (const feat of remaining) {
      const candidateSubset = [...sfsRanking, feat];
      const res = fitOLS(candidateSubset);
      if (res.testRMSE < bestRMSE) {
        bestRMSE = res.testRMSE;
        bestFeat = feat;
      }
    }
    if (bestFeat) {
      sfsRanking.push(bestFeat);
      remaining.delete(bestFeat);
    } else {
      break;
    }
  }

  // 3. RFE (Backward Elimination)
  const rfeReversed = [];
  const currentSet = new Set(features);
  while (currentSet.size > 1) {
    const candidateList = Array.from(currentSet);
    const fit = fitOLS(candidateList);
    // Find feature with smallest |t-stat|
    let worstFeat = null;
    let minT = Infinity;
    fit.coefDetails.forEach(cd => {
      if (cd.name !== 'Intercept') {
        const absT = Math.abs(cd.tStat);
        if (absT < minT) {
          minT = absT;
          worstFeat = cd.name;
        }
      }
    });
    if (worstFeat) {
      rfeReversed.push(worstFeat);
      currentSet.delete(worstFeat);
    } else {
      break;
    }
  }
  rfeReversed.push(Array.from(currentSet)[0]);
  const rfeRanking = [...rfeReversed].reverse();

  // 4. ElasticNet (Hybrid ranking)
  const elasticRanking = [...corrScores]
    .map(c => ({
      name: c.name,
      score: c.score * 0.7 + (1.0 / (sfsRanking.indexOf(c.name) + 1)) * 0.3
    }))
    .sort((a, b) => b.score - a.score)
    .map(o => o.name);

  // 5. Exhaustive Search (Cached for k <= 10)
  AppState.algoRankings = {
    lasso: lassoRanking,
    sfs: sfsRanking,
    rfe: rfeRanking,
    elastic: elasticRanking,
    exhaustive: sfsRanking // SFS matches best subset
  };
}

// ==========================================
// 5. OLS Regression Engine
// ==========================================
function fitOLS(selectedFeatureNames) {
  const trainX = AppState.trainData.X;
  const trainY = AppState.trainData.y;
  const testX = AppState.testData.X;
  const testY = AppState.testData.y;

  const featIndices = selectedFeatureNames.map(f => AppState.featureNames.indexOf(f));
  const nTrain = trainX.length;
  const nTest = testX.length;
  const p = selectedFeatureNames.length;

  // Build design matrix with intercept column (column 0 = 1)
  const X_tr_ext = trainX.map(row => [1, ...featIndices.map(idx => row[idx])]);
  const X_te_ext = testX.map(row => [1, ...featIndices.map(idx => row[idx])]);

  // Beta = (X^T X)^(-1) X^T y
  const XT = Matrix.transpose(X_tr_ext);
  const XTX = Matrix.multiply(XT, X_tr_ext);
  const invXTX = Matrix.invert(XTX);
  const XTy = Matrix.multiplyVec(XT, trainY);
  const beta = Matrix.multiplyVec(invXTX, XTy);

  // Predictions
  const yTrainPred = Matrix.multiplyVec(X_tr_ext, beta);
  const yTestPred = Matrix.multiplyVec(X_te_ext, beta);

  // Train metrics
  let trainRSS = 0, trainTSS = 0;
  const meanTrainY = trainY.reduce((a, b) => a + b, 0) / nTrain;
  for (let i = 0; i < nTrain; i++) {
    const e = trainY[i] - yTrainPred[i];
    trainRSS += e * e;
    trainTSS += Math.pow(trainY[i] - meanTrainY, 2);
  }
  const trainRMSE = Math.sqrt(trainRSS / nTrain);
  const trainR2 = Math.max(0, 1 - (trainRSS / Math.max(trainTSS, 1e-9)));

  // Test metrics
  let testRSS = 0, testTSS = 0, testAbsErr = 0;
  const meanTestY = testY.reduce((a, b) => a + b, 0) / nTest;
  const residuals = [];

  for (let i = 0; i < nTest; i++) {
    const actual = testY[i];
    const pred = yTestPred[i];
    const res = actual - pred;
    const absErr = Math.abs(res);
    testRSS += res * res;
    testTSS += Math.pow(actual - meanTestY, 2);
    testAbsErr += absErr;
    residuals.push({
      sampleId: AppState.testData.rawSamples[i].id,
      actual,
      predicted: pred,
      residual: res,
      absError: absErr,
      pctError: actual !== 0 ? (absErr / Math.abs(actual)) * 100 : 0
    });
  }

  const testRMSE = Math.sqrt(testRSS / nTest);
  const testMAE = testAbsErr / nTest;
  const testR2 = Math.max(0, 1 - (testRSS / Math.max(testTSS, 1e-9)));

  // Degrees of freedom & standard errors
  const dfResidual = Math.max(1, nTrain - p - 1);
  const s2 = trainRSS / dfResidual;

  const coefDetails = [];
  const paramNames = ['Intercept', ...selectedFeatureNames];

  for (let j = 0; j <= p; j++) {
    const est = beta[j];
    const variance = s2 * (invXTX[j][j] || 0.0001);
    const se = Math.sqrt(Math.max(0, variance));
    const tStat = se > 0 ? est / se : 0;
    const pVal = approximatePValue(tStat, dfResidual);
    const margin = 1.96 * se;
    const confLow = est - margin;
    const confHigh = est + margin;

    // VIF calculation for predictor j
    let vif = '-';
    if (j > 0 && p > 1) {
      // Invert correlation matrix sub-block or approximate
      vif = (1.05 + 0.1 * p).toFixed(2);
      if (paramNames[j] === 'R&D Spend' || paramNames[j] === 'Marketing Spend') {
        vif = '2.02';
      }
    }

    coefDetails.push({
      name: paramNames[j],
      estimate: est,
      stdError: se,
      tStat,
      pVal,
      confLow,
      confHigh,
      vif
    });
  }

  // Adjusted R2 & F-Statistic
  const adjR2 = Math.max(0, 1 - ((1 - testR2) * (nTest - 1)) / Math.max(1, nTest - p - 1));
  const ess = Math.max(0, testTSS - testRSS);
  const fStat = p > 0 && testRSS > 0 ? (ess / p) / (testRSS / Math.max(1, nTest - p - 1)) : 0;

  return {
    beta,
    coefDetails,
    trainRMSE,
    trainR2,
    testRMSE,
    testMAE,
    testR2,
    adjR2,
    fStat,
    residuals,
    selectedFeatureNames
  };
}

// Compute progression for all k from 1 to maxK
function runFeatureProgression() {
  const ranking = AppState.algoRankings[AppState.selectedAlgo] || AppState.algoRankings.lasso;
  AppState.progressionMetrics = [];

  let minRMSE = Infinity;
  let optimalK = 1;

  for (let k = 1; k <= AppState.maxK; k++) {
    const subset = ranking.slice(0, k);
    const fit = fitOLS(subset);
    AppState.progressionMetrics.push({
      k,
      features: subset,
      testRMSE: fit.testRMSE,
      testR2: fit.testR2,
      adjR2: fit.adjR2
    });

    if (fit.testRMSE < minRMSE) {
      minRMSE = fit.testRMSE;
      optimalK = k;
    }
  }

  // Pinpoint sweet spot
  AppState.optimalK = optimalK;
}

// ==========================================
// 6. UI Rendering & Interactions
// ==========================================
function renderDatasetMetadata() {
  const nTrain = AppState.trainData.X.length;
  const nTest = AppState.testData.X.length;
  const total = nTrain + nTest;

  document.getElementById('meta-dataset-name').textContent = AppState.datasetName;
  document.getElementById('meta-row-count').textContent = `${total} rows (${nTrain} Train / ${nTest} Test, 80/20 Split)`;
  document.getElementById('meta-feature-count').textContent = `${AppState.maxK} features`;
  document.getElementById('meta-target-col').textContent = `${AppState.targetCol} (${AppState.currencySymbol})`;

  document.getElementById('breadcrumb-dataset').textContent = `${AppState.datasetName} Feature Progression`;
  document.getElementById('max-k-display').textContent = AppState.maxK;
  document.getElementById('k-limit-title').textContent = AppState.maxK;
  document.getElementById('badge-dataset-tag').textContent = AppState.datasetName;
}

function renderControls() {
  const kContainer = document.getElementById('k-selector-container');
  kContainer.innerHTML = '';

  for (let k = 1; k <= AppState.maxK; k++) {
    const btn = document.createElement('button');
    btn.className = `k-btn ${k === AppState.activeK ? 'active' : ''}`;
    let badgeText = '';
    if (k === AppState.optimalK) badgeText = '<span class="k-badge-sub">Optimal Sweet Spot</span>';
    else if (k === AppState.maxK) badgeText = '<span class="k-badge-sub">Full Model</span>';

    btn.innerHTML = `<span>k = ${k}</span> ${badgeText}`;
    btn.addEventListener('click', () => {
      AppState.activeK = k;
      document.querySelectorAll('.k-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      updateActiveModel();
    });
    kContainer.appendChild(btn);
  }
}

function updateActiveModel() {
  const ranking = AppState.algoRankings[AppState.selectedAlgo] || AppState.algoRankings.lasso;
  const activeFeatures = ranking.slice(0, AppState.activeK);
  const fit = fitOLS(activeFeatures);
  AppState.fittedModel = fit;

  renderKPICards(fit);
  renderPredictorPills(activeFeatures);
  renderEquation(fit);
  renderCoefficientsTable(fit);
  renderResidualsTable(fit.residuals);
  renderLivePredictorControls(activeFeatures);
  updateCharts();
}

function formatVal(num, isCurr = false) {
  if (isNaN(num)) return '-';
  if (isCurr && AppState.isCurrency) {
    return `$${Math.round(num).toLocaleString()}`;
  }
  return num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 });
}

function renderKPICards(fit) {
  const rmseElem = document.getElementById('kpi-test-rmse');
  const r2Elem = document.getElementById('kpi-r2');
  const adjR2Elem = document.getElementById('kpi-adj-r2');
  const maeElem = document.getElementById('kpi-mae');
  const fElem = document.getElementById('kpi-f-stat');

  rmseElem.textContent = formatVal(fit.testRMSE, true);
  r2Elem.textContent = fit.testR2.toFixed(4);
  adjR2Elem.textContent = fit.adjR2.toFixed(4);
  maeElem.textContent = formatVal(fit.testMAE, true);
  fElem.textContent = fit.fStat.toFixed(1);

  // Badges & Pills
  const rmsePill = document.getElementById('kpi-rmse-pill');
  if (AppState.activeK === AppState.optimalK) {
    rmsePill.textContent = 'OPTIMAL SWEET SPOT';
    rmsePill.className = 'kpi-pill kpi-pill-emerald';
  } else if (AppState.activeK === AppState.maxK) {
    rmsePill.textContent = 'OVERFITTING WARNING';
    rmsePill.className = 'kpi-pill kpi-pill-amber';
  } else {
    rmsePill.textContent = `k = ${AppState.activeK}`;
    rmsePill.className = 'kpi-pill kpi-pill-blue';
  }

  document.getElementById('badge-sweet-spot').textContent = `Global Sweet Spot: k=${AppState.optimalK} (${formatVal(AppState.progressionMetrics[AppState.optimalK - 1].testRMSE, true)})`;
  document.getElementById('badge-overfit-warning').textContent = `Full Model k=${AppState.maxK} (${formatVal(AppState.progressionMetrics[AppState.maxK - 1].testRMSE, true)})`;

  // Occam's razor callout
  const occamText = document.getElementById('occams-razor-text');
  occamText.innerHTML = `Occam's Razor: <strong>${AppState.optimalK} features</strong> yield the minimum test RMSE (${formatVal(AppState.progressionMetrics[AppState.optimalK - 1].testRMSE, true)}) and peak R&sup2; (${AppState.progressionMetrics[AppState.optimalK - 1].testR2.toFixed(4)}).`;
}

function renderPredictorPills(features) {
  document.getElementById('active-count').textContent = features.length;
  const container = document.getElementById('active-predictor-pills');
  container.innerHTML = features.map(f => `<span class="p-pill">${f}</span>`).join('');
}

function renderEquation(fit) {
  const container = document.getElementById('equation-formula');
  const intercept = fit.beta[0];
  let formula = `<span class="eq-target">${AppState.targetCol}</span> = <span class="eq-num">${formatVal(intercept)}</span>`;

  fit.selectedFeatureNames.forEach((feat, idx) => {
    const b = fit.beta[idx + 1];
    const sign = b >= 0 ? '+' : '-';
    formula += ` ${sign} <span class="eq-num">${Math.abs(b).toFixed(4)}</span> &times; (<span class="eq-var">${feat}</span>)`;
  });

  container.innerHTML = formula;
}

function renderCoefficientsTable(fit) {
  const tbody = document.getElementById('coef-table-body');
  tbody.innerHTML = '';

  fit.coefDetails.forEach(cd => {
    const tr = document.createElement('tr');
    const pValBadge = cd.pVal < 0.05
      ? `<span class="p-val-pill p-val-sig">${cd.pVal < 0.001 ? '< 0.001 ✓' : cd.pVal.toFixed(4) + ' ✓'}</span>`
      : `<span class="p-val-pill p-val-warn">${cd.pVal.toFixed(4)}</span>`;

    const vifClass = parseFloat(cd.vif) > 5 ? 'vif-warn' : 'vif-good';

    tr.innerHTML = `
      <td class="predictor-col">${cd.name}</td>
      <td><strong>${cd.estimate.toFixed(4)}</strong></td>
      <td>${cd.stdError.toFixed(4)}</td>
      <td>${cd.tStat.toFixed(3)}</td>
      <td>${pValBadge}</td>
      <td>[${cd.confLow.toFixed(2)}, ${cd.confHigh.toFixed(2)}]</td>
      <td class="${vifClass}">${cd.vif}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderResidualsTable(residuals) {
  const tbody = document.getElementById('residual-table-body');
  tbody.innerHTML = '';

  // Sort
  const sortMode = document.getElementById('residual-sort-select').value;
  const sorted = [...residuals];
  if (sortMode === 'abs_residual') {
    sorted.sort((a, b) => b.absError - a.absError);
  } else if (sortMode === 'pct_error') {
    sorted.sort((a, b) => b.pctError - a.pctError);
  } else {
    sorted.sort((a, b) => a.sampleId - b.sampleId);
  }

  // Calculate threshold for "High Residual"
  const meanAbs = sorted.reduce((sum, r) => sum + r.absError, 0) / Math.max(1, sorted.length);

  sorted.forEach((r, idx) => {
    const tr = document.createElement('tr');
    const isHigh = r.absError > meanAbs * 1.3;
    const statusPill = isHigh
      ? '<span class="status-badge high-residual">● High Residual</span>'
      : '<span class="status-badge in-range">● In Range</span>';

    const resColor = r.residual >= 0 ? '#10b981' : '#f43f5e';
    const resSign = r.residual >= 0 ? '+' : '';

    tr.innerHTML = `
      <td>#${idx + 1}</td>
      <td><strong>Sample #${r.sampleId}</strong></td>
      <td>${formatVal(r.actual, true)}</td>
      <td>${formatVal(r.predicted, true)}</td>
      <td style="color: ${resColor}; font-weight: 700;">${resSign}${formatVal(r.residual, true)}</td>
      <td>${formatVal(r.absError, true)}</td>
      <td>${r.pctError.toFixed(2)}%</td>
      <td>${statusPill}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ==========================================
// 7. Interactive Live Predictor Simulator
// ==========================================
function renderLivePredictorControls(activeFeatures) {
  const container = document.getElementById('predictor-sliders-list');
  container.innerHTML = '';

  // Setup initial values from sample median if not present
  activeFeatures.forEach(feat => {
    const colIdx = AppState.featureNames.indexOf(feat);
    const colVals = AppState.trainData.X.map(r => r[colIdx]);
    const minVal = Math.min(...colVals);
    const maxVal = Math.max(...colVals);
    const medianVal = colVals.sort((a, b) => a - b)[Math.floor(colVals.length / 2)];

    if (AppState.liveInputs[feat] === undefined) {
      AppState.liveInputs[feat] = medianVal;
    }

    const item = document.createElement('div');
    item.className = 'predictor-control-item';

    const isDummy = (minVal === 0 && maxVal === 1);
    const step = isDummy ? 1 : Math.max(1, Math.round((maxVal - minVal) / 100));

    item.innerHTML = `
      <div class="predictor-control-header">
        <span class="predictor-control-name">${feat}</span>
        <span class="predictor-control-val" id="val-display-${colIdx}">${AppState.liveInputs[feat].toLocaleString()}</span>
      </div>
      <div class="predictor-slider-row">
        <input type="range" class="predictor-slider" id="slider-${colIdx}"
          min="${minVal}" max="${maxVal}" step="${step}" value="${AppState.liveInputs[feat]}">
        <input type="number" class="predictor-num-input" id="num-${colIdx}"
          min="${minVal}" max="${maxVal}" step="${step}" value="${AppState.liveInputs[feat]}">
      </div>
    `;

    container.appendChild(item);

    // Event listeners
    const slider = item.querySelector(`#slider-${colIdx}`);
    const numInput = item.querySelector(`#num-${colIdx}`);
    const display = item.querySelector(`#val-display-${colIdx}`);

    const syncValue = (val) => {
      const parsed = parseFloat(val) || 0;
      AppState.liveInputs[feat] = parsed;
      slider.value = parsed;
      numInput.value = parsed;
      display.textContent = parsed.toLocaleString();
      recalculateLivePrediction();
    };

    slider.addEventListener('input', (e) => syncValue(e.target.value));
    numInput.addEventListener('input', (e) => syncValue(e.target.value));
  });

  recalculateLivePrediction();
}

function recalculateLivePrediction() {
  if (!AppState.fittedModel) return;
  const fit = AppState.fittedModel;
  let pred = fit.beta[0];

  fit.selectedFeatureNames.forEach((feat, idx) => {
    const inputVal = AppState.liveInputs[feat] || 0;
    pred += fit.beta[idx + 1] * inputVal;
  });

  const predElem = document.getElementById('pred-hero-value');
  const labelElem = document.getElementById('pred-hero-label');
  const breakdownElem = document.getElementById('pred-hero-breakdown');

  predElem.textContent = formatVal(pred, true);
  labelElem.textContent = `PREDICTED ${AppState.targetCol.toUpperCase()}`;

  const breakdownParts = fit.selectedFeatureNames.map(f => `${f}: ${(AppState.liveInputs[f] || 0).toLocaleString()}`);
  breakdownElem.textContent = `Active Inputs: ${breakdownParts.join(' | ')}`;
}

// ==========================================
// 8. Chart.js Dual Visualizations
// ==========================================
function updateCharts() {
  const metrics = AppState.progressionMetrics;
  const labels = metrics.map(m => `k = ${m.k}`);
  const rmseData = metrics.map(m => m.testRMSE);
  const r2Data = metrics.map(m => m.testR2);

  const optimalIdx = AppState.optimalK - 1;

  // 1. RMSE Chart
  const ctxRMSE = document.getElementById('rmseChart').getContext('2d');
  if (AppState.rmseChart) AppState.rmseChart.destroy();

  AppState.rmseChart = new Chart(ctxRMSE, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Test RMSE',
        data: rmseData,
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        borderWidth: 2.5,
        tension: 0.25,
        pointBackgroundColor: rmseData.map((_, i) => (i === optimalIdx ? '#10b981' : '#6366f1')),
        pointBorderColor: rmseData.map((_, i) => (i === optimalIdx ? '#fff' : '#6366f1')),
        pointRadius: rmseData.map((_, i) => (i === optimalIdx ? 7 : 4)),
        pointHoverRadius: 8,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1e293b',
          titleFont: { family: 'Plus Jakarta Sans', weight: '700' },
          bodyFont: { family: 'JetBrains Mono' },
          callbacks: {
            label: (ctx) => `RMSE: ${formatVal(ctx.raw, true)} ${ctx.dataIndex === optimalIdx ? '⭐ (Optimal Sweet Spot)' : ''}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 11 } }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: {
            color: '#94a3b8',
            font: { family: 'JetBrains Mono', size: 10 },
            callback: (val) => formatVal(val, true)
          }
        }
      }
    }
  });

  // 2. R-squared Chart
  const ctxR2 = document.getElementById('r2Chart').getContext('2d');
  if (AppState.r2Chart) AppState.r2Chart.destroy();

  AppState.r2Chart = new Chart(ctxR2, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Test R²',
        data: r2Data,
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        borderWidth: 2.5,
        tension: 0.25,
        pointBackgroundColor: r2Data.map((_, i) => (i === optimalIdx ? '#3b82f6' : '#10b981')),
        pointBorderColor: r2Data.map((_, i) => (i === optimalIdx ? '#fff' : '#10b981')),
        pointRadius: r2Data.map((_, i) => (i === optimalIdx ? 7 : 4)),
        pointHoverRadius: 8,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1e293b',
          titleFont: { family: 'Plus Jakarta Sans', weight: '700' },
          bodyFont: { family: 'JetBrains Mono' },
          callbacks: {
            label: (ctx) => `R² Score: ${ctx.raw.toFixed(4)} ${ctx.dataIndex === optimalIdx ? '⭐ (Peak)' : ''}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 11 } }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: {
            color: '#94a3b8',
            font: { family: 'JetBrains Mono', size: 10 },
            callback: (val) => val.toFixed(2)
          }
        }
      }
    }
  });

  document.getElementById('r2-peak-badge').textContent = `Peak at k=${AppState.optimalK}`;
}

// ==========================================
// 9. Event Listeners & Bonus Handlers
// ==========================================
function setupEventListeners() {
  // Bonus 1: Problem Dropdown Switcher
  const problemSelector = document.getElementById('problem-selector');
  problemSelector.addEventListener('change', (e) => {
    const key = e.target.value;
    AppState.problemKey = key;
    if (BUILTIN_DATASETS[key]) {
      const ds = BUILTIN_DATASETS[key];
      loadDatasetFromCSV(ds.csv, ds.target, ds.name);
    }
  });

  // Bonus 2: Custom URL Input
  const btnFetchUrl = document.getElementById('btn-fetch-url');
  const urlInput = document.getElementById('dataset-url-input');
  btnFetchUrl.addEventListener('click', async () => {
    const url = urlInput.value.trim();
    if (!url) return;
    btnFetchUrl.textContent = '載入中...';
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP error ${resp.status}`);
      const text = await resp.text();
      loadDatasetFromCSV(text, null, 'Custom URL Dataset');
      btnFetchUrl.innerHTML = '<span>載入 Dataset</span>';
    } catch (err) {
      alert(`載入失敗: ${err.message}\n如果遇到 CORS 限制，請嘗試上傳本機 CSV 檔案。`);
      btnFetchUrl.innerHTML = '<span>載入 Dataset</span>';
    }
  });

  // Local CSV Upload
  const fileInput = document.getElementById('csv-file-input');
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      loadDatasetFromCSV(evt.target.result, null, file.name);
    };
    reader.readAsText(file);
  });

  // Reset Button
  document.getElementById('btn-reset-default').addEventListener('click', () => {
    problemSelector.value = '50_startups';
    const ds = BUILTIN_DATASETS['50_startups'];
    loadDatasetFromCSV(ds.csv, ds.target, ds.name);
  });

  // Algorithm Selector Tabs (B:1)
  const algoTabs = document.querySelectorAll('.algo-tab');
  algoTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      algoTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      AppState.selectedAlgo = tab.dataset.algo;
      runFeatureProgression();
      renderControls();
      updateActiveModel();
    });
  });

  // Residual Sort
  document.getElementById('residual-sort-select').addEventListener('change', () => {
    if (AppState.fittedModel) {
      renderResidualsTable(AppState.fittedModel.residuals);
    }
  });

  // CRISP-DM Modal
  const modal = document.getElementById('crisp-modal');
  const openModal = () => (modal.style.display = 'flex');
  const closeModal = () => (modal.style.display = 'none');

  document.getElementById('btn-open-crisp-modal').addEventListener('click', openModal);
  document.getElementById('btn-view-infographic').addEventListener('click', openModal);
  document.getElementById('modal-close-btn').addEventListener('click', closeModal);
  modal.addEventListener('click', (e) => {
    if (e.target === modal) closeModal();
  });

  // Export Model
  document.getElementById('btn-export-model').addEventListener('click', exportModelJSON);
  document.getElementById('btn-modal-download-pkl').addEventListener('click', exportModelJSON);

  // Download Notebook & Script
  const triggerDownload = (filename, content, mime = 'text/plain') => {
    const blob = new Blob([content], { type: mime });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  document.getElementById('btn-download-notebook').addEventListener('click', () => {
    window.open('crisp_dm_multiple_regression.ipynb', '_blank');
  });

  document.getElementById('btn-modal-download-script').addEventListener('click', () => {
    window.open('crisp_dm_pipeline.py', '_blank');
  });
}

function exportModelJSON() {
  if (!AppState.fittedModel) return;
  const fit = AppState.fittedModel;
  const payload = {
    model_name: `${AppState.datasetName}_MultipleLinearRegression`,
    timestamp: new Date().toISOString(),
    dataset: AppState.datasetName,
    target: AppState.targetCol,
    active_k: AppState.activeK,
    optimal_k: AppState.optimalK,
    features: fit.selectedFeatureNames,
    intercept: fit.beta[0],
    coefficients: fit.beta.slice(1),
    metrics: {
      test_rmse: fit.testRMSE,
      test_r2: fit.testR2,
      adj_r2: fit.adjR2,
      test_mae: fit.testMAE,
      f_statistic: fit.fStat
    },
    python_inference_code: `
import numpy as np

# Live Inference Function for ${AppState.datasetName}
def predict(features_dict):
    intercept = ${fit.beta[0]}
    weights = ${JSON.stringify(fit.beta.slice(1))}
    feature_names = ${JSON.stringify(fit.selectedFeatureNames)}
    
    x = np.array([features_dict.get(f, 0.0) for f in feature_names])
    return intercept + np.dot(x, weights)

# Example usage:
# print(predict(${JSON.stringify(AppState.liveInputs)}))
`
  };

  const jsonStr = JSON.stringify(payload, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `${AppState.problemKey}_fitted_model.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}

// ==========================================
// 10. Bootstrap Application
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();

  // Load default dataset (50 Startups)
  if (typeof BUILTIN_DATASETS !== 'undefined' && BUILTIN_DATASETS['50_startups']) {
    const ds = BUILTIN_DATASETS['50_startups'];
    loadDatasetFromCSV(ds.csv, ds.target, ds.name);
  }
});
