# 📊 Data Science & ML Portfolio — Shubham Khicha

A collection of end-to-end data science and machine learning projects covering NLP, LLM/RAG systems, supervised ML, time-series forecasting, LLM fine-tuning, model robustness evaluation, credit risk analytics, marketing mix modeling, and optimization.

## Projects

### 1. 📄 Insurance Document NLP Extractor

- **Tech:** Python, SpaCy, PDFPlumber, TF-IDF, Scikit-learn, Streamlit
- **What:** Extracts structured entities (policy number, insurer, premium, dates, sum insured) from unstructured insurance PDFs via regex + SpaCy NER, cross-checked against a TF-IDF/cosine-similarity schema matcher
- **Result:** 18/18 fields (100%) extracted correctly across the 3 bundled sample PDFs; 94% TF-IDF schema-match agreement; 45 passing tests

### 2. 🤖 LLM-Powered Document Q&A Tool (RAG)

- **Tech:** Python, LangChain, FAISS, Gemini/OpenAI API, Streamlit
- **What:** Retrieval-Augmented Generation system for querying business documents — PDF ingestion, chunking, embeddings + FAISS retrieval, and a citation-attaching LLM answer chain (offline TF-IDF embedding fallback when no API key is set)
- **Result:** Retrieval correctly ranks the relevant document per query across the 3 bundled sample PDFs (16 chunks total) with zero API key required; 34 passing tests, with the LLM/embedding layer mocked for fully offline testing

### 3. 🏠 House Price Prediction — End-to-End ML Pipeline

- **Tech:** Python, Scikit-learn, XGBoost, Pandas, Matplotlib, Streamlit
- **What:** Complete ML pipeline on the California Housing dataset (used as a free, openly-disclosed stand-in for Ames Housing) with EDA, feature engineering, and a 4-model comparison
- **Result:** Best RMSE $48,576 / R² 0.82 with XGBoost (tuned Random Forest via RandomizedSearchCV: RMSE $50,610 / R² 0.80); 35 passing tests

### 4. ⚡ Energy Consumption Forecasting — Time-Series ML

- **Tech:** Python, Scikit-learn, XGBoost, SARIMA (Statsmodels), Pandas, Matplotlib, Streamlit
- **What:** Time-series forecasting on the PJME hourly energy consumption dataset with ADF stationarity testing, calendar/lag/rolling-window feature engineering, and an XGBoost-vs-SARIMA comparison
- **Result:** XGBoost achieved ≈0.86% MAPE (RMSE ≈362 MW) on hourly data; SARIMA baseline on a daily-resampled subset achieved ≈8.17% MAPE; 29 passing tests

### 5. 🔧 LoRA Fine-Tuning — Document Field Extraction

- **Tech:** Python, PyTorch, Hugging Face Transformers, PEFT, bitsandbytes
- **What:** Fine-tunes an open-weight LLM (Qwen2.5-0.5B-Instruct) with a LoRA adapter to extract structured JSON fields from synthetic invoice/ID/receipt documents; includes a from-scratch training loop, 4-bit quantized loading, and adapter merging
- **Result:** Quantifies base-model vs. fine-tuned gains via exact-match and field-level accuracy

### 6. 🛡️ Robustness Evaluation Harness

- **Tech:** Python, PyTorch, Hugging Face Transformers, Scikit-learn
- **What:** Reusable harness that stress-tests text classifiers against OCR-style noise, token dropout, case/whitespace corruption, and adversarial synonym swaps across a severity sweep
- **Result:** Wilson 95% confidence intervals and an AUC-style robustness summary per perturbation, tested against both a CPU TF-IDF/LogReg baseline and a zero-shot LLM classifier

### 7. 🏦 Loan Portfolio Risk & ECL Analytics Dashboard

- **Tech:** Python, SQL (SQLite), Pandas, Scikit-learn, Streamlit
- **What:** ETL pipeline over a simulated loan book computing delinquency buckets, empirical roll-rate matrices, and Expected Credit Loss under a simplified IFRS 9 / Ind AS 109 3-stage framework, with a logistic regression early-warning score and a stress-test simulator
- **Result:** 49 passing tests; PD derived from roll-rate matrices (not assumed), live ECL recomputation under PD/LGD shocks

### 8. 📈 Marketing Mix Modeling & Sales Forecasting Tool

- **Tech:** Python, Statsmodels, Scikit-learn, Streamlit
- **What:** Adstock (carryover) and Hill-saturation (diminishing returns) transforms feeding a Ridge regression that attributes weekly sales to TV/digital/promotions spend, with channel contribution decomposition and an interactive budget scenario simulator
- **Result:** R^2 ~= 0.92 on the bundled synthetic dataset, 39 passing tests, automated Excel client report

### 9. 🗺️ Sales Force Optimization & Territory Allocation Model

- **Tech:** Python, SciPy, Pandas, Linear Programming
- **What:** Linear program (solved exactly via `scipy.optimize.linprog`) allocating reps to territories to maximize net revenue subject to coverage and capacity constraints, benchmarked against a naive baseline
- **Result:** +114.7% net revenue vs. baseline on the bundled instance, 21 passing tests including an independent recomputation of the objective value

### 10. 💊 Pharma Customer Segmentation & Targeting Analytics

- **Tech:** Python, Scikit-learn, K-Means, PCA, Streamlit
- **What:** PCA + K-Means clustering on synthetic physician prescription data (with hidden, internally consistent behavioral archetypes) to identify high-value customer segments, with an elbow + silhouette-validated k selection and a decile-based targeting model scoring physicians for commercial outreach
- **Result:** k=4 segments at silhouette ≈ 0.209 on the bundled dataset, 60 passing tests including degenerate/edge-case clustering inputs, automated Excel client report

## Skills Demonstrated

- Exploratory Data Analysis (EDA) and feature engineering
- NLP: Named Entity Recognition, TF-IDF, Cosine Similarity
- LLM integration: OpenAI/Gemini APIs, RAG pipeline, Prompt Engineering
- LLM fine-tuning: LoRA/PEFT, quantization, adapter merging
- Model evaluation: robustness testing, confidence intervals, benchmarking
- ML models: Linear/Ridge/Logistic Regression, Random Forest, XGBoost, SARIMA
- Risk analytics: delinquency/roll-rate modeling, IFRS 9 / Ind AS 109 ECL staging, PD/LGD/EAD
- Marketing analytics: adstock/carryover modeling, saturation curves, channel attribution
- Optimization: Linear Programming (SciPy), constrained allocation, sensitivity analysis
- Customer analytics: unsupervised segmentation (K-Means, PCA dimensionality reduction), cluster validation (elbow method, silhouette score), decile-based targeting/scoring models
- SQL: parameterised queries, joins, views, window functions
- Deployment: Streamlit web applications, automated Excel reporting (openpyxl)
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Statsmodels, SciPy, SpaCy, LangChain, PyTorch, Hugging Face Transformers/PEFT

## Contact

- **Email:** [khichashubham@gmail.com](mailto:khichashubham@gmail.com)
- **LinkedIn:** linkedin.com/in/shubham-khicha
- **GitHub:** github.com/skhicha

