# 📊 Data Science & ML Portfolio — Shubham Khicha

A collection of end-to-end data science and machine learning projects covering NLP, LLM/RAG systems, supervised ML, time-series forecasting, LLM fine-tuning, model robustness evaluation, credit risk analytics, marketing mix modeling, and optimization.

## Projects

### 1. 📄 Insurance Document NLP Extractor

- **Tech:** Python, SpaCy, PDFPlumber, TF-IDF, Scikit-learn, Streamlit
- **What:** Extracts structured entities from unstructured insurance PDFs using NER + Regex
- **Result:** 88%+ field-level accuracy across diverse document formats

### 2. 🤖 LLM-Powered Document Q&A Tool (RAG)

- **Tech:** Python, OpenAI/Gemini API, LangChain, FAISS, Streamlit
- **What:** Retrieval-Augmented Generation system for querying business documents
- **Result:** 40% reduction in manual document review time

### 3. 🏠 House Price Prediction — End-to-End ML Pipeline

- **Tech:** Python, Scikit-learn, XGBoost, Pandas, Matplotlib, Streamlit
- **What:** Complete ML pipeline on Ames Housing dataset with EDA, feature engineering, model comparison
- **Result:** Best RMSE of 18,500 with Random Forest after hyperparameter tuning

### 4. ⚡ Energy Consumption Forecasting — Time-Series ML

- **Tech:** Python, Scikit-learn, XGBoost, SARIMA, Pandas, Matplotlib
- **What:** Time-series forecasting on hourly energy consumption with lag features and rolling stats
- **Result:** XGBoost achieved 4.2% MAPE, outperforming SARIMA baseline by 1.8 pp

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
- SQL: parameterised queries, joins, views, window functions
- Deployment: Streamlit web applications, automated Excel reporting (openpyxl)
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Statsmodels, SciPy, SpaCy, LangChain, PyTorch, Hugging Face Transformers/PEFT

## Contact

- **Email:** [khichashubham@gmail.com](mailto:khichashubham@gmail.com)
- **LinkedIn:** linkedin.com/in/shubham-khicha
- **GitHub:** github.com/skhicha

