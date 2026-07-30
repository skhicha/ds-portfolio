# 📊 Data Science & ML Portfolio — Shubham Khicha

A collection of end-to-end data science and machine learning projects covering NLP, LLM/RAG systems, supervised ML, time-series forecasting, LLM fine-tuning, and model robustness evaluation.

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

## Skills Demonstrated

- Exploratory Data Analysis (EDA) and feature engineering
- NLP: Named Entity Recognition, TF-IDF, Cosine Similarity
- LLM integration: OpenAI/Gemini APIs, RAG pipeline, Prompt Engineering
- LLM fine-tuning: LoRA/PEFT, quantization, adapter merging
- Model evaluation: robustness testing, confidence intervals, benchmarking
- ML models: Linear/Ridge Regression, Random Forest, XGBoost, SARIMA
- Deployment: Streamlit web applications
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SpaCy, LangChain, PyTorch, Hugging Face Transformers/PEFT

## Contact

- **Email:** [khichashubham@gmail.com](mailto:khichashubham@gmail.com)
- **LinkedIn:** linkedin.com/in/shubham-khicha
- **GitHub:** github.com/skhicha

