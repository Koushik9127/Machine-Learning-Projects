# 🧠 Machine Learning & Generative AI Projects Portfolio

Welcome to my curated collection of Machine Learning and Generative AI projects. Each project is designed with real-world applicability and production-level practices—spanning classical ML pipelines, deep learning architectures, transformer-based NLP, and LLM-powered applications.

This portfolio reflects my hands-on expertise in:
- Data preprocessing and feature engineering
- Model training, tuning, and evaluation
- Explainability (SHAP, LIME)
- FastAPI/Streamlit-based deployment
- MLOps tools (MLflow, Docker, GitHub Actions)
- LLMs and generative AI with LangChain + OpenAI APIs

---

## 📁 Project Portfolio

| Folder | Project | Description | Tech Stack |
|--------|---------|-------------|------------|
| `customer_churn_pipeline/` | Customer Churn Prediction | Predict customer attrition using behavioral and service features | XGBoost, SHAP, MLflow, FastAPI, Streamlit |
| `resume_qna_bot/` | Resume QnA Chatbot | LLM-powered chatbot that answers questions from uploaded resumes | LangChain, OpenAI, FAISS, Streamlit |
| `loan_default_prediction/` | Loan Default Classifier | Predict likelihood of loan default using borrower and loan data | RandomForest, BERT, TensorFlow, SQL |
| `ecommerce_review_classifier/` | Ecommerce Review Classifier | Classify product reviews as positive or negative using text embeddings | TF-IDF, SVM, PyTorch LSTM, Hugging Face |
| `genai_projects_suite/` | GenAI Toolkit | A suite of LLM tools: PDF Q&A, Job Bot, README search, Idea generator | LangChain, OpenAI, Streamlit |
| `project-breast-cancer-prediction/` | Breast Cancer Prediction | Diagnose cancer using structured medical data | scikit-learn, XGBoost |
| `project-cell-nuclei-segmentation/` | Cell Nuclei Segmentation | Segment nuclei in microscope images for pathology automation | OpenCV, TensorFlow, U-Net |
| `project-customer-intelligence-platform/` | Customer Intelligence Platform | Analyze and visualize customer insights for business strategy | pandas, seaborn, sklearn |
| `project-patient-care-triage/` | Patient Care Triage | Automate medical triage using symptom and condition data | scikit-learn, healthcare APIs |
| `project-text-summarizer-app/` | Text Summarizer App | Summarize long documents into concise versions using transformers | Hugging Face, T5/BART, Streamlit |

---

## 🛠 Tech Stack & Tools

### ▶️ Machine Learning
`scikit-learn`, `XGBoost`, `RandomForest`, `SVM`, `GradientBoosting`, `PCA`

### 🧠 Deep Learning
`TensorFlow`, `PyTorch`, `Keras`, `CNN`, `LSTM`, `Transfer Learning`

### 📚 NLP & Transformers
`Hugging Face`, `BERT`, `DistilBERT`, `T5`, `Tokenizers`, `Self-Attention`

### 💡 Generative AI & LLMs
`LangChain`, `OpenAI GPT-4`, `FAISS`, `Prompt Engineering`

### 🛠️ Deployment & MLOps
`FastAPI`, `Streamlit`, `MLflow`, `Docker`, `GitHub Actions`

### 📊 Data & Visualization
`pandas`, `NumPy`, `SQL`, `matplotlib`, `seaborn`, `Tableau`, `Power BI`

---

## ⚙️ Quick Start Guide

Clone the repository and set up the virtual environment:

```bash
git clone https://github.com/Koushik9127/Machine-Learning-Projects.git
cd Machine-Learning-Projects
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
Run Churn Prediction App:
cd customer_churn_pipeline
pip install -r requirements.txt
python src/train.py
streamlit run app/main.py
📦 Project Structure

Machine-Learning-Projects/
├── project-name/
│   ├── notebooks/
│   ├── src/
│   ├── app/
│   └── requirements.txt
├── README.md
└── LICENSE
📄 License

Licensed under the MIT License.
Feel free to explore, fork, and extend the work.

