# Machine Learning & Generative AI Projects Portfolio

This repository contains a collection of end-to-end ML and GenAI projects, each built with production-level practices including data preprocessing, modeling, evaluation, explainability, deployment, and LLM-based applications.

---

## 📁 Project Overview

| Folder | Project | Highlights |
|--------|---------|------------|
| `customer_churn_pipeline/` | Customer Churn Prediction | XGBoost, SHAP, MLflow, FastAPI, Streamlit |
| `resume_qna_bot/` | Resume QnA Chatbot | LangChain, OpenAI, PDF embeddings, Streamlit |
| `loan_default_prediction/` | Loan Default Classifier | RandomForest, BERT, TensorFlow, SQL |
| `ecommerce_review_classifier/` | Ecommerce Review Classifier | TF-IDF+SVM, PyTorch LSTM, Hugging Face |
| `genai_projects_suite/` | GenAI Toolkit | Chat with PDF, README Search, Job Alert Bot, Idea Generator |

---

## 🧠 Tools & Libraries by Category

### ▶️ Machine Learning
scikit-learn, XGBoost, SVM, Random Forest, Gradient Boosting, PCA

### 🧠 Deep Learning
TensorFlow, PyTorch, Keras, CNN, RNN, LSTM, Transfer Learning

### 📚 NLP & Transformers
Hugging Face Transformers, BERT, DistilBERT, Tokenizers, Self-Attention

### 💡 GenAI & LLMs
LangChain, OpenAI GPT, FAISS, Prompt Engineering, Encoder-Decoder models

### 🛠️ Deployment & MLOps
FastAPI, Streamlit, Docker, GitHub Actions, MLflow

### 🧮 Data & Visualization
Pandas, SQL, Matplotlib, Seaborn, Tableau, Power BI

---

## ⚙️ Example Workflow

### ▶️ Run Churn Prediction Pipeline

```bash
cd customer_churn_pipeline
pip install -r requirements.txt
python src/train.py             # Train model and log to MLflow
streamlit run app/main.py      # Launch prediction UI

Each folder contains:

notebooks/: EDA + experiments
src/: model scripts
app/: Streamlit or FastAPI interface
requirements.txt: environment setup
📦 Installation & Setup

git clone https://github.com/Koushik9127/Machine-Learning-Projects.git
cd Machine-Learning-Projects
python -m venv venv && source venv/bin/activate
Enter any project folder and follow its README.md to run.

📄 License

This repository is licensed under the MIT License.
Feel free to explore, fork, or extend.

