Customer Churn Prediction Pipeline

Overview
Predicts customer churn using Random Forest and XGBoost. Helps businesses identify at-risk customers.

Features
- Automated data preprocessing (missing values, encoding, scaling)
- Model training using Stratified K-Fold Cross Validation
- Feature importance visualization
- Model evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
- Exported pipeline using joblib

Tech Stack
Python 3.10+, scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn

Workflow
1. Data Preprocessing → Cleaning & Encoding
2. Feature Engineering → Scaling, Imbalance Handling
3. Model Training → Random Forest & XGBoost
4. Evaluation → Cross-Validation Metrics
5. Export → Save model as .pkl

How to Run
pip install -r requirements.txt
python src/train.py

Folder Structure
customer_churn_pipeline/
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
├── requirements.txt
├── README.md
└── .gitignore

Future Improvements
- SHAP interpretability
- Streamlit dashboard deployment
- Model monitoring (MLOps-ready)

Author
Koushik Paturu
paturu02@outlook.com
https://github.com/Koushik9127
