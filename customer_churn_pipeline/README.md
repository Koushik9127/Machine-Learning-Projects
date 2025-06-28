
# Customer Churn Prediction – End‑to‑End ML Pipeline

This repo contains a **production‑ready** customer‑churn pipeline:

```
customer_churn_pipeline/
├── data/                 # Raw & processed data (add your CSV here)
├── notebooks/            # EDA + model exploration (Jupyter/IPynb)
├── src/
│   ├── train.py          # Model training & MLflow logging
│   └── utils.py          # Helper functions
├── models/               # Saved model artefacts (`model.pkl`)
├── app/
│   └── main.py           # Streamlit UI to serve predictions
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Clone repo & install deps
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Put your dataset
cp <your CSV> data/churn.csv

# 3. Train model & log to MLflow
python src/train.py

# 4. Launch the Streamlit app
streamlit run app/main.py
```

### Features

- **Pandas** + **scikit‑learn** data pipeline
- **MLflow** experiment tracking
- **Streamlit** prediction UI
- **Dockerfile** for container deployment (optional – add later)
- Fully portable to AWS/GCP/Azure

---

> Replace placeholder dataset with your own or public churn dataset (e.g., Telco Customer Churn from Kaggle).

