🛍️ Mall Customer Segmentation Pipeline

An end-to-end Machine Learning project that leverages K-Means Clustering to segment mall customers based on their Annual Income and Spending Score. The project includes interactive Exploratory Data Analysis (EDA), model serialization, a FastAPI backend endpoint, an interactive Streamlit dashboard, and production-ready deployment configurations.

📌 Business Problem & Overview

Targeted marketing yields higher conversion rates than broad campaigns. By analyzing customer purchasing patterns, businesses can tailor promotional strategies to specific segments.

This application groups customers into 5 distinct business personas:

Sensible: Moderate Income, Moderate Spending Score

Careless: Low Income, High Spending Score

Target / Stars: High Income, High Spending Score

Careful: High Income, Low Spending Score

Miser / Saver: Low Income, Low Spending Score

🏗️ Architecture

Mall-Customer-Segmentation/
├── data/                       # Raw customer dataset
│   └── Mall_Customers.csv
├── models/                     # Saved model artifacts
│   ├── scaler.pkl              # Fitted StandardScaler instance
│   └── kmeans_model.pkl        # Trained K-Means clustering model
├── .gitignore                  # File exclusions for Git tracking
├── app.py                      # Streamlit frontend dashboard
├── customer_segmentation.ipynb # Data exploration & research notebook
├── main.py                     # FastAPI REST API backend
└── requirements.txt            # Python dependencies


🛠️ Tech Stack & Dependencies

Language: Python 3.13+

Data Processing & ML: Pandas, NumPy, Scikit-learn, Joblib

Visualization: Plotly Express, Seaborn, Matplotlib

API Backend: FastAPI, Uvicorn, Pydantic

Frontend: Streamlit

🚀 Quick Start Guide

1. Prerequisites & Setup

Clone the repository and set up a virtual environment:

git clone https://github.com/Sam-Ake/Mall-Customer-Segmentation.git
cd Mall-Customer-Segmentation

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt


2. Running the FastAPI Backend (main.py)

To launch the REST API server:

python -m uvicorn main:app --reload


Swagger Documentation: Navigate to http://127.0.0.1:8000/docs to test endpoints interactively.

POST /segment: Accepts annual_income and spending_score in JSON format to return predicted cluster IDs and segment names.

3. Running the Streamlit Dashboard (app.py)

To open the interactive dashboard:

streamlit run app.py


Access the application in your browser at http://localhost:8501.

🌐 Deploying Streamlit Dashboard to Streamlit Cloud

To host your Streamlit app live for free:

Sign in to Streamlit Community Cloud using your GitHub account.

Click Create app (or New app).

Fill in the deployment details:

Repository: Sam-Ake/Mall-Customer-Segmentation

Branch: main

Main file path: app.py

Click Deploy!

📜 License

This project is open-source and available under the MIT License.