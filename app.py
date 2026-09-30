import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

st.title("🛍️ Mall Customer Segmentation Dashboard")
st.write("Analyze and segment mall customers using K-Means Clustering.")

# Load models
@st.cache_resource
def load_models():
    scaler = joblib.load("models/scaler.pkl")
    kmeans = joblib.load("models/kmeans_model.pkl")
    return scaler, kmeans

try:
    scaler, kmeans = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")

segment_labels = {
    0: 'Sensible (Moderate Income, Moderate Spending)',
    1: 'Careless (Low Income, High Spending)',
    2: 'Target / Stars (High Income, High Spending)',
    3: 'Careful (High Income, Low Spending)',
    4: 'Miser / Saver (Low Income, Low Spending)'
}

# Sidebar options
st.sidebar.header("Navigation")
option = st.sidebar.radio("Choose Mode", ["Single Prediction", "Batch CSV Upload & Visualization"])

if option == "Single Prediction":
    st.subheader("Predict Segment for a Single Customer")
    
    income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=70)
    spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=85)
    
    if st.button("Predict Segment"):
        scaled_input = scaler.transform([[income, spending]])
        cluster = int(kmeans.predict(scaled_input)[0])
        label = segment_labels.get(cluster, "Unknown")
        
        st.success(f"**Predicted Cluster:** Cluster {cluster}")
        st.info(f"**Segment Category:** {label}")

elif option == "Batch CSV Upload & Visualization":
    st.subheader("Upload Dataset for Batch Segmentation")
    uploaded_file = st.file_uploader("Upload CSV (Must contain 'Annual Income (k$)' and 'Spending Score (1-100)')", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        if 'Annual Income (k$)' in data.columns and 'Spending Score (1-100)' in data.columns:
            scaled_features = scaler.transform(data[['Annual Income (k$)', 'Spending Score (1-100)']])
            data['Cluster'] = kmeans.predict(scaled_features)
            data['Segment_Name'] = data['Cluster'].map(segment_labels)
            
            st.write("### Segmented Data Preview")
            st.dataframe(data.head())
            
            # Interactive Plotly Chart
            fig = px.scatter(
                data, 
                x='Annual Income (k$)', 
                y='Spending Score (1-100)', 
                color='Segment_Name',
                title="Interactive Customer Segments",
                hover_data=['Age'] if 'Age' in data.columns else None
            )
            st.plotly_chart(fig)
            
            # Summary Table
            st.write("### Segment Summary")
            summary = data.groupby('Segment_Name').size().reset_index(name='Customer Count')
            st.table(summary)
        else:
            st.error("Uploaded CSV must contain 'Annual Income (k$)' and 'Spending Score (1-100)' columns.")