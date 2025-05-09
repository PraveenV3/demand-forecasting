import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model_training import TimeSeriesTransformer
import joblib

st.title("Municipal Waste Demand Forecasting Dashboard")

@st.cache_data
def load_weekly_data():
    return pd.read_csv("data/processed/weekly_waste.csv")

@st.cache_data
def load_seasonal_data():
    return pd.read_csv("data/processed/seasonal_waste.csv")

@st.cache_resource
def load_model():
    model_dir = "models"
    model_path = os.path.join(model_dir, "transformer_model.pth")
    scaler_path = os.path.join(model_dir, "scaler.save")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model or scaler not found. Please train the model first.")
        return None, None
    model = TimeSeriesTransformer(input_dim=0)  # placeholder, will update input_dim later
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler

def main():
    data_type = st.sidebar.selectbox("Select Data Type", ["Weekly", "Seasonal"])
    if data_type == "Weekly":
        data = load_weekly_data()
    else:
        data = load_seasonal_data()

    areas = data['area'].unique() if 'area' in data.columns else []
    selected_area = st.sidebar.selectbox("Select Area", areas)

    if data_type == "Weekly":
        filtered_data = data[data['area'] == selected_area]
        fig = px.line(filtered_data, x='week', y='net_weight_kg', title=f"Weekly Waste Volume - {selected_area}")
        st.plotly_chart(fig)
    else:
        filtered_data = data[data['area'] == selected_area]
        fig = px.bar(filtered_data, x='season', y='net_weight_kg', title=f"Seasonal Waste Volume - {selected_area}")
        st.plotly_chart(fig)

    st.write("Forecasting functionality will be added here.")

if __name__ == "__main__":
    main()
