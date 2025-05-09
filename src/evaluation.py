import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from model_training import TimeSeriesTransformer, WasteDataset

def load_data(file_path):
    df = pd.read_csv(file_path)
    df_grouped = df.groupby(['year', 'week', 'area'], as_index=False)['net_weight_kg'].sum()
    pivot_df = df_grouped.pivot(index=['year', 'week'], columns='area', values='net_weight_kg').fillna(0)
    return pivot_df

def evaluate_model(data_path="data/processed/weekly_waste.csv", model_dir="models", batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(data_path)
    scaler_path = os.path.join(model_dir, "scaler.save")
    scaler = None
    if os.path.exists(scaler_path):
        import joblib
        scaler = joblib.load(scaler_path)
    else:
        print("Scaler not found, cannot evaluate properly.")
        return

    data_scaled = scaler.transform(data.values)
    dataset = WasteDataset(data_scaled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = TimeSeriesTransformer(input_dim=data.shape[1])
    model_path = os.path.join(model_dir, "transformer_model.pth")
    if not os.path.exists(model_path):
        print("Model not found, please train the model first.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    criterion = torch.nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.permute(1, 0, 2).to(device)
            y_batch = y_batch.to(device)
            output = model(X_batch)[-1]
            loss = criterion(output, y_batch)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Validation MSE Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    evaluate_model()
