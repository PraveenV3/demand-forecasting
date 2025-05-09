import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Define a simple Transformer-based model for time series forecasting
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, input_dim)  # Predict all features per time step

    def forward(self, src):
        # src shape: (seq_len, batch_size, input_dim)
        src = self.input_proj(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

class WasteDataset(Dataset):
    def __init__(self, data, seq_length=12):
        self.seq_length = seq_length
        self.data = data
        self.X, self.y = self.create_sequences(data)

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i+self.seq_length])
            y.append(data[i+self.seq_length])
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

def load_data(file_path):
    df = pd.read_csv(file_path)
    # Assuming the data has columns: year, week, area, waste_type, net_weight_kg
    # For simplicity, aggregate by week and area, ignoring waste_type for now
    df_grouped = df.groupby(['year', 'week', 'area'], as_index=False)['net_weight_kg'].sum()
    # Pivot to have areas as features
    pivot_df = df_grouped.pivot(index=['year', 'week'], columns='area', values='net_weight_kg').fillna(0)
    return pivot_df

def train_model(data_path="data/processed/weekly_waste.csv", model_dir="models", epochs=50, batch_size=16, lr=0.001):
    import matplotlib.pyplot as plt
    os.makedirs(model_dir, exist_ok=True)
    data = load_data(data_path)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.values)

    dataset = WasteDataset(data_scaled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(input_dim=data.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_losses = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.permute(1, 0, 2).to(device)  # seq_len, batch, features
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)[-1]  # Take last time step output
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        training_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Plot training loss curve
    os.makedirs("visualizations", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), training_losses, marker='o')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig("visualizations/training_loss.png")
    plt.close()

    model_path = os.path.join(model_dir, "transformer_model.pth")
    torch.save(model.state_dict(), model_path)
    scaler_path = os.path.join(model_dir, "scaler.save")
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    train_model()
