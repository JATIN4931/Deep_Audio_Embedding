import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

from audio_utils import load_audio, spectrogram
from model import ConditionedAutoencoder

# ---------------- SETTINGS ----------------
AUDIO_PATH = "audio1.wav"
WINDOW_SIZE = 20  # Changed to 20 to be divisible by 4 (after pooling)
HISTORY_LENGTH = 5
BATCH_SIZE = 128 # Increased for better covariance estimation
EPOCHS = 100
LEARNING_RATE = 1e-3
ALPHA = 0.5 # Reduced forecasting weight
BETA = 0.1 # Weight for decorrelation loss

class AudioDataset(Dataset):
    def __init__(self, spectrogram_data, window_size, history_length):
        self.data = spectrogram_data
        self.window_size = window_size
        self.history_length = history_length
        self.n_frames = self.data.shape[1]
        
        # Valid indices must allow for history and next step
        # We need:
        # t (current)
        # t+1 (next)
        # t-history_length+1 ... t (history)
        
        # So start index must be at least history_length - 1
        # And end index must be at most n_frames - window_size - 1 (for t+1)
        
        self.valid_indices = range(history_length - 1, self.n_frames - window_size - 1)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # idx is the index of the *start* of the current window t
        # But we mapped valid_indices to be the start column index
        
        start_col = self.valid_indices[idx]
        
        # Current window x_t
        x_t = self.data[:, start_col : start_col + self.window_size]
        
        # Next window x_{t+1} (shifted by 1 frame)
        x_next = self.data[:, start_col + 1 : start_col + 1 + self.window_size]
        
        # History windows
        # We need a sequence of L windows.
        # The last one is x_t.
        # The one before is x_{t-1}, etc.
        history = []
        for i in range(self.history_length):
            # history[0] is oldest, history[-1] is current (x_t)
            # offset = i - (self.history_length - 1)
            # if i=0 (oldest), offset = -(4) -> start_col - 4
            # if i=4 (current), offset = 0 -> start_col
            
            offset = i - (self.history_length - 1)
            h_start = start_col + offset
            h_window = self.data[:, h_start : h_start + self.window_size]
            history.append(h_window)
            
        history = np.array(history)
        
        # Add channel dimension (1, Freq, Time)
        x_t = x_t[np.newaxis, :, :]
        x_next = x_next[np.newaxis, :, :]
        history = history[:, np.newaxis, :, :]
        
        return torch.FloatTensor(x_t), torch.FloatTensor(x_next), torch.FloatTensor(history)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    if not os.path.exists(AUDIO_PATH):
        print(f"Error: {AUDIO_PATH} not found.")
        return

    print("Loading audio...")
    y, sr = load_audio(AUDIO_PATH)
    spec = spectrogram(y, sr) # (Freq, Time)
    
    # Crop to 512 frequency bins (drop the highest frequency bin)
    if spec.shape[0] > 512:
        spec = spec[:512, :]
    elif spec.shape[0] < 512:
        # Pad if less (unlikely with n_fft=1024)
        pad = np.zeros((512 - spec.shape[0], spec.shape[1]))
        spec = np.vstack((spec, pad))
        
    # Normalize spectrogram to [0, 1]
    spec_min = spec.min()
    spec_max = spec.max()
    spec = (spec - spec_min) / (spec_max - spec_min + 1e-8)
    
    print(f"Spectrogram shape: {spec.shape}")
    
    dataset = AudioDataset(spec, WINDOW_SIZE, HISTORY_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Initialize Model
    freq_bins = spec.shape[0]
    model = ConditionedAutoencoder(freq_bins, WINDOW_SIZE).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse_loss = nn.MSELoss()
    # cosine_loss = nn.CosineEmbeddingLoss() # Removed

    # 3. Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        recon_loss_total = 0
        forecast_loss_total = 0
        
        for x_t, x_next, history in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            history = history.to(device) # (Batch, Hist_Len, 1, Freq, Time)
            
            optimizer.zero_grad()
            
            # Forward pass
            
            # 1. Encode current and next to get true velocity
            z_t = model.encoder(x_t)
            with torch.no_grad():
                z_next = model.encoder(x_next)
            
            v_true = z_next - z_t
            
            # 2. Encode history for forecasting
            # Flatten batch and history dims to encode all windows
            b, h_len, c, f, t = history.shape
            history_flat = history.view(-1, c, f, t)
            z_history_flat = model.encoder(history_flat)
            z_history = z_history_flat.view(b, h_len, -1)
            
            # 3. Forecast
            v_hat = model.forecaster(z_history)
            
            # 4. Reconstruct
            x_recon = model.decoder(z_t)
            
            # Losses
            l_recon = mse_loss(x_recon, x_t)
            
            # Use MSE for forecasting instead of Cosine
            l_forecast = mse_loss(v_hat, v_true)
            
            # Decorrelation Loss (Forces dimensions to be independent)
            z_centered = z_t - z_t.mean(dim=0)
            cov = (z_centered.T @ z_centered) / (z_t.size(0) - 1 + 1e-8)
            target_cov = torch.eye(z_t.size(1), device=device)
            l_decorr = mse_loss(cov, target_cov)
            
            loss = l_recon + ALPHA * l_forecast + BETA * l_decorr
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            recon_loss_total += l_recon.item()
            forecast_loss_total += l_forecast.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.6f} (Recon={recon_loss_total/len(dataloader):.6f}, Forecast={forecast_loss_total/len(dataloader):.6f}, Decorr={l_decorr.item():.6f})")

    # 4. Save Model
    torch.save(model.state_dict(), "conditioned_autoencoder.pth")
    print("Model saved to conditioned_autoencoder.pth")

if __name__ == "__main__":
    train()
