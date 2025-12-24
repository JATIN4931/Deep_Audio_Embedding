import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_height, input_width, latent_dim=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Calculate flattened size dynamically
        self.h_out = input_height // 4
        self.w_out = input_width // 4
        self.flatten_size = 64 * self.h_out * self.w_out
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.Tanh() # Force output to [-1, 1]
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z

class Decoder(nn.Module):
    def __init__(self, input_height, input_width, latent_dim=3):
        super().__init__()
        self.h_out = input_height // 4
        self.w_out = input_width // 4
        self.flatten_size = 64 * self.h_out * self.w_out
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.flatten_size),
            nn.ReLU()
        )
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid() # Output range [0, 1] for spectrogram
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 64, self.h_out, self.w_out)
        x = self.deconv(x)
        return x

class Forecaster(nn.Module):
    def __init__(self, latent_dim=3, hidden_dim=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, history):
        # history shape: (batch, seq_len, latent_dim)
        lstm_out, _ = self.lstm(history)
        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]
        v_hat = self.fc(last_out)
        return v_hat

class ConditionedAutoencoder(nn.Module):
    def __init__(self, input_height, input_width, latent_dim=3):
        super().__init__()
        self.encoder = Encoder(input_height, input_width, latent_dim)
        self.decoder = Decoder(input_height, input_width, latent_dim)
        self.forecaster = Forecaster(latent_dim)

    def forward(self, x, history=None):
        z = self.encoder(x)
        recon = self.decoder(z)
        
        v_hat = None
        if history is not None:
            v_hat = self.forecaster(history)
            
        return z, recon, v_hat