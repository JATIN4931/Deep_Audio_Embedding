import numpy as np
import torch
import os
from scipy.ndimage import gaussian_filter1d
from audio_utils import load_audio, spectrogram
from model import ConditionedAutoencoder
import config

def load_and_process_audio(audio_path):
    print("Loading audio...")
    y, sr = load_audio(audio_path)
    spec = spectrogram(y, sr) # (Freq, Time)

    # Crop to 512 frequency bins
    if spec.shape[0] > 512:
        spec = spec[:512, :]
    elif spec.shape[0] < 512:
        pad = np.zeros((512 - spec.shape[0], spec.shape[1]))
        spec = np.vstack((spec, pad))

    # Normalize spectrogram (same as training)
    spec_min = spec.min()
    spec_max = spec.max()
    spec = (spec - spec_min) / (spec_max - spec_min + 1e-8)
    
    return y, sr, spec

def generate_trajectory(spec, model_path, window_size):
    freq_bins, time_steps = spec.shape
    print(f"Spectrogram shape: {spec.shape}")

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionedAutoencoder(freq_bins, window_size).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained model.")
    else:
        print("Warning: Trained model not found. Using random weights.")

    model.eval()

    # Generate Trajectory
    print("Generating trajectory...")
    trajectory = []
    batch_size = 64
    windows = []

    # Create sliding windows
    for i in range(time_steps - window_size):
        window = spec[:, i : i + window_size]
        windows.append(window)

    windows = np.array(windows) # (N, Freq, Time)
    windows_tensor = torch.FloatTensor(windows).unsqueeze(1).to(device) # (N, 1, Freq, Time)

    # Process in batches
    with torch.no_grad():
        for i in range(0, len(windows_tensor), batch_size):
            batch = windows_tensor[i : i + batch_size]
            z = model.encoder(batch)
            trajectory.append(z.cpu().numpy())

    trajectory = np.concatenate(trajectory, axis=0) # (N, 3)
    
    return trajectory, windows

def process_trajectory_fallback(trajectory, windows):
    # --- FALLBACK: PCA if Model Collapsed ---
    # Check if the trajectory is essentially 1D (straight line)
    # We do this by looking at the singular values of the covariance matrix
    centered = trajectory - np.mean(trajectory, axis=0)
    cov = np.cov(centered.T)
    eigenvalues, _ = np.linalg.eig(cov)
    sorted_evals = np.sort(eigenvalues)[::-1]
    variance_ratio = sorted_evals / (np.sum(sorted_evals) + 1e-8)

    print(f"Latent Variance Ratios: {variance_ratio}")

    # Force PCA if variance is low OR if the user is seeing a dot (variance < 1e-4)
    total_variance = np.sum(np.var(trajectory, axis=0))
    print(f"Total Variance: {total_variance}")

    if variance_ratio[0] > 0.90 or total_variance < 1e-3:
        print("WARNING: Model output is collapsed (straight line or dot). Switching to PCA for visualization.")
        try:
            from sklearn.decomposition import PCA
            # Flatten windows for PCA: (N, Freq*Time)
            windows_flat = windows.reshape(len(windows), -1)
            pca = PCA(n_components=3)
            trajectory = pca.fit_transform(windows_flat)
            print("PCA generated trajectory successfully.")
        except ImportError:
            print("sklearn not found. Using random projection.")
            # Random projection fallback
            windows_flat = windows.reshape(len(windows), -1)
            projection_matrix = np.random.randn(windows_flat.shape[1], 3)
            trajectory = windows_flat @ projection_matrix
    
    return trajectory

def post_process_trajectory(trajectory):
    # --- POST-PROCESSING (Normalization & Smoothing) ---
    # 1. Center and Normalize (Fixes the huge coordinates)
    mean = np.mean(trajectory, axis=0)
    std = np.std(trajectory, axis=0)
    std[std < 1e-6] = 1.0 # Avoid division by zero
    trajectory = (trajectory - mean) / std

    # 2. Apply Gaussian Smoothing (Creates the organic "Bird's Nest" look)
    # Sigma=3 provides a good balance between detail and smoothness
    trajectory[:, 0] = gaussian_filter1d(trajectory[:, 0], sigma=3)
    trajectory[:, 1] = gaussian_filter1d(trajectory[:, 1], sigma=3)
    trajectory[:, 2] = gaussian_filter1d(trajectory[:, 2], sigma=3)

    # 3. Scale to match the reference image style (approx -0.15 to 0.15)
    # Normalize to unit sphere first, then scale
    max_norm = np.max(np.linalg.norm(trajectory, axis=1))
    if max_norm > 1e-6:
        trajectory = trajectory / max_norm
    trajectory = trajectory * 0.15 # Scale to fit the plot limits exactly

    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Trajectory min/max: {trajectory.min()}, {trajectory.max()}")
    
    return trajectory
