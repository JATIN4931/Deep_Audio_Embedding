import matplotlib.pyplot as plt

# ---------------- SETTINGS ----------------
plt.style.use('default')     # Clean white background
TRAIL_LENGTH = 150       # Smooth ribbon length
ROTATION_SPEED = 0.2     # Slower rotation
SMOOTH_SIGMA = 2.8       # Ribbon smoothness
AUDIO_PATH = "audio2.wav"
MODEL_PATH = "conditioned_autoencoder.pth"
WINDOW_SIZE = 20         # Must match training
HOP_LENGTH = 512         # Must match audio_utils.py
