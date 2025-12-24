# Deep Audio Embedding 

This project visualizes audio by mapping spectrograms into a dynamic 3D latent space using a **Conditioned Autoencoder**. The result is an organic, evolving trajectory that reacts to the music, often resembling a "bird's nest" structure.

## 🎵 Features

- **Deep Learning based Visualization**: Uses a Convolutional Autoencoder to compress audio spectrograms into a 3D latent space.
- **Dynamic Trajectory**: The visualization draws a path that evolves with the music.
- **Real-time Synchronization**: The visual elements are perfectly synced to the audio playback.
- **Fallback Mechanism**: Includes a PCA fallback to ensure interesting visuals even if the neural network model hasn't fully converged.

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/JATIN4931/Deep_Audio_Embedding.git
    cd Deep_Audio_Embedding
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirement.txt
    ```

## 🚀 Usage

### 1. Train the Model (Important!)
The pre-trained model weights (`conditioned_autoencoder.pth`) are **not included** in this repository due to file size limits. You must train the model on your audio file first.

Run the training script:
```bash
python train.py
```
This will:
- Load the audio file specified in `config.py`.
- Train the autoencoder for the specified number of epochs.
- Save the weights to `conditioned_autoencoder.pth`.

### 2. Run the Visualization
Once the model is trained (or if you want to use the PCA fallback), run the main script:
```bash
python main.py
```
This will open a window showing the 3D latent space trajectory, waveform, and spectrogram in real-time.

## ⚙️ Configuration

You can adjust settings in `config.py`:

- `AUDIO_PATH`: Path to your input `.wav` file (e.g., `"audio2.wav"`).
- `WINDOW_SIZE`: Size of the spectrogram window (must match training).
- `TRAIL_LENGTH`: How long the tail of the visualization should be.
- `ROTATION_SPEED`: Speed of the 3D camera rotation.

## 📂 Project Structure

- **`main.py`**: Entry point for the visualization.
- **`train.py`**: Script to train the neural network.
- **`model.py`**: PyTorch definition of the Conditioned Autoencoder.
- **`config.py`**: Global configuration settings.
- **`data_processing.py`**: Handles audio loading, preprocessing, and model inference.
- **`visualization.py`**: Handles the Matplotlib 3D animation logic.
- **`audio_utils.py`**: Helper functions for audio loading and spectrogram generation.

## 📝 Note on Model Weights
If you see a `conditioned_autoencoder.pth` file in your local directory after running `train.py`, **do not commit it** to GitHub if it is larger than 100MB, as GitHub will reject it. The `.gitignore` should handle this, but be aware that users are expected to train their own model locally.
