import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import numpy as np
import time
from audio_utils import play_audio
import config

def setup_plot(freq_bins, window_size):
    # ---------------- CINEMATIC LAYOUT ----------------
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[4, 0.5, 1])

    # 1. 3D Space (The Manifold)
    ax3d = fig.add_subplot(gs[0], projection="3d")

    # Dynamic Line (starts empty, draws over time)
    line3d, = ax3d.plot([], [], [], color='black', alpha=0.8, lw=1.0)

    # Moving point (Green dot)
    scat3d = ax3d.scatter([], [], [], color='green', s=50, alpha=1.0, edgecolors='black') 

    # Axis settings
    ax3d.set_xlabel('Latent X')
    ax3d.set_ylabel('Latent Y')
    ax3d.set_zlabel('Latent Z')
    ax3d.set_title("3D LATENT SPACE TRAJECTORY", pad=-10)

    # Set limits based on trajectory
    # Fixed limits to keep the "Bird's Nest" centered and stable
    limit = 0.15
    ax3d.set_xlim(-limit, limit)
    ax3d.set_ylim(-limit, limit)
    ax3d.set_zlim(-limit, limit)

    # 2. Waveform Overlay
    ax_wav = fig.add_subplot(gs[1])
    chunk_size = 2048
    line_wav, = ax_wav.plot(np.linspace(0, 1, chunk_size), np.zeros(chunk_size), color='black', lw=0.5)
    ax_wav.axis('off')

    # 3. Spectrogram Heatmap
    ax_spec = fig.add_subplot(gs[2])
    # Show a window of the spectrogram
    spec_display = ax_spec.imshow(np.zeros((freq_bins, window_size)), aspect='auto', origin='lower', cmap='magma')
    ax_spec.axis('off')
    
    return fig, ax3d, line3d, scat3d, line_wav, spec_display, chunk_size

def run_animation(fig, line3d, scat3d, line_wav, spec_display, chunk_size, trajectory, y, sr, spec):
    ax3d = line3d.axes
    
    # ---------------- ANIMATION ----------------
    play_audio(y, sr)
    start_time = time.time()

    def update(frame):
        elapsed_time = time.time() - start_time
        
        # Calculate exact frame index based on time
        # frame_idx = time * sample_rate / hop_length
        idx = int(elapsed_time * sr / config.HOP_LENGTH)
        
        if idx >= len(trajectory):
            return
            
        # Update Dynamic Line (Draw path from start to current)
        # Draw full history from start (0) to current index
        start_idx = 0 
        line3d.set_data(trajectory[start_idx:idx+1, 0], trajectory[start_idx:idx+1, 1])
        line3d.set_3d_properties(trajectory[start_idx:idx+1, 2])

        # Current point
        current_z = trajectory[idx]
        
        # Update scatter point
        scat3d._offsets3d = ([current_z[0]], [current_z[1]], [current_z[2]])
        
        # Rotate camera
        ax3d.view_init(elev=20, azim=elapsed_time * config.ROTATION_SPEED * 10)
        
        # Update Waveform (approximate chunk)
        audio_idx = int(elapsed_time * sr)
        if audio_idx + chunk_size < len(y):
            chunk = y[audio_idx : audio_idx + chunk_size]
            line_wav.set_ydata(chunk)
            
        # Update Spectrogram
        # Map trajectory index back to spectrogram column
        spec_col = idx # Since trajectory[i] corresponds to window starting at i
        if spec_col + config.WINDOW_SIZE < spec.shape[1]:
            spec_window = spec[:, spec_col : spec_col + config.WINDOW_SIZE]
            spec_display.set_data(spec_window)

    ani = FuncAnimation(fig, update, interval=20, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
