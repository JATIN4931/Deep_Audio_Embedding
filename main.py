import config
import data_processing
import visualization

def main():
    # 1. Load and Process Audio
    y, sr, spec = data_processing.load_and_process_audio(config.AUDIO_PATH)
    
    # 2. Generate Trajectory
    trajectory, windows = data_processing.generate_trajectory(spec, config.MODEL_PATH, config.WINDOW_SIZE)
    
    # 3. Fallback Check (PCA)
    trajectory = data_processing.process_trajectory_fallback(trajectory, windows)
    
    # 4. Post-Processing
    trajectory = data_processing.post_process_trajectory(trajectory)
    
    # 5. Setup Visualization
    freq_bins = spec.shape[0]
    fig, ax3d, line3d, scat3d, line_wav, spec_display, chunk_size = visualization.setup_plot(freq_bins, config.WINDOW_SIZE)
    
    # 6. Run Animation
    visualization.run_animation(fig, line3d, scat3d, line_wav, spec_display, chunk_size, trajectory, y, sr, spec)

if __name__ == "__main__":
    main()