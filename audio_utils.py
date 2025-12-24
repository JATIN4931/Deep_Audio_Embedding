import librosa
import numpy as np
import sounddevice as sd

def load_audio(path, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def spectrogram(y, sr, n_fft=1024, hop_length=512):
    # CNN ke liye output shape consistent hona chahiye
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # S shape: (Freq_bins, Time_steps)
    return S 

def play_audio_stream(sr):
    stream = sd.OutputStream(samplerate=sr, channels=1)
    stream.start()
    return stream

def play_audio(y, sr):
    sd.play(y, sr)