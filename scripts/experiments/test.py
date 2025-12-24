import numpy as np
import librosa
import sounddevice as sd

# ----------------------------
# CONFIGURATION
# ----------------------------
filename = "./Condition 3 Hyd2.wav"  # input file
step_hz = 5.0  # frequency quantization step
play_volume = 0.5  # master volume
window_size = 4096  # STFT window size
hop_size = window_size // 4  # overlap
sr_target = None  # None = use file's native sampling rate

# ----------------------------
# LOAD AUDIO
# ----------------------------
y, sr = librosa.load(filename, sr=sr_target, mono=True)
y = y.astype(np.float32)
duration = y.shape[0] / sr
print(f"Loaded {filename}, {duration:.2f}s at {sr} Hz")


# ----------------------------
# VECTORIZED STFT SNAP FUNCTION
# ----------------------------
def stft_snap_freq_fast(y, window_size, hop_size, step_hz, sr):
    N = window_size
    window = np.hanning(N)
    y_out = np.zeros_like(y, dtype=np.float32)
    norm = np.zeros_like(y, dtype=np.float32)

    # Precompute FFT bin frequencies and their snapped target bins
    freqs = np.fft.rfftfreq(N, 1 / sr)
    target_bins = np.round(np.round(freqs / step_hz) * step_hz * N / sr).astype(int)
    target_bins = np.clip(target_bins, 0, len(freqs) - 1)  # safety

    for start in range(0, len(y) - N, hop_size):
        segment = y[start:start + N] * window
        Y = np.fft.rfft(segment)

        # Vectorized bin assignment
        Y_quant = np.zeros_like(Y, dtype=np.complex64)
        np.add.at(Y_quant, target_bins, Y)  # adds each Y[i] to the corresponding snapped bin

        # IFFT
        segment_out = np.fft.irfft(Y_quant, n=N).real

        # Overlap-add
        y_out[start:start + N] += segment_out * window
        norm[start:start + N] += window ** 2

    # Normalize by overlap
    y_out /= np.maximum(norm, 1e-8)
    return y_out


# ----------------------------
# PROCESS
# ----------------------------
y_quant = stft_snap_freq_fast(y, window_size, hop_size, step_hz, sr)
y_quant /= np.max(np.abs(y_quant))
y_quant *= play_volume

# ----------------------------
# PLAY AUDIO
# ----------------------------
print("Playing quantized audio...")
sd.play(y_quant, sr)
sd.wait()
print("Done.")
