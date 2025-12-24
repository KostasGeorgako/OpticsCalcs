# file: laser_audio_demo_fast.py
import numpy as np
import sounddevice as sd
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, sosfilt, hilbert


input_file = "./Condition 3 Hyd1.wav"  # your hydrophone file
# input_file = "./bohemian_cut.mp3"  # your hydrophone file
sr_target = 8192  # downsample for performance, still covers 50-1000Hz
method = "filterbank"  # choose "heterodyne" or "filterbank"


y, sr = librosa.load(input_file, sr=sr_target, mono=True)
print(f"Loaded {input_file}, {len(y)/sr:.2f}s, sr={sr}")


def butter_lowpass(cutoff, fs, order=4):
    b, a = butter(order, cutoff/(fs/2), btype='low')
    return b, a

def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    return lfilter(b, a, data)

def bandpass_sos(low, high, sr, order=4):
    nyq = sr / 2
    # ensure low < high and normalized frequencies within (0,1)
    low_n = max(low/nyq, 1e-6)
    high_n = min(high/nyq, 0.999)
    if high_n <= low_n:
        high_n = low_n + 1e-6
    from scipy.signal import butter
    return butter(order, [low_n, high_n], btype='bandpass', output='sos')

def envelope(x):
    analytic_signal = hilbert(x)
    return np.abs(analytic_signal)


def heterodyne_down(x, sr, f_shift=500.0, lp_cut=300.0):
    t = np.arange(len(x))/sr
    x_mixed = x * np.cos(2*np.pi*f_shift*t)
    x_lp = lowpass_filter(x_mixed, lp_cut, sr)
    x_lp /= np.max(np.abs(x_lp))+1e-12
    return x_lp


def filterbank_am(x, sr, bands=None, carriers=None, env_lp_cut=50):
    # safe linear bands if none provided
    if bands is None:
        bands = [(50 + i*60, 50 + (i+1)*60) for i in range(16)]  # 50-1010Hz
    if carriers is None:
        carriers = np.linspace(50, 250, len(bands))
    y_out = np.zeros_like(x)
    for (low,high), fc in zip(bands, carriers):
        sos = bandpass_sos(low, high, sr)
        band = sosfilt(sos, x)
        env = envelope(band)
        # smooth envelope
        b,a = butter(4, env_lp_cut/(sr/2), btype='low')
        env_smooth = lfilter(b,a,env)
        env_smooth /= np.max(env_smooth)+1e-12
        t = np.arange(len(x))/sr
        carrier = np.sin(2*np.pi*fc*t)
        y_out += env_smooth * carrier

    # y_out /= np.max(np.abs(y_out))+1e-12

    # 1. RMS normalize
    target_rms = 0.1
    current_rms = np.sqrt(np.mean(y_out ** 2))
    y_out = y_out * (target_rms / (current_rms + 1e-12))

    # 2. Soft clip
    max_amp = 0.3
    y_out = max_amp * np.tanh(y_out / max_amp)

    return y_out

# ---------------------------
# Process audio
# ---------------------------
if method == "heterodyne":
    print("Processing with Heterodyne Downshift...")
    y_out = heterodyne_down(y, sr, f_shift=500, lp_cut=250)
elif method == "filterbank":
    print("Processing with Filterbank AM...")
    y_out = filterbank_am(y, sr, env_lp_cut=30)
else:
    raise ValueError("Unknown method")

# ---------------------------
# Play audio
# ---------------------------
print("Playing processed audio...")
sd.play(y_out, sr)
sd.wait()
print("Done.")

# ---------------------------
# Plot original vs processed FFT
# ---------------------------
plt.figure(figsize=(12,5))
fft_orig = np.abs(np.fft.rfft(y))
fft_proc = np.abs(np.fft.rfft(y_out))
freqs = np.fft.rfftfreq(len(y), 1/sr)
plt.semilogy(freqs, fft_orig+1e-12, label='Original')
plt.semilogy(freqs, fft_proc+1e-12, label='Processed')
plt.xlim(0, 1200)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title(f"FFT: {method}")
plt.legend()
plt.show()
