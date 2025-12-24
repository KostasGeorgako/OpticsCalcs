from metavision_core.event_io import EventsIterator
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, resample
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import sounddevice as sd

# =======================
# USER PARAMETERS
# =======================
file_path = r"C:\Users\kpgeo\Documents\metavision\recordings\recording_2025-12-15_15-11-57.raw"
file_path = r"C:\Users\kpgeo\Documents\metavision\recordings\recording_2025-12-15_15-01-30.raw"
file_path = r"C:\Users\kpgeo\Documents\metavision\recordings\recording_2025-12-15_15-23-22.raw"
window_us = 1000          # time bin (µs). Smaller = higher temporal resolution, more noise
min_events_per_bin = 5   # skip very sparse bins
smooth_sigma = 2.0       # Gaussian smoothing for velocity (in samples)
hp_cutoff = 20.0         # high-pass cutoff (Hz) to remove drift
max_audio_freq = 8000.0  # max expected audio content (Hz), used for plotting only
target_fs_audio = 44100  # final WAV / playback sample rate
N_dominant = 100         # number of dominant FFT peaks to print

# =======================
# LOAD EVENTS & BUILD TRAJECTORY
# =======================
events_it = EventsIterator(file_path, delta_t=window_us)
centers = []
times = []

for ev in events_it:
    if ev.size < min_events_per_bin:
        continue

    # centroid of all events in this bin
    centers.append([np.mean(ev['x']), np.mean(ev['y'])])

    # use mean timestamp of bin for better timing
    times.append(np.mean(ev['t']))

centers = np.array(centers)
times = np.array(times) * 1e-6  # µs -> s

if len(centers) < 10:
    raise SystemExit("Not enough data for motion analysis.")

# =======================
# PCA: MAIN MOTION AXIS
# =======================
# subtract global mean
centers_mean = centers.mean(axis=0)
centers_centered = centers - centers_mean

# use middle portion for PCA to reduce drift effects
start = len(centers_centered) // 4
end   = 3 * len(centers_centered) // 4
cov = np.cov(centers_centered[start:end].T)
eigvals, eigvecs = np.linalg.eig(cov)
principal_axis = eigvecs[:, np.argmax(eigvals)]

# project all centers onto that axis -> 1D position signal
proj = centers_centered @ principal_axis  # shape (N,)

# =======================
# VELOCITY ESTIMATION
# =======================
dt = np.diff(times)
dx = np.diff(proj)

# guard against tiny dt
median_dt = np.median(dt)
valid = dt > (0.1 * median_dt)
velocity = np.zeros_like(dx)
velocity[valid] = dx[valid] / dt[valid]
velocity = np.nan_to_num(velocity)

# effective sample rate of the motion signal
fs = 1.0 / np.mean(dt[valid])
print(f"Estimated motion sampling rate: {fs:.1f} Hz")

# align time array with velocity (one shorter)
t_vel = times[1:]

# =======================
# SMOOTH VELOCITY
# =======================
if smooth_sigma > 0:
    velocity = gaussian_filter1d(velocity, sigma=smooth_sigma)

# =======================
# HIGHPASS FILTER (REMOVE DRIFT)
# =======================
def highpass(x, fs, f_hp=30.0):
    nyq = fs / 2.0
    f_hp = max(1.0, min(f_hp, nyq * 0.9))
    b, a = butter(2, f_hp / nyq, btype="high")
    return filtfilt(b, a, x)

from scipy.signal import butter, filtfilt

def bandpass(x, fs, f_low=80.0, f_high=4000.0):
    nyq = fs / 2.0
    f_low = max(1.0, min(f_low, nyq * 0.9))
    f_high = max(f_low * 1.1, min(f_high, nyq * 0.9))
    b, a = butter(4, [f_low/nyq, f_high/nyq], btype="band")
    return filtfilt(b, a, x)


# audio = highpass(velocity, fs, f_hp=hp_cutoff)
audio = bandpass(velocity, fs, f_low=20.0, f_high=10000.0)

# normalize to [-1, 1]
audio -= np.mean(audio)
max_abs = np.max(np.abs(audio)) + 1e-12
audio /= max_abs

# =======================
# SAVE NATIVE-RATE WAV
# =======================
native_fs_int = int(round(fs))
write("dvs_motion_audio_native.wav", native_fs_int, np.int16(audio * 32767))
print(f"Saved dvs_motion_audio_native.wav @ {native_fs_int} Hz")

# =======================
# FFT ANALYSIS
# =======================
fft = np.abs(np.fft.rfft(audio))
freqs = np.fft.rfftfreq(len(audio), 1/fs)

# limit FFT display to max_audio_freq
fft_mask = freqs <= max_audio_freq
freqs_plot = freqs[fft_mask]
fft_plot = fft[fft_mask]

idx = np.argsort(fft_plot)[-N_dominant:][::-1]
print("Dominant frequencies (Hz):", freqs_plot[idx])

plt.figure(figsize=(10, 4))
plt.plot(freqs_plot, fft_plot)
plt.scatter(freqs_plot[idx], fft_plot[idx], color='red', s=10)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT of Motion-Based Event Audio")
plt.grid(True)
plt.tight_layout()
plt.show()

# =======================
# RESAMPLE TO 44.1 kHz & PLAY
# =======================
num_samples_44k = int(len(audio) * target_fs_audio / fs)
audio_resampled = resample(audio, num_samples_44k)

write("dvs_motion_audio_44k.wav", target_fs_audio, np.int16(audio_resampled * 32767))
print(f"Saved dvs_motion_audio_44k.wav @ {target_fs_audio} Hz")

print("Playing reconstructed audio...")
sd.play(audio_resampled, samplerate=target_fs_audio)
sd.wait()
