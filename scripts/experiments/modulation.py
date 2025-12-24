import numpy as np
from scipy.signal import hilbert, butter, sosfilt
import sounddevice as sd
import librosa

# ---------------- Parameters ----------------
fs = 44100                  # playback sample rate
global_gain = 0.4          # safe playback volume (0.1-0.2)
band_gain = 8.0             # per-band amplification for audibility
snippet_seconds = 100       # play only first few seconds for quick test; set 0 for full song

# MP3 input file
filename = "bohemian_cut.mp3"      # <-- replace with your song

# ---------------- Load MP3 ----------------
print("Loading MP3...")
audio, sr = librosa.load(filename, sr=fs, mono=True)
audio = audio / np.max(np.abs(audio))  # normalize
print(f"Loaded {len(audio)/fs:.1f} sec of audio.")

# Optional snippet for fast testing
if snippet_seconds > 0:
    audio = audio[:fs*snippet_seconds]

# ---------------- Define Bands ----------------
bands = [(20,200), (200,500), (500,1000), (1000,2000), (2000,4000)]
num_bands = len(bands)

# Carriers: audible range for laptop (300–1500 Hz)
carriers = np.linspace(300, 1500, num_bands)

# ---------------- Helper: Bandpass Filter ----------------
def bandpass_filter(x, low, high, fs):
    sos = butter(4, [low/(fs/2), high/(fs/2)], btype='band', output='sos')
    return sosfilt(sos, x)

# ---------------- Process Each Band ----------------
t = np.arange(len(audio)) / fs
output = np.zeros_like(audio)

print("Processing bands...")
for (low, high), carrier in zip(bands, carriers):
    band = bandpass_filter(audio, low, high, fs)
    envelope = np.abs(hilbert(band))
    output += envelope * np.sin(2*np.pi*carrier*t) * band_gain
print("Processing done.")

# ---------------- Normalize & Apply Global Gain ----------------
output = output / (np.max(np.abs(output)) + 1e-9)  # prevent clipping
output = output * global_gain
output = output.astype(np.float32)

# Debug info
print("Output stats:", np.min(output), np.max(output), np.mean(output))

# ---------------- Playback ----------------
print("Playing processed audio...")
sd.play(output, fs)
sd.wait()
print("Done!")
