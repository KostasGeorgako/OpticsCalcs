import sounddevice as sd
import numpy as np
from scipy.signal import butter, lfilter

# --- DEVICE SELECTION ---
INPUT_DEVICE = 1    # Intel Smart Sound Technology microphone array
OUTPUT_DEVICE = 4   # Realtek Headphones (or speakers)

# --- AUDIO SETTINGS ---
FS = 44100          # Sampling rate
BLOCKSIZE = 1024    # Lower = less latency, higher = more stable
CHANNELS = 1        # Mono
GAIN = 1.8          # Adjust mic loudness (safe range: 1.0–3.0)

# --- FILTER: High-pass to reduce low-frequency feedback ---
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

b_hp, a_hp = butter_highpass(cutoff=180, fs=FS, order=4)

def highpass(data):
    return lfilter(b_hp, a_hp, data)

# --- STREAM CALLBACK ---
def callback(indata, outdata, frames, time, status):
    if status:
        print("Status:", status)

    # 1) get microphone audio
    mic = indata[:, 0]

    # 2) apply high-pass filter (reduces feedback)
    filtered = highpass(mic)

    # 3) apply gain boost
    boosted = filtered * GAIN

    # 4) prevent clipping
    boosted = np.clip(boosted, -1.0, 1.0)

    # 5) send to output
    outdata[:, 0] = boosted


# --- RUN AUDIO STREAM ---
with sd.Stream(
    samplerate=FS,
    blocksize=BLOCKSIZE,
    channels=CHANNELS,
    dtype='float32',
    device=(INPUT_DEVICE, OUTPUT_DEVICE),
    callback=callback
):
    print("Real-time mic → speaker running with filtering & gain. Press Ctrl+C to stop.")
    while True:
        sd.sleep(1000)
