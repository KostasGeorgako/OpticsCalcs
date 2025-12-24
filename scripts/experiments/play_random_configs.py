import numpy as np
import sounddevice as sd
import time

# ------------------------
# Settings
# ------------------------
sample_rate = 44100  # Hz
N = 5  # number of random frequencies per config
freq_min = 50  # Hz
freq_max = 300  # Hz
T = 2.0  # seconds per configuration
master_volume = 0.3

# ------------------------
# Generate and play indefinitely
# ------------------------
print("Starting random multi-frequency playback. Press Ctrl+C to stop.")

try:
    while True:
        # --- Random frequencies ---
        freqs = np.random.uniform(freq_min, freq_max, N)

        # --- Random amplitudes normalized to sum 1 ---
        amps = np.random.rand(N)
        amps /= np.sum(amps) + 1e-12

        # --- Time array ---
        t = np.linspace(0, T, int(sample_rate * T), endpoint=False)

        # --- Generate audio signal ---
        audio = np.zeros_like(t)
        for f, a in zip(freqs, amps):
            audio += a * np.sin(2 * np.pi * f * t)

        # --- Normalize and scale volume ---
        audio /= np.max(np.abs(audio)) + 1e-12
        audio *= master_volume

        # --- Play audio ---
        sd.play(audio, samplerate=sample_rate)
        sd.wait()

        print(f"Played config: freq={freqs}, amps={amps}")

except KeyboardInterrupt:
    print("\nStopped by user.")
    sd.stop()
