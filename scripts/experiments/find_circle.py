import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time
import itertools

# ------------------------
# Settings
# ------------------------
sample_rate = 44100
duration = 2.0  # seconds per test
master_volume = 0.3
top_n_per_batch = 5  # how many best combos per batch to play immediately
final_top_n = 10  # final best to play at the end

# Candidate parameters
freq_candidates = np.arange(90, 110, 0.5)  # Hz
phase_candidates = np.deg2rad([80, 85, 90, 95, 100])
amp_candidates = [(1.0, 1.0), (1.0, 0.95), (1.0, 0.9)]

# Batch settings
batch_fraction = 0.2  # process 20% at a time


# ------------------------
# Function to score circularity
# ------------------------
def circularity_score(x, y):
    x_std = np.std(x)
    y_std = np.std(y)
    ratio = min(x_std, y_std) / max(x_std, y_std)
    return ratio


# ------------------------
# Generate all candidate combinations
# ------------------------
all_combos = list(itertools.product(freq_candidates, freq_candidates, phase_candidates, amp_candidates))
# Filter out too-distant frequencies (>5 Hz)
all_combos = [c for c in all_combos if abs(c[0] - c[1]) <= 5.0]

# ------------------------
# Split into batches
# ------------------------
batch_size = max(1, int(len(all_combos) * batch_fraction))
batches = [all_combos[i:i + batch_size] for i in range(0, len(all_combos), batch_size)]

# ------------------------
# Main batch processing
# ------------------------
all_best = []

t_global = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

for batch_idx, batch in enumerate(batches):
    print(f"\nProcessing batch {batch_idx + 1}/{len(batches)} ({len(batch)} combinations)...")
    batch_results = []

    for combo in tqdm(batch, desc=f"Batch {batch_idx + 1}", ncols=100):
        f1, f2, phase, (a1, a2) = combo
        x = a1 * np.sin(2 * np.pi * f1 * t_global)
        y = a2 * np.sin(2 * np.pi * f2 * t_global + phase)
        score = circularity_score(x, y)
        batch_results.append((score, f1, f2, phase, a1, a2))

    # Sort batch results and keep top_n_per_batch
    batch_results.sort(reverse=True, key=lambda r: r[0])
    top_batch = batch_results[:top_n_per_batch]

    # Play top batch immediately
    for i, (score, f1, f2, phase, a1, a2) in enumerate(top_batch):
        print(
            f"Batch top {i + 1}: f1={f1}, f2={f2}, phase={np.rad2deg(phase):.1f}°, amps=({a1},{a2}), score={score:.3f}")

        x = a1 * np.sin(2 * np.pi * f1 * t_global)
        y = a2 * np.sin(2 * np.pi * f2 * t_global + phase)
        audio = x + y
        audio /= np.max(np.abs(audio)) + 1e-12
        audio *= master_volume

        # Show plot
        plt.figure(figsize=(4, 4))
        plt.plot(x, y)
        plt.title(f"Batch {batch_idx + 1} top {i + 1} | Score {score:.3f}")
        plt.axis("equal")
        plt.show(block=False)

        # Play audio
        sd.play(audio, sample_rate)
        sd.wait()
        plt.close()
        time.sleep(0.3)

    # Add batch top to global best
    all_best.extend(top_batch)

# ------------------------
# Final global scoring
# ------------------------
print("\nRescoring all best candidates globally...")
all_best.sort(reverse=True, key=lambda r: r[0])
final_best = all_best[:final_top_n]

print("\nFinal top combinations:")
for i, (score, f1, f2, phase, a1, a2) in enumerate(final_best):
    print(f"{i + 1}: f1={f1}, f2={f2}, phase={np.rad2deg(phase):.1f}°, amps=({a1},{a2}), score={score:.3f}")

    x = a1 * np.sin(2 * np.pi * f1 * t_global)
    y = a2 * np.sin(2 * np.pi * f2 * t_global + phase)
    audio = x + y
    audio /= np.max(np.abs(audio)) + 1e-12
    audio *= master_volume

    # Show plot
    plt.figure(figsize=(4, 4))
    plt.plot(x, y)
    plt.title(f"Final top {i + 1} | Score {score:.3f}")
    plt.axis("equal")
    plt.show(block=False)

    # Play audio
    sd.play(audio, sample_rate)
    sd.wait()
    plt.close()
    time.sleep(0.3)

# ------------------------
# Save to CSV
# ------------------------
df = pd.DataFrame(final_best, columns=["Score", "Freq1", "Freq2", "PhaseRad", "Amp1", "Amp2"])
df.to_csv("best_circle_combinations.csv", index=False)
print("\nSaved final top combinations to best_circle_combinations.csv")
