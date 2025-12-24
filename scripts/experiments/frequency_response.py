import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------------------- CHORD PRESETS --------------------
# Frequencies in Hz for the 4th/5th octave (approx)
chords = {
    # C family
    "C":    [261.63, 329.63, 392.00],                 # C  E  G
    "Cm":   [261.63, 311.13, 392.00],                 # C  Eb G
    "C7":   [261.63, 329.63, 392.00, 466.16],         # C  E  G  Bb
    "Cm7":  [261.63, 311.13, 392.00, 466.16],         # C  Eb G  Bb

    # D family
    "D":    [293.66, 369.99, 440.00],                 # D  F# A
    "Dm":   [293.66, 349.23, 440.00],                 # D  F  A
    "D7":   [293.66, 369.99, 440.00, 523.25],         # D  F# A  C
    "Dm7":  [293.66, 349.23, 440.00, 523.25],         # D  F  A  C

    # E family
    "E":    [329.63, 415.30, 493.88],                 # E  G# B
    "Em":   [329.63, 392.00, 493.88],                 # E  G  B
    "E7":   [329.63, 415.30, 493.88, 587.33],         # E  G# B  D
    "Em7":  [329.63, 392.00, 493.88, 587.33],         # E  G  B  D

    # F family
    "F":    [349.23, 440.00, 523.25],                 # F  A  C
    "Fm":   [349.23, 415.30, 523.25],                 # F  Ab C
    "F7":   [349.23, 440.00, 523.25, 587.33],         # F  A  C  Eb
    "Fm7":  [349.23, 415.30, 523.25, 587.33],         # F  Ab C  Eb

    # G family
    "G":    [196.00, 246.94, 293.66],                 # G  B  D
    "Gm":   [196.00, 233.08, 293.66],                 # G  Bb D
    "G7":   [196.00, 246.94, 293.66, 349.23],         # G  B  D  F
    "Gm7":  [196.00, 233.08, 293.66, 349.23],         # G  Bb D  F

    # A family
    "A":    [220.00, 277.18, 329.63],                 # A  C# E
    "Am":   [220.00, 261.63, 329.63],                 # A  C  E
    "A7":   [220.00, 277.18, 329.63, 392.00],         # A  C# E  G
    "Am7":  [220.00, 261.63, 329.63, 392.00],         # A  C  E  G

    # B family
    "B":    [246.94, 311.13, 369.99],                 # B  D# F#
    "Bm":   [246.94, 293.66, 369.99],                 # B  D  F#
    "B7":   [246.94, 311.13, 369.99, 440.00],         # B  D# F# A
    "Bm7":  [246.94, 293.66, 369.99, 440.00],         # B  D  F# A
}


def play_chord(chord_name):
    freqs = chords.get(chord_name)
    if not freqs:
        return
    delete_all()  # clear existing tones
    with state_lock:
        for f in freqs:
            frequencies.append(f)
            gains.append(1.0)
            phases.append(0.0)
            active_flags.append(True)
    for f in freqs:
        add_freq_row(f)


# --------------------
# AUDIO SETTINGS
# --------------------
sample_rate = 44100
master_volume = 0.3
plot_buffer_size = 1024  # Number of samples to display

# --------------------
# SHARED STATE
# --------------------
frequencies = []       # [float]
gains = []             # [0..inf floats]
phases = []            # per-tone phase
active_flags = []      # True/False
latest_buf = np.zeros(plot_buffer_size, dtype=np.float32)
state_lock = threading.Lock()

# --------------------
# AUDIO CALLBACK
# --------------------
def audio_callback(outdata, frames, time, status):
    global frequencies, phases, gains, active_flags, latest_buf

    buf = np.zeros(frames, dtype=np.float32)

    with state_lock:
        freqs = list(frequencies)
        phs = list(phases)
        g = list(gains)
        active = list(active_flags)

        for i, f in enumerate(freqs):
            if not active[i]:
                continue
            phase = phs[i]
            inc = 2 * np.pi * f / sample_rate
            idx = np.arange(frames, dtype=np.float32)
            buf += g[i] * np.sin(phase + inc * idx).astype(np.float32)
            phs[i] = (phase + inc * frames) % (2 * np.pi)

        phases[:] = phs

    buf *= master_volume
    outdata[:] = buf.reshape(-1, 1)

    # Update buffer for plotting
    with state_lock:
        latest_buf = buf[-plot_buffer_size:].copy()

# Start audio stream
stream = sd.OutputStream(
    samplerate=sample_rate,
    channels=1,
    callback=audio_callback,
    blocksize=0
)
stream.start()

# --------------------
# GUI FUNCTIONS
# --------------------
def add_frequency(event=None):
    """Add new frequency and create slider."""
    try:
        f = float(entry.get())
        if f <= 0:
            return
    except:
        return

    entry.delete(0, tk.END)

    with state_lock:
        frequencies.append(f)
        gains.append(1.0)
        phases.append(0.0)
        active_flags.append(True)

    add_freq_row(f)

def add_freq_row(freq):
    frame = tk.Frame(freq_frame)
    frame.pack(fill="x", pady=2)

    row = {"frame": frame, "freq": freq}

    var_active = tk.BooleanVar(value=True)
    row["active"] = var_active

    row_index = len(rows)
    row["index"] = row_index

    cb = tk.Checkbutton(frame, variable=var_active,
        command=lambda r=row: toggle_active(r["index"]))
    cb.pack(side="left")

    lbl = tk.Label(frame, text=f"{freq:.2f} Hz")
    lbl.pack(side="left", padx=5)

    slider = tk.Scale(frame, from_=0, to=5, resolution=0.01,
                      orient="horizontal", length=200,
                      command=lambda val, r=row: update_gain(r["index"], val))
    row["slider"] = slider
    slider.set(1.0)
    slider.pack(side="left", padx=5)

    rows.append(row)

def update_gain(idx, val):
    """Update gain for one frequency."""
    with state_lock:
        if idx < len(gains):
            gains[idx] = float(val)

def toggle_active(idx):
    """Toggle active/inactive state."""
    with state_lock:
        if idx < len(active_flags):
            active_flags[idx] = rows[idx]["active"].get()

def delete_selected():
    to_delete = [i for i, row in enumerate(rows) if row["active"].get()]
    for idx in reversed(to_delete):

        rows[idx]["frame"].destroy()

        with state_lock:
            frequencies.pop(idx)
            gains.pop(idx)
            phases.pop(idx)
            active_flags.pop(idx)

        rows.pop(idx)

    # RENORMALIZE INDICES
    for i, row in enumerate(rows):
        row["index"] = i

def delete_all():
    # Remove all GUI widgets
    for row in rows:
        row["frame"].destroy()
    rows.clear()

    # Clear audio lists safely
    with state_lock:
        frequencies.clear()
        gains.clear()
        phases.clear()
        active_flags.clear()

# --------------------
# GUI SETUP
# --------------------
root = tk.Tk()
root.title("Mixer")

# Left panel
left_frame = tk.Frame(root)
left_frame.pack(side="left", padx=10, pady=10)

entry = tk.Entry(left_frame)
entry.pack(pady=5)
entry.bind("<Return>", add_frequency)

btn_add = tk.Button(left_frame, text="Add Frequency", command=add_frequency)
btn_add.pack(pady=5)

freq_frame = tk.Frame(left_frame)
freq_frame.pack(pady=10)

rows = []  # list of dictionaries with keys: frame, freq, slider, active, index

btn_del_sel = tk.Button(left_frame, text="Delete Selected", command=delete_selected)
btn_del_sel.pack(pady=3)

btn_del_all = tk.Button(left_frame, text="Delete All", command=delete_all)
btn_del_all.pack(pady=3)

# Right panel: waveform plot
fig, ax = plt.subplots(figsize=(5,3))
line, = ax.plot(np.zeros(plot_buffer_size))
ax.set_ylim(-5, 5)
ax.set_xlim(0, plot_buffer_size)
ax.set_title("Output Waveform")
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")
plt.tight_layout()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side="right", padx=10, pady=10)

def update_plot():
    with state_lock:
        line.set_ydata(latest_buf)
    canvas.draw()
    root.after(30, update_plot)  # update ~33 times/sec

update_plot()

# -------------------- Preset Chord Buttons --------------------
preset_frame = tk.LabelFrame(left_frame, text="Chord Presets")
preset_frame.pack(pady=10)

# Define families in order
families = ["C", "D", "E", "F", "G", "A", "B"]

# For each family, create a vertical frame for its chords
for col, family in enumerate(families):
    family_frame = tk.Frame(preset_frame)
    family_frame.grid(row=0, column=col, padx=5, pady=5, sticky="n")

    # Select all chords starting with this family letter
    family_chords = [name for name in chords.keys() if name.startswith(family)]

    # Create buttons stacked vertically
    for chord_name in family_chords:
        btn = tk.Button(family_frame, text=chord_name, width=6,
                        command=lambda c=chord_name: play_chord(c))
        btn.pack(pady=2)



root.mainloop()

stream.stop()
stream.close()
