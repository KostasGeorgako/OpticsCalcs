import numpy as np
import sounddevice as sd
import time

sample_rate = 44100
volume = 0.2

# -----------------------------------------
# NOTE → FREQUENCY MAP (A4 = 440 Hz)
# -----------------------------------------
notes = {
    "C4": 261.63,
    "C#4": 277.18,
    "D4": 293.66,
    "D#4": 311.13,
    "E4": 329.63,
    "F4": 349.23,
    "F#4": 369.99,
    "G4": 392.00,
    "G#4": 415.30,
    "A4": 440.00,
    "A#4": 466.16,
    "B4": 493.88,

    "C5": 523.25,
    "D5": 587.33,
    "E5": 659.25,
    "F5": 698.46,
    "G5": 783.99,
}

# -----------------------------------------
# DAISY BELL MELODY (first verse)
# Each tuple: (NOTE, DURATION in seconds)
# -----------------------------------------
melody = [
    ("E4", 0.40), ("D4", 0.40), ("C4", 0.80),        # Dai-sy
    ("E4", 0.40), ("D4", 0.40), ("C4", 0.80),        # Dai-sy

    ("G4", 0.40), ("E4", 0.40), ("D4", 0.40), ("C4", 0.60),  # Give me your an-
    ("G4", 0.40), ("E4", 0.40), ("D4", 0.40), ("C4", 0.60),  # -swer, do!

    ("E4", 0.40), ("G4", 0.40), ("C5", 0.80),        # I'm half
    ("B4", 0.40), ("G4", 0.40), ("E4", 0.80),        # cra-zy

    ("E4", 0.40), ("D4", 0.40), ("C4", 0.80),        # All for the
    ("E4", 0.40), ("D4", 0.40), ("C4", 1.00),        # love of you!
]

# -----------------------------------------
# PLAY A SINGLE NOTE
# -----------------------------------------
def play_tone(freq, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t).astype(np.float32)
    sd.play(volume * wave, samplerate=sample_rate)
    sd.wait()  # wait until done


# -----------------------------------------
# MAIN: PLAY THE MELODY
# -----------------------------------------
print("Playing Daisy Bell…")
for note, dur in melody:
    freq = notes[note]
    play_tone(freq, dur)

print("Done.")
