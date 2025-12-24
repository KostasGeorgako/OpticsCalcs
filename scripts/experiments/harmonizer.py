import tkinter as tk
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
BLOCK_TIME = 0.05  # seconds

# Equal-tempered note frequencies for octave 2 and up
# We generate dynamically using semitone math relative to A4 = 440 Hz
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

A4_FREQ = 440.0
A4_INDEX = NOTE_NAMES.index('A') + 12 * 4  # A4 is 9th note + 4*12 = 57 semitones from C0


def note_to_freq(name, octave):
    """Compute frequency of note name at given octave (scientific pitch)."""
    # Determine semitone index relative to C0
    name_index = NOTE_NAMES.index(name)
    semitone_index = name_index + 12 * octave
    # Frequency relative to A4
    freq = A4_FREQ * 2 ** ((semitone_index - A4_INDEX) / 12.0)
    return freq


# Map keyboard keys to natural notes
KEY_TO_NOTE = {
    'a': 'C',
    's': 'D',
    'd': 'E',
    'f': 'F',
    'g': 'G',
    'h': 'A',
    'j': 'B',
}


class KeyboardChordApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Keyboard Chords")
        self.root.geometry("500x200")
        label = tk.Label(self.root, text=(
            "Press A S D F G H J for roots (C D E F G A B)\n"
            "Hold '#' (hash) together to make it sharp (#)\n"
            "Then optionally hold 'm' for minor, '7' for dominant-7, or both for minor-7.\n"
            "Release to silence.\n"
            "Frequencies now stay within 50-1500 Hz range."
        ), font=("Arial", 12))
        label.pack(expand=True)

        self.pressed = set()
        self.phases = {}  # freq -> phase
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback,
            blocksize=int(SAMPLE_RATE * BLOCK_TIME),
        )
        self.stream.start()

        self.root.focus_set()
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_key_press(self, event):
        k = event.keysym.lower()
        if k == 'escape':
            self.on_close()
        else:
            self.pressed.add(k)

    def on_key_release(self, event):
        k = event.keysym.lower()
        if k in self.pressed:
            self.pressed.remove(k)

    def get_current_chord(self):
        # find root key
        root_key = None
        for key in KEY_TO_NOTE:
            if key in self.pressed:
                root_key = key
                break

        if root_key is None:
            return None

        note = KEY_TO_NOTE[root_key]

        # apply sharp if '#' in pressed
        if '#' in self.pressed:
            idx = NOTE_NAMES.index(note)
            note = NOTE_NAMES[(idx + 1) % 12]

        # choose chord variant
        minor = 'm' in self.pressed
        seven = '7' in self.pressed

        variant = 'maj'
        if seven and minor:
            variant = 'm7'
        elif seven:
            variant = '7'
        elif minor:
            variant = 'min'

        return (note, variant)

    def get_chord_octave(self, note, variant):
        """Dynamically choose octave so all chord tones stay in 50-1500 Hz."""
        idx = NOTE_NAMES.index(note)

        # Test different base octaves
        for base_octave in range(1, 6):
            freqs = self.chord_freqs(note, base_octave, variant)
            if all(50 <= f <= 1500 for f in freqs):
                return base_octave

        # Fallback to octave 3 if none fit perfectly
        return 3

    def chord_freqs(self, note, octave, variant):
        idx = NOTE_NAMES.index(note)
        # intervals in semitones
        if variant in ('maj', '7'):
            third_int = 4
        else:
            third_int = 3
        fifth_int = 7
        seventh_int = 10  # minor 7th above root

        freqs = []
        root_freq = note_to_freq(note, octave)
        freqs.append(root_freq)

        # third
        n3_idx = (idx + third_int) % 12
        octave3 = octave + (idx + third_int) // 12
        n3 = NOTE_NAMES[n3_idx]
        freqs.append(note_to_freq(n3, octave3))

        # fifth
        n5_idx = (idx + fifth_int) % 12
        octave5 = octave + (idx + fifth_int) // 12
        n5 = NOTE_NAMES[n5_idx]
        freqs.append(note_to_freq(n5, octave5))

        if variant in ('7', 'm7'):
            n7_idx = (idx + seventh_int) % 12
            octave7 = octave + (idx + seventh_int) // 12
            n7 = NOTE_NAMES[n7_idx]
            freqs.append(note_to_freq(n7, octave7))

        return freqs

    def audio_callback(self, outdata, frames, time_info, status):
        chord = self.get_current_chord()
        if chord is None:
            outdata[:] = np.zeros((frames, 1), dtype=np.float32)
            return

        note, variant = chord
        octave = self.get_chord_octave(note, variant)
        freqs = self.chord_freqs(note, octave, variant)

        t = np.arange(frames) / SAMPLE_RATE
        wave = np.zeros(frames, dtype=np.float32)

        for f in freqs:
            phase = self.phases.get(f, 0.0)
            inc = 2 * np.pi * f / SAMPLE_RATE
            samples = np.sin(phase + inc * np.arange(frames))
            wave += samples
            self.phases[f] = (phase + inc * frames) % (2 * np.pi)

        # simple normalization
        maxval = np.max(np.abs(wave)) + 1e-9
        wave = 0.3 * wave / maxval
        outdata[:] = wave.reshape(-1, 1).astype(np.float32)

    def on_close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = KeyboardChordApp()
    app.run()
