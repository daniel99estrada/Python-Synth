import numpy as np
import sounddevice as sd
from scipy import signal
import pygame
import threading
import time
from queue import Queue
import sys

class Filters:
    @staticmethod
    def low_pass(waveform, cutoff_freq, sample_rate=44100, order=5):
        """
        Apply a low-pass filter to the waveform
        cutoff_freq: frequency cutoff in Hz
        order: filter order - higher orders have sharper cutoff but more CPU intensive
        """
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, waveform)
    
    @staticmethod
    def high_pass(waveform, cutoff_freq, sample_rate=44100, order=5):
        """
        Apply a high-pass filter to the waveform
        cutoff_freq: frequency cutoff in Hz
        order: filter order - higher orders have sharper cutoff but more CPU intensive
        """
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        return signal.filtfilt(b, a, waveform)
    
    @staticmethod
    def band_pass(waveform, lowcut, highcut, sample_rate=44100, order=5):
        """
        Apply a band-pass filter to the waveform
        lowcut: lower frequency cutoff in Hz
        highcut: upper frequency cutoff in Hz
        """
        nyquist = sample_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, waveform)

class ADSR:
    def __init__(self, attack=0.1, decay=0.1, sustain=0.7, release=0.2):
        """
        Initialize ADSR envelope
        Parameters are in seconds, sustain is amplitude level (0-1)
        """
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        
    def generate(self, duration, sample_rate):
        """Generate ADSR envelope"""
        total_samples = int(duration * sample_rate)
        attack_samples = int(self.attack * sample_rate)
        decay_samples = int(self.decay * sample_rate)
        release_samples = int(self.release * sample_rate)
        sustain_samples = total_samples - attack_samples - decay_samples - release_samples
        
        if sustain_samples < 0:
            sustain_samples = 0
            attack_samples = int(total_samples * 0.3)
            decay_samples = int(total_samples * 0.2)
            release_samples = total_samples - attack_samples - decay_samples
        
        attack = np.linspace(0, 1, attack_samples)
        decay = np.linspace(1, self.sustain, decay_samples)
        sustain = np.ones(sustain_samples) * self.sustain
        release = np.linspace(self.sustain, 0, release_samples)
        
        return np.concatenate([attack, decay, sustain, release])

class Effects:
    @staticmethod
    def tremolo(waveform, rate=5, depth=0.5, sample_rate=44100):
        """Apply tremolo effect (amplitude modulation)"""
        t = np.linspace(0, len(waveform)/sample_rate, len(waveform))
        modulator = (1 - depth/2) + (depth/2) * np.sin(2 * np.pi * rate * t)
        return waveform * modulator
    
    @staticmethod
    def vibrato(waveform, rate=5, depth=0.3, sample_rate=44100):
        """Apply vibrato effect (frequency modulation)"""
        t = np.linspace(0, len(waveform)/sample_rate, len(waveform))
        depth_samples = int(depth * sample_rate / 1000)
        delay = depth_samples * np.sin(2 * np.pi * rate * t)
        
        indices = np.arange(len(waveform))
        shifted_indices = indices + delay
        valid_indices = (shifted_indices >= 0) & (shifted_indices < len(waveform))
        result = np.zeros_like(waveform)
        result[valid_indices] = np.interp(shifted_indices[valid_indices], indices, waveform)
        return result

class Synthesizer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.amplitude = 0.3
        self.adsr = ADSR()
        self.effects = Effects()
        self.filters = Filters()
        
    def generate_frequency(self, frequency, duration):
        """Generate time points for a given duration and frequency."""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        return t, 2 * np.pi * frequency * t
        
    def sine_wave(self, frequency, duration):
        """Generate a sine wave."""
        t, omega = self.generate_frequency(frequency, duration)
        return self.amplitude * np.sin(omega)
        
    def square_wave(self, frequency, duration):
        """Generate a square wave."""
        t, omega = self.generate_frequency(frequency, duration)
        return self.amplitude * signal.square(omega)
        
    def sawtooth_wave(self, frequency, duration):
        """Generate a sawtooth wave."""
        t, omega = self.generate_frequency(frequency, duration)
        return self.amplitude * signal.sawtooth(omega)
        
    def triangle_wave(self, frequency, duration):
        """Generate a triangle wave."""
        t, omega = self.generate_frequency(frequency, duration)
        return self.amplitude * signal.sawtooth(omega, 0.5)
        
    def apply_envelope(self, waveform, duration):
        """Apply ADSR envelope to waveform."""
        envelope = self.adsr.generate(duration, self.sample_rate)
        return waveform * envelope
    
    def generate_tone(self, frequency, duration, wave_type='sine', 
                     use_adsr=True, tremolo=False, vibrato=False,
                     low_pass=None, high_pass=None):
        """
        Generate a tone with optional effects and filters
        low_pass: cutoff frequency for low-pass filter (Hz)
        high_pass: cutoff frequency for high-pass filter (Hz)
        """
        wave_functions = {
            'sine': self.sine_wave,
            'square': self.square_wave,
            'sawtooth': self.sawtooth_wave,
            'triangle': self.triangle_wave
        }
        wave_function = wave_functions.get(wave_type)
        if not wave_function:
            raise ValueError(f"Invalid wave type. Choose from: {list(wave_functions.keys())}")
            
        waveform = wave_function(frequency, duration)
        
        # Apply filters
        if low_pass is not None:
            waveform = self.filters.low_pass(waveform, low_pass, self.sample_rate)
        if high_pass is not None:
            waveform = self.filters.high_pass(waveform, high_pass, self.sample_rate)
        
        # Apply ADSR envelope
        if use_adsr:
            waveform = self.apply_envelope(waveform, duration)
            
        # Apply effects
        if tremolo:
            waveform = self.effects.tremolo(waveform, sample_rate=self.sample_rate)
        if vibrato:
            waveform = self.effects.vibrato(waveform, sample_rate=self.sample_rate)
            
        return waveform
        
    def play_tone(self, waveform):
        """Play the generated waveform."""
        sd.play(waveform, self.sample_rate)
        sd.wait()
        
    def generate_chord(self, frequencies, duration, wave_type='sine', 
                      use_adsr=True, tremolo=False, vibrato=False,
                      low_pass=None, high_pass=None):
        """Generate a chord from multiple frequencies with optional filters."""
        waves = [self.generate_tone(f, duration, wave_type, use_adsr, 
                                  tremolo, vibrato, low_pass, high_pass) 
                for f in frequencies]
        return np.mean(waves, axis=0)
        
    def play_sequence(self, notes, durations, wave_type='sine', 
                     use_adsr=True, tremolo=False, vibrato=False,
                     low_pass=None, high_pass=None):
        """Play a sequence of notes with optional filters."""
        for note, duration in zip(notes, durations):
            if isinstance(note, (list, tuple)):
                wave = self.generate_chord(note, duration, wave_type, use_adsr, 
                                        tremolo, vibrato, low_pass, high_pass)
            else:
                wave = self.generate_tone(note, duration, wave_type, use_adsr, 
                                       tremolo, vibrato, low_pass, high_pass)
            self.play_tone(wave)

class SynthesizerUI:
    def __init__(self):

        self.synth = Synthesizer()
        
        # Initialize Pygame
        pygame.init()
        self.width = 800
        self.height = 400
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Synthesizer")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        
        # Synthesizer settings
        self.sample_rate = 44100
        self.current_octave = 4
        self.base_frequency = 261.63  # C4
        
        # Key mappings (Natural notes)
        self.natural_keys = {
            pygame.K_a: ('C', 0),
            pygame.K_s: ('D', 2),
            pygame.K_d: ('E', 4),
            pygame.K_f: ('F', 5),
            pygame.K_g: ('G', 7),
            pygame.K_h: ('A', 9),
            pygame.K_j: ('B', 11),
            pygame.K_k: ('C', 12),
        }
        
        # Sharp/flat keys
        self.sharp_keys = {
            pygame.K_w: ('C#', 1),
            pygame.K_e: ('D#', 3),
            pygame.K_t: ('F#', 6),
            pygame.K_y: ('G#', 8),
            pygame.K_u: ('A#', 10),
        }
        
        # Active notes
        self.active_notes = set()
        
        # Sound queue for async playback
        self.sound_queue = Queue()
        self.running = True
        
        # Start sound thread
        self.sound_thread = threading.Thread(target=self._sound_worker)
        self.sound_thread.start()
    
    def _sound_worker(self):
        """Background thread for playing sounds"""
        while self.running:
            try:
                waveform = self.sound_queue.get(timeout=0.1)
                sd.play(waveform, self.sample_rate)
                sd.wait()
            except:
                continue
    
    def get_frequency(self, semitones):
        """Calculate frequency based on semitones from C4"""
        return self.base_frequency * (2 ** ((self.current_octave - 4) + semitones/12))
    
    def generate_tone(self, frequency, duration=0.1):
        """Generate a simple sine wave tone"""
        samples = np.arange(int(duration * self.sample_rate))
        waveform = np.sin(2 * np.pi * frequency * samples / self.sample_rate)
        return waveform * 0.3  # Reduce amplitude to avoid clipping
    
    def draw_keyboard(self):
        """Draw the piano keyboard with active notes highlighted"""
        self.screen.fill(self.WHITE)
        
        # Draw octave number
        font = pygame.font.Font(None, 36)
        octave_text = f"Octave: {self.current_octave}"
        text_surface = font.render(octave_text, True, self.BLACK)
        self.screen.blit(text_surface, (20, 20))
        
        # Draw key mappings
        key_text = font.render("Keys: A S D F G H J K (natural notes) | W E T Y U (sharp notes) | Z/X (octave down/up)", True, self.BLACK)
        self.screen.blit(key_text, (20, 60))
        
        # Draw piano keys
        key_width = 60
        key_height = 200
        start_x = 100
        start_y = 150
        
        # Draw white keys
        for i, (key, (note, _)) in enumerate(self.natural_keys.items()):
            x = start_x + i * key_width
            color = self.RED if chr(key) in self.active_notes else self.WHITE
            pygame.draw.rect(self.screen, color, (x, start_y, key_width - 2, key_height))
            pygame.draw.rect(self.screen, self.BLACK, (x, start_y, key_width - 2, key_height), 2)
            
            # Draw note name
            note_text = font.render(note, True, self.BLACK)
            self.screen.blit(note_text, (x + 20, start_y + key_height - 30))
        
        # Draw black keys
        black_key_width = 40
        black_key_height = 120
        for key, (note, semitones) in self.sharp_keys.items():
            index = semitones - 1
            if index not in [2, 6, 9]:  # Skip positions where there are no black keys
                x = start_x + (index * key_width / 2) + (key_width - black_key_width) / 2
                color = self.BLUE if chr(key) in self.active_notes else self.BLACK
                pygame.draw.rect(self.screen, color, (x, start_y, black_key_width, black_key_height))
                
                # Draw note name
                note_text = font.render(note, True, self.WHITE)
                self.screen.blit(note_text, (x + 5, start_y + black_key_height - 30))
        
        pygame.display.flip()
    
    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                elif event.type == pygame.KEYDOWN:
                    # Octave control
                    if event.key == pygame.K_z:
                        self.current_octave = max(0, self.current_octave - 1)
                    elif event.key == pygame.K_x:
                        self.current_octave = min(8, self.current_octave + 1)
                    
                    # Natural keys
                    if event.key in self.natural_keys:
                        note_char = chr(event.key)
                        if note_char not in self.active_notes:
                            self.active_notes.add(note_char)
                            _, semitones = self.natural_keys[event.key]
                            freq = self.get_frequency(semitones)
                            self.sound_queue.put(self.synth.generate_tone(freq, 1.0, wave_type='sawtooth', low_pass=1000))
                    
                    # Sharp/flat keys
                    if event.key in self.sharp_keys:
                        note_char = chr(event.key)
                        if note_char not in self.active_notes:
                            self.active_notes.add(note_char)
                            _, semitones = self.sharp_keys[event.key]
                            freq = self.get_frequency(semitones)                        
                            self.sound_queue.put(self.synth.generate_tone(freq, 1.0, wave_type='sawtooth', low_pass=1000))
                
                elif event.type == pygame.KEYUP:
                    if event.key in self.natural_keys:
                        self.active_notes.discard(chr(event.key))
                    if event.key in self.sharp_keys:
                        self.active_notes.discard(chr(event.key))
            
            self.draw_keyboard()
            clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    synth_ui = SynthesizerUI()
    synth_ui.run()
