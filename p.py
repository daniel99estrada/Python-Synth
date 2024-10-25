
import numpy as np
import sounddevice as sd
from scipy import signal
import time

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

def main():
    # Initialize synthesizer
    synth = Synthesizer()
    
    # Define frequencies (Hz)
    C4 = 261.63
    E4 = 329.63
    G4 = 392.00
    A4 = 440.00
    
    print("1. Playing raw sawtooth wave...")
    synth.play_tone(synth.generate_tone(C4, 1.0, wave_type='sawtooth'))
    time.sleep(0.5)
    
    print("2. Playing sawtooth wave with low-pass filter (cutoff: 1000 Hz)...")
    synth.play_tone(synth.generate_tone(C4, 1.0, wave_type='sawtooth', low_pass=1000))
    time.sleep(0.5)
    
    print("3. Playing sawtooth wave with high-pass filter (cutoff: 500 Hz)...")
    synth.play_tone(synth.generate_tone(C4, 1.0, wave_type='sawtooth', high_pass=500))
    time.sleep(0.5)
    
    print("4. Playing melody with low-pass filter...")
    notes = [C4, E4, G4, A4]
    durations = [0.5, 0.5, 0.5, 1.0]
    synth.play_sequence(notes, durations, wave_type='square', low_pass=800)
    
    print("5. Playing chord with both filters...")
    chord = synth.generate_chord([C4, E4, G4], 2.0, wave_type='sawtooth', 
                               low_pass=2000, high_pass=200)
    synth.play_tone(chord)
    
    print("6. Playing filtered sound with effects...")
    complex_tone = synth.generate_tone(A4, 2.0, wave_type='square',
                                     tremolo=True, vibrato=True, 
                                     low_pass=1500)
    synth.play_tone(complex_tone)

if __name__ == "__main__":
    main()