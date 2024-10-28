import numpy as np
import sounddevice as sd
import mido
import math
from collections import defaultdict
import threading
import time

class LFO:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.phase = 0.0
        self.rate = 2.0  # Hz
        self.depth = 0.5  # Semitones
        self.wave_type = 'sine'
    
    def set_rate(self, rate):
        """Set LFO rate in Hz (0.1 to 20 Hz)"""
        self.rate = max(0.1, min(20.0, rate))
    
    def set_depth(self, depth):
        """Set LFO depth in semitones (0 to 12)"""
        self.depth = max(0.0, min(12.0, depth))
    
    def set_wave_type(self, wave_type):
        """Set LFO waveform type"""
        if wave_type in ['sine', 'triangle', 'square']:
            self.wave_type = wave_type
    
    def generate_sample(self):
        """Generate next LFO sample and update phase"""
        if self.wave_type == 'sine':
            value = np.sin(self.phase)
        elif self.wave_type == 'triangle':
            value = 2 * abs(2 * (self.phase / (2 * np.pi) - np.floor(0.5 + self.phase / (2 * np.pi)))) - 1
        else:  # square
            value = np.sign(np.sin(self.phase))
        
        # Update phase
        self.phase += 2 * np.pi * self.rate / self.sample_rate
        self.phase %= 2 * np.pi
        
        return value * self.depth
    
class Filter:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.cutoff = 1000.0  # Default cutoff frequency
        self.resonance = 0.7  # Default resonance (0.0 to 1.0)
        self.type = 'lowpass'  # Default filter type
        
        # Filter state variables
        self.low = 0.0
        self.band = 0.0
        self.high = 0.0
    
    def set_cutoff(self, frequency):
        """Set filter cutoff frequency"""
        self.cutoff = max(20.0, min(20000.0, frequency))
    
    def set_resonance(self, resonance):
        """Set filter resonance (0.0 to 1.0)"""
        self.resonance = max(0.0, min(0.99, resonance))
    
    def set_type(self, filter_type):
        """Set filter type (lowpass, highpass, bandpass)"""
        if filter_type in ['lowpass', 'highpass', 'bandpass']:
            self.type = filter_type
    
    def process(self, input_signal):
        """Process audio through the filter"""
        output = np.zeros_like(input_signal)
        
        # Pre-calculate filter coefficients
        f = 2.0 * np.sin(np.pi * self.cutoff / self.sample_rate)
        q = 1.0 - self.resonance
        
        for i in range(len(input_signal)):
            # Update filter state
            self.high = input_signal[i] - self.low - q * self.band
            self.band += f * self.high
            self.low += f * self.band
            
            # Output based on filter type
            if self.type == 'lowpass':
                output[i] = self.low
            elif self.type == 'highpass':
                output[i] = self.high
            else:  # bandpass
                output[i] = self.band
        
        return output

class NoteState:
    def __init__(self, velocity, sample_rate, adsr):
        self.velocity = velocity
        self.phase = 0.0
        self.start_time = time.time()
        self.released_time = None
        self.is_released = False
        self.adsr = adsr
        self.sample_rate = sample_rate
        self.total_samples = 0

    def get_envelope_value(self, num_samples):
        """Calculate ADSR envelope value for the current state"""
        current_time = self.total_samples / self.sample_rate
        
        if self.is_released:
            release_time = (time.time() - self.released_time)
            if release_time >= self.adsr.release:
                return 0
            return self.adsr.sustain * (1 - release_time / self.adsr.release)
        
        if current_time < self.adsr.attack:
            return current_time / self.adsr.attack
        
        if current_time < (self.adsr.attack + self.adsr.decay):
            decay_progress = (current_time - self.adsr.attack) / self.adsr.decay
            return 1.0 + (self.adsr.sustain - 1.0) * decay_progress
        
        return self.adsr.sustain

class ADSR:
    def __init__(self, attack=0.1, decay=0.1, sustain=0.7, release=0.2):
        self.attack = max(0.001, attack)
        self.decay = max(0.001, decay)
        self.sustain = max(0.0, min(1.0, sustain))
        self.release = max(0.001, release)

    def set_adsr(self, attack, decay, sustain, release):
        """
        Sets the ADSR envelope parameters for the synthesizer
        
        Parameters:
        attack (float): Attack time in seconds
        decay (float): Decay time in seconds
        sustain (float): Sustain level (0-1)
        release (float): Release time in seconds
        """
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        

class EnvelopeFilter(Filter):
    def __init__(self, sample_rate):
        super().__init__(sample_rate)
        self.base_cutoff = 1000.0
        self.envelope_amount = 0.5
        self.min_cutoff = 20.0
        self.max_cutoff = 20000.0
        self.lfo_mod = 0.0
        self.adsr_mix = 0.5  # New ADSR mix control

    def set_adsr_mix(self, mix_amount):
        """Set the mix between static filter and ADSR-controlled filter (0.0 to 1.0)"""
        self.adsr_mix = max(0.0, min(1.0, mix_amount))
    
    def set_envelope_amount(self, amount):
        """Set the amount of envelope modulation (0.0 to 1.0)"""
        self.envelope_amount = max(0.0, min(1.0, amount))
    
    def set_base_cutoff(self, frequency):
        """Set the base cutoff frequency"""
        self.base_cutoff = max(self.min_cutoff, min(self.max_cutoff, frequency))
    
    def calculate_cutoff(self, envelope_value):
        """Calculate cutoff frequency with ADSR mix control"""
        # Scale the envelope value exponentially
        exp_env = envelope_value ** 2
        
        # Calculate envelope modulation with mix control
        mod_range = self.max_cutoff - self.base_cutoff
        env_modulation = mod_range * exp_env * self.envelope_amount * self.adsr_mix
        
        # Mix between static and modulated filter
        static_cutoff = self.base_cutoff
        modulated_cutoff = self.base_cutoff + env_modulation + self.lfo_mod
        
        # Crossfade between static and modulated cutoff
        current_cutoff = (static_cutoff * (1 - self.adsr_mix) + 
                         modulated_cutoff * self.adsr_mix)
        
        return max(self.min_cutoff, min(self.max_cutoff, current_cutoff))
    
    def process(self, input_signal, envelope_value):
        """Process audio through the filter with envelope modulation"""
        output = np.zeros_like(input_signal)
        
        # Calculate current cutoff based on envelope
        self.cutoff = self.calculate_cutoff(envelope_value)
        
        # Pre-calculate filter coefficients
        f = 2.0 * np.sin(np.pi * self.cutoff / self.sample_rate)
        q = 1.0 - self.resonance
        
        for i in range(len(input_signal)):
            # Update filter state
            self.high = input_signal[i] - self.low - q * self.band
            self.band += f * self.high
            self.low += f * self.band
            
            # Output based on filter type
            if self.type == 'lowpass':
                output[i] = self.low
            elif self.type == 'highpass':
                output[i] = self.high
            else:  # bandpass
                output[i] = self.band
        
        return output

class MIDISynthesizer:
    def __init__(self, sample_rate=44100, buffer_size=512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.active_notes = {}
        self.is_playing = False
        self.stream = None
        self.last_frame = np.zeros(buffer_size)
        self.wave_type = 'sine'
        
        # Initialize ADSR, EnvelopeFilter, and LFO
        self.adsr = ADSR(attack=0.1, decay=0.1, sustain=0.7, release=0.2)
        self.filter = EnvelopeFilter(sample_rate)
        self.lfo = LFO(sample_rate)
    
    def set_lfo_params(self, rate=None, depth=None, wave_type=None):
        """Set LFO parameters"""
        if rate is not None:
            self.lfo.set_rate(rate)
        if depth is not None:
            self.lfo.set_depth(depth)
        if wave_type is not None:
            self.lfo.set_wave_type(wave_type)
    
    def generate_continuous_tone(self, frequency, num_samples, note_state):
        # Generate array of frequencies with LFO modulation
        lfo_mod = np.array([self.lfo.generate_sample() for _ in range(num_samples)])
        # Convert semitone modulation to frequency multiplier
        freq_mod = 2 ** (lfo_mod / 12)
        modulated_freq = frequency * freq_mod
        
        # Generate phase increments for varying frequency
        phase_increments = 2 * np.pi * modulated_freq / self.sample_rate
        # Calculate phase array
        phases = note_state.phase + np.cumsum(phase_increments)
        
        # Generate waveform
        if self.wave_type == 'sine':
            wave = np.sin(phases)
        elif self.wave_type == 'square':
            wave = np.sign(np.sin(phases))
        elif self.wave_type == 'sawtooth':
            wave = 2 * (phases / (2 * np.pi) - np.floor(0.5 + phases / (2 * np.pi)))
        elif self.wave_type == 'triangle':
            wave = 2 * np.abs(2 * (phases / (2 * np.pi) - np.floor(0.5 + phases / (2 * np.pi)))) - 1
        
        # Apply envelope
        envelope = note_state.get_envelope_value(num_samples)
        wave *= envelope * note_state.velocity * 0.3
        
        # Update phase for next buffer
        note_state.phase = phases[-1] % (2 * np.pi)
        note_state.total_samples += num_samples
        
        return wave
    
    def set_filter_params(self, base_cutoff=None, resonance=None, type=None, 
                         envelope_amount=None, adsr_mix=None):
        """Set filter parameters including envelope amount and ADSR mix"""
        if base_cutoff is not None:
            self.filter.set_base_cutoff(base_cutoff)
        if resonance is not None:
            self.filter.set_resonance(resonance)
        if type is not None:
            self.filter.set_type(type)
        if envelope_amount is not None:
            self.filter.set_envelope_amount(envelope_amount)
        if adsr_mix is not None:
            self.filter.set_adsr_mix(adsr_mix)
        
    def generate_waveform(self, frequency, phase, num_samples):
        t = np.linspace(0, num_samples/self.sample_rate, num_samples, endpoint=False)
        omega = 2 * np.pi * frequency * t + phase
        
        if self.wave_type == 'sine':
            return np.sin(omega) 
        elif self.wave_type == 'square':
            return np.sign(np.sin(omega))
        elif self.wave_type == 'sawtooth':
            return 2 * (omega / (2 * np.pi) - np.floor(0.5 + omega / (2 * np.pi)))
        elif self.wave_type == 'triangle':
            return 2 * np.abs(2 * (omega / (2 * np.pi) - np.floor(0.5 + omega / (2 * np.pi)))) - 1
        
        return np.sin(omega)
    
    def start_audio_stream(self):
        def callback(outdata, frames, time, status):
            if status:
                print(status)
            
            try:
                if self.active_notes:
                    combined = np.zeros(frames)
                    max_envelope = 0.0
                    notes_to_remove = []
                    
                    # First pass: generate audio and find maximum envelope value
                    for note, note_state in self.active_notes.items():
                        freq = midi_to_frequency(note)
                        wave = self.generate_continuous_tone(
                            freq,
                            frames,
                            note_state
                        )
                        
                        combined += wave
                        max_envelope = max(max_envelope, note_state.get_envelope_value(frames))
                        
                        if note_state.is_released and note_state.get_envelope_value(frames) <= 0:
                            notes_to_remove.append(note)
                    
                    for note in notes_to_remove:
                        del self.active_notes[note]
                    
                    # Apply envelope-controlled filter to combined signal
                    if len(self.active_notes) > 0:
                        combined = self.filter.process(combined, max_envelope)
                        combined = np.tanh(combined)  # Soft limiting
                    
                    outdata[:] = combined.reshape(-1, 1)
                else:
                    fade_out = np.linspace(1, 0, frames)
                    outdata[:] = (self.last_frame * fade_out).reshape(-1, 1)
                
                self.last_frame = outdata[:, 0]
                
            except Exception as e:
                print(f"Error in audio callback: {e}")
                outdata.fill(0)


        try:
            self.stream = sd.OutputStream(
                channels=1,
                callback=callback,
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                latency='low',
                device=sd.default.device['output']
            )
            self.stream.start()
            self.is_playing = True
        except Exception as e:
            print(f"Error starting audio stream: {e}")

    def stop_audio_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False
        self.active_notes.clear()

    def note_on(self, note, velocity):
        self.active_notes[note] = NoteState(velocity / 127.0, self.sample_rate, self.adsr)
        if not self.is_playing:
            self.start_audio_stream()

    def note_off(self, note):
        if note in self.active_notes:
            self.active_notes[note].is_released = True
            self.active_notes[note].released_time = time.time()

    def set_waveform(self, wave_type):
        if wave_type in ['sine', 'square', 'sawtooth', 'triangle']:
            self.wave_type = wave_type

    def set_adsr(self, attack, decay, sustain, release):
        self.adsr.set_adsr(attack, decay, sustain, release)

def midi_to_frequency(note_number):
    return 440.0 * (2.0 ** ((note_number - 69) / 12.0))

def listen_for_midi_input(midi_synth):
    try:
        input_names = mido.get_input_names()
        if not input_names:
            print("No MIDI devices found. Please connect a MIDI keyboard.")
            return

        print(f"Available MIDI input devices: {input_names}")
        print(f"Listening for MIDI input on {input_names[0]}...")
        
        with mido.open_input(input_names[0]) as midi_in:
            for msg in midi_in:
                if msg.type == 'note_on':
                    if msg.velocity > 0:
                        midi_synth.note_on(msg.note, msg.velocity)
                    else:
                        midi_synth.note_off(msg.note)
                elif msg.type == 'note_off':
                    midi_synth.note_off(msg.note)
    except Exception as e:
        print(f"Error in MIDI input handling: {e}")
        midi_synth.stop_audio_stream()

def main():    
    midi_synth = MIDISynthesizer(buffer_size=512)
    midi_synth.set_waveform('triangle')
    midi_synth.set_adsr(attack=0.5, decay=0.3, sustain=0.7, release=0.6)

    midi_synth.set_filter_params(
        resonance=0.3,
        base_cutoff=500,
        type="lowpass",
        envelope_amount=0,  # Increased for more dramatic effect
        adsr_mix=0.6         # 60% envelope modulation, 40% static filter
    )
    
    # Set up LFO
    midi_synth.set_lfo_params(
        rate=1.0,    # 5 Hz
        depth=0,   # 0.5 semitones
        wave_type='triangle'
    )
    
    try:
        listen_for_midi_input(midi_synth)
    except KeyboardInterrupt:
        print("\nStopping synthesizer...")
        midi_synth.stop_audio_stream()
    finally:
        print("Cleanup complete")

if __name__ == "__main__":
    main()