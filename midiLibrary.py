import numpy as np
import sounddevice as sd
import mido
import math
from collections import defaultdict
import threading
import time

    
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
        print("Press Ctrl+C to stop...")
        
        with mido.open_input(input_names[0]) as midi_in:
            for msg in midi_in:
                if msg.type == 'note_on':
                    if msg.velocity > 0:
                        midi_synth.note_on(msg.note, msg.velocity)
                    else:
                        midi_synth.note_off(msg.note)
                elif msg.type == 'note_off':
                    midi_synth.note_off(msg.note)
    except KeyboardInterrupt:
        print("\nStopping MIDI input...")
    except Exception as e:
        print(f"Error in MIDI input handling: {e}")
    finally:
        midi_synth.stop_audio_stream()
        print("MIDI input stopped")
        print(f"Error in MIDI input handling: {e}")
        midi_synth.stop_audio_stream()
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
        self.base_cutoff = 1000.0  # Base cutoff frequency
        self.envelope_amount = 0.5  # Amount of envelope modulation (0.0 to 1.0)
        self.min_cutoff = 20.0
        self.max_cutoff = 20000.0
    
    def set_envelope_amount(self, amount):
        """Set the amount of envelope modulation (0.0 to 1.0)"""
        self.envelope_amount = max(0.0, min(1.0, amount))
    
    def set_base_cutoff(self, frequency):
        """Set the base cutoff frequency"""
        self.base_cutoff = max(self.min_cutoff, min(self.max_cutoff, frequency))
    
    def calculate_cutoff(self, envelope_value):
        """Calculate the current cutoff frequency based on envelope value"""
        # Scale the envelope value exponentially for more musical results
        exp_env = envelope_value ** 2
        
        # Calculate modulation amount
        mod_range = self.max_cutoff - self.base_cutoff
        modulation = mod_range * exp_env * self.envelope_amount
        
        # Apply modulation to base cutoff
        current_cutoff = self.base_cutoff + modulation
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

class SubOscillator:
    def __init__(self):
        self.wave_type = 'square'  # Sub oscillators often use square waves
        self.mix = 0.5  # Mix level (0.0 to 1.0)
    
    def set_mix(self, mix):
        """Set sub oscillator mix level (0.0 to 1.0)"""
        self.mix = max(0.0, min(1.0, mix))
    
    def set_wave_type(self, wave_type):
        """Set sub oscillator waveform type"""
        if wave_type in ['sine', 'square', 'sawtooth', 'triangle']:
            self.wave_type = wave_type

    def generate_waveform(self, phase):
        """Generate waveform for given phase"""
        if self.wave_type == 'sine':
            return np.sin(phase)
        elif self.wave_type == 'square':
            return np.sign(np.sin(phase))
        elif self.wave_type == 'sawtooth':
            return 2 * (phase / (2 * np.pi) - np.floor(0.5 + phase / (2 * np.pi)))
        elif self.wave_type == 'triangle':
            return 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(0.5 + phase / (2 * np.pi)))) - 1
        return np.sin(phase)
    
class MIDISynthesizer:
    def __init__(self, sample_rate=44100, buffer_size=512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.active_notes = {}
        self.is_playing = False
        self.stream = None
        self.last_frame = np.zeros(buffer_size)
        self.wave_type = 'sine'
        
        # Add master volume control
        self.master_volume = 0.7  # Keep headroom for multiple notes
        
        # Voice limiting
        self.max_voices = 8  # Prevent too many simultaneous notes
        
        # Initialize components
        self.adsr = ADSR(attack=0.1, decay=0.1, sustain=0.7, release=0.2)
        self.filter = EnvelopeFilter(sample_rate)
        self.lfo = LFO(sample_rate)
        self.sub_osc = SubOscillator()
        
        # Add DC offset filter state
        self.dc_filter_x1 = 0
        self.dc_filter_y1 = 0

    def dc_filter(self, signal):
        """Remove DC offset using a high-pass filter"""
        R = 0.995  # Filter coefficient
        output = np.zeros_like(signal)
        x1, y1 = self.dc_filter_x1, self.dc_filter_y1
        
        for i in range(len(signal)):
            y1 = R * y1 + signal[i] - x1
            x1 = signal[i]
            output[i] = y1
            
        self.dc_filter_x1 = x1
        self.dc_filter_y1 = y1
        return output

    
    def set_sub_osc_params(self, mix=None, wave_type=None):
        """Set sub oscillator parameters"""
        if mix is not None:
            self.sub_osc.set_mix(mix)
        if wave_type is not None:
            self.sub_osc.set_wave_type(wave_type)
    
    def generate_waveform(self, phase, wave_type):
        """Generate waveform of specified type"""
        if wave_type == 'sine':
            return np.sin(phase)
        elif wave_type == 'square':
            return np.sign(np.sin(phase))
        elif wave_type == 'sawtooth':
            return 2 * (phase / (2 * np.pi) - np.floor(0.5 + phase / (2 * np.pi)))
        elif wave_type == 'triangle':
            return 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(0.5 + phase / (2 * np.pi)))) - 1
        return np.sin(phase)
    
    def generate_continuous_tone(self, frequency, num_samples, note_state):
        # Generate array of frequencies with LFO modulation
        lfo_mod = np.array([self.lfo.generate_sample() for _ in range(num_samples)])
        freq_mod = 2 ** (lfo_mod / 12)
        modulated_freq = frequency * freq_mod
        
        # Generate phase increments for main oscillator
        phase_increments = 2 * np.pi * modulated_freq / self.sample_rate
        main_phases = note_state.phase + np.cumsum(phase_increments)
        
        # Generate main oscillator waveform
        main_wave = self.generate_waveform(main_phases, self.wave_type)
        
        # Generate sub oscillator waveform (one octave below)
        sub_phases = main_phases / 2
        sub_wave = self.sub_osc.generate_waveform(sub_phases)
        
        # Mix the two waves with improved scaling
        main_mix = 1.0 - (self.sub_osc.mix * 0.5)
        sub_mix = self.sub_osc.mix * 0.3  # Reduced sub oscillator level
        
        wave = (main_wave * main_mix) + (sub_wave * sub_mix)
        
        # Apply envelope with improved scaling
        envelope = note_state.get_envelope_value(num_samples)
        # Scale velocity curve for better dynamic response
        velocity_curve = note_state.velocity ** 1.5  # Non-linear velocity response
        wave *= envelope * velocity_curve * 0.2  # Reduced overall level
        
        # Update phase for next buffer
        note_state.phase = main_phases[-1] % (2 * np.pi)
        note_state.total_samples += num_samples
        
        return wave
  
    def dc_filter(self, signal):
        """Remove DC offset using a high-pass filter"""
        R = 0.995  # Filter coefficient
        output = np.zeros_like(signal)
        x1, y1 = self.dc_filter_x1, self.dc_filter_y1
        
        for i in range(len(signal)):
            y1 = R * y1 + signal[i] - x1
            x1 = signal[i]
            output[i] = y1
            
        self.dc_filter_x1 = x1
        self.dc_filter_y1 = y1
        return output
     
    def set_lfo_params(self, rate=None, depth=None, wave_type=None):
        """Set LFO parameters"""
        if rate is not None:
            self.lfo.set_rate(rate)
        if depth is not None:
            self.lfo.set_depth(depth)
        if wave_type is not None:
            self.lfo.set_wave_type(wave_type)
    

    def set_filter_params(self, base_cutoff=None, resonance=None, type=None, envelope_amount=None):
        """Set filter parameters including envelope amount"""
        if base_cutoff is not None:
            self.filter.set_base_cutoff(base_cutoff)
        if resonance is not None:
            self.filter.set_resonance(resonance)
        if type is not None:
            self.filter.set_type(type)
        if envelope_amount is not None:
            self.filter.set_envelope_amount(envelope_amount)

    def stop_audio_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False
        self.active_notes.clear()

    def note_on(self, note, velocity):
        # Implement voice stealing if we exceed max_voices
        if len(self.active_notes) >= self.max_voices:
            # Find the oldest note and remove it
            oldest_note = min(self.active_notes.items(), 
                            key=lambda x: x[1].start_time)[0]
            del self.active_notes[oldest_note]
            
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

    def apply_soft_clipper(self, signal, threshold=0.8):
        """Apply soft clipping to prevent harsh digital distortion"""
        return np.tanh(signal / threshold) * threshold
            
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

def main():    
    midi_synth = MIDISynthesizer(buffer_size=512)
    midi_synth.set_waveform('triangle')
    midi_synth.set_adsr(attack=0.6, decay=0.3, sustain=0.7, release=0.7)
    midi_synth.set_filter_params(resonance=0.3, base_cutoff=1000, type="lowpass")
    
    # Set up LFO
    midi_synth.set_lfo_params(
        rate=2.0,    # 5 Hz
        depth=0.2,   # 0.5 semitones
        wave_type='sine'
    )
    
    # Set up sub oscillator
    midi_synth.set_sub_osc_params(
        mix=0,     # Equal mix between main and sub
        wave_type='square'  # Classic square wave sub
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
