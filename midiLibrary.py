import numpy as np
import sounddevice as sd
import mido
import time

class LFO:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.phase = 0.0
        self.rate = 2.0
        self.depth = 0.5
        self.filter_depth = 2000
        self.wave_type = 'sine'
        self.pitch_mix = 0.5
        self.filter_mix = 0.5
    
    def set_params(self, rate=None, depth=None, filter_depth=None, wave_type=None, pitch_mix=None, filter_mix=None):
        if rate is not None:
            self.rate = max(0.1, min(20.0, rate))
        if depth is not None:
            self.depth = max(0.0, min(12.0, depth))
        if filter_depth is not None:
            self.filter_depth = max(0.0, min(5000.0, filter_depth))
        if wave_type in ['sine', 'triangle', 'square']:
            self.wave_type = wave_type
        if pitch_mix is not None:
            self.pitch_mix = max(0.0, min(1.0, pitch_mix))
        if filter_mix is not None:
            self.filter_mix = max(0.0, min(1.0, filter_mix))
    
    def generate_value(self):
        if self.wave_type == 'sine':
            value = np.sin(self.phase)
        elif self.wave_type == 'triangle':
            value = 2 * abs(2 * (self.phase / (2 * np.pi) - np.floor(0.5 + self.phase / (2 * np.pi)))) - 1
        else:  # square
            value = np.sign(np.sin(self.phase))
        
        self.phase = (self.phase + 2 * np.pi * self.rate / self.sample_rate) % (2 * np.pi)
        
        return {
            'pitch': value * self.depth * self.pitch_mix,
            'filter': value * self.filter_depth * self.filter_mix
        }

class ADSR:
    def __init__(self, attack=0.1, decay=0.1, sustain=0.7, release=0.2):
        self.set_params(attack, decay, sustain, release)

    def set_params(self, attack, decay, sustain, release):
        self.attack = max(0.001, attack)
        self.decay = max(0.001, decay)
        self.sustain = max(0.0, min(1.0, sustain))
        self.release = max(0.001, release)

class Filter:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.cutoff = 1000.0
        self.resonance = 0.7
        self.type = 'lowpass'
        self.low = self.band = self.high = 0.0
        self.base_cutoff = 1000.0
        self.envelope_amount = 0.5
        self.min_cutoff = 20.0
        self.max_cutoff = 20000.0
        self.lfo_mod = 0.0
    
    def set_params(self, base_cutoff=None, resonance=None, type=None, envelope_amount=None):
        if base_cutoff is not None:
            self.base_cutoff = max(self.min_cutoff, min(self.max_cutoff, base_cutoff))
        if resonance is not None:
            self.resonance = max(0.0, min(0.99, resonance))
        if type in ['lowpass', 'highpass', 'bandpass']:
            self.type = type
        if envelope_amount is not None:
            self.envelope_amount = max(0.0, min(1.0, envelope_amount))
    
    def calculate_cutoff(self, envelope_value):
        exp_env = envelope_value ** 2
        mod_range = self.max_cutoff - self.base_cutoff
        env_modulation = mod_range * exp_env * self.envelope_amount
        return max(self.min_cutoff, min(self.max_cutoff, 
                                      self.base_cutoff + env_modulation + self.lfo_mod))
    
    def process(self, input_signal, envelope_value=1.0):
        self.cutoff = self.calculate_cutoff(envelope_value)
        f = 2.0 * np.sin(np.pi * self.cutoff / self.sample_rate)
        q = 1.0 - self.resonance
        
        output = np.zeros_like(input_signal)
        for i in range(len(input_signal)):
            self.high = input_signal[i] - self.low - q * self.band
            self.band += f * self.high
            self.low += f * self.band
            
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

    def get_envelope(self):
        current_time = self.total_samples / self.sample_rate
        
        if self.is_released:
            release_time = time.time() - self.released_time
            return 0 if release_time >= self.adsr.release else self.adsr.sustain * (1 - release_time / self.adsr.release)
        
        if current_time < self.adsr.attack:
            return current_time / self.adsr.attack
        
        if current_time < (self.adsr.attack + self.adsr.decay):
            decay_progress = (current_time - self.adsr.attack) / self.adsr.decay
            return 1.0 + (self.adsr.sustain - 1.0) * decay_progress
        
        return self.adsr.sustain

class MIDISynthesizer:
    WAVE_TYPES = {
        'sine': lambda x: np.sin(x),
        'square': lambda x: np.sign(np.sin(x)),
        'sawtooth': lambda x: 2 * (x / (2 * np.pi) - np.floor(0.5 + x / (2 * np.pi))),
        'triangle': lambda x: 2 * np.abs(2 * (x / (2 * np.pi) - np.floor(0.5 + x / (2 * np.pi)))) - 1
    }

    def __init__(self, sample_rate=44100, buffer_size=512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.active_notes = {}
        self.is_playing = False
        self.stream = None
        self.last_frame = np.zeros(buffer_size)
        self.wave_type = 'sine'
        
        self.adsr = ADSR()
        self.filter = Filter(sample_rate)
        self.lfo = LFO(sample_rate)

    def set_waveform(self, waveform):
        self.wave_type = waveform

    def generate_tone(self, frequency, num_samples, note_state):
        pitch_mod = np.zeros(num_samples)
        filter_mod = np.zeros(num_samples)
        
        for i in range(num_samples):
            lfo_values = self.lfo.generate_value()
            pitch_mod[i] = lfo_values['pitch']
            filter_mod[i] = lfo_values['filter']
        
        freq_mod = 2 ** (pitch_mod / 12)
        modulated_freq = frequency * freq_mod
        phase_increments = 2 * np.pi * modulated_freq / self.sample_rate
        phases = note_state.phase + np.cumsum(phase_increments)
        
        wave = self.WAVE_TYPES[self.wave_type](phases)
        envelope = note_state.get_envelope()
        wave *= envelope * note_state.velocity * 0.3
        
        self.filter.lfo_mod = filter_mod[-1]
        note_state.phase = phases[-1] % (2 * np.pi)
        note_state.total_samples += num_samples
        
        return wave

    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(status)
        
        try:
            if self.active_notes:
                combined = np.zeros(frames)
                max_envelope = 0.0
                notes_to_remove = []
                
                for note, state in self.active_notes.items():
                    freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
                    wave = self.generate_tone(freq, frames, state)
                    combined += wave
                    max_envelope = max(max_envelope, state.get_envelope())
                    
                    if state.is_released and state.get_envelope() <= 0:
                        notes_to_remove.append(note)
                
                for note in notes_to_remove:
                    del self.active_notes[note]
                
                if self.active_notes:
                    combined = self.filter.process(combined, max_envelope)
                    combined = np.tanh(combined)
                
                outdata[:] = combined.reshape(-1, 1)
            else:
                outdata[:] = (self.last_frame * np.linspace(1, 0, frames)).reshape(-1, 1)
            
            self.last_frame = outdata[:, 0]
            
        except Exception as e:
            print(f"Error in audio callback: {e}")
            outdata.fill(0)

    def start(self):
        try:
            self.stream = sd.OutputStream(
                channels=1,
                callback=self.audio_callback,
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                latency='low',
                device=sd.default.device['output']
            )
            self.stream.start()
            self.is_playing = True
        except Exception as e:
            print(f"Error starting audio stream: {e}")

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False
        self.active_notes.clear()

    def note_on(self, note, velocity):
        self.active_notes[note] = NoteState(velocity / 127.0, self.sample_rate, self.adsr)
        if not self.is_playing:
            self.start()

    def note_off(self, note):
        if note in self.active_notes:
            self.active_notes[note].is_released = True
            self.active_notes[note].released_time = time.time()



def main():
    synth = MIDISynthesizer()
    synth.wave_type = 'sawtooth'
    synth.adsr.set_params(0.5, 0.3, 0.7, 0.6)
    synth.filter.set_params(base_cutoff=200, resonance=0.3, type="lowpass", envelope_amount=0)
    synth.lfo.set_params(rate=0.5, depth=0.5, filter_depth=500, wave_type='sine',
                        pitch_mix=0.8, filter_mix=0.5)
    
    try:
        input_names = mido.get_input_names()
        if not input_names:
            print("No MIDI devices found.")
            return

        print(f"Available MIDI devices: {input_names}")
        print(f"Using device: {input_names[0]}")
        
        with mido.open_input(input_names[0]) as midi_in:
            for msg in midi_in:
                if msg.type == 'note_on':
                    if msg.velocity > 0:
                        synth.note_on(msg.note, msg.velocity)
                    else:
                        synth.note_off(msg.note)
                elif msg.type == 'note_off':
                    synth.note_off(msg.note)
    except KeyboardInterrupt:
        print("\nStopping synthesizer...")
    finally:
        synth.stop()
        print("Cleanup complete")

if __name__ == "__main__":
    main()