from p import *
import numpy as np
import sounddevice as sd
import mido
import math
from collections import defaultdict
import threading
import time

class MIDISynthesizer:
    def __init__(self, base_synth, buffer_size=1024):
        self.base_synth = base_synth
        self.sample_rate = base_synth.sample_rate
        self.buffer_size = buffer_size
        self.active_notes = defaultdict(lambda: 0)
        self.is_playing = False
        self.stream = None
        self.last_frame = np.zeros(buffer_size)
        self.phase = defaultdict(float)  # Track phase for each note
        
    def generate_continuous_tone(self, frequency, num_samples, velocity):
        """Generate a continuous tone with phase continuity"""
        phase = self.phase[frequency]
        t = np.linspace(0, num_samples/self.sample_rate, num_samples, endpoint=False)
        
        # Generate the wave with phase continuity
        wave = 0.3 * velocity * np.sin(2 * np.pi * frequency * t + phase)
        
        # Update the phase for next time
        self.phase[frequency] = (phase + 2 * np.pi * frequency * num_samples/self.sample_rate) % (2 * np.pi)
        
        return wave

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
                    
                    # Dynamic scaling based on number of active notes
                    if len(self.active_notes) > 1:
                        scaling_factor = 1.0 / np.sqrt(len(self.active_notes))
                        combined *= scaling_factor
                    
                    # Apply envelope-controlled filter
                    if len(self.active_notes) > 0:
                        combined = self.filter.process(combined, max_envelope)
                        
                        # Apply DC offset filter
                        combined = self.dc_filter(combined)
                        
                        # Apply soft clipping and master volume
                        combined = self.apply_soft_clipper(combined)
                        combined *= self.master_volume
                    
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
        """Stop the audio output stream"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False
        self.active_notes.clear()
        self.phase.clear()

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

    def apply_soft_clipper(self, signal, threshold=0.8):
        """Apply soft clipping to prevent harsh digital distortion"""
        return np.tanh(signal / threshold) * threshold
    
    def note_off(self, note):
        """Handle note-off MIDI message"""
        if note in self.active_notes:
            del self.active_notes[note]
            freq = midi_to_frequency(note)
            if freq in self.phase:
                del self.phase[freq]
            if not self.active_notes:
                self.stop_audio_stream()

def midi_to_frequency(note_number):
    """Convert MIDI note number to frequency in Hz"""
    return 440.0 * (2.0 ** ((note_number - 69) / 12.0))

def list_audio_devices():
    """Print available audio devices"""
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    print(f"\nDefault output device: {sd.default.device['output']}")

def listen_for_midi_input(midi_synth):
    """Listen for MIDI input and handle note on/off events"""
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
    # List available audio devices
    list_audio_devices()
    
    # Initialize the base synthesizer
    base_synth = Synthesizer()
    
    # Create MIDI-enabled synthesizer wrapper with smaller buffer
    midi_synth = MIDISynthesizer(base_synth, buffer_size=1024)
    
    try:
        listen_for_midi_input(midi_synth)
    except KeyboardInterrupt:
        print("\nStopping synthesizer...")
        midi_synth.stop_audio_stream()
    finally:
        print("Cleanup complete")

if __name__ == "__main__":
    main()