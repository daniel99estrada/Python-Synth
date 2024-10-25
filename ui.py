import streamlit as st
import numpy as np
from scipy import signal
import sounddevice as sd
import threading
from dataclasses import dataclass
from collections import defaultdict
import queue
import contextlib
import sys

# MIDI backend selection
MIDI_BACKEND = 'DEFAULT'
try:
    import mido
    MIDI_BACKEND = 'MIDO'
except ImportError:
    try:
        import pygame.midi
        MIDI_BACKEND = 'PYGAME'
        pygame.midi.init()
    except ImportError:
        st.error("No MIDI backend available. Please install either 'python-rtmidi' or 'pygame'")

from midiLibrary import *

@dataclass
class ADSR:
    attack: float
    decay: float
    sustain: float
    release: float

class MIDIHandler:
    def __init__(self):
        self.backend = MIDI_BACKEND
        
    def get_input_names(self):
        if self.backend == 'MIDO':
            return mido.get_input_names()
        elif self.backend == 'PYGAME':
            names = []
            for i in range(pygame.midi.get_count()):
                info = pygame.midi.get_device_info(i)
                if info[2]:  # is_input
                    names.append(f"MIDI Device {i}: {info[1].decode()}")
            return names
        return []

    def open_input(self, device_name):
        if self.backend == 'MIDO':
            return mido.open_input(device_name)
        elif self.backend == 'PYGAME':
            device_id = int(device_name.split(':')[0].split()[-1])
            return pygame.midi.Input(device_id)
        return None

class StreamlitMIDISynth:
    def __init__(self):
        self.midi_synth = None
        self.attack = 0.1
        self.decay = 0.1
        self.sustain = 0.7
        self.release = 0.2
        self.waveform = "sine"
        self.midi_queue = queue.Queue()
        self.running = True
        self.lock = threading.Lock()
        self.midi_handler = MIDIHandler()
        
        # Initialize the synth first
        self.setup_midi_synth()
        # Then set up the interface
        self.setup_streamlit_interface()

    def setup_streamlit_interface(self):
        st.title("MIDI Synthesizer Controls")

        # Initialize session state for MIDI status
        if "midi_status" not in st.session_state:
            st.session_state.midi_status = f"Initializing... (Using {MIDI_BACKEND} backend)"

        # Display MIDI status
        st.write(st.session_state.midi_status)

        # Create columns for controls
        col1, col2, col3 = st.columns(3)
        
        # ADSR Controls
        with col1:
            st.subheader("ADSR")
            new_attack = st.slider("Attack (s)", 0.0, 2.0, self.attack, step=0.1)
            new_decay = st.slider("Decay (s)", 0.0, 2.0, self.decay, step=0.1)
            new_sustain = st.slider("Sustain", 0.0, 1.0, self.sustain, step=0.1)
            new_release = st.slider("Release (s)", 0.0, 3.0, self.release, step=0.1)
            
            if st.button("Update ADSR"):
                with self.lock:
                    self.attack = new_attack
                    self.decay = new_decay
                    self.sustain = new_sustain
                    self.release = new_release
                    self.update_adsr()
        
        # Waveform Controls
        with col2:
            st.subheader("Waveform")
            new_waveform = st.radio(
                "Select Wave",
                ["sine", "triangle", "sawtooth", "square"],
                index=["sine", "triangle", "sawtooth", "square"].index(self.waveform)
            )
            
            if st.button("Update Waveform"):
                with self.lock:
                    self.waveform = new_waveform
                    self.update_waveform()
        
        # MIDI Status and Device Selection
        with col3:
            st.subheader("MIDI Status")
            self.status_placeholder = st.empty()
            self.midi_devices = st.empty()
            
            if st.button("Refresh MIDI Devices"):
                self.refresh_midi_devices()

    def setup_midi_synth(self):
        try:
            self.midi_synth = MIDISynthesizer(buffer_size=512)
            # Start MIDI listening in a separate thread
            self.midi_thread = threading.Thread(target=self.run_midi_listener)
            self.midi_thread.daemon = True
            self.midi_thread.start()

            # Start the MIDI processing thread
            self.process_thread = threading.Thread(target=self.process_midi_messages)
            self.process_thread.daemon = True
            self.process_thread.start()

        except Exception as e:
            st.error(f"Failed to initialize MIDI synthesizer: {str(e)}")
            self.midi_synth = None

    def update_adsr(self):
        if self.midi_synth is None:
            st.warning("MIDI synthesizer not initialized")
            return
            
        try:
            with self.lock:
                self.midi_synth.set_adsr(
                    attack=self.attack,
                    decay=self.decay,
                    sustain=self.sustain,
                    release=self.release
                )
            st.success("ADSR settings updated!")
        except Exception as e:
            st.error(f"Failed to update ADSR settings: {str(e)}")

    def update_waveform(self):
        if self.midi_synth is None:
            st.warning("MIDI synthesizer not initialized")
            return
            
        try:
            with self.lock:
                self.midi_synth.set_waveform(self.waveform.lower())
            st.success(f"Waveform changed to {self.waveform}")
        except Exception as e:
            st.error(f"Failed to update waveform: {str(e)}")

    def refresh_midi_devices(self):
        try:
            input_names = self.midi_handler.get_input_names()
            if input_names:
                self.midi_devices.write(f"Available MIDI devices: {', '.join(input_names)}")
            else:
                self.midi_devices.write("No MIDI devices found. Please connect a MIDI keyboard.")
        except Exception as e:
            st.error(f"Failed to refresh MIDI devices: {str(e)}")

    def process_midi_messages(self):
        """Process MIDI messages from the queue"""
        while self.running:
            try:
                msg = self.midi_queue.get(timeout=0.1)
                with self.lock:
                    if msg['type'] == 'note_on':
                        self.midi_synth.note_on(msg['note'], msg['velocity'])
                    elif msg['type'] == 'note_off':
                        self.midi_synth.note_off(msg['note'])
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing MIDI message: {e}")

    def run_midi_listener(self):
        """Listen for MIDI messages and add them to the queue"""
        try:
            input_names = self.midi_handler.get_input_names()
            if not input_names:
                st.session_state.midi_status = "⚠️ No MIDI devices found"
                return

            st.session_state.midi_status = "✅ MIDI device connected"
            midi_in = self.midi_handler.open_input(input_names[0])
            
            if self.midi_handler.backend == 'MIDO':
                self._run_mido_listener(midi_in)
            else:
                self._run_pygame_listener(midi_in)

        except Exception as e:
            st.session_state.midi_status = f"⚠️ MIDI Error: {str(e)}"

    def _run_mido_listener(self, midi_in):
        with midi_in:
            for msg in midi_in:
                if not self.running:
                    break
                
                if msg.type == 'note_on':
                    self.midi_queue.put({
                        'type': 'note_on',
                        'note': msg.note,
                        'velocity': msg.velocity
                    })
                elif msg.type in ['note_off', 'note_on'] and msg.velocity == 0:
                    self.midi_queue.put({
                        'type': 'note_off',
                        'note': msg.note
                    })

    def _run_pygame_listener(self, midi_in):
        while self.running:
            if midi_in.poll():
                events = midi_in.read(10)
                for event in events:
                    data = event[0]
                    status = data[0] & 0xF0
                    note = data[1]
                    velocity = data[2] if len(data) > 2 else 0
                    
                    if status == 0x90:  # Note On
                        self.midi_queue.put({
                            'type': 'note_on',
                            'note': note,
                            'velocity': velocity
                        })
                    elif status == 0x80 or (status == 0x90 and velocity == 0):  # Note Off
                        self.midi_queue.put({
                            'type': 'note_off',
                            'note': note
                        })

    def __del__(self):
        """Cleanup method to ensure threads are properly stopped"""
        self.running = False
        if hasattr(self, 'midi_thread'):
            self.midi_thread.join(timeout=1)
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=1)

def main():
    synth_app = StreamlitMIDISynth()

if __name__ == "__main__":
    main()