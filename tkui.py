from midiLibrary import *
import tkinter as tk
from tkinter import ttk
import threading
import mido

class SynthesizerGUI:
    def __init__(self, midi_synth):
        self.midi_synth = midi_synth
        self.root = tk.Tk()
        self.root.title("MIDI Synthesizer Control")
        self.root.geometry("400x800")  # Increased height for filter controls
        
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # MIDI Device Selection
        ttk.Label(main_frame, text="MIDI Device", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, columnspan=4, pady=(0, 5)
        )
        
        # Device selection frame
        device_frame = ttk.Frame(main_frame)
        device_frame.grid(row=1, column=0, columnspan=4, pady=(0, 10))
        
        self.device_var = tk.StringVar()
        self.device_menu = ttk.Combobox(
            device_frame,
            textvariable=self.device_var,
            state='readonly',
            width=30
        )
        self.device_menu.grid(row=0, column=0, padx=5)
        
        ttk.Button(
            device_frame,
            text="Refresh",
            command=self.refresh_devices
        ).grid(row=0, column=1, padx=5)
        
        ttk.Button(
            device_frame,
            text="Connect",
            command=self.connect_device
        ).grid(row=0, column=2, padx=5)
        
        # Connection status
        self.connection_var = tk.StringVar(value="No MIDI device connected")
        ttk.Label(
            main_frame,
            textvariable=self.connection_var,
            font=('Arial', 10, 'italic')
        ).grid(row=2, column=0, columnspan=4, pady=(0, 10))
        
        # Waveform selection
        ttk.Label(main_frame, text="Waveform", font=('Arial', 12, 'bold')).grid(
            row=3, column=0, columnspan=4, pady=(10, 5)
        )
        
        waveforms = ['sine', 'square', 'sawtooth', 'triangle']
        self.wave_var = tk.StringVar(value='sine')
        
        for i, wave in enumerate(waveforms):
            ttk.Radiobutton(
                main_frame, 
                text=wave.capitalize(),
                variable=self.wave_var,
                value=wave,
                command=self.update_waveform
            ).grid(row=4, column=i, padx=5)
        
        # Filter Controls
        ttk.Label(main_frame, text="Filter", font=('Arial', 12, 'bold')).grid(
            row=5, column=0, columnspan=4, pady=(20, 10)
        )
        
        # Filter Type Selection
        filter_types = ['lowpass', 'highpass', 'bandpass']
        self.filter_type_var = tk.StringVar(value='lowpass')
        
        filter_type_frame = ttk.Frame(main_frame)
        filter_type_frame.grid(row=6, column=0, columnspan=4, pady=(0, 10))
        
        for i, f_type in enumerate(filter_types):
            ttk.Radiobutton(
                filter_type_frame,
                text=f_type.capitalize(),
                variable=self.filter_type_var,
                value=f_type,
                command=self.update_filter
            ).grid(row=0, column=i, padx=10)
        
        # Filter parameter sliders
        self.filter_vars = {
            'Cutoff': tk.DoubleVar(value=1000.0),
            'Resonance': tk.DoubleVar(value=0.7)
        }
        
        # Cutoff frequency (logarithmic scale)
        ttk.Label(main_frame, text="Cutoff Frequency").grid(row=7, column=0, columnspan=4)
        cutoff_slider = ttk.Scale(
            main_frame,
            from_=0,  # We'll convert this to Hz logarithmically
            to=100,
            variable=self.filter_vars['Cutoff'],
            orient=tk.HORIZONTAL,
            length=300,
            command=self.update_cutoff
        )
        cutoff_slider.grid(row=8, column=0, columnspan=4, pady=(0, 10))
        
        # Cutoff display (in Hz)
        self.cutoff_display = tk.StringVar(value="1000 Hz")
        ttk.Label(
            main_frame,
            textvariable=self.cutoff_display,
            font=('Arial', 10)
        ).grid(row=9, column=0, columnspan=4)
        
        # Resonance
        ttk.Label(main_frame, text="Resonance").grid(row=10, column=0, columnspan=4)
        resonance_slider = ttk.Scale(
            main_frame,
            from_=0.0,
            to=0.99,
            variable=self.filter_vars['Resonance'],
            orient=tk.HORIZONTAL,
            length=300,
            command=lambda _: self.update_filter()
        )
        resonance_slider.grid(row=11, column=0, columnspan=4, pady=(0, 10))
        
        # ADSR Controls
        ttk.Label(main_frame, text="ADSR Envelope", font=('Arial', 12, 'bold')).grid(
            row=12, column=0, columnspan=4, pady=(20, 10)
        )
        
        # ADSR sliders
        self.adsr_vars = {
            'Attack': tk.DoubleVar(value=0.1),
            'Decay': tk.DoubleVar(value=0.1),
            'Sustain': tk.DoubleVar(value=0.7),
            'Release': tk.DoubleVar(value=0.2)
        }
        
        for i, (param, var) in enumerate(self.adsr_vars.items()):
            ttk.Label(main_frame, text=param).grid(row=13+i*2, column=0, columnspan=4)
            slider = ttk.Scale(
                main_frame,
                from_=0.001 if param != 'Sustain' else 0.0,
                to=2.0 if param != 'Sustain' else 1.0,
                variable=var,
                orient=tk.HORIZONTAL,
                length=300,
                command=lambda _: self.update_adsr()
            )
            slider.grid(row=14+i*2, column=0, columnspan=4, pady=(0, 10))


        
        # Status display
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(
            main_frame, 
            textvariable=self.status_var,
            font=('Arial', 10, 'italic')
        )
        status_label.grid(row=21, column=0, columnspan=4, pady=(20, 0))
        
        # Initialize device list
        self.midi_input = None
        self.refresh_devices()
        
        # Start periodic device check
        self.root.after(1000, self.check_connection)
    
    def update_cutoff(self, *args):
        """Update cutoff frequency with logarithmic scaling"""
        # Convert linear slider value (0-100) to logarithmic frequency (20-20000 Hz)
        slider_val = self.filter_vars['Cutoff'].get()
        freq = 20.0 * (1000.0 ** (slider_val / 100.0))
        self.cutoff_display.set(f"{freq:.0f} Hz")
        self.midi_synth.set_filter_params(cutoff=freq)
        self.status_var.set("Filter cutoff updated")
    
    def update_filter(self, *args):
        """Update all filter parameters"""
        self.midi_synth.set_filter_params(
            filter_type=self.filter_type_var.get(),
            resonance=self.filter_vars['Resonance'].get()
        )
        self.status_var.set("Filter parameters updated")
    
    def refresh_devices(self):
        """Refresh the list of available MIDI devices"""
        devices = mido.get_input_names()
        self.device_menu['values'] = devices
        if devices:
            if not self.device_var.get() or self.device_var.get() not in devices:
                self.device_var.set(devices[0])
            self.status_var.set("MIDI devices found")
        else:
            self.device_var.set('')
            self.connection_var.set("No MIDI devices available")
            self.status_var.set("No MIDI devices found")
    
    def connect_device(self):
        """Attempt to connect to the selected MIDI device"""
        if self.midi_input:
            try:
                self.midi_input.close()
            except:
                pass
            self.midi_input = None
        
        device_name = self.device_var.get()
        if device_name:
            try:
                self.midi_input = mido.open_input(device_name)
                self.connection_var.set(f"Connected to: {device_name}")
                self.status_var.set("Successfully connected")
                
                # Start MIDI listening thread
                self.midi_thread = threading.Thread(
                    target=self.listen_for_midi,
                    daemon=True
                )
                self.midi_thread.start()
            except Exception as e:
                self.connection_var.set("Connection failed")
                self.status_var.set(f"Error: {str(e)}")
    
    def check_connection(self):
        """Periodically check MIDI connection status"""
        if self.midi_input is None:
            self.refresh_devices()
        self.root.after(1000, self.check_connection)
    
    def listen_for_midi(self):
        """Listen for MIDI input messages"""
        try:
            while True:
                if self.midi_input is None:
                    break
                for msg in self.midi_input.iter_pending():
                    if msg.type == 'note_on':
                        if msg.velocity > 0:
                            self.midi_synth.note_on(msg.note, msg.velocity)
                        else:
                            self.midi_synth.note_off(msg.note)
                    elif msg.type == 'note_off':
                        self.midi_synth.note_off(msg.note)
                time.sleep(0.001)  # Prevent CPU overload
        except Exception as e:
            self.status_var.set(f"MIDI Error: {str(e)}")
    
    def update_waveform(self):
        """Update the synthesizer's waveform type"""
        self.midi_synth.set_waveform(self.wave_var.get())
        self.status_var.set(f"Waveform changed to {self.wave_var.get()}")
    
    def update_adsr(self):
        """Update the synthesizer's ADSR envelope parameters"""
        self.midi_synth.set_adsr(
            attack=self.adsr_vars['Attack'].get(),
            decay=self.adsr_vars['Decay'].get(),
            sustain=self.adsr_vars['Sustain'].get(),
            release=self.adsr_vars['Release'].get()
        )
        self.status_var.set("ADSR parameters updated")
    
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()
        # Cleanup on exit
        if self.midi_input:
            self.midi_input.close()

def main():
    # Create synth instance
    midi_synth = MIDISynthesizer(buffer_size=512)
    
    # Create and run GUI
    gui = SynthesizerGUI(midi_synth)
    gui.run()

if __name__ == "__main__":
    main()