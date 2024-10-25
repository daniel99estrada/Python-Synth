import mido
import math

# Define a dictionary that maps MIDI note numbers to note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_note_name(note_number):
    """Convert a MIDI note number to a musical note name."""
    octave = note_number // 12 - 1  # Octave calculation
    note = NOTE_NAMES[note_number % 12]  # Get the note name
    return f"{note}{octave}"

def midi_to_frequency(note_number):
    """Convert a MIDI note number to its corresponding frequency in Hz."""
    return 440.0 * math.pow(2.0, (note_number - 69) / 12.0)

def print_note_info(note_number):
    """Print the note name and frequency for the given MIDI note number."""
    note_name = midi_to_note_name(note_number)
    frequency = midi_to_frequency(note_number)
    print(f"Note: {note_name}, Frequency: {frequency:.2f} Hz")

def listen_for_midi_input():
    """Listen for MIDI input from a musical keyboard."""
    input_names = mido.get_input_names()
    if not input_names:
        print("No MIDI devices found. Please connect a MIDI keyboard.")
        return

    print(f"Available MIDI input devices: {input_names}")
    with mido.open_input(input_names[0]) as midi_in:
        print(f"Listening for MIDI input on {input_names[0]}...")
        for msg in midi_in:
            if msg.type == 'note_on' and msg.velocity > 0:  # Only process note_on events with velocity
                print_note_info(msg.note)

if __name__ == "__main__":
    listen_for_midi_input()
