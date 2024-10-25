from flask import Flask, jsonify, request
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

# Global synthesizer instance
midi_synth = None

@app.route('/api/waveform', methods=['POST'])
def set_waveform():
    data = request.get_json()
    if 'type' in data:
        midi_synth.set_waveform(data['type'])
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Missing waveform type'})

@app.route('/api/adsr', methods=['POST'])
def set_adsr():
    data = request.get_json()
    midi_synth.set_adsr(
        float(data.get('attack', 0.1)),
        float(data.get('decay', 0.1)),
        float(data.get('sustain', 0.7)),
        float(data.get('release', 0.2))
    )
    return jsonify({'status': 'success'})

@app.route('/api/note', methods=['POST'])
def trigger_note():
    data = request.get_json()
    if data.get('state') == 'on':
        midi_synth.note_on(data['note'], int(data.get('velocity', 100)))
    else:
        midi_synth.note_off(data['note'])
    return jsonify({'status': 'success'})

def start_web_server(synth_instance):
    global midi_synth
    midi_synth = synth_instance
    app.run(port=5000)

# Modify main() in your original code to include:
def main():
    midi_synth = MIDISynthesizer(buffer_size=512)
    midi_synth.set_waveform('triangle')
    midi_synth.set_adsr(attack=0.5, decay=0.3, sustain=0.7, release=0.6)
    
    # Start web server in a separate thread
    web_thread = threading.Thread(target=start_web_server, args=(midi_synth,))
    web_thread.daemon = True
    web_thread.start()
    
    try:
        listen_for_midi_input(midi_synth)
    except KeyboardInterrupt:
        print("\nStopping synthesizer...")
        midi_synth.stop_audio_stream()
    finally:
        print("Cleanup complete")