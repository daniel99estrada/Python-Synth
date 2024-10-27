import asyncio
import websockets
import json
from midiLibrary import * # Your existing synth code
import sounddevice as sd
import mido

class SynthServer:
    def __init__(self):
        self.synth = MIDISynthesizer(sample_rate=44100, buffer_size=512)
        self.setup_synth()
        
    def setup_synth(self):
        """Initial synth setup"""
        self.synth.set_waveform('sawtooth')
        self.synth.set_adsr(attack=0.5, decay=0.3, sustain=0.7, release=0.6)
        self.synth.set_filter_params(
            resonance=0.3,
            base_cutoff=0.7,
            filter_type="lowpass",
            envelope_amount=0.1
        )
        
    async def handle_message(self, websocket):
        """Handle incoming WebSocket messages"""
        async for message in websocket:
            try:
                data = json.loads(message)
                command = data.get('command')
                params = data.get('params', {})
                
                if command == 'set_waveform':
                    self.synth.set_waveform(params['waveform'])
                    
                elif command == 'set_adsr':
                    self.synth.set_adsr(
                        attack=params['attack'],
                        decay=params['decay'],
                        sustain=params['sustain'],
                        release=params['release']
                    )
                    
                elif command == 'set_filter':
                    self.synth.set_filter_params(
                        base_cutoff=params.get('cutoff'),
                        resonance=params.get('resonance'),
                        filter_type=params.get('type'),
                        envelope_amount=params.get('envelopeAmount')
                    )
                    
                # Send confirmation back to client
                await websocket.send(json.dumps({
                    'status': 'success',
                    'command': command
                }))
                
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    'status': 'error',
                    'message': 'Invalid JSON format'
                }))
            except Exception as e:
                await websocket.send(json.dumps({
                    'status': 'error',
                    'message': str(e)
                }))

    async def start_server(self):
        """Start the WebSocket server"""
        server = await websockets.serve(
            self.handle_message,
            "localhost",
            8765  # WebSocket port
        )
        print("WebSocket server started on ws://localhost:8765")
        await server.wait_closed()

def main():
    # Initialize and start the server
    server = SynthServer()
    
    # Start MIDI input handling in a separate thread
    import threading
    midi_thread = threading.Thread(
        target=listen_for_midi_input,
        args=(server.synth,),
        daemon=True
    )
    midi_thread.start()
    
    # Run the WebSocket server
    asyncio.get_event_loop().run_until_complete(server.start_server())

if __name__ == "__main__":
    main()