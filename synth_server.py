import asyncio
import websockets
import json
from midiLibrary import MIDISynthesizer  # Your existing synth code
import sounddevice as sd
import mido
import logging

class SynthServer:
    def __init__(self):
        self.synth = MIDISynthesizer(sample_rate=44100, buffer_size=512)
        self.setup_synth()
        self.clients = set()
        self.logger = self.setup_logger()
        
    def setup_logger(self):
        logger = logging.getLogger('SynthServer')
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        return logger
        
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
    
    async def register(self, websocket):
        self.clients.add(websocket)
        self.logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
    async def unregister(self, websocket):
        self.clients.remove(websocket)
        self.logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def broadcast_state(self):
        """Broadcast current synth state to all connected clients"""
        if not self.clients:
            return
            
        state = {
            'waveform': self.synth.current_waveform,
            'adsr': self.synth.get_adsr_params(),
            'filter': self.synth.get_filter_params()
        }
        
        message = json.dumps({
            'type': 'state_update',
            'data': state
        })
        
        await asyncio.gather(
            *[client.send(message) for client in self.clients]
        )

    async def handle_message(self, websocket):
        """Handle incoming WebSocket messages"""
        try:
            await self.register(websocket)
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    command = data.get('command')
                    params = data.get('params', {})
                    
                    self.logger.debug(f"Received command: {command} with params: {params}")
                    
                    if command == 'set_waveform':
                        self.synth.set_waveform(params['waveform'])
                        
                    elif command == 'set_adsr':
                        self.synth.set_adsr(**params)
                        
                    elif command == 'set_filter':
                        self.synth.set_filter_params(**params)
                        
                    # Send confirmation back to client
                    await websocket.send(json.dumps({
                        'status': 'success',
                        'command': command
                    }))
                    
                    # Broadcast updated state to all clients
                    await self.broadcast_state()
                    
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'status': 'error',
                        'message': 'Invalid JSON format'
                    }))
                except Exception as e:
                    self.logger.error(f"Error handling message: {str(e)}")
                    await websocket.send(json.dumps({
                        'status': 'error',
                        'message': str(e)
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Client connection closed")
        finally:
            await self.unregister(websocket)

    async def start_server(self):
        """Start the WebSocket server"""
        async with websockets.serve(
            self.handle_message,
            "localhost",
            8765,
            ping_interval=30,
            ping_timeout=10
        ) as server:
            self.logger.info("WebSocket server started on ws://localhost:8765")
            await asyncio.Future()  # run forever

def listen_for_midi_input(synth):
    """MIDI input handling function"""
    try:
        with mido.open_input() as midi_in:
            for msg in midi_in:
                if msg.type == 'note_on':
                    synth.note_on(msg.note, msg.velocity)
                elif msg.type == 'note_off':
                    synth.note_off(msg.note)
    except Exception as e:
        logging.error(f"MIDI input error: {str(e)}")

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
    asyncio.run(server.start_server())

if __name__ == "__main__":
    main()