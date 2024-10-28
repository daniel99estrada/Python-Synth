import { useState, useEffect, useCallback } from 'react';

const defaultSettings = {
  waveform: 'sawtooth',
  adsr: {
    attack: 0.5,
    decay: 0.3,
    sustain: 0.7,
    release: 0.6
  },
  filter: {
    base_cutoff: 0.7,
    resonance: 0.3,
    envelope_amount: 0.1,
    type: 'lowpass'
  },
  lfo: {
    rate: 0.7,
    pitch_depth: 0.3,
    filter_depth: 0.1,
    wave_type: 'sine',
  }
};

// WebSocket hook remains the same
const useWebSocket = () => {
  const [ws, setWs] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const socket = new WebSocket('ws://localhost:8765');

    socket.onopen = () => {
      console.log('Connected to WebSocket server');
      setIsConnected(true);
      setError(null);
    };

    socket.onclose = () => {
      console.log('Disconnected from WebSocket server');
      setIsConnected(false);
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Failed to connect to synth server');
    };

    setWs(socket);

    return () => {
      socket.close();
    };
  }, []);

  const sendMessage = useCallback((command, params) => {
    if (ws && isConnected) {
      ws.send(JSON.stringify({ command, params }));
    }
  }, [ws, isConnected]);

  return { sendMessage, isConnected, error };
};

const App = () => {
  const [settings, setSettings] = useState(defaultSettings);
  const { sendMessage, isConnected, error } = useWebSocket();

  const updateADSR = (parameter, value) => {
    const newValue = parseFloat(value);
    setSettings(prev => ({
      ...prev,
      adsr: {
        ...prev.adsr,
        [parameter]: newValue
      }
    }));
    
    sendMessage('set_adsr', {
      ...settings.adsr,
      [parameter]: newValue
    });
  };

  const updateFilter = (parameter, value) => {
    const newValue = parameter === 'type' ? value : parseFloat(value);
    setSettings(prev => ({
      ...prev,
      filter: {
        ...prev.filter,
        [parameter]: newValue
      }
    }));
    
    sendMessage('set_filter', {
      ...settings.filter,
      [parameter]: newValue
    });
  };

  const updateLFO = (parameter, value) => {
    const newValue = parameter === 'wave_type' ? value : parseFloat(value);
    setSettings(prev => ({
      ...prev,
      lfo: {
        ...prev.lfo,
        [parameter]: newValue
      }
    }));
    
    if (parameter === 'pitch_mix' || parameter === 'filter_mix') {
      sendMessage('set_lfo_params', {
        ...settings.lfo,
        [parameter]: newValue
      });
    }
  };

  const setWaveform = (waveform) => {
    setSettings(prev => ({
      ...prev,
      waveform
    }));
    
    sendMessage('set_waveform', { waveform });
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white py-8">
      <div className="max-w-4xl mx-auto px-4">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold">MIDI Synthesizer</h1>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
        
        {error && (
          <div className="bg-red-500 text-white p-4 rounded-lg mb-6">
            {error}
          </div>
        )}
        
        <div className="bg-gray-800 rounded-lg p-6 shadow-lg space-y-8">
          {/* Waveform Selection */}
          <div>
            <h2 className="text-xl font-semibold mb-4">Waveform</h2>
            <div className="grid grid-cols-4 gap-3">
              {['sine', 'square', 'sawtooth', 'triangle'].map((wave) => (
                <button
                  key={wave}
                  onClick={() => setWaveform(wave)}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-colors
                    ${settings.waveform === wave 
                      ? 'bg-blue-500 text-white' 
                      : 'bg-gray-700 hover:bg-gray-600'}`}
                >
                  {wave}
                </button>
              ))}
            </div>
          </div>

          {/* ADSR Controls */}
          <div>
            <h2 className="text-xl font-semibold mb-4">ADSR Envelope</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {Object.entries(settings.adsr).map(([param, value]) => (
                <div key={param}>
                  <label className="block text-sm font-medium mb-2 capitalize">
                    {param}: {value.toFixed(2)}
                  </label>
                  <input 
                    type="range" 
                    min="0" 
                    max="1" 
                    step="0.01" 
                    value={value}
                    onChange={(e) => updateADSR(param, e.target.value)}
                    className="w-full"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Filter Controls */}
          <div>
            <h2 className="text-xl font-semibold mb-4">Filter</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {Object.entries(settings.filter).map(([param, value]) => (
                param === 'type' ? (
                  <div key={param}>
                    <label className="block text-sm font-medium mb-2 capitalize">
                      {param}
                    </label>
                    <select
                      value={value}
                      onChange={(e) => updateFilter(param, e.target.value)}
                      className="w-full bg-gray-700 rounded-md px-3 py-2"
                    >
                      <option value="lowpass">Lowpass</option>
                      <option value="highpass">Highpass</option>
                      <option value="bandpass">Bandpass</option>
                    </select>
                  </div>
                ) : (
                  <div key={param}>
                    <label className="block text-sm font-medium mb-2 capitalize">
                      {param.replace('_', ' ')}: {value.toFixed(2)}
                    </label>
                    <input 
                      type="range" 
                      min="0" 
                      max="1" 
                      step="0.01" 
                      value={value}
                      onChange={(e) => updateFilter(param, e.target.value)}
                      className="w-full"
                    />
                  </div>
                )
              ))}
            </div>
          </div>

          {/* LFO Controls */}
          <div>
            <h2 className="text-xl font-semibold mb-4">LFO</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* LFO Wave Type Selection */}
              <div>
                <label className="block text-sm font-medium mb-2">Wave Type</label>
                <select
                  value={settings.lfo.wave_type}
                  onChange={(e) => updateLFO('wave_type', e.target.value)}
                  className="w-full bg-gray-700 rounded-md px-3 py-2"
                >
                  <option value="sine">Sine</option>
                  <option value="triangle">Triangle</option>
                  <option value="square">Square</option>
                </select>
              </div>

              {/* LFO Parameters */}
              {['rate', 'pitch_depth', 'filter_depth'].map((param) => (
                <div key={param}>
                  <label className="block text-sm font-medium mb-2 capitalize">
                    {param.replace('_', ' ')}: {settings.lfo[param].toFixed(2)}
                  </label>
                  <input 
                    type="range" 
                    min="0" 
                    max="1" 
                    step="0.01" 
                    value={settings.lfo[param]}
                    onChange={(e) => updateLFO(param, e.target.value)}
                    className="w-full"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Settings Display */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h2 className="text-xl font-semibold mb-2">Current Settings</h2>
            <pre className="text-sm font-mono overflow-x-auto">
              {JSON.stringify(settings, null, 2)}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;