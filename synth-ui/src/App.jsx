import { useState } from 'react';

const defaultSettings = {
  waveform: 'sawtooth',
  adsr: {
    attack: 0.5,
    decay: 0.3,
    sustain: 0.7,
    release: 0.6
  },
  filter: {
    cutoff: 0.7,
    resonance: 0.3,
    envelopeAmount: 0.1,
    type: 'lowpass'
  }
};

function App() {
  const [settings, setSettings] = useState(defaultSettings);

  const updateADSR = (parameter, value) => {
    setSettings(prev => ({
      ...prev,
      adsr: {
        ...prev.adsr,
        [parameter]: parseFloat(value)
      }
    }));
  };

  const updateFilter = (parameter, value) => {
    setSettings(prev => ({
      ...prev,
      filter: {
        ...prev.filter,
        [parameter]: parameter === 'type' ? value : parseFloat(value)
      }
    }));
  };

  const setWaveform = (waveform) => {
    setSettings(prev => ({
      ...prev,
      waveform
    }));
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white py-8">
      <div className="max-w-4xl mx-auto px-4">
        <h1 className="text-3xl font-bold text-center mb-8">MIDI Synthesizer</h1>
        
        <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
          {/* Waveform Selection */}
          <div className="mb-8">
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
          <div className="mb-8">
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
          <div className="mb-8">
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
                      {param}: {value.toFixed(2)}
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
}

export default App;