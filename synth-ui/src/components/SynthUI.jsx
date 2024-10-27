import React, { useState } from 'react';
import { Slider } from "./ui/slider";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { WaveSignalIcon, SlidersIcon, Volume2Icon } from 'lucide-react';

const SynthUI = () => {
  const [synthParams, setSynthParams] = useState({
    waveform: 'sawtooth',
    attack: 0.5,
    decay: 0.3,
    sustain: 0.7,
    release: 0.6,
    filterCutoff: 0.7,
    filterResonance: 0.3,
    filterEnvAmount: 0.1
  });

  const handleParamChange = (param, value) => {
    setSynthParams(prev => ({
      ...prev,
      [param]: value
    }));
    // In a real implementation, this would call methods on the synth instance
  };

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <WaveSignalIcon className="w-6 h-6" />
            MIDI Synthesizer Controls
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Waveform Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Waveform</label>
            <Select 
              value={synthParams.waveform}
              onValueChange={(value) => handleParamChange('waveform', value)}
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="sine">Sine</SelectItem>
                <SelectItem value="square">Square</SelectItem>
                <SelectItem value="sawtooth">Sawtooth</SelectItem>
                <SelectItem value="triangle">Triangle</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* ADSR Controls */}
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Volume2Icon className="w-5 h-5" />
                ADSR Envelope
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Attack: {synthParams.attack.toFixed(2)}s</label>
                <Slider
                  value={[synthParams.attack]}
                  onValueChange={([value]) => handleParamChange('attack', value)}
                  min={0}
                  max={2}
                  step={0.01}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Decay: {synthParams.decay.toFixed(2)}s</label>
                <Slider
                  value={[synthParams.decay]}
                  onValueChange={([value]) => handleParamChange('decay', value)}
                  min={0}
                  max={2}
                  step={0.01}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Sustain: {synthParams.sustain.toFixed(2)}</label>
                <Slider
                  value={[synthParams.sustain]}
                  onValueChange={([value]) => handleParamChange('sustain', value)}
                  min={0}
                  max={1}
                  step={0.01}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Release: {synthParams.release.toFixed(2)}s</label>
                <Slider
                  value={[synthParams.release]}
                  onValueChange={([value]) => handleParamChange('release', value)}
                  min={0}
                  max={2}
                  step={0.01}
                />
              </div>
            </CardContent>
          </Card>

          {/* Filter Controls */}
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <SlidersIcon className="w-5 h-5" />
                Filter Controls
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Cutoff: {synthParams.filterCutoff.toFixed(2)}</label>
                <Slider
                  value={[synthParams.filterCutoff]}
                  onValueChange={([value]) => handleParamChange('filterCutoff', value)}
                  min={0}
                  max={1}
                  step={0.01}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Resonance: {synthParams.filterResonance.toFixed(2)}</label>
                <Slider
                  value={[synthParams.filterResonance]}
                  onValueChange={([value]) => handleParamChange('filterResonance', value)}
                  min={0}
                  max={1}
                  step={0.01}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Envelope Amount: {synthParams.filterEnvAmount.toFixed(2)}</label>
                <Slider
                  value={[synthParams.filterEnvAmount]}
                  onValueChange={([value]) => handleParamChange('filterEnvAmount', value)}
                  min={0}
                  max={1}
                  step={0.01}
                />
              </div>
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </div>
  );
};

export default SynthUI;