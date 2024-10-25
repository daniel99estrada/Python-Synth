// src/SynthController.js
import React, { useState, useEffect, useCallback } from 'react';
import { SliderComponent } from "./ui/slider"; // No need for 'components' in the path
import Button from "./ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";
import { SelectComponent } from "./ui/select";
import { Activity } from 'lucide-react';

const SynthController = () => {
  const [adsr, setADSR] = useState({
    attack: 0.5,
    decay: 0.3,
    sustain: 0.7,
    release: 0.6,
  });

  const [waveform, setWaveform] = useState('triangle');
  const [activeNotes, setActiveNotes] = useState(new Set());

  const updateADSR = useCallback(async (param, value) => {
    const newADSR = { ...adsr, [param]: value };
    setADSR(newADSR);

    try {
      await fetch('http://localhost:5000/api/adsr', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newADSR),
      });
    } catch (error) {
      console.error('Failed to update ADSR:', error);
    }
  }, [adsr]);

  const updateWaveform = useCallback(async (value) => {
    setWaveform(value);
    try {
      await fetch('http://localhost:5000/api/waveform', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: value }),
      });
    } catch (error) {
      console.error('Failed to update waveform:', error);
    }
  }, []);

  const handleNoteOn = useCallback(async (note) => {
    if (!activeNotes.has(note)) {
      const newActiveNotes = new Set(activeNotes);
      newActiveNotes.add(note);
      setActiveNotes(newActiveNotes);

      try {
        await fetch('http://localhost:5000/api/note', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ note, state: 'on', velocity: 100 }),
        });
      } catch (error) {
        console.error('Failed to trigger note:', error);
      }
    }
  }, [activeNotes]);

  const handleNoteOff = useCallback(async (note) => {
    const newActiveNotes = new Set(activeNotes);
    newActiveNotes.delete(note);
    setActiveNotes(newActiveNotes);

    try {
      await fetch('http://localhost:5000/api/note', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ note, state: 'off' }),
      });
    } catch (error) {
      console.error('Failed to release note:', error);
    }
  }, [activeNotes]);

  // Keyboard handler
  useEffect(() => {
    const keyMap = {
      'a': 60, // Middle C
      'w': 61,
      's': 62,
      'e': 63,
      'd': 64,
      'f': 65,
      't': 66,
      'g': 67,
      'y': 68,
      'h': 69,
      'u': 70,
      'j': 71,
      'k': 72  // One octave up
    };

    const handleKeyDown = (e) => {
      const note = keyMap[e.key.toLowerCase()];
      if (note) handleNoteOn(note);
    };

    const handleKeyUp = (e) => {
      const note = keyMap[e.key.toLowerCase()];
      if (note) handleNoteOff(note);
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [handleNoteOn, handleNoteOff]);

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-6 h-6" />
            Synthesizer Control Panel
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Waveform Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Waveform</label>
            <SelectComponent 
              value={waveform} 
              onValueChange={updateWaveform} 
              options={[
                { value: 'sine', label: 'Sine' },
                { value: 'square', label: 'Square' },
                { value: 'sawtooth', label: 'Sawtooth' },
                { value: 'triangle', label: 'Triangle' },
              ]}
            />
          </div>

          {/* ADSR Controls */}
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Attack: {adsr.attack.toFixed(2)}s</label>
              <SliderComponent
                value={[adsr.attack]}
                onValueChange={([value]) => updateADSR('attack', value)}
                min={0.001}
                max={2}
                step={0.01}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Decay: {adsr.decay.toFixed(2)}s</label>
              <SliderComponent
                value={[adsr.decay]}
                onValueChange={([value]) => updateADSR('decay', value)}
                min={0.001}
                max={2}
                step={0.01}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Sustain: {adsr.sustain.toFixed(2)}</label>
              <SliderComponent
                value={[adsr.sustain]}
                onValueChange={([value]) => updateADSR('sustain', value)}
                min={0}
                max={1}
                step={0.01}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Release: {adsr.release.toFixed(2)}s</label>
              <SliderComponent
                value={[adsr.release]}
                onValueChange={([value]) => updateADSR('release', value)}
                min={0.001}
                max={2}
                step={0.01}
              />
            </div>
          </div>

          {/* Play Button */}
          <Button onClick={() => handleNoteOn(60)}>Play Middle C</Button>
        </CardContent>
      </Card>
    </div>
  );
};

export default SynthController;
