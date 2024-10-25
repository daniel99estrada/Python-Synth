// src/components/ui/slider.js
import * as Slider from '@radix-ui/react-slider';

export const SliderComponent = ({ value, onValueChange, min, max, step }) => (
  <Slider.Root
    className="relative flex items-center select-none touch-none w-full h-5"
    value={value}
    onValueChange={onValueChange}
    min={min}
    max={max}
    step={step}
    aria-label="Slider"
  >
    <Slider.Track className="bg-gray-300 relative h-2 rounded">
      <Slider.Range className="absolute bg-blue-500 h-full rounded" />
    </Slider.Track>
    <Slider.Thumb className="h-4 w-4 bg-white border border-gray-400 rounded" />
  </Slider.Root>
);
