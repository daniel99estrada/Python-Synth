import { Select } from "@radix-ui/react-select";

export const SelectComponent = ({ value, onValueChange, options }) => (
  <Select.Root value={value} onValueChange={onValueChange}>
    <Select.Trigger className="border rounded p-2">
      <Select.Value placeholder="Select waveform" />
    </Select.Trigger>
    <Select.Content>
      {options.map((option) => (
        <Select.Item key={option.value} value={option.value} className="p-2 hover:bg-gray-200">
          {option.label}
        </Select.Item>
      ))}
    </Select.Content>
  </Select.Root>
);
