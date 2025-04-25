#!/usr/bin/env python3
import numpy as np
from scipy.io import wavfile

# Action sound (key selection) - short beep
def generate_action_sound():
    sample_rate = 44100  # 44.1 kHz
    duration = 0.1  # 100 ms
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate a higher pitched beep
    frequency = 880  # A5
    beep = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Apply a quick fade out
    fade = np.ones_like(beep)
    fade_samples = int(sample_rate * 0.05)  # 50ms fade
    fade[-fade_samples:] = np.linspace(1, 0, fade_samples)
    beep = beep * fade
    
    # Convert to 16-bit PCM
    beep_int16 = (beep * 32767).astype(np.int16)
    
    # Write to file
    wavfile.write('static/sounds/action.wav', sample_rate, beep_int16)
    print("Generated action sound")

# Selection sound (row selection) - double beep with lower pitch
def generate_selection_sound():
    sample_rate = 44100  # 44.1 kHz
    duration = 0.3  # 300 ms
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate a lower pitched double beep
    frequency = 440  # A4
    silence_duration = 0.05  # 50ms silent gap
    silence_samples = int(sample_rate * silence_duration)
    
    # First part of the beep
    first_part_duration = 0.1  # 100ms
    first_part_samples = int(sample_rate * first_part_duration)
    
    # Second part of the beep
    second_part_duration = 0.1  # 100ms
    second_part_samples = int(sample_rate * second_part_duration)
    
    # Create signal
    beep = np.zeros_like(t)
    
    # First beep
    first_beep_indices = np.arange(0, first_part_samples)
    beep[first_beep_indices] = 0.5 * np.sin(2 * np.pi * frequency * t[first_beep_indices])
    
    # Second beep
    second_beep_start = first_part_samples + silence_samples
    second_beep_indices = np.arange(second_beep_start, second_beep_start + second_part_samples)
    second_beep_indices = second_beep_indices[second_beep_indices < len(t)]
    
    if len(second_beep_indices) > 0:
        t_for_second = np.linspace(0, second_part_duration, len(second_beep_indices), False)
        beep[second_beep_indices] = 0.5 * np.sin(2 * np.pi * (frequency * 1.5) * t_for_second)
    
    # Apply fades
    fade_samples = int(sample_rate * 0.02)  # 20ms fade
    
    # Fade in/out for first beep
    if first_part_samples > fade_samples:
        beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
        beep[first_part_samples-fade_samples:first_part_samples] *= np.linspace(1, 0, fade_samples)
    
    # Fade in/out for second beep
    if len(second_beep_indices) > fade_samples * 2:
        beep[second_beep_indices[:fade_samples]] *= np.linspace(0, 1, fade_samples)
        beep[second_beep_indices[-fade_samples:]] *= np.linspace(1, 0, fade_samples)
    
    # Convert to 16-bit PCM
    beep_int16 = (beep * 32767).astype(np.int16)
    
    # Write to file
    wavfile.write('static/sounds/selection.wav', sample_rate, beep_int16)
    print("Generated selection sound")

if __name__ == "__main__":
    generate_action_sound()
    generate_selection_sound()
    print("Sound generation complete")
