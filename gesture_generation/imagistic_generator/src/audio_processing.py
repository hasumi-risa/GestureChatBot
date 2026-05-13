import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import parselmouth as pm

def compute_prosody(audio_filename, time_step=0.04, kernel_size=1):
    audio = pm.Sound(audio_filename)

    # Extract pitch and intensity
    pitch = audio.to_pitch(time_step=time_step)
    intensity = audio.to_intensity(time_step=time_step)

    # Evenly spaced time steps
    times = np.arange(0, audio.get_total_duration() - time_step, time_step)

    # Compute prosodic features at each time step
    pitch_values = np.nan_to_num(
        np.asarray([pitch.get_value_at_time(t) for t in times]))
    intensity_values = np.nan_to_num(
        np.asarray([intensity.get_value(t) for t in times]))

    intensity_values = np.clip(
        intensity_values, np.finfo(intensity_values.dtype).eps, None)

    # Normalize features [Chiu '11]
    pitch_norm = np.clip(np.log(pitch_values + 1) - 4, 0, None)
    intensity_norm = np.clip(np.log(intensity_values) - 3, 0, None)

    fileterd_pitch = gaussian_filter1d(pitch_norm, kernel_size)
    fileterd_intensity = gaussian_filter1d(intensity_norm, kernel_size)

    return fileterd_pitch, fileterd_intensity, times

def visualize(data, time, save_path=None):
    plt.figure()
    plt.plot(time, data)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

if __name__ == "__main__":
    audio_file = "D:/TED_videos/segmented_by_gesture/05jJodDVJRQ/05jJodDVJRQ_4_7/05jJodDVJRQ_4_7.wav"

    pitch, intensity, times = compute_prosody(audio_file)