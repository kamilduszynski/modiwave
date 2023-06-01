## Standard library imports
import sys
import math

## Third party imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

## Local imports

def separate_silence_and_noise(channel, threshold=0.0001):
    silence_parts = np.where(np.abs(channel)<threshold)[0]
    noise_parts = np.where(np.abs(channel)>threshold)[0]
    silence_sample_points = channel[silence_parts].shape[0]
    noise_sample_points = channel[noise_parts].shape[0]
    return noise_parts, noise_sample_points, silence_parts, silence_sample_points

def calculate_time_array(sample_points, sampling_freq):
    time_array = np.arange(0, sample_points, 1)
    time_array = time_array / sampling_freq
    time_array = time_array / 60
    return time_array

def plot_amplitude(time_arrays, sound_arrays):
    # Plot orginal audio, silence and noise parts amplitudes
    fig, axs = plt.subplots(3, 1)
    fig.suptitle('Audio amplitude', fontsize=16)
    fig.set_figheight(10)
    fig.set_figwidth(16)
    axs[0].plot(time_arrays[0], sound_arrays[0], color='r')
    axs[0].set_title('Original audio')
    axs[0].set_ylabel('Amplitude')
    axs[1].plot(time_arrays[1], sound_arrays[1], color='g')
    axs[1].set_title('silence parts')
    axs[1].set_ylabel('Amplitude')
    axs[2].plot(time_arrays[2], sound_arrays[2], color='b')
    axs[2].set_title('noise parts')
    axs[2].set_xlabel('Duration (min)')
    axs[2].set_ylabel('Amplitude')
    plt.show()

def main() -> int:
    myAudio = "audio.wav"

    # Read file and get sampling frequency and sound object
    sampling_freq, sound = wavfile.read(myAudio)

    # Check if wave file is 16bit or 32 bit. 24bit is not supported
    sound_data_type = sound.dtype

    # We can convert our sound array to floating point values ranging from -1 to 1 as follows
    sound = sound / (2.**15)
    max_amplitude = np.abs(sound).max()
    min_amplitude = np.abs(sound).min()

    # Perform checking only on one channel in case audio is mono
    sound_shape = sound.shape
    channel_0 = sound[:,0]

    if sound.shape[1] == 2:
        channel_1 = sound[:,0]

    threshold = 0.0001
    sample_points = sound.shape[0]
    noise_parts, noise_sample_points, silence_parts, silence_sample_points = separate_silence_and_noise(channel_0, threshold)

    time_array = calculate_time_array(sample_points, sampling_freq)
    silence_time_array = calculate_time_array(silence_sample_points, sampling_freq)
    noise_time_array = calculate_time_array(noise_sample_points, sampling_freq)

    original_audio_duration =  math.modf(np.round(sample_points / sampling_freq / 60, 2))
    silence_audio_duration =  math.modf(np.round(silence_sample_points / sampling_freq / 60, 2))
    noise_audio_duration =  math.modf(np.round(noise_sample_points / sampling_freq / 60, 2))
    
    time_arrays = [time_array, silence_time_array, noise_time_array]
    sound_arrays = [sound, sound[silence_parts, :], sound[noise_parts, :]]

    print("#############################################")
    print(f"|sound data type:           {sound_data_type}")
    print(f"|sound data shape:          {sound_shape}")
    print(f"|")
    print(f"|original samples:          {time_array.shape[0]}")
    print(f"|original audio duration:   {int(original_audio_duration[1])}min {int(60*original_audio_duration[0])}s")
    print(f"|")
    print(f"|silence samples:           {silence_time_array.shape[0]}")
    print(f"|silence audio duration:    {int(silence_audio_duration[1])}min {int(60*silence_audio_duration[0])}s")
    print(f"|")
    print(f"|noise samples:             {noise_time_array.shape[0]}")
    print(f"|noise audio duration:      {int(noise_audio_duration[1])}min {int(60*noise_audio_duration[0])}s")
    print(f"|")
    print(f"|sampling frequency:        {sampling_freq}")
    print(f"|min amplitude:             {min_amplitude}")
    print(f"|max amplitude:             {max_amplitude}")
    print("#############################################")

    wavfile.write("audio_silence.wav", sampling_freq, sound[silence_parts])
    wavfile.write("audio_no_silence.wav", sampling_freq, sound[noise_parts])

    if "-p" in sys.argv:
        plot_amplitude(time_arrays, sound_arrays)
    return 0

if __name__ == "__main__":
    sys.exit(main())
