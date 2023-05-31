## Standard library imports
import sys
import math

## Third party imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

## Local imports

def calculate_time_array(sample_points, sampling_freq):
    time_array = np.arange(0, sample_points, 1)
    time_array = time_array / sampling_freq
    time_array = time_array / 60
    return time_array

def plot_amplitude(time_arrays, channels_1, channels_2):
    
    #Plot the tone
    for ind, timeArray in enumerate(time_arrays):
        fig, axs = plt.subplots(2, 1)
        fig.suptitle('Audio amplitude', fontsize=16)
        fig.set_figheight(7)
        fig.set_figwidth(12)
        axs[0].plot(timeArray, channels_1[ind], color='r')
        axs[0].set_title('channel 1')
        axs[0].set_xlabel('Time (min)')
        axs[0].set_ylabel('Amplitude')
        axs[1].plot(timeArray, channels_2[ind], color='g')
        axs[1].set_title('channel 2')
        axs[1].set_xlabel('Time (min)')
        axs[1].set_ylabel('Amplitude')
    plt.show()

def main() -> int:
    myAudio = "audio.wav"

    # Read file and get sampling freq [ usually 44100 Hz ]  and sound object
    sampling_freq, sound = wavfile.read(myAudio)

    # Check if wave file is 16bit or 32 bit. 24bit is not supported
    sound_data_type = sound.dtype

    # We can convert our sound array to floating point values ranging from -1 to 1 as follows
    sound = sound / (2.**15)

    #Check sample points and sound channel for duel channel(5060, 2) or  (5060, ) for mono channel
    sound_shape = sound.shape
    channel_1 = sound[:,0]
    channel_2 = sound[:,1]

    threshold = 0.0001
    max = np.abs(sound).max()
    min = np.abs(sound).min()
    silent_parts = np.where(np.abs(channel_1)<threshold)[0]
    noisy_parts = np.where(np.abs(channel_1)>threshold)[0]

    sample_points = sound.shape[0]
    silent_sample_points = sound[silent_parts].shape[0]
    noisy_sample_points = sound[noisy_parts].shape[0]

    time_array = calculate_time_array(sample_points, sampling_freq)
    silent_time_array = calculate_time_array(silent_sample_points, sampling_freq)
    noisy_time_array = calculate_time_array(noisy_sample_points, sampling_freq)

    audio_duration =  math.modf(np.round(sample_points / sampling_freq / 60, 2))
    silent_audio_duration =  math.modf(np.round(silent_sample_points / sampling_freq / 60, 2))
    noisy_audio_duration =  math.modf(np.round(noisy_sample_points / sampling_freq / 60, 2))
    
    print("#############################################")
    print(f"|sound data type:           {sound_data_type}")
    print(f"|sound data shape:          {sound_shape}")
    print(f"|")
    print(f"|original samples:          {len(time_array)}")
    print(f"|original audio duration:   {int(audio_duration[1])}min {int(60*audio_duration[0])}s")
    print(f"|")
    print(f"|silent samples:            {len(silent_time_array)}")
    print(f"|silent audio duration:     {int(silent_audio_duration[1])}min {int(60*silent_audio_duration[0])}s")
    print(f"|")
    print(f"|noisy samples:             {len(noisy_time_array)}")
    print(f"|noisy audio duration:      {int(noisy_audio_duration[1])}min {int(60*noisy_audio_duration[0])}s")
    print(f"|")
    print(f"|sampling frequency:        {sampling_freq}")
    print(f"|min amplitude:             {min}")
    print(f"|max amplitude:             {max}")
    print("#############################################")

    time_arrays = [time_array, silent_time_array, noisy_time_array]
    channels_1 = [channel_1, channel_1[silent_parts], channel_1[noisy_parts]]
    channels_2 = [channel_2, channel_2[silent_parts], channel_2[noisy_parts]]

    # plot_amplitude(time_arrays, channels_1, channels_2)

    wavfile.write("out.wav", sampling_freq, sound[noisy_parts])
    return 0

if __name__ == "__main__":
    sys.exit(main())
