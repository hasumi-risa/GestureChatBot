import os
import sys
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import subprocess

import librosa
import librosa.display
from tqdm import tqdm
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import parselmouth as pm

def extract_mel(wav, sr=44100, n_mels=128, plot=True): #Output -> (timeframe, mel_dim)
    audio, _ = librosa.load(wav, sr=sr)
    ps = librosa.feature.melspectrogram(y=audio, sr=sr)
    ps_db= librosa.power_to_db(ps, ref=np.max)
    plt.rcParams["figure.figsize"] = (8, 4)
    librosa.display.specshow(ps_db, x_axis='s', y_axis='log')
    plt.show()
    return ps_db

def plotTransition(lists, labels, keyframes=None, sr=None, title=None, save_path=None):
    plt.rcParams["figure.figsize"] = (8, 4)
    for i in range(len(lists)):
        if sr is None:
            if keyframes:
                plt.plot(lists[i], label=labels[i], marker="o", markevery=keyframes[i])
            else:
                plt.plot(lists[i], label=labels[i])
        else:
            time = np.linspace(0, len(lists[i])/sr, len(lists[i]))
            if keyframes:
                plt.plot(time, lists[i], label=labels[i], marker="o", markevery=keyframes[i])
            else:
                plt.plot(time, lists[i], label=labels[i])

    plt.legend()
    plt.xlabel('Time')
    if title:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plotTextAudio(audio_wave, word_list, fps=25, save_path=None):
    start_time = [w[1] * fps for w in word_list]
    words = [w[0] for w in word_list]
    plt.plot(np.arange(len(audio_wave)), audio_wave)
    plt.xticks(start_time, words)
    plt.xticks(rotation=90)
    plt.title("Audio Waveform")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def saveAudioMovie(wav_path, save_path, fps=25):
    x, fs = librosa.load(wav_path, sr=44100)
    video_time = len(x)/fs
    frame_num = int(video_time * fps)
    time = np.linspace(0, video_time, len(x))
    interval = int(len(x)/frame_num)

    save_dir = "./.tmp/wavimgs/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.rcParams["figure.figsize"] = (8, 4)
    cnt = 0
    img_name = save_dir+'frame_{0:03d}.png'
    print("Generating images...")
    for i in tqdm(range(0, len(x), interval)):
        plt.plot(time[:i], x[:i], color='green')
        plt.plot(time[i:], x[i:], color='blue')
        plt.xlabel('Time [sec]')
        plt.savefig(img_name.format(cnt))
        plt.close()
        cnt += 1

    img = cv2.imread(img_name.format(0), cv2.IMREAD_COLOR)
    height, width, channels = img.shape[:3]
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    vout = cv2.VideoWriter('./.tmp/wavimgs/wav.mp4', fourcc, fps, (width, height))    
    for frm in range(cnt):
        img = cv2.imread(img_name.format(frm))
        vout.write(img)
    vout.release()
    
    cmd = "ffmpeg -i {mp4} -i {wav} -c:v copy -c:a aac -strict experimental -map 0:v -map 1:a {output} -y".format(
        mp4='./.tmp/wavimgs/wav.mp4', wav=wav_path, output=save_path)
    subprocess.call(cmd)
    print("Output to {}".format(save_path))

def getAudioEnvelopeGaussian(wav_path, kernel_size=3000, sr=44100):
    x, fs = librosa.load(wav_path, sr=sr)
    envelope = np.abs(signal.hilbert(x))
    audio_wave = gaussian_filter1d(envelope, 3000)
    return audio_wave

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

if __name__ == '__main__':
    wav_path = "./data/1L6l-FiV4xo_8/1L6l-FiV4xo_8.wav"
    # wav_path = "./data/1oNlTrLIjU4_8/1oNlTrLIjU4_8.wav"
    
    # saveAudioMovie(wav_path, save_path="./data/1oNlTrLIjU4_8/1oNlTrLIjU4_8_audio.mp4")

    # mel = extract_mel(wav=wav_path)

    sr = 44100
    audio_wave = getAudioEnvelopeGaussian(wav_path)
    plotTransition([audio_wave], labels=['audio'], sr=sr)
    print()
