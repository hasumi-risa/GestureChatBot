import os
import subprocess
import numpy as np
from scipy import signal
from scipy.signal.filter_design import maxflat
from scipy.ndimage import gaussian_filter1d

from motion_processing import load_kinect_csv, calcAngular
from audio_processing import getAudioEnvelopeGaussian, plotTransition
from src.cmu2kinect import CMUPose2KinectData
from src.plotPose import Plot, display_pose

def plotUpperBody2D(pose3d, save_path, fps=25):
    upper_idx = [2, 20, 4, 5, 6, 8, 9, 10]
    
    pose3d = pose3d[:, upper_idx]
    pose2d = np.delete(pose3d, 2, 2).reshape([-1, len(upper_idx)*2])

    # rotate y axis
    # for frame in range(len(pose2d)):
    #     for joint in range(len(pose2d[frame])):
    #         pose2d[frame][joint][1] *= -1

    poses = []
    for i in range(len(pose2d)):
        pose = []
        for j in range(len(pose2d[i])):
            pose.append(pose2d[i][j])
        poses.append(pose)

    time = np.arange(len(pose2d))
    time = time.reshape(-1, 1)
    poses = np.concatenate([np.array(poses), time], axis=1)

    p = Plot((-0.75, 0.75), (-0.5, 1))
    anim = p.animate(poses, 1000/fps)
    p.save(anim, save_path, fps=fps)

def motionAdjustment(motion, adjust_frame):
    if len(motion) == adjust_frame:
        adjusted_motion = motion
    
    # Interpolation
    elif len(motion) < adjust_frame:
        adjusted_motion = np.zeros([adjust_frame, 25, 3])
        adjusted_motion[0] = motion[0]
        adjusted_motion[len(adjusted_motion)-1] = motion[len(motion)-1]
        interval = (adjust_frame - 2) / (len(motion) - 2)
        interval_sum, count = 0, 0
        e_index, tmp = [], []
        isEmpty = False
        for i in range(1, len(adjusted_motion)-1):
            if i > interval_sum:
                adjusted_motion[i] = motion[count]
                interval_sum += interval
                count += 1
            else:
                e_index.append(i)

        empty_index, tmp = [], []
        for i in range(len(e_index)):
            if i == len(e_index) - 1:
                tmp.append(e_index[i])
                empty_index.append(tmp)
            elif e_index[i]+1 == e_index[i+1]:
                tmp.append(e_index[i])
            else:
                tmp.append(e_index[i])
                empty_index.append(tmp)
                tmp = []

        for e in empty_index:
            unit = (adjusted_motion[e[len(e)-1]+1] - adjusted_motion[e[0]-1]) / (len(e) + 1)
            cnt = 1
            for i in range(e[0], e[len(e)-1] + 1):
                adjusted_motion[i] = adjusted_motion[e[0]-1] + unit*cnt
                cnt += 1

    # Sampling
    else:
        adjusted_motion = []
        decrease_frame = len(motion) - adjust_frame
        if decrease_frame == 1:
            decrease_interval = len(motion) / 2 + 1
        else:
            decrease_interval = len(motion) / decrease_frame
        for i in range(len(motion)):
            if i != 0 and i % decrease_interval == 0:
                continue
            adjusted_motion.append(motion[i])
            if len(adjusted_motion) == adjust_frame:
                break
    
    return adjusted_motion


if __name__ == '__main__':
    # wav_path = "./data/_ryJK294Psw_14/_ryJK294Psw_14.wav"
    # wav_path = "./data/1oNlTrLIjU4_8/1oNlTrLIjU4_8.wav"
    # wav_path = "./data/_vBggxCNNno_4/_vBggxCNNno_4.wav"
    wav_path = "./data/Beat_1L6l-FiV4xo_8/1L6l-FiV4xo_8_human.wav"

    motion_data = './data/Beat_1L6l-FiV4xo_8/1L6l-FiV4xo_8.csv'
    motion_gaussian = 1
    motion_range = [45, 105]

    # motion_data = './data/Beat_0ZfSOArXbGQ_10_4/0ZfSOArXbGQ_10_4.csv'
    # motion_gaussian = 4
    # motion_range = [0, 127]

    csv_save_dir = "./output/csv/"
    mp4_save_dir = "./output/mp4/"
    fps = 25
    sr = 44100  # sampling rate
    # motion_interval = 25   # swing arms every interval frame

    # Extract motion feature
    pose3d = load_kinect_csv(motion_data)
    # Extract Beat motion
    pose3d_cut = pose3d[motion_range[0]: motion_range[1]]


    lelbow_angles = calcAngular(pose3d_cut[:,6,:], pose3d_cut[:,5,:], pose3d_cut[:,4,:])
    relbow_angles = calcAngular(pose3d_cut[:,10,:], pose3d_cut[:,9,:], pose3d_cut[:,8,:])
    plotTransition([relbow_angles, lelbow_angles], ['right', 'left'], title='Elbow Angle')
    lelbow_angles = gaussian_filter1d(lelbow_angles, motion_gaussian)
    relbow_angles = gaussian_filter1d(relbow_angles, motion_gaussian)
    plotTransition([relbow_angles, lelbow_angles], ['right', 'left'], title='Elbow Angle')

    # Extract audio feature
    audio_wave = getAudioEnvelopeGaussian(wav_path, sr=sr)
    interval = int(sr/fps)
    a = audio_wave[::interval]
    # Cut the last part that doesn't say anything.
    for i in range(len(a)-1, 0, -1):
        if a[i] > 1e-4:
            end_idx = i
            break
    audio_wave = a[:end_idx]
    plotTransition([audio_wave], labels=['audio'])


    # Local Maxima
    motion_maxima = signal.argrelmax(relbow_angles, order=1)[0]
    audio_maxima = signal.argrelmax(audio_wave, order=1)[0]


    adjusted_motion = []
    motion_interval = int(len(audio_wave)/len(motion_maxima))
    adjusted_frame = 0
    cnt, last_peak = 0, 0
    for i in range(0, len(audio_wave), motion_interval):
        audio = audio_wave[i:i+motion_interval]
        audio_maxima = signal.argrelmax(audio, order=1)[0]
        if len(audio_maxima) == 0:
            peak_frame = np.argmax(audio)
        else:
            maxid = np.argmax(np.array([audio[m] for m in audio_maxima]))
            peak_frame = audio_maxima[maxid]
        
        print("Peak Frame: ", i+peak_frame)

        # Adjust motion to match audio
        motion = motionAdjustment(pose3d_cut[adjusted_frame:motion_maxima[cnt]], i+peak_frame-last_peak)
        for m in motion:
            adjusted_motion.append(m)
        
        adjusted_frame = motion_maxima[cnt]
        
        cnt += 1
        if cnt >= len(motion_maxima):
            break

        last_peak = i + peak_frame

    motion = motionAdjustment(pose3d_cut[motion_maxima[cnt-1]:], len(audio_wave) - len(adjusted_motion))
    for m in motion:
        adjusted_motion.append(m)

    adjusted_motion = np.array(adjusted_motion)

    # # Confirm
    lelbow_angles = calcAngular(adjusted_motion[:,6,:],  adjusted_motion[:,5,:], adjusted_motion[:,4,:])
    relbow_angles = calcAngular(adjusted_motion[:,10,:], adjusted_motion[:,9,:], adjusted_motion[:,8,:])
    # lelbow_angles = gaussian_filter1d(lelbow_angles, motion_gaussian)
    # relbow_angles = gaussian_filter1d(relbow_angles, motion_gaussian)
    plotTransition([relbow_angles, lelbow_angles], ['right', 'left'], title='Elbow Angle')
    

    file_name = os.path.basename(wav_path)[:-4] + '__beat__' + os.path.basename(motion_data)[:-4]
    
    # Save csv
    save_path = csv_save_dir + file_name + '.csv'
    CMUPose2KinectData(adjusted_motion, save_csv=save_path, fps=fps, isConvert=False)
    print('Saved csv file to {}'.format(save_path))
    
    print("Saving video...")
    plotUpperBody2D(adjusted_motion, "./.tmp/tmp.mp4", fps=fps)

    # Attaching audio
    save_path = mp4_save_dir + file_name + '.mp4'
    cmd = "ffmpeg -i {mp4} -i {wav} -c:v copy -c:a aac -strict experimental -map 0:v -map 1:a {output} -y".format(
        mp4="./.tmp/tmp.mp4", wav=wav_path, output=save_path)
    subprocess.call(cmd)

    os.remove("./.tmp/tmp.mp4")
    
    print("Output to {}".format(save_path))

