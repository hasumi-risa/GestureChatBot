import os, sys
import datetime
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from .audio_processing import compute_prosody
from .src.cmu2kinect import CMUPose2KinectData
from .src.plotPose import Plot
from .src.calc_total_energy import totalEnergy


class BeatGenerator:
    def __init__(self, beat_library_path, fps=25, sr=44100):
        self.fps = fps
        self.sr = sr  # sampling rate
        # motion_interval = 25   # swing arms every interval frame

        beat_library = np.load(beat_library_path, allow_pickle=True)
        self.index_to_gesture_id = beat_library.item()['index_to_gesture_id']
        self.keyframes_dic = beat_library.item()['keyframe_num_to_index']
        self.gesture_list = beat_library.item()['gesture_list']
        self.laban_list = beat_library.item()['laban_list']


    def generate(self, audio_wave):
        audio_keyframe = list(signal.argrelmax(audio_wave, order=1)[0])
        audio_keyframe_num = len(audio_keyframe)
        if audio_keyframe_num == 0:
            audio_keyframe_num = 1
        if not audio_keyframe_num in self.keyframes_dic.keys():
            l = list(self.keyframes_dic.keys())
            index = np.abs(np.asarray(l) - audio_keyframe_num).argmin()
            audio_keyframe_num = l[index]

        audio_length = len(audio_wave)

        # Gesture ID の選出
        # Keyframeの数が音声のKeyframeの数と一致しているものから
        # ランダム or 時間が近いもの or Keyframeの間隔が似ているもの or 波形が近いもの 
        
        # 時間が近いもの
        min_length = 1000
        for i in self.keyframes_dic[audio_keyframe_num]:
            if len(self.gesture_list[i]) - audio_length < min_length:
                index = i

        # Motion feature
        gesture_id = self.index_to_gesture_id[index]
        print('Selected Beat Gesture -> ', gesture_id)
        pose3d = self.gesture_list[index]
        laban = self.laban_list[index]
        keyframe = [int(int(laban[key]['start time'][0]) * self.fps / 1000) for key in list(laban.keys())][:-1]

        # Confirm
        # energy = totalEnergy(pose3d)
        # plotTransition([energy, audio_wave], ['motion energy', 'audio'], [keyframe, audio_keyframe])

        # Adjust motion to audio
        adjusted_motion = []
        pre_mkf, pre_akf = 0, 0
        for mkf,akf in zip(keyframe, audio_keyframe):
            length = akf - pre_akf
            motion = self.motionAdjustment(pose3d[pre_mkf:mkf], length)
            # print("{} -> {}".format(len(pose3d[pre_mkf:mkf]), len(motion)))
            for m in motion:
                adjusted_motion.append(m)
            pre_mkf = mkf
            pre_akf = akf
        motion = self.motionAdjustment(pose3d[pre_mkf:], len(audio_wave)-pre_akf)
        for m in motion:
            adjusted_motion.append(m)
        adjusted_motion = np.array(adjusted_motion)
        self.beat_gesture = adjusted_motion

        # # Confirm
        new_energy = totalEnergy(adjusted_motion)
        new_keyframe = list(signal.argrelmax(new_energy, order=1)[0])
        now = datetime.datetime.now()
        save_path = "./.tmp/beat_energy_{}.png".format(now.strftime('%Y%m%d_%H%M%S'))
        plotTransition([new_energy, audio_wave], ['motion energy', 'audio'], [new_keyframe, audio_keyframe], save_path=save_path)
        
        return self.beat_gesture, laban
        

    def saveMotionFile(self, save_path):
        # Save csv
        CMUPose2KinectData(self.beat_gesture, save_csv=save_path, fps=self.fps, isConvert=False)
        print('Saved csv file to {}'.format(save_path))


    def saveVideo(self, save_path):
        print("Saving video...")
        self.plotUpperBody2D(self.beat_gesture, "./.tmp/tmp.mp4", fps=self.fps)

        # Attaching audio
        cmd = "ffmpeg -i {mp4} -i {wav} -c:v copy -c:a aac -strict experimental -map 0:v -map 1:a {output} -y".format(
            mp4="./.tmp/tmp.mp4", wav=wav_path, output=save_path)
        subprocess.call(cmd)

        os.remove("./.tmp/tmp.mp4")
        
        print("Output to {}".format(save_path))


    def plotUpperBody2D(self, pose3d, save_path, types=None, fps=25):
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
            if types:
                pose.append(types[i])
            poses.append(pose)

        p = Plot((-0.75, 0.75), (-0.5, 1))
        anim = p.animate(poses, 1000/fps)
        p.save(anim, save_path, fps=fps)


    def motionAdjustment(self, motion, adjust_frame, types=None):
        if len(motion) == adjust_frame:
            adjusted_motion = motion
        
        # Interpolation
        elif len(motion) < adjust_frame:
            adjusted_motion = np.zeros([adjust_frame, 25, 3])
            adjusted_motion[0] = motion[0]
            if types:
                adjusted_types = [""] * adjust_frame
                adjusted_types[0] = types[0]
                adjusted_types[len(adjusted_types)-1] = types[len(types)-1]
            adjusted_motion[len(adjusted_motion)-1] = motion[len(motion)-1]
            interval = (adjust_frame - 2) / (len(motion) - 2)
            interval_sum, count = 0, 0
            e_index, tmp = [], []
            isEmpty = False
            for i in range(1, len(adjusted_motion)-1):
                if i > interval_sum:
                    adjusted_motion[i] = motion[count]
                    if types:
                        if count < len(types):  # To Do
                            adjusted_types[i] = types[count]
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
                    if types:
                        adjusted_types[i] = adjusted_types[e[0]-1]
                    cnt += 1

        # Sampling
        else:
            adjusted_motion = []
            adjusted_types = []
            decrease_frame = len(motion) - adjust_frame
            sampling_frames = [int(n) for n in np.arange(0, len(motion), len(motion) / decrease_frame)]
            sampling_frames = sampling_frames[:decrease_frame]

            if len(sampling_frames) != decrease_frame:
                print('Error: samping failed', file=sys.stderr)
                sys.exit()
                
            for i in range(len(motion)):
                if i in sampling_frames:
                    continue
                adjusted_motion.append(motion[i])
                if types:
                    adjusted_types.append(types[i])
        
        if types:
            return np.array(adjusted_motion), adjusted_types
        else:
            return np.array(adjusted_motion)


def plotTransition(lists, labels, keyframes=None, sr=None, title=None, save_path=None):
    figsize = plt.rcParams["figure.figsize"]
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
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    plt.rcParams["figure.figsize"] = figsize


if __name__ == '__main__':
    # wav_path = "./data/_ryJK294Psw_14/_ryJK294Psw_14.wav"
    # wav_path = "./data/1oNlTrLIjU4_8/1oNlTrLIjU4_8.wav"
    # wav_path = "./data/_vBggxCNNno_4/_vBggxCNNno_4.wav"
    wav_path = "./data/38OUCtzkT4Q_7_0.wav"

    # beat_library_path = "./data/beat_library_20211006.npy"
    beat_library_path = "./data/beat_library_20211006_msrabot.npy" # for MSRABot
    audio_feature = 'intensity'
    audio_gaussian = 1

    csv_save_path = "./output/csv/{}.csv".format(os.path.basename(wav_path)[:-4])
    mp4_save_path = "./output/mp4/{}.mp4".format(os.path.basename(wav_path)[:-4])

    bg = BeatGenerator(beat_library_path)

    # Extract audio feature
    pitch, intensity, time = compute_prosody(wav_path)

    if audio_feature == 'pitch':
        audio_wave = pitch
    elif audio_feature == 'intensity':
        audio_wave = intensity

    # Cut the last part that doesn't say anything.
    for i in range(len(audio_wave)-1, 0, -1):
        if audio_wave[i] > 1e-4:
            end_idx = i
            break
    audio_wave = audio_wave[:end_idx]
    audio_wave = gaussian_filter1d(audio_wave, audio_gaussian)
    
    # Generate
    gesture = bg.generate(audio_wave)

    # Save
    bg.saveMotionFile(csv_save_path)
    bg.saveVideo(mp4_save_path)


