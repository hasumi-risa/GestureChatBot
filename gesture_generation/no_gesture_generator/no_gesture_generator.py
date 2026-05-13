import os
import random
import subprocess
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage.measurements import minimum

from .src.cmu2kinect import CMUPose2KinectData
from .src.plotPose import Plot
from .src.calc_total_energy import totalEnergy
from .audio_processing import compute_prosody
from .src.create_no_gesture_library import handsLaban2Index

class NoGestureGenerator:
    def __init__(self, nogesture_library_path, fps=25, sr=44100):
        self.fps = fps
        self.sr = sr  # sampling rate
        # motion_interval = 25   # swing arms every interval frame

        noges_library = np.load(nogesture_library_path, allow_pickle=True)
        self.index_to_gesture_id = noges_library.item()['index_to_gesture_id']
        self.start_laban_dic = noges_library.item()['start_laban_to_index']
        self.end_laban_dic = noges_library.item()['end_laban_to_index']
        self.frame_nums = noges_library.item()['frame_nums']
        self.gesture_list = noges_library.item()['gesture_list']
        self.laban_list = noges_library.item()['laban_list']


    def generate(self, duration, start_laban=None, end_laban=None):
        # Gesture ID の選出
        # start_laban: No-Gesture の始まりのラバノーテーション
        # end_laban: No-Gesture の終わりのラバノーテーション
        # start_laban, end_laban を満たしつつ,durationが近いものを選ぶ

        if start_laban and end_laban:
            
            start_laban_index = handsLaban2Index(start_laban['right wrist'], start_laban['left wrist'])
            end_laban_index = handsLaban2Index(end_laban['right wrist'], end_laban['left wrist'])

            if start_laban_index in self.end_laban_dic.keys():
                end_laban_ids = self.end_laban_dic[start_laban_index]
            else:
                end_laban_ids = self.end_laban_dic[0]

            if end_laban_index in self.start_laban_dic.keys():
                start_laban_ids = self.start_laban_dic[end_laban_index]
            else:
                start_laban_ids = self.start_laban_dic[0]
            laban_ids = list(set(start_laban_ids) & set(end_laban_ids))

            if len(laban_ids) == 0:
                laban_ids = start_laban_ids

        elif start_laban:
            laban_index = handsLaban2Index(start_laban['right wrist'], start_laban['left wrist'])

            if laban_index in self.end_laban_dic.keys():
                laban_ids = self.end_laban_dic[laban_index]
            else:
                laban_ids = self.end_laban_dic[0]

        elif end_laban:
            laban_index = handsLaban2Index(end_laban['right wrist'], end_laban['left wrist'])

            if laban_index in self.start_laban_dic.keys():
                laban_ids = self.start_laban_dic[laban_index]
            else:
                laban_ids = self.start_laban_dic[0]
                
        else:
            laban_ids = np.arange(0, 30)

        minimum = 10000
        for i in laban_ids:
            if minimum > self.frame_nums[i] - duration:
                index = i
                minimum = self.frame_nums[i] - duration

        # Motion feature
        gesture_id = self.index_to_gesture_id[index]
        print('Selected No-Gesture -> ', gesture_id)
        pose3d = self.gesture_list[index]
        laban = self.laban_list[index]

        # Adjust motion to audio
        self.noges_gesture = self.motionAdjustment(pose3d, duration)

        return self.noges_gesture, laban
        

    def saveMotionFile(self, save_path):
        # Save csv
        CMUPose2KinectData(self.noges_gesture, save_csv=save_path, fps=self.fps, isConvert=False)
        print('Saved csv file to {}'.format(save_path))


    def saveVideo(self, save_path):
        print("Saving video...")
        self.plotUpperBody2D(self.noges_gesture, "./.tmp/tmp.mp4", fps=self.fps)

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
            if decrease_frame == 1:
                decrease_interval = len(motion) / 2 + 1
            else:
                decrease_interval = len(motion) / decrease_frame
            for i in range(len(motion)):
                if i != 0 and i % decrease_interval == 0:
                    continue
                adjusted_motion.append(motion[i])
                if types:
                    adjusted_types.append(types[i])
                if len(adjusted_motion) == adjust_frame:
                    break
        
        if types:
            return np.array(adjusted_motion), adjusted_types
        else:
            return np.array(adjusted_motion)


if __name__ == '__main__':
    # wav_path = "./data/_ryJK294Psw_14/_ryJK294Psw_14.wav"
    # wav_path = "./data/1oNlTrLIjU4_8/1oNlTrLIjU4_8.wav"
    # wav_path = "./data/_vBggxCNNno_4/_vBggxCNNno_4.wav"
    wav_path = "./data/38OUCtzkT4Q_7_0.wav"

    noges_library_path = "./data/noges_library_20211006.npy"
    audio_feature = 'intensity'

    csv_save_path = "./output/csv/{}.csv".format(os.path.basename(wav_path)[:-4])
    mp4_save_path = "./output/mp4/{}.mp4".format(os.path.basename(wav_path)[:-4])

    # Extract audio feature
    pitch, intensity, time = compute_prosody(wav_path)

    audio_wave = pitch

    # Cut the last part that doesn't say anything.
    for i in range(len(audio_wave)-1, 0, -1):
        if audio_wave[i] > 1e-4:
            end_idx = i
            break
    audio_wave = audio_wave[:end_idx]

    ngg = NoGestureGenerator(noges_library_path)

    # Generate
    gesture = ngg.generate(len(audio_wave))

    # Save
    ngg.saveMotionFile(csv_save_path)
    ngg.saveVideo(mp4_save_path)


