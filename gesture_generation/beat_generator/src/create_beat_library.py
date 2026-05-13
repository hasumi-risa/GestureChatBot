import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import parselmouth as pm

def load_kinect_csv(csv_file):
    kinect_3d_pose = pd.read_csv(csv_file, header=None)
    pose3d = []
    for i in range(len(kinect_3d_pose)):
        frame_pose = kinect_3d_pose.iloc[i]
        pose = []
        for j in range(len(frame_pose)):
            if j == 0:
                tmp = []
            elif j % 4 == 0:
                pose.append(tmp)
                tmp = []
            else:
                tmp.append(frame_pose[j])
        pose3d.append(pose)
    pose3d = np.array(pose3d)
    return pose3d


def compute_prosody(audio_filename, time_step=0.04):
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

    return pitch_norm, intensity_norm, times


if __name__ == '__main__':

    annotation_data = './data/annotation_results_integrated_20211006.xlsx'
    # glove_file = 'C:\\Users\\b19.teshima\\Documents\\Gesture\\GestureGeneration\\imagistic_gesture_generator\\data\\glove.npy'
    data_dir = 'D:/TED_videos/segmented_by_gesture'
    invalid_data = "D:/TED_videos/invalid_gesture_id.txt"
    save_geslib_path = './data/beat_library_20211006.npy'
    fps = 25

    # Load Data
    annot_df = pd.read_excel(annotation_data)
    
    with open(invalid_data) as f:
        invalid_ids = f.read().splitlines()
    
    print('Adding gestures to gesture library...')
    gesture_id_list = []
    gesture_list = []
    laban_list = []
    prosodic_list = []
    keyframes_dic = {}
    index = 0
    for i in tqdm(range(len(annot_df))):
        gesture_type = annot_df['Gesture Type'].iloc[i]
        if gesture_type != "Beat":
            continue

        gesture_id = annot_df['Gesture ID'].iloc[i]
        if gesture_id in invalid_ids:
            continue


        video_id = annot_df['Video ID'].iloc[i][:11]
        gesture_csv = data_dir + '/' + video_id + '/' + gesture_id + '/' + gesture_id + '.csv'
        laban_json = data_dir + '/' + video_id + '/' + gesture_id + '/' + gesture_id + '.json'
        audio_file = data_dir + '/' + video_id + '/' + gesture_id + '/' + gesture_id + '.wav'

        if not os.path.exists(gesture_csv) or not os.path.exists(laban_json) or not os.path.exists(audio_file):
            continue

        # Load
        pose3d = load_kinect_csv(gesture_csv)
        with open(laban_json, 'r') as f:
            laban = json.load(f)
        laban = laban[list(laban.keys())[0]]
        keyframes = [int(int(laban[key]['start time'][0]) * fps / 1000) for key in list(laban.keys())][:-1]
        keyframe_num = len(keyframes)

        if not keyframe_num in keyframes_dic.keys():
            keyframes_dic[keyframe_num] = [index]
        else:
            keyframes_dic[keyframe_num].append(index)

        pitch, intensity, times = compute_prosody(audio_file)
        if len(pitch) > len(pose3d):
            diff = len(pitch) - len(pose3d)
            pitch = pitch[:-diff]
            intensity = intensity[:-diff]
            times = times[:-diff]
        prosody = {
            "pitch": pitch, 
            "intensity": intensity, 
            "time": times
        }

        gesture_id_list.append(gesture_id)
        gesture_list.append(pose3d)
        laban_list.append(laban)
        prosodic_list.append(prosody)
        index += 1

    gesture_id_dic = {}
    for i in range(len(gesture_id_list)):
        gesture_id_dic[gesture_id_list[i]] = i

    gesture_library = { 
        'index_to_gesture_id': gesture_id_list,
        'gesture_id_to_index': gesture_id_dic,
        'keyframe_num_to_index': keyframes_dic,
        'gesture_list':gesture_list, 
        'laban_list':laban_list,
        'prosodic_list':prosodic_list
        }
                        
    np.save(save_geslib_path, gesture_library)

    print("Saved to", save_geslib_path)



