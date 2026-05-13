import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def laban2index(laban):
    if      laban == ['Forward',            'High']:    return 0
    elif    laban == ['Right Forward',      'High']:    return 1
    elif    laban == ['Right',              'High']:    return 2
    elif    laban == ['Left',               'High']:    return 3
    elif    laban == ['Left Forward',       'High']:    return 4
    elif    laban == ['Place',              'High']:    return 5
    elif    laban == ['Forward',            'Normal']:  return 6
    elif    laban == ['Right Forward',      'Normal']:  return 7
    elif    laban == ['Right',              'Normal']:  return 8
    elif    laban == ['Left',               'Normal']:  return 9
    elif    laban == ['Left Forward',       'Normal']:  return 10
    elif    laban == ['Forward',            'Low']:     return 11
    elif    laban == ['Right Forward',      'Low']:     return 12
    elif    laban == ['Right',              'Low']:     return 13
    elif    laban == ['Left',               'Low']:     return 14
    elif    laban == ['Left Forward',       'Low']:     return 15
    elif    laban == ['Place',              'Low']:     return 16
    else:   return -1


def handsLaban2Index(right, left):
    return laban2index(right) * 100 + laban2index(left)


if __name__ == '__main__':

    annotation_data = './data/annotation_results_integrated_20211006.xlsx'
    # glove_file = 'C:\\Users\\b19.teshima\\Documents\\Gesture\\GestureGeneration\\imagistic_gesture_generator\\data\\glove.npy'
    data_dir = 'D:/TED_videos/segmented_by_gesture'
    invalid_data = "D:/TED_videos/invalid_gesture_id.txt"
    save_geslib_path = './data/no-gesture_library_20211006.npy'
    fps = 25

    # Load Data
    annot_df = pd.read_excel(annotation_data)
    
    with open(invalid_data) as f:
        invalid_ids = f.read().splitlines()
    
    print('Adding gestures to gesture library...')
    gesture_id_list = []
    gesture_list = []
    laban_list = []
    start_laban_dic = {}
    end_laban_dic = {}
    frame_nums = []
    index = 0
    for i in tqdm(range(len(annot_df))):
        gesture_type = annot_df['Gesture Type'].iloc[i]
        if gesture_type != "No-Gesture":
            continue

        gesture_id = annot_df['Gesture ID'].iloc[i]
        if gesture_id in invalid_ids:
            continue

        video_id = annot_df['Video ID'].iloc[i][:11]
        gesture_csv = data_dir + '/' + video_id + '/' + gesture_id + '/' + gesture_id + '.csv'
        laban_json = data_dir + '/' + video_id + '/' + gesture_id + '/' + gesture_id + '.json'

        if not os.path.exists(gesture_csv) or not os.path.exists(laban_json):
            continue

        # Load
        pose3d = load_kinect_csv(gesture_csv)
        with open(laban_json, 'r') as f:
            laban = json.load(f)
        laban = laban[list(laban.keys())[0]]

        if not laban:
            continue

        laban_index = handsLaban2Index(laban['Position0']['right wrist'], laban['Position0']['left wrist'])
        if not laban_index in start_laban_dic.keys():
            start_laban_dic[laban_index] = [index]
        else:
            start_laban_dic[laban_index].append(index)
        
        end = list(laban.keys())[-1]
        laban_index = handsLaban2Index(laban[end]['right wrist'], laban[end]['left wrist'])
        if not laban_index in end_laban_dic.keys():
            end_laban_dic[laban_index] = [index]
        else:
            end_laban_dic[laban_index].append(index)

        gesture_id_list.append(gesture_id)
        gesture_list.append(pose3d)
        laban_list.append(laban)
        frame_nums.append(pose3d.shape[0])
        index += 1

    gesture_id_dic = {}
    for i in range(len(gesture_id_list)):
        gesture_id_dic[gesture_id_list[i]] = i

    gesture_library = { 
        'index_to_gesture_id': gesture_id_list,
        'gesture_id_to_index': gesture_id_dic,
        'start_laban_to_index': start_laban_dic,
        'end_laban_to_index': end_laban_dic,
        'frame_nums': frame_nums,
        'gesture_list':gesture_list, 
        'laban_list':laban_list,
        }
                        
    np.save(save_geslib_path, gesture_library)

    print("Saved to", save_geslib_path)



