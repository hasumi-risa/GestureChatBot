from ast import Num
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm

beat_library_path = "./beat_generator/data/beat_library_20211006.npy"
save_new_path = "./beat_generator/data/beat_library_20211006_msrabot.npy"


beat_library = np.load(beat_library_path, allow_pickle=True).item()
index_to_gesture_id = beat_library['index_to_gesture_id']
keyframes_dic = beat_library['keyframe_num_to_index']
gesture_list = beat_library['gesture_list']
laban_list = beat_library['laban_list']
prosodic_list = beat_library['prosodic_list']
gesture_id_to_index = beat_library['gesture_id_to_index']

delete_indexes = []

# delete the gesture that include Backward position
for i in range(len(laban_list)):
    for kf in list(laban_list[i].keys()):
        re = 'Backward' in laban_list[i][kf]['right elbow'][0]
        rw = 'Backward' in laban_list[i][kf]['right wrist'][0]
        le = 'Backward' in laban_list[i][kf]['left elbow'][0]
        lw = 'Backward' in laban_list[i][kf]['left wrist'][0]
        if re or rw or le or lw:
            delete_indexes.append(i)
            break

def deleteListElements(list, delete_list):
    for i in sorted(delete_list, reverse=True):
        list.pop(i)
    return list

gesture_list = deleteListElements(gesture_list, delete_indexes)
laban_list = deleteListElements(laban_list, delete_indexes)
index_to_gesture_id = deleteListElements(index_to_gesture_id, delete_indexes)
prosodic_list = deleteListElements(prosodic_list, delete_indexes)

idx = 0
for i,key in enumerate(list(gesture_id_to_index.keys())):
    if i in delete_indexes:
        gesture_id_to_index.pop(key)
    else:
        gesture_id_to_index[key] = idx
        idx += 1

keyframes_dic = {}
for i in range(len(laban_list)):
    keyframe_num = len(laban_list[i])
    if not keyframe_num in keyframes_dic.keys():
        keyframes_dic[keyframe_num] = [i]
    else:
        keyframes_dic[keyframe_num].append(i)

gesture_library = { 
    'index_to_gesture_id': index_to_gesture_id,
    'gesture_id_to_index': gesture_id_to_index,
    'keyframe_num_to_index': keyframes_dic,
    'gesture_list':gesture_list, 
    'laban_list':laban_list,
    'prosodic_list':prosodic_list
    }
                    
np.save(save_new_path, gesture_library)

print()
