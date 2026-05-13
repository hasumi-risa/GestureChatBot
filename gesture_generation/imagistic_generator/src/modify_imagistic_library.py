import os
import json
import numpy as np
import pandas as pd

gesture_library = "./imagistic_generator/data/gesture_library_k=35_distmat.npy"
new_gesture_library = "./imagistic_generator/data/gesture_library_k=35_msrabot.npy"

geslib = np.load(gesture_library, allow_pickle=True).item()
gesture_list    = geslib['gesture_list']           # 34    
laban_list      = geslib['laban_list']               # 34
distance_list   = geslib['distance_list']         # 34    
cluster_index   = geslib['cluster_to_index']      # 34        
remark_list     = geslib['remark_list']             # 3262
embvec_list     = geslib['emb_vectors']             # 3262
idx2cluster     = geslib['index_to_cluster']        # 3262        

def deleteListElements(list, delete_list):
    for i in sorted(delete_list, reverse=True):
        list.pop(i)
    return list

new_gesture_list = []
new_laban_list = []
new_distance_list = []
new_cluster_index = []

all_delete_indexes = []
for cluster in range(len(laban_list)):
    delete_indexes = []
    for i in range(len(laban_list[cluster])):
        laban = laban_list[cluster][i]
        laban = laban[list(laban.keys())[0]]

        # delete the gesture that include Backward position
        for kf in list(laban.keys()):
            re = 'Backward' in laban[kf]['right elbow'][0]
            rw = 'Backward' in laban[kf]['right wrist'][0]
            le = 'Backward' in laban[kf]['left elbow'][0]
            lw = 'Backward' in laban[kf]['left wrist'][0]
            if re or rw or le or lw:
                delete_indexes.append(i)
                break

    for idx in delete_indexes:
        all_delete_indexes.append(cluster_index[cluster][idx])

    new_gesture_list.append(deleteListElements(gesture_list[cluster], delete_indexes))
    new_laban_list.append(deleteListElements(laban_list[cluster], delete_indexes))
    new_cluster_index.append(deleteListElements(cluster_index[cluster], delete_indexes))
    distmat = distance_list[cluster].tolist()
    tmp = deleteListElements(distmat, delete_indexes)
    tmp_list = []
    for i in range(len(tmp)):
        tmp_list.append(deleteListElements(tmp[i], delete_indexes))
    new_distance_list.append(np.array(tmp_list))

new_remark_list = deleteListElements(remark_list, all_delete_indexes)
new_embvec_list = deleteListElements(embvec_list, all_delete_indexes)
new_idx2cluster = deleteListElements(idx2cluster, all_delete_indexes)

print()

gesture_library = { 
    'gesture_list'      : new_gesture_list  ,
    'laban_list'        : new_laban_list    ,
    'distance_list'     : new_distance_list ,
    'cluster_to_index'  : new_cluster_index ,
    'remark_list'       : new_remark_list   ,
    'emb_vectors'       : new_embvec_list   ,
    'index_to_cluster'  : new_idx2cluster   ,
    }

np.save(new_gesture_library, gesture_library)

print("Saved to", new_gesture_library)