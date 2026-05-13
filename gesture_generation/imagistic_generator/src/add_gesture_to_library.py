import os
import numpy as np
import pandas as pd
    
import torch
from transformers import BertTokenizer, BertModel
    

def load_kinect_csv(csv_file):
    kinect_3d_pose = pd.read_csv(csv_file)
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


ted_video_dir = "D:/TED_videos/segmented_by_gesture"
gesture_library = "./data/gesture_library_k=30_BERT.npy"
new_gesture_library = "./data/gesture_library_k=30_BERT_q.npy"
add_word = "?"
add_gesture_ids_dists = {
    "0FQXicAGy5U_0_3":  0.4,
    "0g2WE1qXiKM_1_2":  0.8,
    "0ztdofPc8Rw_0_4":  0.2,
    "0ZfSOArXbGQ_9_2":  0.4,
    "1oNlTrLIjU4_13_4": 0,
    "1oNlTrLIjU4_11_4": 0
}


# BERT setting
options_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(options_name)
bert_model = BertModel.from_pretrained(options_name)
bert_model.eval()

print('Loading Gesture Library...')
geslib = np.load(gesture_library, allow_pickle=True)
gesture_list = geslib.item().get('gesture_list')
distance_list = geslib.item().get('distance_list')
remark_list = geslib.item().get('remark_list')
freq_list = geslib.item().get('frequency_list')
embvecs = geslib.item().get('emb_vectors')
cluster_index = geslib.item().get('cluster_index')
mean_pose = geslib.item().get('mean_pose')

print()

# Word Embedding
sw = add_word.split()
sw.insert(0, "[CLS]")
sw.append("[SEP]")
tokens = tokenizer.convert_tokens_to_ids(sw)
tokens_tensor = torch.tensor([tokens])
with torch.no_grad(): # 勾配計算なし
    all_encoder_layers = bert_model(tokens_tensor)
embedding = all_encoder_layers[0]
cls = embedding[:,0,:][0].numpy()


add_gesture_list = []
for gesture_id in list(add_gesture_ids_dists.keys()):
    video_id = gesture_id[:11]
    csv_path = ted_video_dir + '/' + video_id + '/' + gesture_id + '/' + gesture_id + '.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError
    gesture = load_kinect_csv(csv_path)
    add_gesture_list.append(gesture)





gesture_library = { 'gesture_list':gesture_list, 
                    'word_list':np.array(w_list), 
                    'word2idx':word2idx,
                    'frequency_list':np.array(f_list), 
                    'emb_vectors':embvecs, 
                    'cluster_index':cluster_index, 
                    'mean_pose':mean_pose}
np.save(save_geslib_path, gesture_library)




print("Saved to", save_geslib_path)