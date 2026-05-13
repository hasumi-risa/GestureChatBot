import os
import math
import csv
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer


def matchWordToken(clip_words, tokenizer):
    word_list = [w[0] for w in clip_words]
    token = tokenizer(word_list)['input_ids']
    each_token_num = [len(t) - 2 for t in token]
    # token = tokenizer(" ".join(word_list))['input_ids']
    new_word_list = []
    for i in range(len(each_token_num)):
        for j in range(each_token_num[i]):
            word = tokenizer.convert_ids_to_tokens(token[i][j+1])
            if j == 0:
                word_unit = [word, clip_words[i][1], clip_words[i][2]]
            else:
                word_unit = [word, clip_words[i][2], clip_words[i][2]]
            new_word_list.append(word_unit)
    return new_word_list


def type2num(g_type):
    if "No-Gesture" in g_type:
        return 0
    elif "Beat" in g_type:
        return 1
    elif "Imagistic" in g_type:
        return 2
    else:
        print("Unknown gesture type: {}".format(g_type))
        exit(-1)

def insertCLSandSEP(array):
    array.insert(0, 0)
    array.append(0)
    return array

ted_dataset_path = "C:/Users/b19.teshima/Documents/Gesture/3D-Pose-Baseline-LSTM/data/TED_gesture_dataset_3D_interpolate.pickle"
annotation_data = "./data/annotation_results_integrated_20210528.xlsx"
save_dir = './data/'
bert_model = 'bert-base-uncased'
video_fps = 25
fade_word_num = 4
train_test_ratio = 0.80
train_valid_ratio = 0.90    # Train : Valid : Test = 72 : 8 : 20


ted_data = torch.load(ted_dataset_path)
annot_df = pd.read_excel(annotation_data)
tokenizer = BertTokenizer.from_pretrained(bert_model)

save_data = {'token_ids': [], 'label': [], 'video_id':[], 'clip_id':[]}
for vid_data in ted_data:
    ted_id = vid_data['vid']
    for clip_id, clip_data in enumerate(vid_data['clips']):
        video_id = ted_id + '_' + str(clip_id)
        clip_df = annot_df[annot_df['Video ID']==video_id]
        if len(clip_df) == 0:
            continue


        # Prepare whole text
        word_list = matchWordToken(clip_data['words'], tokenizer)
        begin_frames = np.array([w[1] for w in word_list])
        end_frames = np.array([w[2] for w in word_list])
        clip_text = ""
        for w in clip_data['words']:
            clip_text += w[0] + ' '
        token_ids = tokenizer(clip_text)

        clip_start_frame = clip_data['start_frame_no']
        trapez_label = np.zeros((3, len(word_list)))
        for ges_id in range(len(clip_df)):
            type_id = type2num(clip_df["Gesture Type"].iloc[ges_id])
            begin_frame = clip_start_frame + int(clip_df["Start Time"].iloc[ges_id] * video_fps)
            end_frame = clip_start_frame + int(clip_df["End Time"].iloc[ges_id] * video_fps)

            for i,w in enumerate(word_list):
                for j, frm in enumerate(begin_frames - begin_frame):
                    if frm >= 0:
                        begin_index = j
                        break
                for j, frm in enumerate(end_frames - end_frame):
                    if frm >= 0:
                        end_index = j
                        break
            if (begin_frames - begin_frame)[-1] < 0:
                continue
            if (end_frames - end_frame)[-1] < 0:
                end_index = len(end_frames) - 1

            # 台形波形作成
            x = np.arange(0, fade_word_num, 1)
            rising = np.linspace(0, 1, num=fade_word_num+1)
            falling =  np.linspace(1, 0, num=fade_word_num+1)

            # 上り
            for i in range(begin_index - fade_word_num, begin_index):
                if i < 0:
                    continue
                trapez_label[type_id][i] = max(trapez_label[type_id][i], rising[i - (begin_index - fade_word_num)])
            
            # 上底
            for i in range(begin_index, end_index + 1):
                trapez_label[type_id][i] = 1

            # 下り
            for i in range(end_index + 1, end_index + fade_word_num + 1):
                if i >= len(word_list):
                    continue
                trapez_label[type_id][i] = falling[i - (end_index + 1)]

        save_data['token_ids'].append(token_ids['input_ids'])
        save_data['label'].append(trapez_label.tolist())
        save_data['video_id'].append(ted_id)
        save_data['clip_id'].append(video_id)

df = pd.DataFrame(save_data)
save_data['tokenizer'] = tokenizer
torch.save(save_data, save_dir + 'preprocessed_data_all.pth')

labels = []
for clip in df['label']:
    label = []
    label.append(insertCLSandSEP(list(clip[0])))
    label.append(insertCLSandSEP(list(clip[1])))
    label.append(insertCLSandSEP(list(clip[2])))
    labels.append(label)

for i in range(len(df['label'])):
    df['label'].iloc[i] = labels[i]


# Train-test split
df_full_train, df_test = train_test_split(df, train_size = train_test_ratio, random_state = 1)

# Train-valid split
df_train, df_valid = train_test_split(df_full_train, train_size = train_valid_ratio, random_state = 1)

torch.save(df_train, save_dir + '/train_3class.pth')
torch.save(df_valid, save_dir + '/valid_3class.pth')
torch.save(df_test, save_dir + '/test_3class.pth')
print("Saved")

