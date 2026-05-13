import os
import math
import csv
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer
from tqdm import tqdm

def matchWordToken(clip_words, tokenizer):
    word_list = [w[0] for w in clip_words]
    token = tokenizer(word_list)['input_ids']
    each_token_num = [len(t) - 2 for t in token]
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


def findTokensInWordList(tokens, word_list):
    cnt, start, end = 0, 0, 0
    for i,w in enumerate(word_list):
        if end > 0:
            break
        for j in range(cnt, len(tokens)):
            if w == tokens[j]:
                if cnt == 0:
                    start = i
                cnt += 1
                if cnt == len(tokens):
                    end = i
                # print("{}\t{}\t{}".format(w, tokens[j], cnt))     # for debug
                break
            else:
                if cnt == 0:
                    # print("{}\t{}\t{}".format(w, tokens[j], cnt))     # for debug
                    break
                else:
                    cnt = 0
                    # print("{}\t{}\t{}".format(w, tokens[j], cnt))     # for debug
                    break
    return start, end


def insertCLSandSEP(array):
    array.insert(0, 0)
    array.append(0)
    return array

ted_dataset_path = "C:/Users/b19.teshima/Documents/Gesture/3D-Pose-Baseline-LSTM/data/TED_gesture_dataset_3D_interpolate.pickle"
annotation_data = "./data/annotation_results_integrated_20210726_mark.xlsx"
save_dir = './data/'
bert_model = 'bert-base-uncased'
video_fps = 25
fade_word_num = 4
train_test_ratio = 0.80
train_valid_ratio = 0.90    # Train : Valid : Test = 72 : 8 : 20


ted_data = torch.load(ted_dataset_path)
annot_df = pd.read_excel(annotation_data)
tokenizer = BertTokenizer.from_pretrained(bert_model)

save_data = {'text': [], 'token_ids': [], 'label': [], 'video_id':[], 'clip_id':[]}
sentence_length = []
ignore_seq, all_seq = 0, 0
for vid_data in tqdm(ted_data):
    ted_id = vid_data['vid']
    for clip_id, clip_data in enumerate(vid_data['clips']):
        video_id = ted_id + '_' + str(clip_id)

        clip_df = annot_df[annot_df['Video ID']==video_id]
        if len(clip_df) == 0:
            continue

        sentence_length.append(len(clip_data['words']))

        # Prepare whole text
        word_list = matchWordToken(clip_data['words'], tokenizer)
        begin_frames = np.array([w[1] for w in word_list])
        end_frames = np.array([w[2] for w in word_list])
        clip_text = ""
        for w in clip_data['words']:
            clip_text += w[0] + ' '
        token_ids = tokenizer(clip_text)

        clip_start_frame = clip_data['start_frame_no']
        label = np.zeros(len(word_list))
        for ges_id in range(len(clip_df)):
            all_seq += 1
            if clip_df["Gesture Type"].iloc[ges_id] != "Imagistic":
                continue

            begin_frame = clip_start_frame + int(clip_df["Start Time"].iloc[ges_id] * video_fps)
            end_frame = clip_start_frame + int(clip_df["End Time"].iloc[ges_id] * video_fps)
            remarks = tokenizer.convert_ids_to_tokens(tokenizer(str(clip_df['Remarks'].iloc[ges_id]))["input_ids"][1:-1])

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

            # for i in range(begin_index, end_index + 1):
            #     label[i] = int(type_id)

            clip_word_list = [w[0] for w in word_list][begin_index:end_index+1]
            start, end = findTokensInWordList(remarks, clip_word_list)
            if end == 0:
                # print("Failed to find remarks")
                # print("Text:", " ".join(clip_word_list))
                # print("Remarks:", " ".join(remarks))
                ignore_seq += 1
                continue

            for i in range(begin_index + start, begin_index + end + 1):
                label[i] = 1
            
        save_data['text'].append(clip_text)
        save_data['token_ids'].append(token_ids['input_ids'])
        save_data['label'].append(insertCLSandSEP(label.tolist()))
        save_data['video_id'].append(ted_id)
        save_data['clip_id'].append(video_id)

print("Ignored {} / {} seq".format(ignore_seq, all_seq))
df = pd.DataFrame(save_data)
save_data['tokenizer'] = tokenizer
torch.save(save_data, save_dir + 'preprocessed_data_all_class.pth')

# Train-test split
df_full_train, df_test = train_test_split(df, train_size = train_test_ratio, random_state = 1)

# Train-valid split
df_train, df_valid = train_test_split(df_full_train, train_size = train_valid_ratio, random_state = 1)

torch.save(df_train, save_dir + '/train_class_20210706.pth')
torch.save(df_valid, save_dir + '/valid_class_20210706.pth')
torch.save(df_test, save_dir + '/test_class_20210706.pth')
print("Saved")

