import os
import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

train_data_path = './data/train_class_augmented.pth'
annotation_data = "./data/annotation_results_integrated_20210528.xlsx"
save_path = './Data/train_class_augmented_balanced.pth'

annot_df = pd.read_excel(annotation_data)
train_data = torch.load(train_data_path)

def getAllDataBalance(df):
    # 各ジェスチャータイプの数
    beat_num, imgs_num, noge_num = 0, 0, 0
    # 各ジェスチャータイプの総単語数
    beat_num_w, imgs_num_w, noge_num_w = 0, 0, 0
    for i in range(len(df)):
        if "No-Gesture" in df['Gesture Type'].iloc[i]:
            noge_num += 1
            if not pd.isna(df['Text'].iloc[i]):
                noge_num_w += len(df['Text'].iloc[i].split())
        elif "Beat" in df['Gesture Type'].iloc[i]:
            beat_num += 1
            if not pd.isna(df['Text'].iloc[i]):
                beat_num_w += len(df['Text'].iloc[i].split())
        elif "Imagistic" in df['Gesture Type'].iloc[i]:
            imgs_num += 1
            if not pd.isna(df['Text'].iloc[i]):
                imgs_num_w += len(df['Text'].iloc[i].split())

    total_num = beat_num + imgs_num + noge_num
    total_num_w = beat_num_w + imgs_num_w + noge_num_w
    print('Number of Each Gesture Type')
    print('No-Gesture : {}, {} %'.format(noge_num, int(100*noge_num/total_num))) 
    print('Beat       : {}, {} %'.format(beat_num, int(100*beat_num/total_num))) 
    print('Imagistic  : {}, {} %'.format(imgs_num, int(100*imgs_num/total_num))) 
    print('Total      : ', total_num)
    print()
    print('Number of Words of Each Gesture Type')
    print('No-Gesture : {}, {} %'.format(noge_num_w, int(100*noge_num_w/total_num_w))) 
    print('Beat       : {}, {} %'.format(beat_num_w, int(100*beat_num_w/total_num_w))) 
    print('Imagistic  : {}, {} %'.format(imgs_num_w, int(100*imgs_num_w/total_num_w))) 
    print('Total      : ', total_num_w)
    print()

def getWordBalance(data):
    noge_num, beat_num, imgs_num = 0, 0, 0
    for i in range(len(data)):
        label = data['label'].iloc[i]
        noge_num += label.count(0)
        beat_num += label.count(1)
        imgs_num += label.count(2)
    print('Beat\t\t', beat_num)
    print('Imagistic\t', imgs_num)
    print('No-Gesture\t', noge_num)
    print()
    return noge_num, beat_num, imgs_num


getAllDataBalance(annot_df)
noge_num, beat_num, imgs_num = getWordBalance(train_data)

train_data_new = train_data
while beat_num >= imgs_num:
    for i in range(len(train_data)):
        label = train_data['label'].iloc[i]
        noge = label.count(0)
        beat = label.count(1)
        imgs = label.count(2)
        total = noge + beat + imgs
        if total == 0:
            continue
        # Imagistic が過半数あるデータを複製して追加
        if imgs / total > 0.5:
            train_data_new = train_data_new.append(train_data.iloc[i])
            noge_num += noge
            beat_num += beat
            imgs_num += imgs

        if beat_num <= imgs_num:
            break

getWordBalance(train_data_new)

torch.save(train_data_new, save_path)


