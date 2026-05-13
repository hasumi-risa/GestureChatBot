import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from tqdm import tqdm

data_path = './data/train_class.pth'
ted_dataset_path = "C:/Users/b19.teshima/Documents/Gesture/3D-Pose-Baseline-LSTM/data/TED_gesture_dataset_3D_interpolate.pickle"
save_dir = "./images/train_data_class"

data = torch.load(data_path)
gesture_data = torch.load(ted_dataset_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def visualizeTrapezLabel(text, label, class_num=3, save_path=None):
    x = np.arange(len(label[0]))
    plt.figure()
    plt.plot(x, label[0], label='No-Gesture', color='gray')
    if class_num == 3:
        plt.plot(x, label[1], label='Beat', color='blue')
        plt.plot(x, label[2], label='Imagistic', color='red')    
    elif class_num == 5:
        plt.plot(x, label[1], label='Beat', color='blue')
        plt.plot(x, label[2], label='Deictic', color='green')
        plt.plot(x, label[3], label='Iconic', color='orange')
        plt.plot(x, label[4], label='Metaphoric', color='red')
    plt.xticks(x,text)
    plt.xticks(rotation=90)
    plt.ylim(0,1.5)
    plt.xlabel('Text')
    plt.xlabel('Label')
    plt.legend(loc='upper right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def visualizeClassLabel(text, label, class_num=3, save_path=None):
    x = np.arange(len(label))
    each_label = []
    for i in range(class_num):
        each_label.append(np.zeros(len(label)))
    for i in range(class_num):
        for w in range(len(label)):
            if label[w] == i:
                each_label[i][w] = 1 

    plt.figure()
    plt.plot(x, each_label[0], label='No-Gesture', color='gray')
    if class_num == 3:
        plt.plot(x, each_label[1], label='Beat', color='blue')
        plt.plot(x, each_label[2], label='Imagistic', color='red')    
    elif class_num == 5:
        plt.plot(x, each_label[1], label='Beat', color='blue')
        plt.plot(x, each_label[2], label='Deictic', color='green')
        plt.plot(x, each_label[3], label='Iconic', color='orange')
        plt.plot(x, each_label[4], label='Metaphoric', color='red')
    plt.xticks(x,text)
    plt.xticks(rotation=90)
    plt.ylim(0,1.5)
    plt.xlabel('Text')
    plt.xlabel('Label')
    plt.legend(loc='upper right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# save images
for c_idx in tqdm(range(len(data))):
    text = tokenizer.convert_ids_to_tokens(data['token_ids'].iloc[c_idx])
    save_each_dir = save_dir + "/" + data['video_id'].iloc[c_idx] + '/'
    if not os.path.exists(save_each_dir):
        os.makedirs(save_each_dir)
    save_path = save_each_dir + data['clip_id'].iloc[c_idx] + '.png'
    # visualizeTrapezLabel(text, data['label'].iloc[c_idx], class_num=3, save_path=save_path)
    visualizeClassLabel(text, data['label'].iloc[c_idx], class_num=3, save_path=save_path)

