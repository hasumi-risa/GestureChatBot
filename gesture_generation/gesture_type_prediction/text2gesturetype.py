import os
import re
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import seaborn as sns

import torch
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch.nn as nn
from transformers import BertTokenizer
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


from .model import GestureTypePredictor, GestureTypePredictorLSTM
from .model import GestureTypePredictorLinear2, GestureTypePredictorLinear3, GestureTypePredictorLinear8, GestureTypePredictorLinear9, GestureTypePredictorLinear10, GestureTypePredictorLinear20


# ------------ Prepare Data ------------
def createDataLoader(data_path):
    data = torch.load(data_path)
    tag = list(data['label'])
    text_tensor = torch.zeros((len(data), MAX_SEQ_LEN))
    tag_tensor = torch.zeros((len(data), len(tag[0]), MAX_SEQ_LEN))
    token_length = []
    for i in range(len(data)):
        token = data['token_ids'].iloc[i]
        token_length.append(len(token))
        for j in range(len(token)):
            text_tensor[i][j] = token[j]
        for j in range(len(tag[i])):
            for k in range(len(tag[i][j])):
                tag_tensor[i][j][k] = tag[i][j][k]

    print('Max Token Num: {}'.format(max(token_length)))
    data = TensorDataset(text_tensor, tag_tensor)
    data_loader = DataLoader(data, batch_size=batch_size)
    return data_loader


def load_checkpoint(load_path, model, device):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['loss']

def makeContinuous(gesture_types, window_size=5):   
    new_gesture_types = []
    for i in range(len(gesture_types)):
        tmp = []
        for j in range(-int(window_size/2), int(window_size/2)+1):
            if i + j < 0 or i + j >= len(gesture_types):
                continue
            tmp.append(int(gesture_types[i+j]))
        new_gesture_types.append(collections.Counter(tmp).most_common()[0][0])
    return new_gesture_types

    
def _split_into_chunks(text, tokenizer, max_len):
    """Split text into sentence-boundary chunks that each fit within max_len tokens."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]

    chunks = []
    current_sentences = []
    current_len = 2  # [CLS] + [SEP]

    for sent in sentences:
        sent_token_len = len(tokenizer.encode(sent, add_special_tokens=False))
        if current_len + sent_token_len > max_len and current_sentences:
            chunks.append(' '.join(current_sentences))
            current_sentences = [sent]
            current_len = 2 + sent_token_len
        else:
            current_sentences.append(sent)
            current_len += sent_token_len

    if current_sentences:
        chunks.append(' '.join(current_sentences))

    return chunks


def _predict_chunk(chunk, model, tokenizer, device, PAD_INDEX):
    """Run gesture type prediction on a single chunk (must fit within MAX_SEQ_LEN)."""
    chunk_id = tokenizer.encode(chunk, add_special_tokens=True)
    chunk_id = [int(t) for t in chunk_id]
    # Truncate if a single sentence is pathologically long
    if len(chunk_id) > MAX_SEQ_LEN:
        chunk_id = chunk_id[:MAX_SEQ_LEN - 1] + [tokenizer.sep_token_id]
    chunk_tokens = tokenizer.convert_ids_to_tokens(chunk_id)
    padded = chunk_id + [PAD_INDEX] * (MAX_SEQ_LEN - len(chunk_id))
    chunk_tensor = torch.tensor([padded]).to(device)
    with torch.no_grad():
        output = model(chunk_tensor)
        _, predicted = torch.max(output, 2)
    chunk_types = list(predicted[0][:len(chunk_tokens)].cpu().numpy())
    return chunk_tokens, chunk_types


def text2gesturetype(text, model, tokenizer, device):
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    model.eval()

    raw_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(raw_ids) + 2 <= MAX_SEQ_LEN:
        chunks = [text]
    else:
        chunks = _split_into_chunks(text, tokenizer, MAX_SEQ_LEN)

    # Accumulate tokens and types across all chunks, keeping one [CLS]/[SEP] pair
    all_tokens = ['[CLS]']
    all_types = [0]

    for chunk in chunks:
        chunk_tokens, chunk_types = _predict_chunk(chunk, model, tokenizer, device, PAD_INDEX)
        # Strip [CLS] and [SEP] from each chunk before merging
        all_tokens.extend(chunk_tokens[1:-1])
        all_types.extend(chunk_types[1:-1])

    all_tokens.append('[SEP]')
    all_types.append(0)

    text_tokens = all_tokens
    gesture_types = all_types

    for i in range(len(text_tokens)):
        if text_tokens[i] == '?':
            gesture_types[i] = 2
            if i > 0:
                gesture_types[i - 1] = 2

    new_gesture_types = makeContinuous(gesture_types)
    filename = re.sub(r'[\\/:*?"<>|]+','', text[:30])
    visualizeLabel(text_tokens, new_gesture_types, save_path="./.tmp/{}.png".format(filename))
    return text_tokens, new_gesture_types


# ------------ Show Result  ------------
def visualizeLabel(text, y_pred, y_true=None, save_path=None):
    x = np.arange(len(text))

    class_num = 3
    y_true_each, y_pred_each = [], []
    for i in range(class_num):
        if y_true:
            y_true_each.append(np.zeros(len(y_true)))
        y_pred_each.append(np.zeros(len(y_pred)))
    for i in range(class_num):
        for w in range(len(y_pred)):
            if y_true:
                if y_true[w] == i:
                    y_true_each[i][w] = 1
            if y_pred[w] == i:
                y_pred_each[i][w] = 0.7

    plt.figure()

    plt.plot(x, y_pred_each[0][:len(text)], label='No-Gesture(Predict)', color='gray', alpha=1)
    plt.plot(x, y_pred_each[1][:len(text)], label='Non-imagistic(Predict)', color='blue', alpha=1)
    plt.plot(x, y_pred_each[2][:len(text)], label='imagistic(Predict)', color='red', alpha=1)

    if y_true:
        plt.plot(x, y_true_each[0][:len(text)], label='No-Gesture(GT)', color='gray', alpha=0.5, linestyle = "dashed")
        plt.plot(x, y_true_each[1][:len(text)], label='Non-imagistic(GT)', color='blue', alpha=0.5, linestyle = "dashed")
        plt.plot(x, y_true_each[2][:len(text)], label='imagistic(GT)', color='red', alpha=0.5, linestyle = "dashed")

    plt.xticks(x,text)
    plt.xticks(rotation=90)
    plt.ylim(0,1.5)
    plt.xlabel('Text')
    plt.xlabel('Label')
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def ReLU(array):
    array[array < 0] = 0
    return array

def showAttention(text, attention, save_path=None):
    attention = attention[0:len(text),0:len(text)]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.matshow(attention, cmap='Reds')
    ax.set_xticks(np.arange(len(text)))
    ax.set_yticks(np.arange(len(text)))
    ax.set_xticklabels(text, rotation=90)
    ax.set_yticklabels(text)
    ax.xaxis.tick_bottom()
    fig.colorbar(im)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def convert2class(y_true, y_pred):
    t = y_true.transpose(0, 2, 1)
    p = y_pred.transpose(0, 2, 1)
    yt, yp = [], []
    for i in range(len(t)):
        for j in range(len(t[i])):
            if sum(t[i][j]) == 0:
                continue
            max_t = np.argmax(t[i][j])
            max_p = np.argmax(p[i][j])
            isMax = t[i][j] == max(t[i][j])
            # 最大値が2つあった場合
            if list(isMax).count(True) == 2:
                continue
            yt.append(max_t)
            yp.append(max_p)
    return yt, yp


def showConfusionMatrix(y_true, y_pred, labels, save_path=None):
    yt, yp = [], []
    for i in range(len(y_true)):
        yt.append(labels[y_true[i]])
        yp.append(labels[y_pred[i]])
    ax = plt.subplot()
    cm = confusion_matrix(yt, yp, labels=labels)
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def insertCLSandSEP(array):
    array.insert(0, 0)
    array.append(0)
    return array

input_text = "business makes more money if they don't have a safe working environment. "
input_text = "business makes more money if they dont have a safe working environment . thats been the conventional wisdom ."
input_text = "thats been the conventional wisdom . if they dont have a safe working environment, business makes more money ."
input_text = "company makes more money if I didnt have a safe working environment. I have the conventional wisdom."
input_text = "company creates more money if we have a safe working space. I have wisdom."
input_text = "company destroys more money if we have a safe resting space. thats been the knowledge."
input_text = "so what is evolution's answer to the of uncertainty? that's been the conventional wisdom."

# ------------ Parameters ------------
model_path = './gesture_type_prediction/model/model_valid_f1_class_augmented_finetune_Linear.pt'

# Preliminaries
num_class = 3
MAX_SEQ_LEN = 256

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
# UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print("Device : ", device)

# ------------ Evaluation  ------------
# model = GestureTypePredictor(num_class=num_class).to(device)    
# load_checkpoint(model_path, model)

# text_tokens, gesture_types = text2gesturetype(input_text, model)

# print(output)



