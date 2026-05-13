
# Libraries
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

# Preliminaries
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


# Models
import torch.nn as nn
from transformers import BertTokenizer
from model import GestureTypePredictor, GestureTypePredictorLSTM
from model import GestureTypePredictorLinear2, GestureTypePredictorLinear3, GestureTypePredictorLinear8, GestureTypePredictorLinear9, GestureTypePredictorLinear10, GestureTypePredictorLinear20

# Evaluation
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import seaborn as sns



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


# ------------ Function for Evaluation  ------------
def load_checkpoint(load_path, model):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['loss']

    
def text2gesturetype(model, text):
    model.eval()
    text_id = tokenizer.encode(text)
    for i in range(len(text_id), MAX_SEQ_LEN):
        text_id.append(PAD_INDEX)
    text_id = torch.tensor([text_id]).to(device)
    with torch.no_grad():
        output = model(text_id)
        answer = torch.argmax(output, 1)
    return answer


# ------------ Show Result  ------------
def visualizeLabel(text, y_pred, y_true=None, save_path=None):
    x = np.arange(len(text))
    plt.figure()

    plt.plot(x, y_pred[0][:len(text)], label='No-Gesture(Predict)', color='gray', alpha=1)
    plt.plot(x, y_pred[1][:len(text)], label='Non-imagistic(Predict)', color='blue', alpha=1)
    plt.plot(x, y_pred[2][:len(text)], label='imagistic(Predict)', color='red', alpha=1)

    if y_true:
        plt.plot(x, y_true[0][:len(text)], label='No-Gesture(GT)', color='gray', alpha=0.5, linestyle = "dashed")
        plt.plot(x, y_true[1][:len(text)], label='Non-imagistic(GT)', color='blue', alpha=0.5, linestyle = "dashed")
        plt.plot(x, y_true[2][:len(text)], label='imagistic(GT)', color='red', alpha=0.5, linestyle = "dashed")

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

if __name__ == '__main__':

    input_text = "business makes more money if they don't have a safe working environment. "
    input_text = "business makes more money if they dont have a safe working environment . thats been the conventional wisdom ."
    input_text = "thats been the conventional wisdom . if they dont have a safe working environment, business makes more money ."
    input_text = "company makes more money if I didnt have a safe working environment. I have the conventional wisdom."
    input_text = "company creates more money if we have a safe working space. I have wisdom."
    input_text = "company destroys more money if we have a safe resting space. thats been the knowledge."
    input_text = "so what is evolution's answer to the of uncertainty? that's been the conventional wisdom."

    # ------------ Parameters ------------
    mode1 = "3class"    
    mode2 = "1batch"    
    model_mode = 'LSTM'
    model_kind = 'train'            # valid or train
    clip_id = "0iIh5YYDR2o_23"      # for displaying GT
    data_kind = "train"             # for displaying GT (test or train)
    ted_dataset_path = "C:/Users/b19.teshima/Documents/Gesture/3D-Pose-Baseline-LSTM/data/TED_gesture_dataset_3D_interpolate.pickle"


    mses, accs, f1s = [], [], []
    print(model_mode)

    data_dir = './data/'
    model_path = './model/model_{}_{}_{}_{}.pt'.format(model_kind, mode1, mode2, model_mode) 
    batch_size = 1
    MAX_SEQ_LEN = 256

    # Preliminaries
    if mode1 == '3class':
        num_class = 3

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)
    
    if data_kind == 'test':
        data = torch.load(data_dir + 'test_{}.pth'.format(mode1))
    elif data_kind == 'train':
        data = torch.load(data_dir + 'train_{}.pth'.format(mode1 + '_balanced'))

    # ------------ Evaluation  ------------
    eval_data = torch.load(data_dir + '{}_{}.pth'.format(data_kind, mode1))
    if model_mode == 'Linear':
        model = GestureTypePredictor(num_class=num_class).to(device)    
    elif model_mode == 'LSTM':
        model = GestureTypePredictorLSTM(num_class=num_class).to(device)
    elif model_mode == 'Linear2':
        model = GestureTypePredictorLinear2(num_class=num_class).to(device)
    elif model_mode == 'Linear3':
        model = GestureTypePredictorLinear3(num_class=num_class).to(device)
    elif model_mode == 'Linear8':
        model = GestureTypePredictorLinear8(num_class=num_class).to(device)
    elif model_mode == 'Linear9':
        model = GestureTypePredictorLinear9(num_class=num_class).to(device)
    elif model_mode == 'Linear10':
        model = GestureTypePredictorLinear10(num_class=num_class).to(device)
    elif model_mode == 'Linear20':
        model = GestureTypePredictorLinear20(num_class=num_class).to(device)
    load_checkpoint(model_path, model)


    # Input
    if clip_id:
        ted_data = torch.load(ted_dataset_path)
        for i,d in enumerate(ted_data):
            if d['vid'] == clip_id[:11]:
                print(i)
                break
        clip_number = int(clip_id[12:])
        words = [w[0] for w in ted_data[i]['clips'][clip_number]['words']]
        input_text = ' '.join(words)

    tokens = np.zeros([1, 256])
    for i,t in enumerate(tokenizer(input_text)['input_ids']):
        tokens[0][i] = t

    print(input_text)
    tokens = torch.Tensor(tokens).type(torch.LongTensor)
    tokens = tokens.to(device)

    all_token = tokenizer.convert_ids_to_tokens(tokens[0])
    text = []
    for j in range(len(all_token)):
        if all_token[j] == tokenizer.pad_token:
            break
        text.append(all_token[j])

    # Label
    if clip_id:
        y_true = data[data['clip_id']==clip_id].iloc[0]['label']

    # Output
    model.eval()
    output, atten = model(tokens)
    y_pred = output[0][:len(text)]
    y_pred = y_pred.to('cpu').detach().numpy().copy().T

    if clip_id:
        visualizeLabel(text, y_pred, y_true)
    else:
        visualizeLabel(text, y_pred)


    # # Save Images
    # save_dir = './images/prediction/{}_{}_{}_relu/'.format(data_kind, model_mode, mode1)
    # print("Save Images to --> ", save_dir)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # # Save Confusion Matrix
    # save_path = save_dir + 'cm.png' 
    # showConfusionMatrix(yt, yp, labels = ['No-Gesture', 'Non-Imagistic', 'Imagistic'], save_path=save_path)

    # # Save Prediction Results
    # print('Saving Images...')
    # for i in tqdm(range(len(eval_data))):
    #     save_path = save_dir + eval_data['vid_id'].iloc[i] + '_' + str(eval_data['clip_id'].iloc[i]) + '.png' 
    #     if os.path.exists(save_path):
    #         continue
    #     visualizeLabel(texts[i], y_true[i], y_pred[i], save_path=save_path)




