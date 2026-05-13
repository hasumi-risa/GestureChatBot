
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
    tag_tensor = torch.zeros((len(data), MAX_SEQ_LEN))
    token_length = []
    for i in range(len(data)):
        token = data['token_ids'].iloc[i]
        token_length.append(len(token))
        for j in range(len(token)):
            text_tensor[i][j] = token[j]
        for j in range(len(tag[i])):
            tag_tensor[i][j] = tag[i][j]

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

# Evaluation Function
def evaluate(model, test_loader, interval=1):
    texts, y_pred, y_true, attens = [], [], [], []
    model.eval()
    cnt, miss_num, all_num = 0, 0, 0
    print("Evaluating...")
    with torch.no_grad():
        for text, labels in tqdm(test_loader):
            labels = labels.to(device)
            text = text.type(torch.LongTensor)
            text = text.to(device)
                
            output, atten = model(text)
            _, predicted = torch.max(output, 2)
            
            # texts.append
            text_len = []
            for i in range(len(text)):
                all_token = tokenizer.convert_ids_to_tokens(text[i])
                tmp = []
                for j in range(len(all_token)):
                    if all_token[j] == tokenizer.pad_token:
                        break
                    tmp.append(all_token[j])
                text_len.append(len(tmp))

                if cnt % interval == 0:
                    texts.append(tmp)
            
            for i in range(len(predicted)):
                miss_num += torch.count_nonzero(predicted[i][:text_len[i]] - labels[i][:text_len[i]])
                all_num += text_len[i]

                if cnt % interval == 0:
                    y_pred.append(predicted[i][:text_len[i]].tolist())
                    y_true.append(labels[i][:text_len[i]].tolist())

                    # 最初の層のAttentionを取り出し
                    # attens.append(atten[0].tolist())

            cnt += 1

    return texts, y_true, y_pred, attens
    
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
def visualizeClassLabel(text, y_true, y_pred, save_path=None):
    x = np.arange(len(text))

    class_num = 3
    y_true_each, y_pred_each = [], []
    for i in range(class_num):
        y_true_each.append(np.zeros(len(y_true)))
        y_pred_each.append(np.zeros(len(y_pred)))
    for i in range(class_num):
        for w in range(len(y_true)):
            if y_true[w] == i:
                y_true_each[i][w] = 1
            if y_pred[w] == i:
                y_pred_each[i][w] = 0.7

    plt.figure()

    plt.plot(x, y_pred_each[0][:len(text)], label='No-Gesture(Predict)', color='gray', alpha=1)
    plt.plot(x, y_pred_each[1][:len(text)], label='Beat(Predict)', color='blue', alpha=1)
    plt.plot(x, y_pred_each[2][:len(text)], label='Imagistic(Predict)', color='red', alpha=1)
    plt.plot(x, y_true_each[0][:len(text)], label='No-Gesture(GT)', color='gray', alpha=0.5, linestyle = "dashed")
    plt.plot(x, y_true_each[1][:len(text)], label='Beat(GT)', color='blue', alpha=0.5, linestyle = "dashed")
    plt.plot(x, y_true_each[2][:len(text)], label='Imagistic(GT)', color='red', alpha=0.5, linestyle = "dashed")

    plt.xticks(x,text)
    plt.xticks(rotation=90)
    plt.yticks([0.7, 1],['Prediction', 'GT'])
    plt.ylim(0,1.7)
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
        yt.append(labels[int(y_true[i])])
        yp.append(labels[int(y_pred[i])])
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
            

if __name__ == '__main__':
    # ------------ Parameters ------------
    data_kind = "test"      # test or train
    option1 = 'class_augmented'
    option2 = '1batch'
    model_kind = 'train'       # valid or train

    # model_modes = ['LSTM', 'Linear', 'Linear2', 'Linear3', 'Linear8', 'Linear9', 'Linear10', 'Linear20']
    model_modes = ['Linear']
    
    mses, accs, f1s = [], [], []
    for model_mode in model_modes:
        print(model_mode)

        data_dir = './data/'
        model_path = './model/model_{}_{}_{}.pt'.format(model_kind, option1, model_mode) 
        batch_size = 4
        MAX_SEQ_LEN = 256

        # Preliminaries
        # if option2[:6] == '3class':
        #     num_class = 3
        num_class = 3


        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Device : ", device)
        
        if data_kind == 'test':
            data_iter = createDataLoader(data_dir + 'test_class_20210706.pth'.format(option1))
        elif data_kind == 'train':
            data_iter = createDataLoader(data_dir + 'train_{}.pth'.format(option1 + '_balanced'))


        # ------------ Evaluation  ------------
        eval_data = torch.load(data_dir + '{}_{}.pth'.format(data_kind, option1))
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

        texts, y_true, y_pred, attens = evaluate(model, data_iter)

        y_true_all = [int(y) for yt in y_true for y in yt]
        y_pred_all = [int(y) for yp in y_pred for y in yp]

        print('Accuracy : ', accuracy_score(y_true_all, y_pred_all))
        print('F1 Score: ', f1_score(y_true_all, y_pred_all, average='macro'))

        # Save Images
        save_dir = './images/prediction_class/{}_{}_{}_{}_tcl/'.format(data_kind, model_mode, option1, model_kind)
        print("Save Images to --> ", save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save Confusion Matrix
        save_path = save_dir + 'cm.png' 
        showConfusionMatrix(y_true_all, y_pred_all, labels = ['No-Gesture', 'Beat', 'Imagistic'], save_path=save_path)

        # Save Prediction Results
        print('Saving Images...')
        for i in tqdm(range(len(texts))):
            save_path = save_dir + eval_data['clip_id'].iloc[i] + '.png' 
            if os.path.exists(save_path):
                continue
            visualizeClassLabel(texts[i], y_true[i], y_pred[i], save_path=save_path)

    # metrics = pd.DataFrame({'model':model_modes, 'mse':mses, 'acc':accs, 'f1':f1s})
    # metrics.to_excel('./images/evaluation.xlsx')

    print('finish')

    # # confirm attention
    # attentions = []
    # for i in range(len(attens)):
    #     # sum 12 heads
    #     all_attens = attens[i][0]
    #     for j in range(1, len(attens[i])):
    #         all_attens += attens[i][j]
    #     attentions.append(all_attens)
    # attentions = np.array(attentions)

    # save_dir = './Images/3class/{}_{}_attention/'.format(data_kind, model_mode)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # for i in tqdm(range(len(eval_data))):
    #     save_path = save_dir + eval_data['vid_id'].iloc[i] + '_' + str(eval_data['clip_id'].iloc[i]) + '_attention.png' 
    #     showAttention(texts[i], attention6s[i], save_path=save_path)




