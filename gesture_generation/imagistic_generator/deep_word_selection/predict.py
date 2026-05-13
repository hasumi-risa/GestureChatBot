from operator import index
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
import scipy.stats
from sklearn import preprocessing

from .model import GestureWordPredictor


class GestureWordPrediction:
    def __init__(self, model_path, gesture_word_thresh=0.6):
        self.MAX_SEQ_LEN = 256
        self.GESTURE_WORD_THRESH = gesture_word_thresh
        self.model_path = model_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = GestureWordPredictor().to(self.device)  
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict['model_state_dict'])

    def predict(self, text):
        PAD_INDEX = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.model.eval()
        text_id = self.tokenizer.encode(text)
        text_len = len(text_id)
        tokens = self.tokenizer.convert_ids_to_tokens(text_id)
        for i in range(len(text_id), self.MAX_SEQ_LEN):
            text_id.append(PAD_INDEX)
        text_id = torch.tensor([text_id]).type(torch.LongTensor).to(self.device)
        with torch.no_grad():
            output = self.model(text_id)
        predicted = output.reshape(output.shape[0], -1)[0]
        pred = predicted[:text_len].to('cpu').detach().numpy().copy()
                
        pred_zscore = scipy.stats.zscore(pred)
        pred = preprocessing.minmax_scale(pred_zscore)

        return pred, tokens

    def extract_gesture_words(self, text):
        pred, tokens = self.predict(text)
        gesture_words = []
        isGestureWord = False
        for i in range(len(pred)):
            if pred[i] >= self.GESTURE_WORD_THRESH:
                if isGestureWord:
                    tmp.append(tokens[i])
                else:
                    tmp = [tokens[i]]
                    isGestureWord = True
            else:
                if isGestureWord:
                    gesture_words.append(tmp)
                    isGestureWord = False
        return gesture_words, pred, tokens
    
    def visualizePrediction(self, pred, tokens, save_path=None):
        fig, ax = plt.subplots()
        sns.heatmap(data=pred.reshape(-1, 1), cmap='OrRd', annot=True, vmin=0, vmax=1)
        ax.set_xticklabels(['Predicted'], rotation=0)
        ax.set_yticks(np.arange(len(pred)))
        ax.set_yticklabels(tokens, rotation=0)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    input_text = "business makes more money if they don't have a safe working environment. "
    input_text = "business makes more money if they dont have a safe workoutputing environment . thats been the conventional wisdom ."
    input_text = "thats been the conventional wisdom . if they dont have a safe working environment, business makes more money ."
    input_text = "company makes more money if I didnt have a safe working environment. I have the conventional wisdom."
    input_text = "company creates more money if we have a safe working space. I have wisdom."
    input_text = "company destroys more money if we have a safe resting space. thats been the knowledge."
    input_text = "so what is evolution's answer to the of uncertainty? that's been the conventional wisdom."


    model_path = './model_data/model_train_20210726_Linear.pt'


    gwp = GestureWordPrediction(model_path)
    gesture_words = gwp.extract_gesture_words(input_text)
    
    for i in range(len(gesture_words)):
        print(gesture_words[i])


