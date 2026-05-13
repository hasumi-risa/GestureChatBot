import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer
import matplotlib.pyplot as plt

class CustomLoss(nn.Module):
    def __init__(self, device, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.crossentropy = nn.CrossEntropyLoss().to(device)

    def forward(self, output, label):
        loss1 = self.crossentropy(output, label)
        output = torch.argmax(output, axis=1)
        loss2 = torch.count_nonzero(output[1:] - output[:-1]) / len(output)
        return loss1 + self.alpha + loss2

class GestureWordPredictor(nn.Module):

    def __init__(self):
        super().__init__()
        options_name = "bert-base-uncased"
        self.MAX_SEQ_LEN = 256
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        embed_dim = BertConfig.from_pretrained(options_name).hidden_size
        self.bert_model = BertModel.from_pretrained(options_name)
        self.fc = nn.Linear(embed_dim, 1)
        self.sm = nn.Softmax()

    def forward(self, text):
        # embedding
        outputs = self.bert_model(text)
        embedded = outputs.last_hidden_state
        # CLS = embedded[:,0,:]
        fc_out = self.fc(embedded)
        fc_out = fc_out.reshape(-1, self.MAX_SEQ_LEN)
        sm_out = torch.Tensor().to(self.device)
        for i in range(len(text)):
            fc_out[i][:torch.where(text[i]==0)[0][0]]
            o = self.sm(fc_out[i][:torch.where(text[i]==0)[0][0]])
            o = torch.cat([o, torch.zeros(self.MAX_SEQ_LEN - o.shape[0]).to(self.device)])
            sm_out = torch.cat((sm_out, o.reshape(1, -1)), 0)
        return sm_out

class GestureWordPredictorLSTM(nn.Module):

    def __init__(self, hidden_dim=128, bidirectional=True):
        super().__init__()
        options_name = "bert-base-uncased"
        embed_dim = BertConfig.from_pretrained(options_name).hidden_size
        self.bert_model = BertModel.from_pretrained(options_name)
        if bidirectional:
            self.lstm = nn.LSTM(embed_dim, int(hidden_dim/2), bidirectional=True)
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=False)
            self.fc = nn.Linear(hidden_dim, 1)
        self.sm = nn.Softmax()

    def forward(self, text):
        # embedding
        embedded, _, attentions = self.bert_model(text, output_attentions=True)
        lstm_out, _ = self.lstm(embedded)
        fc_out = self.fc(lstm_out)
        output = self.sm(fc_out)
        return output


class GestureWordPredictorLinear2(nn.Module):
    def __init__(self):
        super().__init__()
        options_name = "bert-base-uncased"
        embed_dim = BertConfig.from_pretrained(options_name).hidden_size
        self.bert_model = BertModel.from_pretrained(options_name)
        self.fc1 = nn.Linear(embed_dim, 128)        # 768 -> 128
        self.fc2 = nn.Linear(128, 1)   # 128 -> 1
        self.sm = nn.Softmax()

    def forward(self, text):
        # embedding
        embedded, _, attentions = self.bert_model(text, output_attentions=True)
        # CLS = embedded[:,0,:]
        fc1_out = F.relu(self.fc1(embedded))
        fc2_out = self.fc2(fc1_out)
        out = self.sm(fc2_out)
        return out