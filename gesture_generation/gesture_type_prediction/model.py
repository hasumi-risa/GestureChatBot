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

class GestureTypePredictor(nn.Module):

    def __init__(self, num_class=3):
        super().__init__()
        options_name = "bert-base-uncased"
        embed_dim = BertConfig.from_pretrained(options_name).hidden_size
        self.num_class = num_class
        self.bert_model = BertModel.from_pretrained(options_name)
        self.fc = nn.Linear(embed_dim, self.num_class)

    def forward(self, text):
        # embedding
        outputs = self.bert_model(text)
        embedded = outputs.last_hidden_state
        output = self.fc(embedded)
        return output

class GestureTypePredictorLSTM(nn.Module):

    def __init__(self, num_class=3, hidden_dim=128, bidirectional=True):
        super().__init__()
        options_name = "bert-base-uncased"
        embed_dim = BertConfig.from_pretrained(options_name).hidden_size
        self.num_class = num_class
        self.bert_model = BertModel.from_pretrained(options_name)
        if bidirectional:
            self.lstm = nn.LSTM(embed_dim, int(hidden_dim/2), bidirectional=True)
            self.fc = nn.Linear(hidden_dim, self.num_class)
        else:
            self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=False)
            self.fc = nn.Linear(hidden_dim, self.num_class)
        

    def forward(self, text):
        # embedding
        embedded, _, attentions = self.bert_model(text, output_attentions=True)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output, attentions


class GestureTypePredictorLinear2(nn.Module):
    def __init__(self, num_class=3):
        super().__init__()
        options_name = "bert-base-uncased"
        embed_dim = BertConfig.from_pretrained(options_name).hidden_size
        self.num_class = num_class
        self.bert_model = BertModel.from_pretrained(options_name)
        self.fc1 = nn.Linear(embed_dim, 128)        # 768 -> 128
        self.fc2 = nn.Linear(128, self.num_class)   # 128 -> 3

    def forward(self, text):
        # embedding
        embedded, _, attentions = self.bert_model(text, output_attentions=True)
        # CLS = embedded[:,0,:]
        fc1_out = F.relu(self.fc1(embedded))
        fc2_out = self.fc2(fc1_out)
        return fc2_out, attentions

class GestureTypePredictorLinear3(nn.Module):
    def __init__(self, num_class=3):
        super().__init__()
        options_name = "bert-base-uncased"
        embed_dim = BertConfig.from_pretrained(options_name).hidden_size
        self.num_class = num_class
        self.bert_model = BertModel.from_pretrained(options_name)
        self.fc1 = nn.Linear(embed_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, self.num_class)

    def forward(self, text, epoch=None):
        # embedding
        embedded, _, attentions = self.bert_model(text, output_attentions=True)
        # CLS = embedded[:,0,:]
        fc1_out = F.relu(self.fc1(embedded))
        fc2_out = F.relu(self.fc2(fc1_out))
        fc3_out = self.fc3(fc2_out)

        if epoch is not None:
            if epoch % 5 == 0:
                save_path = "./Images/vector/linear3_1batch/epoch{}/".format(epoch)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                idx = 0
                visualizeVector(fc1_out, text, idx, save_path=save_path+'linear3_1.png')
                visualizeVector(fc2_out, text, idx, save_path=save_path+'linear3_2.png')
                visualizeVector(fc3_out, text, idx, save_path=save_path+'linear3_3.png')

        return fc3_out, attentions


class GestureTypePredictorLinear8(nn.Module):
    def __init__(self, num_class=3):
        super().__init__()
        options_name = "bert-base-uncased"
        embed_dim = BertConfig.from_pretrained(options_name).hidden_size
        self.num_class = num_class
        self.bert_model = BertModel.from_pretrained(options_name)
        self.fc1 = nn.Linear(embed_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, 8)
        self.fc8 = nn.Linear(8, self.num_class)

    def forward(self, text):
        # embedding
        embedded, _, attentions = self.bert_model(text, output_attentions=True)
        # CLS = embedded[:,0,:]
        x = F.relu(self.fc1(embedded))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x, attentions

class GestureTypePredictorLinear9(nn.Module):
    def __init__(self, num_class=3):
        super().__init__()
        options_name = "bert-base-uncased"
        embed_dim = BertConfig.from_pretrained(options_name).hidden_size
        self.num_class = num_class
        self.bert_model = BertModel.from_pretrained(options_name)
        self.fc1 = nn.Linear(embed_dim, 512)
        self.fc2 = nn.Linear(512, 340)
        self.fc3 = nn.Linear(340, 170)
        self.fc4 = nn.Linear(170, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, 8)
        self.fc9 = nn.Linear(8, self.num_class)

    def forward(self, text):
        # embedding
        embedded, _, attentions = self.bert_model(text, output_attentions=True)
        # CLS = embedded[:,0,:]
        x1 = F.relu(self.fc1(embedded))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        x4 = F.relu(self.fc4(x3))
        x5 = F.relu(self.fc5(x4))
        x6 = F.relu(self.fc6(x5))
        x7 = F.relu(self.fc7(x6))
        x8 = F.relu(self.fc8(x7))
        x9 = self.fc9(x8)
        return x9, attentions, x1

class GestureTypePredictorLinear10(nn.Module):
    def __init__(self, num_class=3):
        super().__init__()
        options_name = "bert-base-uncased"
        embed_dim = BertConfig.from_pretrained(options_name).hidden_size
        self.num_class = num_class
        self.bert_model = BertModel.from_pretrained(options_name)
        self.fc1  = nn.Linear(embed_dim, 512)
        self.fc2  = nn.Linear(512, 340)
        self.fc3  = nn.Linear(340, 170)
        self.fc4  = nn.Linear(170, 128)
        self.fc5  = nn.Linear(128, 96)
        self.fc6  = nn.Linear(96, 64)
        self.fc7  = nn.Linear(64, 32)
        self.fc8  = nn.Linear(32, 16)
        self.fc9  = nn.Linear(16, 8)
        self.fc10 = nn.Linear(8, self.num_class)

    def forward(self, text, epoch=None):
        # embedding
        embedded, _, attentions = self.bert_model(text, output_attentions=True)
        # CLS = embedded[:,0,:]

        fc1_out = F.relu(self.fc1(embedded))
        fc2_out = F.relu(self.fc2(fc1_out))
        fc3_out = F.relu(self.fc3(fc2_out))
        fc4_out = F.relu(self.fc4(fc3_out))
        fc5_out = F.relu(self.fc5(fc4_out))
        fc6_out = F.relu(self.fc6(fc5_out))
        fc7_out = F.relu(self.fc7(fc6_out))
        fc8_out = F.relu(self.fc8(fc7_out))
        fc9_out = F.relu(self.fc9(fc8_out))
        fc10_out = self.fc10(fc9_out)

        if epoch is not None:
            if epoch % 5 == 0:
                save_path = "./Images/vector/linear10_1batch/epoch{}/".format(epoch)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                idx = 0
                visualizeVector(fc1_out , text, idx, save_path=save_path+'linear10_1.png')
                visualizeVector(fc2_out , text, idx, save_path=save_path+'linear10_2.png')
                visualizeVector(fc3_out , text, idx, save_path=save_path+'linear10_3.png')
                visualizeVector(fc4_out , text, idx, save_path=save_path+'linear10_4.png')
                visualizeVector(fc5_out , text, idx, save_path=save_path+'linear10_5.png')
                visualizeVector(fc6_out , text, idx, save_path=save_path+'linear10_6.png')
                visualizeVector(fc7_out , text, idx, save_path=save_path+'linear10_7.png')
                visualizeVector(fc8_out , text, idx, save_path=save_path+'linear10_8.png')
                visualizeVector(fc9_out , text, idx, save_path=save_path+'linear10_9.png')
                visualizeVector(fc10_out, text, idx, save_path=save_path+'linear10_10.png')
        
        return fc10_out, attentions

class GestureTypePredictorLinear20(nn.Module):
    def __init__(self, num_class=3):
        super().__init__()
        options_name = "bert-base-uncased"
        embed_dim = BertConfig.from_pretrained(options_name).hidden_size
        self.num_class = num_class
        self.bert_model = BertModel.from_pretrained(options_name)
        self.fc1  = nn.Linear(embed_dim, 760)
        self.fc2  = nn.Linear(760, 720)
        self.fc3  = nn.Linear(720, 680)
        self.fc4  = nn.Linear(680, 640)
        self.fc5  = nn.Linear(640, 600)
        self.fc6  = nn.Linear(600, 560)
        self.fc7  = nn.Linear(560, 520)
        self.fc8  = nn.Linear(520, 480)
        self.fc9  = nn.Linear(480, 440)
        self.fc10 = nn.Linear(440, 400)
        self.fc11 = nn.Linear(400, 360)
        self.fc12 = nn.Linear(360, 320)
        self.fc13 = nn.Linear(320, 280)
        self.fc14 = nn.Linear(280, 240)
        self.fc15 = nn.Linear(240, 200)
        self.fc16 = nn.Linear(200, 160)
        self.fc17 = nn.Linear(160, 120)
        self.fc18 = nn.Linear(120, 80)
        self.fc19 = nn.Linear(80,  40)
        self.fc20 = nn.Linear(40, self.num_class)

    def forward(self, text):
        # embedding
        embedded, _, attentions = self.bert_model(text, output_attentions=True)
        # CLS = embedded[:,0,:]
        x = F.relu(self.fc1 (embedded))
        x = F.relu(self.fc2 (x))
        x = F.relu(self.fc3 (x))
        x = F.relu(self.fc4 (x))
        x = F.relu(self.fc5 (x))
        x = F.relu(self.fc6 (x))
        x = F.relu(self.fc7 (x))
        x = F.relu(self.fc8 (x))
        x = F.relu(self.fc9 (x))
        x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        x = F.relu(self.fc13(x))
        x = F.relu(self.fc14(x))
        x = F.relu(self.fc15(x))
        x = F.relu(self.fc16(x))
        x = F.relu(self.fc17(x))
        x = F.relu(self.fc18(x))
        x = F.relu(self.fc19(x))
        x = self.fc20(x)
        return x, attentions


def visualizeVector(data, text, idx, save_path=None):
    data = data[idx].to('cpu').detach().numpy()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text = tokenizer.convert_ids_to_tokens(text[idx])
    row_labels = [t for t in text if t != '[PAD]']
    data = data[:len(row_labels)]
    
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data.T, cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(data.shape[0]))
    ax.set_xticklabels(row_labels, rotation=90)
    plt.rcParams['figure.subplot.bottom'] = 0.15
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()