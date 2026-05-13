# Libraries
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Preliminaries
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Models
import torch.nn as nn
from transformers import BertTokenizer
from model import GestureTypePredictor, GestureTypePredictorLSTM
from model import GestureTypePredictorLinear2, GestureTypePredictorLinear3, GestureTypePredictorLinear8, GestureTypePredictorLinear9, GestureTypePredictorLinear10, GestureTypePredictorLinear20

# Training
import torch.optim as optim

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns



# ------------ Parameters ------------
data_mode = '3class_1batch'
model_mode = 'Linear2'
data_dir = './Data/'
model_dir = './Model/'
# model_path = model_dir + "model_train.pt" 
model_path = None
num_epochs = 50
batch_size = 1
lr = 2e-5
MAX_SEQ_LEN = 256
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preliminaries
if data_mode[:6] == '3class':
    num_class = 3
elif data_mode[:6] == '5class':
    num_class = 5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device : ", device)




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

train_iter = createDataLoader(data_dir + 'train_{}.pth'.format(data_mode[:6] + '_balanced'))
valid_iter = createDataLoader(data_dir + 'valid_{}.pth'.format(data_mode[:6]))



# ------------ Function for Training  ------------
# Save and Load Functions
def save_checkpoint(save_path, model, loss):
    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'loss': loss}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['loss']

def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_metrics(load_path):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

def removePadding(text, data):
    end_index = (text == 0).nonzero()[0][0]
    return data[0:end_index]


# Training Function
def train(model,
          optimizer,
          criterion,
          train_loader = train_iter,
          valid_loader = valid_iter,
          num_epochs = num_epochs,
          eval_every = len(train_iter) // 10,
          file_path = model_dir,
          best_valid_loss = float("Inf")):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    local_step = 0
    best_train_loss = float("Inf")
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        local_step = 0
        for text, labels in train_loader:
            labels = labels.transpose(1, 2)
            labels = labels.to(device)
            text = text.type(torch.LongTensor)
            text = text.to(device)

            if local_step == 0:
                output, attn = model(text, epoch)
            else:
                output, attn = model(text)
            output = output.to(device)

            batch_loss = 0
            for i in range(len(output)):
                out = removePadding(text[i], output[i])
                lab = removePadding(text[i], labels[i])
                loss = criterion(out, lab)
                optimizer.zero_grad()
                if i != len(output) - 1:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                optimizer.step()
                batch_loss += loss.item()

            # update running values
            running_loss += batch_loss
            global_step += 1
            local_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    

                    # validation loop
                    for text, labels in valid_loader:
                        labels = labels.transpose(1, 2)
                        labels = labels.to(device)
                        text = text.type(torch.LongTensor)
                        text = text.to(device)
                        
                        output, attn = model(text)
                        output = output.to(device)
                        loss = criterion(output, labels)

                        batch_loss = 0
                        for i in range(len(output)):
                            out = removePadding(text[i], output[i])
                            lab = removePadding(text[i], labels[i])
                            loss = criterion(out, lab)
                            batch_loss += loss.item()
                        
                        valid_running_loss += batch_loss

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + 'model_valid_{}_{}.pt'.format(data_mode, model_mode), model, best_valid_loss)
                if best_train_loss > average_train_loss:
                    best_train_loss = average_train_loss
                    save_checkpoint(file_path + 'model_train_{}_{}.pt'.format(data_mode, model_mode), model, best_train_loss)
    
    save_metrics(file_path + 'metrics_{}_{}.pt'.format(data_mode, model_mode), train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')




# ------------ Training ------------
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
if model_path:
    load_checkpoint(model_path, model)

# ファインチューニングの設定
# まずは全部OFF
# for param in model.parameters():
#     param.requires_grad = False
# # BERTの最後の層だけ更新ON
# for param in model.bert_title.encoder.layer[-1].parameters():
#     param.requires_grad = True
# for param in model.bert_content.encoder.layer[-1].parameters():
#     param.requires_grad = True
# 追加部分のところもON
# if model_mode == 'Linear':
#     for param in model.fc.parameters():
#         param.requires_grad = True
# elif model_mode == 'LSTM':
#     for param in model.lstm.parameters():
#         param.requires_grad = True
# elif model_mode == 'Linear2':
#     for param in model.fc1.parameters():
#         param.requires_grad = True
#     for param in model.fc2.parameters():
#         param.requires_grad = True
# elif model_mode == 'Linear3':
#     for param in model.fc1.parameters():
#         param.requires_grad = True
#     for param in model.fc2.parameters():
#         param.requires_grad = True
#     for param in model.fc3.parameters():
#         param.requires_grad = True
# elif model_mode == 'Linear8':
#     for param in model.fc1.parameters():
#         param.requires_grad = True
#     for param in model.fc2.parameters():
#         param.requires_grad = True
#     for param in model.fc3.parameters():
#         param.requires_grad = True
#     for param in model.fc4.parameters():
#         param.requires_grad = True
#     for param in model.fc5.parameters():
#         param.requires_grad = True
#     for param in model.fc6.parameters():
#         param.requires_grad = True
#     for param in model.fc7.parameters():
#         param.requires_grad = True
#     for param in model.fc8.parameters():
#         param.requires_grad = True


optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss().to(device)

train(model=model, optimizer=optimizer, criterion=criterion)




# ------------ Show Loss Graph ------------
train_loss_list, valid_loss_list, global_steps_list = load_metrics(model_dir + '/metrics_{}_{}.pt'.format(data_mode, model_mode))
plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Valid')
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.legend()
plt.show() 


# loss graph by epoch
step_per_epoch = 340
train_loss_list, valid_loss_list, global_steps_list = load_metrics(model_dir + '/metrics_{}_{}.pt'.format(data_mode, model_mode))
epochs = [(n-1) / step_per_epoch for n in global_steps_list]
plt.plot(epochs, train_loss_list, label='Train')
plt.plot(epochs, valid_loss_list, label='Valid')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show() 

