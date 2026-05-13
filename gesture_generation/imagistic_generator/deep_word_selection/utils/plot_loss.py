import matplotlib.pyplot as plt
import torch

option1 = '20210726'
model_mode = 'Linear'
model_dir = './model/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_metrics(load_path):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


# ------------ Show Loss Graph ------------
# train_loss_list, valid_loss_list, global_steps_list = load_metrics(model_dir + '/metrics_{}_{}.pt'.format(option1, model_mode))
# plt.plot(global_steps_list, train_loss_list, label='Train')
# plt.plot(global_steps_list, valid_loss_list, label='Valid')
# plt.xlabel('Global Steps')
# plt.ylabel('Loss')
# plt.legend()
# plt.show() 


# loss graph by epoch
max_epoch = 50
train_loss_list, valid_loss_list, global_steps_list = load_metrics(model_dir + '/metrics_{}_{}.pt'.format(option1, model_mode))
step_per_epoch = global_steps_list[-1] / max_epoch
epochs = [(n-1) / step_per_epoch for n in global_steps_list]
plt.plot(epochs, train_loss_list, label='Train')
plt.plot(epochs, valid_loss_list, label='Valid')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.ylim(-0.00, 0.25)
plt.legend()
plt.show() 