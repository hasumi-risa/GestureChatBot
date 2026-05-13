import torch
import pandas as pd
from sklearn.model_selection import train_test_split

data_path = "./data/preprocessed_data_all_class_augmented.pth"
save_dir = "./data"
train_test_ratio = 0.80
train_valid_ratio = 0.90    # Train : Valid : Test = 72 : 8 : 20

data = torch.load(data_path)
df = pd.DataFrame(data)

# Train-test split
df_full_train, df_test = train_test_split(df, train_size = train_test_ratio, random_state = 1)

# Train-valid split
df_train, df_valid = train_test_split(df_full_train, train_size = train_valid_ratio, random_state = 1)

torch.save(df_train, save_dir + '/train_class_augmented.pth')
torch.save(df_valid, save_dir + '/valid_class_augmented.pth')
torch.save(df_test, save_dir + '/test_class_augmented.pth')
print("Saved")