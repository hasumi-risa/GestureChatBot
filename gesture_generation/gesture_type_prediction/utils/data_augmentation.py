import pandas as pd

import torch
import nlpaug.augmenter.word as naw
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from nltk.tokenize import word_tokenize

def delete_space(text):
    stop_words = [' .', ' ,', ' !', ' ?']
    for sw in stop_words:
        text = text.replace(sw, sw[1:])
    return text

def add_space(text):
    stop_words = ['.', ',', '!', '?', ':', ';']
    replace_words = [' .', ' ,', ' !', ' ?', ' :', ' ;']
    for i in range(len(stop_words)):
        text = text.replace(stop_words[i], replace_words[i])
    return text

processed_data = "./data/preprocessed_data_all_class.pth"
save_path = "./data/preprocessed_data_all_class_augmented.pth"
scale_factor = 30


data = pd.DataFrame(torch.load(processed_data))
tokenizer = data['tokenizer'].iloc[0]
aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")

ignore_num, all_num = 0, 0
augmented_data = {'text': [], 'token_ids': [], 'label': [], 'video_id':[], 'clip_id':[]}
for i in tqdm(range(len(data))):
    text = add_space(data['text'].iloc[i])
    
    word_list = text.split()
    word_list.insert(0, tokenizer.cls_token)
    word_list.append(tokenizer.sep_token)
    word_ids = tokenizer(word_list)['input_ids']
    each_token_num = [len(t) - 2 for t in word_ids]

    label = data['label'].iloc[i]
    word_label = []
    sum_n = 0
    for n in each_token_num:
        word_label.append(label[sum_n])
        sum_n += n

    # Augmentation with BERT
    aug_texts = aug.augment(text, n=scale_factor)
    
    for aug_text in aug_texts:
        aug_text = add_space(aug_text)
        aug_word_list = aug_text.split()
        aug_word_list.insert(0, tokenizer.cls_token)
        aug_word_list.append(tokenizer.sep_token)

        if len(word_list) != len(aug_word_list):
            # print("aaa")
            ignore_num += 1
            all_num += 1
            continue

        aug_label, aug_ids = [], []
        for j in range(len(each_token_num)):
            aug_id = tokenizer(aug_word_list[j])['input_ids']
            aug_token_num = len(aug_id) - 2
            for k in range(aug_token_num):
                aug_label.append(word_label[j])
                aug_ids.append(aug_id[k+1])
        
        if len(aug_ids) != len(aug_label):
            ignore_num += 1
            all_num += 1
            print('bbb')
            continue

        augmented_data['text'].append(aug_text)
        augmented_data['token_ids'].append(aug_ids)
        augmented_data['label'].append(aug_label)
        augmented_data['video_id'].append(data['video_id'].iloc[i])
        augmented_data['clip_id'].append(data['clip_id'].iloc[i])
        all_num += 1
    
    augmented_data['text'].append(data['text'].iloc[i])
    augmented_data['token_ids'].append(data['token_ids'].iloc[i])
    augmented_data['label'].append(data['label'].iloc[i])
    augmented_data['video_id'].append(data['video_id'].iloc[i])
    augmented_data['clip_id'].append(data['clip_id'].iloc[i])

print('Ignored {}/{} data'.format(ignore_num, all_num))
print('Augmented {} --> {}'.format(len(data), len(augmented_data['text'])))
augmented_data['tokenizer'] = tokenizer
torch.save(augmented_data, save_path)
print("Saved to {}".format(save_path))