import wave
import random
import numpy as np
import pandas as pd

import corenlp
import torch
from transformers import BertTokenizer, BertModel
from sklearn.neighbors import NearestNeighbors

from .text2speech import text2speech
from .split_sentence import splitSentence


def selectByBERTNN(segmented_words, knn_model, tokenizer, bert_model, gesture_list, distance_list, remark_list, cluster_index, audio_time, fps, mode='Gentle'):
    gesture_seq = []            # gesture seaquence
    original_remark_seq = []    # original remark sequence
    nn_remark_seq = []          # NN remark sequence
    cluster_seq = []            # cluster index sequence
    dists_seq = []             # distances between nearest phrases
    order = []
    for i,sw in enumerate(segmented_words):
        order.append(i)
        text = ' '.join(sw)

        # Word Embedding
        sw.insert(0, "[CLS]")
        sw.append("[SEP]")
        tokens = tokenizer.convert_tokens_to_ids(sw)
        tokens_tensor = torch.tensor([tokens])
        with torch.no_grad(): # 勾配計算なし
            all_encoder_layers = bert_model(tokens_tensor)
        embedding = all_encoder_layers[0]
        cls = embedding[:,0,:][0].numpy()
            
        # select cluster by nearest neighbor
        dists, index = knn_model.kneighbors(cls.reshape(1, -1))
        dists_seq.append(dists)
        index = index[0][0]
        nn_remark_seq.append(remark_list[index])
        original_remark_seq.append(text)
        cluster = cluster_index[index]
        cluster_seq.append(cluster)

        # Randomly select gesture from clusters according to a normal distribution with mean 0 and standard deviation np.std(dist)
        dist = distance_list[cluster]
        rand = np.random.normal(loc=0, scale=np.std(dist))
        idx = np.argmin(np.abs(np.array(dist) - rand))
        gesture_seq.append(gesture_list[cluster][idx])

    df = pd.DataFrame({
        "original_words": original_remark_seq,
        "nn_words": nn_remark_seq,
        "clusters": cluster_seq,
        "gestures": gesture_seq,
        "distances": dists_seq,
        "order": order
    })

    # Decide which gestures to output in order of distance.
    gesture_seq = []        
    original_remark_seq = []
    nn_remark_seq = []      
    cluster_seq = []        
    average_time = np.mean(np.array([d.shape[0] for d in df['gestures']]))
    gesture_num = int((audio_time * fps) / average_time)
    if mode == 'Aggressive':
        gesture_num += 2
    word_interval = int(len(df)/gesture_num) + 1
    for i in range(0, len(df), word_interval):
        d = df.iloc[i:i+word_interval].sort_values('distances').iloc[0]
        gesture_seq.append(d['gestures'])
        original_remark_seq.append(d['original_words'])
        nn_remark_seq.append(d['nn_words'])
        cluster_seq.append(d['clusters'])
    
    return gesture_seq, original_remark_seq, nn_remark_seq, cluster_seq

def selectByRandom(segmented_words, knn_model, tokenizer, bert_model, gesture_list, distance_list, remark_list, cluster_index, audio_time, fps, mode='Gentle'):
    gesture_seq = []            # gesture seaquence
    original_remark_seq = []    # original remark sequence
    nn_remark_seq = []          # NN remark sequence
    cluster_seq = []            # cluster index sequence

    average_time = 40
    gesture_num = int((audio_time * fps) / average_time)
    if mode == 'Aggressive':
        gesture_num += 1
    word_interval = int(len(segmented_words)/gesture_num) + 1

    # Random Word Selection
    selected_words = []
    for i in range(0, len(segmented_words), word_interval):
        selected_words.append(random.choice(segmented_words[i:i+word_interval]))


    for i,sw in enumerate(selected_words):
        text = ' '.join(sw)

        # Word Embedding
        sw.insert(0, "[CLS]")
        sw.append("[SEP]")
        tokens = tokenizer.convert_tokens_to_ids(sw)
        tokens_tensor = torch.tensor([tokens])
        with torch.no_grad(): # 勾配計算なし
            all_encoder_layers = bert_model(tokens_tensor)
        embedding = all_encoder_layers[0]
        cls = embedding[:,0,:][0].numpy()
            
        # select cluster by nearest neighbor
        dists, index = knn_model.kneighbors(cls.reshape(1, -1))
        index = index[0][0]
        nn_remark_seq.append(remark_list[index])
        original_remark_seq.append(text)
        cluster = cluster_index[index]
        cluster_seq.append(cluster)

        # Randomly select gesture from clusters according to a normal distribution with mean 0 and standard deviation np.std(dist)
        dist = distance_list[cluster]
        rand = np.random.normal(loc=0, scale=np.std(dist))
        idx = np.argmin(np.abs(np.array(dist) - rand))
        gesture_seq.append(gesture_list[cluster][idx])

    return gesture_seq, original_remark_seq, nn_remark_seq, cluster_seq


def selectByTFIDF(weights, segmented_words, knn_model, 
    tokenizer, bert_model, gesture_list, distance_list, 
    remark_list, cluster_index, audio_time, mode='Gentle'):

    gesture_seq = []            # gesture seaquence
    original_remark_seq = []    # original remark sequence
    nn_remark_seq = []          # NN remark sequence
    cluster_seq = []            # cluster index sequence

    average_time = 50/25  # 50 frame , 25 fps
    gesture_num = int(audio_time / average_time)
    if gesture_num == 0:
        gesture_num = 1
    if mode == 'Aggressive':
        gesture_num += 1
    if mode == 'Very Aggressive':
        gesture_num += 2
    word_interval = int(len(segmented_words)/gesture_num) + 1

    print('Gesture Num: ', gesture_num)

    segmented_weights = []
    word_count = 0
    for sw in segmented_words:
        segmented_weights.append(max(weights[word_count:word_count+len(sw)]))
        word_count += len(sw)
    
    selected_words = []
    for i in range(0, len(segmented_words), word_interval):
        index = np.argmax(segmented_weights[i:i+word_interval])
        selected_words.append(segmented_words[i:i+word_interval][index])

    for i,sw in enumerate(selected_words):
        text = ' '.join(sw)

        # Word Embedding
        sw.insert(0, "[CLS]")
        sw.append("[SEP]")
        tokens = tokenizer.convert_tokens_to_ids(sw)
        tokens_tensor = torch.tensor([tokens])
        with torch.no_grad(): # 勾配計算なし
            all_encoder_layers = bert_model(tokens_tensor)
        embedding = all_encoder_layers[0]
        cls = embedding[:,0,:][0].numpy()
            
        # select cluster by nearest neighbor
        dists, index = knn_model.kneighbors(cls.reshape(1, -1))
        index = index[0][0]
        nn_remark_seq.append(remark_list[index])
        original_remark_seq.append(text)
        cluster = cluster_index[index]
        cluster_seq.append(cluster)

        # Randomly select gesture from clusters according to a normal distribution with mean 0 and standard deviation np.std(dist)
        dist = distance_list[cluster]
        rand = np.random.normal(loc=0, scale=np.std(dist))
        idx = np.argmin(np.abs(np.array(dist) - rand))
        gesture_seq.append(gesture_list[cluster][idx])

    return gesture_seq, original_remark_seq, nn_remark_seq, cluster_seq


def selectGesture(gesture_words, knn_model, 
    tokenizer, bert_model, gesture_list, distance_list, 
    laban_list, cluster2index, index2cluster, audio_time, mode='Gentle', fps=25):
    
    gesture_seq = []            # gesture seaquence
    laban_seq = []            # laban seaquence
    indexes = []  

    # average_time = 50/fps  # 50 frame , 25 fps
    # gesture_num = int(audio_time / average_time)
    # if gesture_num == 0:
    #     gesture_num = 1
    # if mode == 'Aggressive':
    #     gesture_num += 1
    # if mode == 'Very Aggressive':
    #     gesture_num += 2

    for i,gw in enumerate(gesture_words):
        if gw[0] == '[CLS]' and len(gw) == 1:
            continue

        text = ' '.join(gw)

        # Word Embedding
        if not "[CLS]" in gw:
            gw.insert(0, "[CLS]")
        if not "[SEP]" in gw:
            gw.append("[SEP]")
        tokens = tokenizer.convert_tokens_to_ids(gw)
        tokens_tensor = torch.tensor([tokens])
        with torch.no_grad(): # 勾配計算なし
            all_encoder_layers = bert_model(tokens_tensor)
        embedding = all_encoder_layers[0]
        cls = embedding[:,0,:][0].numpy()
            
        # select cluster by nearest neighbor
        dists, index = knn_model.kneighbors(cls.reshape(1, -1))
        index = index[0][0]
        indexes.append(index)
        cluster = index2cluster[index]

        # cluster_index = np.array(cluster2index[cluster])
        # distmat_idx = np.where(cluster_index==index)

        # cluster_distmat[cluster][distmat_idx]

        # # Select the gesture with the closest audio duration from the cluster
        arr = []
        time = audio_time * fps / len(gesture_words)
        for i in range(len(gesture_list[cluster])):
            arr.append(np.abs(len(gesture_list[cluster][i]) - time))
        idx = np.argmin(arr)

        # Randomly select gesture from clusters according to a normal distribution with mean 0 and standard deviation np.std(dist)
        # dist = distance_list[cluster]
        # rand = np.random.normal(loc=0, scale=np.std(dist))
        # arr = np.abs(np.array(dist) - rand)
        # idx = np.argmin(arr)


        # # Select the smallest n_keyframe gesture (for MSRABot)
        # kf_len_list = np.array([len(laban[list(laban.keys())[0]].keys()) for laban in laban_list[cluster]])
        # idx = random.choice(np.where(kf_len_list == kf_len_list.min())[0])


        gesture_seq.append(gesture_list[cluster][idx])
        laban_seq.append(laban_list[cluster][idx])

    return gesture_seq, laban_seq, indexes

if __name__ == '__main__':
    input_text = "all I have to do is build a platform and all these people are going to put their stuff on top and I sit back and roll it in ?"

    gesture_library = "./data/gesture_library_k=30_BERT.npy"
    nlp_base_dir = './src/NLP/'
    corenlp_dir = nlp_base_dir + "stanford-corenlp-full-2013-06-20/"
    properties_file = nlp_base_dir + "user.properties"
    video_save_dir = "./output/videos/"
    pose_save_dir = "./output/csv/"
    sentence_segment = 4
    interpolate_frame = 10
    fps = 25


    print('Loading Gesture Library...')
    geslib = np.load(gesture_library, allow_pickle=True)
    gesture_list = geslib.item().get('gesture_list')
    distance_list = geslib.item().get('distance_list')
    remark_list = geslib.item().get('remark_list')
    freq_list = geslib.item().get('frequency_list')
    embvecs = geslib.item().get('emb_vectors')
    cluster_index = geslib.item().get('cluster_index')
    mean_pose = geslib.item().get('mean_pose')

    # NN setting
    knn_model = NearestNeighbors(n_neighbors=1).fit(embvecs) 

    # BERT setting
    options_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(options_name)
    bert_model = BertModel.from_pretrained(options_name)
    bert_model.eval()

    # Create audio file from input text
    input_audio = './.tmp/input_audio.wav'
    text2speech(input_text, input_audio)

    # Adjustment Gestures to Input Audio
    wr = wave.open(input_audio, 'r')
    fr = wr.getframerate()
    fn = wr.getnframes()
    audio_time = fn / fr - 0.7

    # Sentence Segmentation (Stanford Parser)
    parser = corenlp.StanfordCoreNLP(
        corenlp_path=corenlp_dir,
        properties=properties_file)
    segmented_words = splitSentence(input_text, max_word_num=sentence_segment, parser=parser)