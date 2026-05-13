import nltk
import torch
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.neighbors import NearestNeighbors

from src.motion_interpolation import linear_interpolation
from src.plot3D import plot3D, animate3D
from src.plotPose import Plot, display_pose

def plotUpperBody2D(pose3d, save_path, words=None, clusters=None):
    upper_idx = [2, 20, 4, 5, 6, 8, 9, 10]
    
    pose3d = pose3d[:, upper_idx]
    pose2d = np.delete(pose3d, 2, 2).reshape([-1, len(upper_idx)*2])

    # rotate y axis
    # for frame in range(len(pose2d)):
    #     for joint in range(len(pose2d[frame])):
    #         pose2d[frame][joint][1] *= -1

    if words:
        poses = []
        for i in range(len(pose2d)):
            pose = []
            for j in range(len(pose2d[i])):
                pose.append(pose2d[i][j])
            pose.append(words[i])
            pose.append(clusters[i])
            poses.append(pose)


    # time = np.arange(len(pose2d))
    # time = time.reshape(-1, 1)
    # pose2d = np.concatenate([pose2d, time], axis=1)

    p = Plot((-1, 1), (-1, 1))
    anim = p.animate(poses, 1000/25)
    p.save(anim, save_path, fps=25)


if __name__ == '__main__':

    input_text = "all I have to do is build a platform and all these people are going to put their stuff on top and I sit back and roll it in ?"
    gesture_library = "../../Labanotation/LabanSuiteBeta/GestureAuthoringTools/LabanEditor/data_output/kmedoids/imgistic_by_phrase_k=21/imgistic_by_phrase_k=21.npy"
    glove_file = './data/glove.npy'
    interpolate_frame = 5


    valid_part_of_speech = ['DT', 'IN', 'JJ', 'JJS', 'NN', 'NNS', 'PDT', 'PRP$', 'RB', 'RBR', 'RBS', 
                        'RP', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WRB']
    invalid_word = ['a', 'an', 'the', 'of', 'for', 'by', 'am', 'are', 'is', 'was', 'were', 'be', 'being', 'been', 'it']

    print('Loading Gesture Library...')
    geslib = np.load(gesture_library, allow_pickle=True)
    gesture_list = geslib.item().get('gesture_list')
    word_list = geslib.item().get('word_list')
    embvecs = geslib.item().get('emb_vectors')
    cluster_index = geslib.item().get('cluster_index')
    mean_pose = geslib.item().get('mean_pose')

    print('Loading Glove...')
    glove_data = np.load(glove_file, allow_pickle=True)
    emb_vec = glove_data.item().get('emb_vec')
    word2idx = glove_data.item().get('word2idx')
    idx2word = glove_data.item().get('idx2word')
    del glove_data


    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    stemmer = nltk.stem.PorterStemmer()

    knn_model = NearestNeighbors(n_neighbors=1).fit(embvecs) 

    part_of_speech = nltk.pos_tag(nltk.word_tokenize(input_text))
    part_of_speech = [part_of_speech[i][1] for i in range(len(part_of_speech))]
    words = input_text.split()
    gesture_seq, word_seq, cluster_seq = [], [], []
    for i in range(len(words)):
        # word = stemmer.stem(words[i])
        word = words[i]
        if not part_of_speech[i] in valid_part_of_speech or word in invalid_word:
            continue
        word_seq.append(word)

        # nearest neighbor
        vec = emb_vec[word2idx[word]]
        dists, index = knn_model.kneighbors(vec.reshape(1, -1))

        cluster = cluster_index[index[0][0]]
        cluster_seq.append(cluster)
        gesture = gesture_list[cluster]
        gesture_seq.append(gesture)

    gesture = linear_interpolation(gesture_seq, interpolate_frame=interpolate_frame, mean_pose=mean_pose)

    # Merge gestures
    clusters, words = [], []
    for i in range(len(gesture_seq)):
        for j in range(len(gesture_seq[i])):
            clusters.append(cluster_seq[i])
            words.append(word_seq[i])

        if i == len(gesture_seq) - 1:
            break

        for j in range(interpolate_frame):
            clusters.append('interpolation')
            words.append(' ')

    save_path = "./tmp_interpolated_5frame.mp4"
    plotUpperBody2D(gesture, save_path, words=words, clusters=clusters)



    print()


