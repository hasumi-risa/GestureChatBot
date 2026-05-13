import os
import nltk
import torch
import wave
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from pyflann import FLANN, set_distance_type

from src.motion_interpolation import linear_interpolation
from src.plot3D import plot3D, animate3D
from src.plotPose import Plot, display_pose
from src.text2speech import text2speech

def plotUpperBody2D(pose3d, save_path, input_words=None, nn_words=None, clusters=None, fps=25):
    upper_idx = [2, 20, 4, 5, 6, 8, 9, 10]
    
    pose3d = pose3d[:, upper_idx]
    pose2d = np.delete(pose3d, 2, 2).reshape([-1, len(upper_idx)*2])

    # rotate y axis
    # for frame in range(len(pose2d)):
    #     for joint in range(len(pose2d[frame])):
    #         pose2d[frame][joint][1] *= -1

    poses = []
    for i in range(len(pose2d)):
        pose = []
        for j in range(len(pose2d[i])):
            pose.append(pose2d[i][j])
        pose.append(input_words[i])
        pose.append(nn_words[i])
        pose.append(clusters[i])
        poses.append(pose)


    # time = np.arange(len(pose2d))
    # time = time.reshape(-1, 1)
    # pose2d = np.concatenate([pose2d, time], axis=1)

    p = Plot((-0.75, 0.75), (-0.5, 1))
    anim = p.animate(poses, 1000/fps)
    p.save(anim, save_path, fps=fps)


if __name__ == '__main__':

    input_text = "all I have to do is build a platform and all these people are going to put their stuff on top and I sit back and roll it in?"
    # input_text = "I can dodge what I don't want and pull in what I want."
    # input_text = "they keep spinning with the same axis, indefinitely. Hubble kind of rotates around them, and so it can orient itself."
    # input_text = "a doughnut or a half-moon shape with a large, central hole."
    # input_text = "you'd see that picture of dog poop high up in the search results"

    gesture_library = "../../Labanotation/LabanSuiteBeta/GestureAuthoringTools/LabanEditor/data_output/kmedoids/by_gesture_k=16/gesture_library_k=16.npy"
    glove_file = './data/glove.npy'
    sentence_segment = 7
    interpolate_frame = 10


    valid_part_of_speech = ['DT', 'IN', 'JJ', 'JJS', 'NN', 'NNS', 'PDT', 'PRP$', 'RB', 'RBR', 'RBS', 
                        'RP', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WRB']
    invalid_word = ['a', 'an', 'the', 'of', 'for', 'by', 'in', 'that', 'am', 'are', 'is', 'was', 'were', 'be', 'being', 'been', 'it']

    print('Loading Gesture Library...')
    geslib = np.load(gesture_library, allow_pickle=True)
    gesture_list = geslib.item().get('gesture_list')
    word_list = geslib.item().get('word_list')
    word2idx_gl = geslib.item().get('word2idx')
    freq_list = geslib.item().get('frequency_list')
    embvecs = geslib.item().get('emb_vectors')
    cluster_index = geslib.item().get('cluster_index')
    mean_pose = geslib.item().get('mean_pose')

    print('Loading Glove...')
    glove_data = np.load(glove_file, allow_pickle=True)
    emb_vec = glove_data.item().get('emb_vec')
    word2idx_glove = glove_data.item().get('word2idx')
    del glove_data


    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    stemmer = nltk.stem.PorterStemmer()

    knn_model = NearestNeighbors(n_neighbors=1).fit(embvecs) 
    set_distance_type('manhattan') # defaultはeuclidean
    flann = FLANN()
    flann.build_index(embvecs)

    words = nltk.word_tokenize(input_text)
    part_of_speech = nltk.pos_tag(words)
    part_of_speech = [part_of_speech[i][1] for i in range(len(part_of_speech))]

    # Sentence Segmentation (uniformly)
    segmented_words, segmented_poss, ws, ps = [], [], [], []
    for i in range(len(words)):
        ws.append(words[i])
        ps.append(part_of_speech[i])
        if (i+1) % sentence_segment == 0 or i == len(words) - 1:
            segmented_words.append(ws)
            segmented_poss.append(ps)
            ws = []
            ps = []

    gesture_seq, original_word_seq, nn_word_seq, cluster_seq = [], [], [], []
    for sw, sp in zip(segmented_words, segmented_poss):
        nn_word_list_scikt, nn_word_list_flann = [], []
        for word, pos in zip(sw, sp):
            if not pos in valid_part_of_speech or word in invalid_word:
                nn_word_list_scikt.append(None)
                nn_word_list_flann.append(None)
                continue
            # nearest neighbor
            vec = emb_vec[word2idx_glove[word.lower()]]
            dists, index = knn_model.kneighbors(vec.reshape(1, -1))
            nn_word_list_scikt.append(word_list[index[0][0]])
            # index, dists = flann.nn_index(vec.reshape(1, -1), num_neighbors=1)
            # nn_word_list_flann.append(word_list[index[0]])

        freqs = np.array([freq_list[word2idx_gl[w]] if w != None else 0 for w in nn_word_list_scikt])
        ori_max_freq_word = sw[np.argmax(freqs)]
        nn_max_freq_word = nn_word_list_scikt[np.argmax(freqs)]
        nn_word_seq.append(nn_max_freq_word)
        original_word_seq.append(ori_max_freq_word)

        # if the word does not exist in the gesture library 
        if nn_max_freq_word == None:
            continue

        cluster = cluster_index[word2idx_gl[nn_max_freq_word]]
        cluster_seq.append(cluster)
        gesture = gesture_list[cluster]
        gesture_seq.append(gesture)

    gesture = linear_interpolation(gesture_seq, interpolate_frame=interpolate_frame, mean_pose=mean_pose)

    # Merge gestures
    clusters, input_words, nn_words = [], [], []
    for i in range(len(gesture_seq)):
        for j in range(len(gesture_seq[i])):
            clusters.append(cluster_seq[i])
            input_words.append(original_word_seq[i])
            nn_words.append(nn_word_seq[i])

        if i == len(gesture_seq) - 1:
            break

        for j in range(interpolate_frame):
            clusters.append('interpolation')
            input_words.append(' ')
            nn_words.append(' ')

    # Create audio file from input text
    input_audio = './tmp.wav'
    text2speech(input_text, input_audio)

    # Adjustment Gestures to Input Audio
    wr = wave.open(input_audio, 'r')
    fr = wr.getframerate()
    fn = wr.getnframes()
    audio_time = fn / fr - 0.7
    fps = gesture.shape[0] / audio_time
    
    file_name = input_text[:10]
    print("Saving video...")
    plotUpperBody2D(gesture, "./tmp.mp4", input_words=input_words, nn_words=nn_words, clusters=clusters, fps=fps)
    
    # Attaching audio
    save_path = "./output/byges_{}_{}word_interpolated_{}frame.mp4".format(file_name, sentence_segment, interpolate_frame)
    save_path = save_path.replace(' ', '_')
    cmd = "ffmpeg -i {mp4} -i {wav} -c:v copy -c:a aac -strict experimental -map 0:v -map 1:a {output} -y".format(
        mp4="./tmp.mp4", wav=input_audio, output=save_path)
    subprocess.call(cmd)

    os.remove("./tmp.mp4")
    
    print("Output to {}".format(save_path))


