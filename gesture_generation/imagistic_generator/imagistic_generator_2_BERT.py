import os
import wave
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

import torch
from transformers import BertTokenizer, BertModel
import corenlp

from src.motion_interpolation import linear_interpolation
from src.plot3D import plot3D, animate3D
from src.plotPose import Plot, display_pose
from src.poseVisualizer import visualizePose
from src.text2speech import text2speech
from src.cmu2kinect import CMUPose2KinectData
from src.split_sentence import splitSentence
from src.select_gesture_phrase import selectByBERTNN, selectByRandom

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

    # input_text = "He rotated the wheel slowly."
    # input_text = "I can dodge what I don't want and pull in what I want."
    # input_text = "they keep spinning with the same axis, indefinitely. Hubble kind of rotates around them, and so it can orient itself"
    # input_text = "a doughnut or a half-moon shape with a large, central hole."
    # input_text = "all I have to do is build a platform and all these people are going to put their stuff on top and I sit back and roll it in ?"
    # input_text = "you'd see that picture of dog poop high up in the search results"
    input_text = "you live in the past while I'll create the future"

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


    # Sentence Segmentation (uniformly)
    # segmented_words = []
    # words = tokenizer.tokenize(input_text)
    # ws = []
    # for i in range(len(words)):
    #     ws.append(words[i])
    #     if (i+1) % sentence_segment == 0 or i == len(words) - 1:
    #         segmented_words.append(ws)
    #         ws = []

    # Sentence Segmentation (Stanford Parser)
    parser = corenlp.StanfordCoreNLP(
        corenlp_path=corenlp_dir,
        properties=properties_file)
    segmented_words = splitSentence(input_text, max_word_num=sentence_segment, parser=parser)
    

    # Gesture Selection
    print('Selecting Gestures...')
    gesture_seq, original_remark_seq, nn_remark_seq, cluster_seq = selectByBERTNN(
        segmented_words, knn_model, tokenizer, bert_model, gesture_list, distance_list, remark_list, cluster_index, audio_time, fps, mode='Aggressive'
    )
    # gesture_seq, original_remark_seq, nn_remark_seq, cluster_seq = selectByRandom(
    #     segmented_words, knn_model, tokenizer, bert_model, gesture_list, distance_list, remark_list, cluster_index, audio_time, fps, mode='Gentle'
    # )

    gesture = linear_interpolation(gesture_seq, interpolate_frame=interpolate_frame, mean_pose=mean_pose)
    fps = gesture.shape[0] / audio_time

    # Merge gestures
    clusters, input_words, nn_words = [], [], []
    for i in range(len(gesture_seq)):
        for j in range(len(gesture_seq[i])):
            clusters.append(cluster_seq[i])
            input_words.append(original_remark_seq[i])
            nn_words.append(nn_remark_seq[i])

        if i == len(gesture_seq) - 1:
            break

        for j in range(interpolate_frame):
            clusters.append('interpolation')
            input_words.append(' ')
            nn_words.append(' ')



    file_name = input_text[:10] + "_interpo_{}frame".format(interpolate_frame)
    file_name = file_name.replace(' ', '_')
    
    # Save csv
    save_path = pose_save_dir + file_name + '.csv'
    CMUPose2KinectData(gesture, save_csv=save_path, fps=fps, isConvert=False)
    print('Saved csv file to {}'.format(save_path))
    
    print("Saving video...")
    plotUpperBody2D(gesture, "./tmp.mp4", input_words=input_words, nn_words=nn_words, clusters=clusters, fps=fps)

    # Attaching audio
    save_path = video_save_dir + file_name + '.mp4'
    cmd = ["ffmpeg", "-i", "./tmp.mp4", "-i", input_audio, "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", "-map", "0:v", "-map", "1:a", save_path, "-y"]
    subprocess.call(cmd)

    os.remove("./tmp.mp4")
    
    print("Output to {}".format(save_path))


