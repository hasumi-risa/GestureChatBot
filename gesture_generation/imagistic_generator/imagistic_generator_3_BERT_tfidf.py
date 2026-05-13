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
import nltk
import corenlp

from .src.motion_interpolation import linearInterpolation, motionAdjustment
from .src.plot3D import plot3D, animate3D
from .src.plotPose import Plot, display_pose
from .src.poseVisualizer import visualizePose
from .src.text2speech import text2speech
from .src.cmu2kinect import CMUPose2KinectData
from .src.split_sentence import splitSentence
from .src.select_gesture_phrase import selectByBERTNN, selectByRandom, selectByTFIDF

def plotUpperBody2D(pose3d, save_path, input_words=None, nn_words=None, clusters=None, fps=25):
    upper_idx = [2, 20, 4, 5, 6, 8, 9, 10]
    
    pose3d = pose3d[:, upper_idx]
    pose2d = np.delete(pose3d, 2, 2).reshape([-1, len(upper_idx)*2])

    # rotate y axis
    for frame in range(len(pose2d)):
        for joint in range(len(pose2d[frame])):
            if joint % 2 == 0:
                pose2d[frame][joint] *= -1

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


def generate_imagistic_gesture(input_text, gesture_library, tfidf_weight, mode, audio_time, fps, parser=None, sentence_segment=4, interpolate_frame=10):
    gesture_list = gesture_library.item().get('gesture_list')
    distance_list = gesture_library.item().get('distance_list')
    remark_list = gesture_library.item().get('remark_list')
    embvec_list = gesture_library.item().get('emb_vectors')
    word2weight = np.load(tfidf_weight, allow_pickle=True)[()]

    embvecs = []
    remarks = []
    cluster_index = []
    for i,vecs in enumerate(embvec_list):
        for j,vec in enumerate(vecs):
            cluster_index.append(i)
            embvecs.append(vec)
            remarks.append(remark_list[i][j])
    embvecs = np.array(embvecs)

    # NN setting
    knn_model = NearestNeighbors(n_neighbors=1).fit(embvecs) 

    # BERT setting
    options_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(options_name)
    bert_model = BertModel.from_pretrained(options_name)
    bert_model.eval()

    # Sentence Segmentation (Stanford Parser)
    segmented_words = splitSentence(input_text, max_word_num=sentence_segment, parser=parser)

    # Gesture Selection
    print('Selecting Gestures...')
    tokens = nltk.word_tokenize(input_text)
    weights = []
    for i in range(len(tokens)):
        word = str.lower(tokens[i])
        if word in word2weight.keys():
            weights.append(word2weight[word])
        else:
            weights.append(0)
    weights = np.array(weights)
        

    # gesture_seq, original_remark_seq, nn_remark_seq, cluster_seq = selectByBERTNN(
    #     segmented_words, knn_model, tokenizer, bert_model, gesture_list, distance_list, remarks, cluster_index, audio_time, fps, mode=mode)

    # gesture_seq, original_remark_seq, nn_remark_seq, cluster_seq = selectByRandom(
    #     segmented_words, knn_model, tokenizer, bert_model, gesture_list, distance_list, remarks, cluster_index, audio_time, fps, mode=mode)

    gesture_seq, original_remark_seq, nn_remark_seq, cluster_seq = selectByTFIDF(
        weights, segmented_words, knn_model, tokenizer, bert_model, gesture_list, distance_list, remarks, cluster_index, audio_time, mode=mode)

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

    # Interpolation between gestures
    frame_num = int(audio_time * fps)
    gesture = linearInterpolation(gesture_seq, interpolate_frame=interpolate_frame)
    # Adjust to match fps
    gesture, input_words, nn_words, clusters = motionAdjustment(gesture, frame_num, input_words, nn_words, clusters)

    return gesture, original_remark_seq, nn_remark_seq, cluster_seq 


if __name__ == "__main__":
    # input_text = "He rotated the wheel slowly."
    # input_text = "I can dodge what I don't want and pull in what I want."
    # input_text = "they keep spinning with the same axis, indefinitely. Hubble kind of rotates around them, and so it can orient itself"
    # input_text = "a doughnut or a half-moon shape with a large, central hole."
    # input_text = "all I have to do is build a platform and all these people are going to put their stuff on top and I sit back and roll it in ?"
    # input_text = "you'd see that picture of dog poop high up in the search results"
    # input_text = "you live in the past while I'll create the future"
    # input_text = "whilst up above , on the surface , hms bageye patrolled"    # 05jJodDVJRQ_2_2
    # input_text = "us, in its final show and official introduction," # _B1JmOerYmY_5_6
    # input_text = "and the radiation of flowering plants, or angiosperms, onto land."    # 2NWpMqD8Qyk_2_1
    # input_text = "you can imagine if he pushed the rock on different hills, at least he would have some sense of progress" # 5aH2Ppjpcho_10_0
    input_text = "we have a lot of youth on this continent."    # 21hgbMa_sVc_4_1

    # input_audio = None
    input_audio = "./data/for_evaluation/21hgbMa_sVc_4_1/21hgbMa_sVc_4_1.wav"
    gesture_library = "./data/gesture_library_k=31.npy"
    nlp_base_dir = './src/NLP/'
    corenlp_dir = nlp_base_dir + "stanford-corenlp-full-2013-06-20/"
    properties_file = nlp_base_dir + "user.properties"
    tfidf_weight = "./data/tfidf_TED_weight_0.95.npy"
    mp4_save_path = "./data/for_evaluation/21hgbMa_sVc_4_1/21hgbMa_sVc_4_1-2d-pose_generated.mp4"
    csv_save_path = "./data/for_evaluation/21hgbMa_sVc_4_1/21hgbMa_sVc_4_1_generated.csv"
    # video_save_dir = "./output/videos/"
    # pose_save_dir = "./output/csv/"

    # mode = 'Aggressive'
    mode = 'Gentle'
    sentence_segment = 4
    interpolate_frame = 10
    fps = 60

    file_name = input_text[:10] + "_{}".format(mode)
    file_name = file_name.replace(' ', '_')

    print('Loading Gesture Library...')
    geslib = np.load(gesture_library, allow_pickle=True)

    parser = corenlp.StanfordCoreNLP(corenlp_path=corenlp_dir, properties=properties_file)

    if input_audio:
        # Adjustment Gestures to Input Audio
        wr = wave.open(input_audio, 'r')
        fr = wr.getframerate()
        fn = wr.getnframes()
        audio_time = fn / fr
    else:
        # create voice from input text
        input_audio = './.tmp/wav/tmp.wav'
        text2speech(input_text, input_audio)

        # Adjustment Gestures to Input Audio
        wr = wave.open(input_audio, 'r')
        fr = wr.getframerate()
        fn = wr.getnframes()
        audio_time = fn / fr - 0.5

    gesture, input_words, nn_words, clusters  \
        = generate_imagistic_gesture(input_text, geslib, tfidf_weight, mode, parser=parser)

    # Save csv
    # csv_save_path = pose_save_dir + file_name + '.csv'
    CMUPose2KinectData(gesture, save_csv=csv_save_path, fps=fps, isConvert=False)
    print('Saved csv file to {}'.format(csv_save_path))

    print("Saving video...")
    plotUpperBody2D(gesture, "./tmp.mp4", input_words=input_words, nn_words=nn_words, clusters=clusters, fps=fps)

    # Attaching audio
    # mp4_save_path = video_save_dir + file_name + '.mp4'
    cmd = ["ffmpeg", "-i", "./tmp.mp4", "-i", input_audio, "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", "-map", "0:v", "-map", "1:a", mp4_save_path, "-y"]
    subprocess.call(cmd)

    os.remove("./tmp.mp4")

    print("Output to {}".format(mp4_save_path))