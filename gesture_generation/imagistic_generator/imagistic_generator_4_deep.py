import os
import wave
import subprocess
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy import signal

from transformers import BertTokenizer, BertModel

from .src.motion_interpolation import linearInterpolation, motionAdjustment, motionAndWordsAdjustment
from .src.plot3D import plot3D, animate3D
from .src.plotPose import Plot, display_pose
from .src.poseVisualizer import visualizePose
from .src.text2speech import text2speech
from .src.cmu2kinect import CMUPose2KinectData
from .src.split_sentence import splitSentence
from .src.select_gesture_phrase import selectByBERTNN, selectByRandom, selectGesture
# from .sec.audio_processing import compute_prosody
from .deep_word_selection.predict import GestureWordPrediction

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

def adjustMotionToAudio(audio_wave, pose3d, motion_keyframes, motion_interval=16, fps=25):
    adjusted_motion = []
    adjusted_frame = 0
    cnt, last_peak = 0, 0
    for i in range(0, len(audio_wave), motion_interval):
        audio = audio_wave[i:i+motion_interval]
        audio_maxima = signal.argrelmax(audio, order=1)[0]
        if len(audio_maxima) == 0:
            peak_frame = np.argmax(audio)
        else:
            maxid = np.argmax(np.array([audio[m] for m in audio_maxima]))
            peak_frame = audio_maxima[maxid]
        
        # print("Peak Frame: ", i+peak_frame)

        # Adjust motion to match audio

        motion = motionAdjustment(pose3d[adjusted_frame:motion_keyframes[cnt]], i+peak_frame-last_peak)
        for m in motion:
            adjusted_motion.append(m)
        
        adjusted_frame = motion_keyframes[cnt]
        
        cnt += 1
        if cnt >= len(motion_keyframes):
            break

        last_peak = i + peak_frame

    motion = motionAdjustment(pose3d[motion_keyframes[cnt-1]:], len(audio_wave) - len(adjusted_motion))
    for m in motion:
        adjusted_motion.append(m)
        
    adjusted_motion = np.array(adjusted_motion)
    
    # # Confirm
    # from scipy.ndimage import gaussian_filter1d
    # kernel_size = 1
    # v = pose3d[1:] - pose3d[:-1]
    # v_ = adjusted_motion[1:] - adjusted_motion[:-1]
    # l = gaussian_filter1d(np.linalg.norm(v[:,6], axis=1), kernel_size)
    # r = gaussian_filter1d(np.linalg.norm(v[:,10], axis=1), kernel_size)
    # l_ = gaussian_filter1d(np.linalg.norm(v_[:,6], axis=1), kernel_size)
    # r_ = gaussian_filter1d(np.linalg.norm(v_[:,10], axis=1), kernel_size)
    # plt.figure()
    # plt.plot(np.arange(len(audio_wave)), audio_wave, label="audio")
    # plt.plot(np.arange(len(l)), l, label="original left hand")
    # plt.plot(np.arange(len(r)), r, label="original right hand")
    # plt.plot(np.arange(len(l_)), l_, label="adjusted left hand")
    # plt.plot(np.arange(len(r_)), r_, label="adjusted right hand")
    # plt.legend()
    # plt.show()

    # Timestamp for kinect data
    msec = 1/fps * 1000
    timestamp = np.arange(0, len(pose3d)*msec, msec)
    timestamp = timestamp.reshape(-1, 1)
    
    # Extra data for kinect data
    # 0 -> no-detection,    2 -> detected
    cmu2kinect = [20, 2, 0, 4, 5, 6, 12, 13, 14, 8, 9, 10, 16, 17, 18]
    kinect_extra = np.zeros((len(pose3d), 25))
    for i in range(len(kinect_extra)):
        for j in range(len(kinect_extra[i])):
            if j in cmu2kinect or j in [1, 3, 15, 19]:
                kinect_extra[i][j] = 2
            else:
                kinect_extra[i][j] = 0

    # Concat timestamp and data
    kinect_extra = kinect_extra.reshape(-1, 25, 1)
    tmp = np.concatenate([pose3d, kinect_extra], axis=2)
    tmp = tmp.reshape(-1, 25 * 4)
    kinect_csv_data = np.concatenate([timestamp, tmp], axis=1)


    # Save
    kinect_csv_data = pd.DataFrame(kinect_csv_data)
    for i in range(0, 101, 4):
        kinect_csv_data[i] = kinect_csv_data[i].astype('int')
    kinect_csv_data.to_csv("./tmp_bef.csv", header=False, index=False)

    return adjusted_motion

def generate_imagistic_gesture(input_text, audio_wave, geslib,
    gwp, mode, fps, sentence_segment=4, interpolate_frame=10):

    audio_time = len(audio_wave) / fps

    if audio_time == 0:
        return [], None, None


    gesture_list    = geslib.item().get('gesture_list')
    laban_list      = geslib.item().get('laban_list')
    distance_list   = geslib.item().get('distance_list')
    remark_list     = geslib.item().get('remark_list')
    index2cluster   = geslib.item().get('index_to_cluster')
    embvecs         = geslib.item().get('emb_vectors')
    cluster_distmat = geslib.item().get('distance_list')
    cluster2index = geslib.item().get('cluster_to_index')

    # NN setting
    knn_model = NearestNeighbors(n_neighbors=1).fit(embvecs) 

    # predict words that gestures likely appear
    gesture_words, pred, token = gwp.extract_gesture_words(input_text)
    gwp.visualizePrediction(pred, token, save_path=".tmp/word_selection_{}.png".format(input_text[:10]))

    # ?は強制的にImagistic
    for i in range(len(token)):
        if token[i] == '?':
            gesture_words.append(['?'])

    # BERT setting
    options_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(options_name)
    bert_model = BertModel.from_pretrained(options_name)
    bert_model.eval()

    # Sentence Segmentation (Stanford Parser)

    gesture_seq, laban_seq, indexes = selectGesture(
        gesture_words, knn_model, tokenizer, bert_model, gesture_list, distance_list, 
        laban_list, cluster2index, index2cluster, audio_time, mode=mode, fps=fps)

    nn_remark_seq, cluster_seq= [], []
    for i,index in enumerate(indexes):
        print("Original Words: {}".format(gesture_words[i]))
        print('NN words: {}'.format(remark_list[index]))
        nn_remark_seq.append(remark_list[index])
        cluster = index2cluster[index]
        cluster_seq.append(cluster)

    # Merge gestures
    clusters, input_words, nn_words = [], [], []
    for i in range(len(gesture_seq)):
        for j in range(len(gesture_seq[i])):
            clusters.append(cluster_seq[i])
            nn_words.append(nn_remark_seq[i])

        if i == len(gesture_seq) - 1:
            break

        for j in range(interpolate_frame):
            clusters.append('interpolation')
            input_words.append(' ')
            nn_words.append(' ')

    keyframes = []
    frame = 0
    for i,gesture in enumerate(gesture_seq):
        gesture_id = list(laban_seq[i].keys())[0]
        if i == 0:
            start_laban = laban_seq[i][gesture_id]['Position0']
        if i == len(gesture_seq) - 1:
            end_laban = laban_seq[i][gesture_id][list(laban_seq[-1][gesture_id].keys())[-1]]
        for key in list(laban_seq[i][gesture_id].keys()):
            kf = int((int(laban_seq[i][gesture_id][key]['start time'][0]) / 1000) * fps)
            keyframes.append(frame + kf)
        frame += len(gesture) + interpolate_frame

    # Interpolation between gestures
    frame_num = int(audio_time * fps)
    gesture = linearInterpolation(gesture_seq, interpolate_frame=interpolate_frame)

    if gesture.shape[0] == 0:
        return gesture, None, None


    # Adjust to match fps
    # adjusted_motion = adjustMotionToAudio(audio_wave, gesture, keyframes)
    adjusted_motion = motionAdjustment(gesture, frame_num)

    return adjusted_motion, start_laban, end_laban


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
    gesture_library = "./data/gesture_library_k=31_laban.npy"
    nlp_base_dir = './src/NLP/'
    corenlp_dir = nlp_base_dir + "stanford-corenlp-full-2013-06-20/"
    properties_file = nlp_base_dir + "user.properties"
    tfidf_weight = "./data/tfidf_TED_weight_0.95.npy"
    mp4_save_path = "./data/for_evaluation/21hgbMa_sVc_4_1/21hgbMa_sVc_4_1-2d-pose_generated.mp4"
    csv_save_path = "./data/for_evaluation/21hgbMa_sVc_4_1/21hgbMa_sVc_4_1_generated.csv"
    gwp_model_path = './deep_word_selection/model_data/model_train_20210726_Linear.pt'

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
    gesture_list    = geslib.item().get('gesture_list')
    distance_list   = geslib.item().get('distance_list')
    remark_list     = geslib.item().get('remark_list')
    embvec_list     = geslib.item().get('emb_vectors')
    embvecs, remarks, cluster_index = [], [], []
    for i,vecs in enumerate(embvec_list):
        for j,vec in enumerate(vecs):
            cluster_index.append(i)
            embvecs.append(vec)
            remarks.append(remark_list[i][j])
    embvecs = np.array(embvecs)
    word2weight = np.load(tfidf_weight, allow_pickle=True)[()]
    gwp = GestureWordPrediction(gwp_model_path)

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
        = generate_imagistic_gesture(input_text, gesture_list, distance_list, remarks, cluster_index,
            embvecs, gwp, mode, duration, FPS, parser=None, sentence_segment=SENTENCE_SEGMENT_NUM, interpolate_frame=INTERPOLATE_FRAME)

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