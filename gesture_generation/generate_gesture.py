import os
import re
import wave
import subprocess
import numpy as np

import torch
from scipy.ndimage import gaussian_filter1d
from transformers import BertTokenizer

from gesture_type_prediction.model import GestureTypePredictor
from gesture_type_prediction.text2gesturetype import load_checkpoint, text2gesturetype, visualizeLabel
# from imagistic_generator.imagistic_generator_3_BERT_tfidf import generate_imagistic_gesture, text2speech
from imagistic_generator.imagistic_generator_4_deep import generate_imagistic_gesture, text2speech, GestureWordPrediction
from imagistic_generator.src.motion_interpolation import linearInterpolation, splineInterpolation
from imagistic_generator.src.cmu2kinect import CMUPose2KinectData
from beat_generator.beat_generator_2_library import BeatGenerator
from beat_generator.audio_processing import compute_prosody
from beat_generator.motion_processing import load_kinect_csv
from beat_generator.src.speech2text import speech2text
from no_gesture_generator.no_gesture_generator import NoGestureGenerator

from utils.match_words import getWordTime



#####################################  Input  ######################################

# input_text = "so what is evolution's answer to the of uncertainty? that's been the conventional wisdom."
# input_text = "our story, therefore, needs two dimensions of time, a long arc of time that is our lifespan, and"
# input_text = "all I have to do is build a platform and all these people are going to put their stuff on top and I sit back and roll it in ?"
# input_text = "you can imagine if he pushed the rock on different hills, at least he would have some sense of progress"
# input_text = "whilst up above , on the surface , hms bageye patrolled"    # 05jJodDVJRQ_2_2
# input_text = "us, in its final show and official introduction," # _B1JmOerYmY_5_6
# input_text = "and the radiation of flowering plants, or angiosperms, onto land."    # 2NWpMqD8Qyk_2_1
# input_text = "you can imagine if he pushed the rock on different hills, at least he would have some sense of progress" # 5aH2Ppjpcho_10_0
# input_text = "we have a lot of youth on this continent."    # 21hgbMa_sVc_4_1
# input_text = "just like inanimate matter cooled down to near absolute zero, where quantum effects play a very important role." # _qgSz1UmcBM_9
# input_text = "I met there with people from the page that told me, Okay, you're going to be in Europe, I'm coming. I'm coming from France, from Holland, from Germany" # 6Lp-NMaU0r8_24
# input_text = "More than 50 people have come to stay in the 18th-century watchhouse he lives in with his cat, Squeak." # kTqgiF4HmgQ_0
# input_text = "in business. But again, this path that we've been on is not getting us where we need to go." # 0iIh5YYDR2o_18
# input_text = "really scary Death Valley period in which many companies instead fail. But what really interests me, especially nowadays and because of what's happening politically around the world," # 3r1IPsldbBg_1
input_text = "Please make yourself feel at home."
# input_text = "It's just that everything turned out so perfectly it's almost hard to believe."
# input_text = "The robot is capable of vastly more than almost anyone knows."
# input_text = "I haven’t tried it yet because I seldom go on picnics, but I’d love to try."
# input_text = "I like going to the beach. What do you do with your family?"

#####################################  Parameters  ######################################

# Input files
gtp_model_path = './gesture_type_prediction/model/model_valid_f1_class_augmented_finetune_Linear.pt'
gwp_model_path = './imagistic_generator/deep_word_selection/model_data/model_train_20210726_Linear.pt'
imag_library_path = "./imagistic_generator/data/gesture_library_k=35_msrabot.npy" # for MSRABot
beat_library_path = "./beat_generator/data/beat_library_20211006_msrabot.npy" # for MSRABot
noges_library_path = "./no_gesture_generator/data/no_gesture_library_20211006.npy"
sample_beat_data = './beat_generator/data/Beat_1L6l-FiV4xo_8_sample/sample_beat_1L6l-FiV4xo_8.csv'


# Parameters
MODE = "Gentle"
AUDIO_FEATURE = "intensity"
# AUDIO_GAUSSIAN = 1
AUDIO_GAUSSIAN = 2 # for MSRABot
CLASS_NUM = 3
MAX_SEQ_LEN = 256
SENTENCE_SEGMENT_NUM = 4
INTERPOLATION_FRAME = 10
FPS = 60
ORIGINAL_FPS = 25
INTERPOLATE_FRAME = 7
BEAT_MOTION_INTERVAL = 20


#####################################  Preliminaries  ######################################

# BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device : ", device)

# Load Gesture Type Prediction Model
model = GestureTypePredictor(num_class=CLASS_NUM).to(device)    
load_checkpoint(gtp_model_path, model, device)

# Load Gesture Library
print('Loading Gesture Library...')
imglib = np.load(imag_library_path, allow_pickle=True)
# parser = corenlp.StanfordCoreNLP(corenlp_path=corenlp_dir, properties=properties_file)

bg = BeatGenerator(beat_library_path)
ngg = NoGestureGenerator(noges_library_path)

gwp = GestureWordPrediction(gwp_model_path)

def generateGesture(input_text, input_audio=None, save_csv_path=None, save_mp4_path=None):

    # Convert text to speech audio
    print('Converting the text to speech audio...')
    filename = re.sub(r'[\\/:*?"<>|]+','',input_text[:30])
    tmp_wav_path = "./gestureBotDesignKit/src/Libraries/gestureBot/web/audio/{}.wav".format(filename)
    
    if not input_audio:
        text2speech(input_text, tmp_wav_path)

    # Extract audio feature
    print('Extracting the duration of each word...')

    if input_audio:
        pitch, intensity, time = compute_prosody(input_audio)
    else:
        pitch, intensity, time = compute_prosody(tmp_wav_path)

    if AUDIO_FEATURE == 'pitch':
        audio_wave = pitch
    elif AUDIO_FEATURE == 'intensity':
        audio_wave = intensity

    # Cut the last part that doesn't say anything.
    for i in range(len(audio_wave)-1, 0, -1):
        if audio_wave[i] > 1e-4:
            end_idx = i
            break
    audio_wave = audio_wave[:end_idx]
    audio_wave = gaussian_filter1d(audio_wave, AUDIO_GAUSSIAN)
    frame_num = int(FPS * len(audio_wave) / ORIGINAL_FPS)

    # plotTextAudio(audio_wave, word_list, save_path="./.tmp/audio_text.png")   # for Debug
    # plotTransition([audio_wave], labels=['audio'])

    #####################################  Gesture Type Prediction  ######################################
    print("Predicting Gesture Types...")
    text_tokens, gesture_types = text2gesturetype(input_text, model, tokenizer, device)
    visualizeLabel(text_tokens, gesture_types, save_path="./.tmp/gesture_type.png") # for Debug
    gesture_types = gesture_types[1:-1]
    bert_tokens = text_tokens[1:len(text_tokens)-1]

    type_order = []         # the order of gesture types
    preidx = 0
    each_word_time = []
    for i in range(1, len(gesture_types)):
        if gesture_types[i] != gesture_types[i-1]:
            each_word_time.append(bert_tokens[preidx:i])
            type_order.append(gesture_types[i-1])
            preidx = i
    each_word_time.append(bert_tokens[preidx:])
    type_order.append(gesture_types[i])

    each_word_num = [len(w) for w in each_word_time]
    frame_per_word = len(audio_wave) / sum(each_word_num)

    # type_order = [2 for i in range(len(type_order))]


    #####################################  Gesture Generation  ######################################
    print("Generating Gestures...")
    gesture_seq, laban_seq, durations = [], [], []
    start_frame, end_frame = 0, int(each_word_num[0] * frame_per_word)
    for i in range(len(type_order)):
        if i > 0:
            start_frame = end_frame
            end_frame = start_frame + int(each_word_num[i] * frame_per_word)
        if end_frame == start_frame:
            continue

        # Beat Generation
        if type_order[i] == 1:
            print("Beat: {}".format(" ".join(each_word_time[i])))
            a = audio_wave[start_frame:end_frame]
            gesture, laban = bg.generate(a)
            kf = list(laban.keys())
            gesture_seq.append(gesture)
            laban_seq.append([laban[kf[0]], laban[kf[-1]]])

        # Imagistic Generation
        elif type_order[i] == 2:
            text = " ".join(each_word_time[i])  # full BERT tokens, not first chars
            print("Imagistic: {}".format(text))
            if i == 0:
                a = audio_wave[:end_frame-INTERPOLATE_FRAME]
            elif i == len(type_order) - 1:
                a = audio_wave[start_frame+INTERPOLATE_FRAME:end_frame]
            else:
                a = audio_wave[start_frame+INTERPOLATE_FRAME:end_frame-INTERPOLATE_FRAME]

            gesture, sl, el = generate_imagistic_gesture(text, a, imglib, gwp, MODE,
                ORIGINAL_FPS, sentence_segment=SENTENCE_SEGMENT_NUM, interpolate_frame=INTERPOLATE_FRAME)
            if len(gesture) != 0:
                gesture_seq.append(gesture)
                laban_seq.append([sl, el])
            else:
                # Fall back to No-Gesture to keep gesture_seq in sync with type_order
                durations.append([start_frame, end_frame])
                gesture_seq.append([])
                laban_seq.append([])

        # No-Gesture
        else:
            print("No-Gesture")
            durations.append([start_frame, end_frame])
            gesture_seq.append([])
            laban_seq.append([])

    cnt = 0
    for i in range(len(gesture_seq)):
        if len(gesture_seq[i]) == 0:
            start_frame = durations[cnt][0]
            end_frame = durations[cnt][1]
            duration = end_frame - start_frame
            if i == 0:
                if len(laban_seq) <= 1:
                    gesture, laban = ngg.generate(duration)
                else:
                    if len(laban_seq[i+1]) == 0:
                        gesture, laban = ngg.generate(duration)
                    else:
                        gesture, laban = ngg.generate(duration, end_laban=laban_seq[i+1][0])
            elif i == len(gesture_seq) - 1:
                gesture, laban = ngg.generate(duration, start_laban=laban_seq[i-1][1])
            else:
                gesture, laban = ngg.generate(duration, start_laban=laban_seq[i-1][1], end_laban=laban_seq[i+1][0])
            gesture_seq[i] = gesture
            laban_seq[i] = [[], laban[list(laban.keys())[-1]]]
            cnt += 1
            
            # for i in range(len(gesture)): type_seq.append("No-Gesture")
            # for i in range(INTERPOLATE_FRAME): type_seq.append("Interpolation")
    

    
    # Interpolate gestures
    gesture = linearInterpolation(gesture_seq, interpolate_frame=INTERPOLATE_FRAME)
    # gesture = splineInterpolation(gesture_seq, interpolate_frame=interpolate_frame)

    # Adjust to match fps
    gesture = bg.motionAdjustment(gesture, frame_num)


    #####################################  Saving  ######################################

    # Save csv
    if save_csv_path:
        print("Saving csv... ")
        CMUPose2KinectData(gesture, save_csv=save_csv_path, fps=FPS, isConvert=False)
        print('Saved csv file to {}'.format(save_csv_path))

    if save_mp4_path:
        print("Saving video...")
        tmp_mp4_path = "./.tmp/generated_gesture.mp4"
        bg.plotUpperBody2D(gesture, tmp_mp4_path, fps=FPS)

        # Attaching audio
        # mp4_save_path = video_save_dir + file_name + '.mp4'
        if input_audio:
            cmd = ["ffmpeg", "-i", tmp_mp4_path, "-i", input_audio, "-c:v", "copy", "-c:a", "aac", "-strict", \
            "experimental", "-map", "0:v", "-map", "1:a", save_mp4_path, "-y", "-hide_banner", "-loglevel", "error"]
        else:
            cmd = ["ffmpeg", "-i", tmp_mp4_path, "-i", tmp_wav_path, "-c:v", "copy", "-c:a", "aac", "-strict", \
            "experimental", "-map", "0:v", "-map", "1:a", save_mp4_path, "-y", "-hide_banner", "-loglevel", "error"]
        subprocess.call(cmd)

        print("Output to {}".format(save_mp4_path))

if __name__ == "__main__":

    # input_audio = "./imagistic_generator/data/for_evaluation/21hgbMa_sVc_4_1/21hgbMa_sVc_4_1.wav"
    input_audio = None

    # Output files
    save_csv_path = "./output/csv/{}_{}.csv".format(input_text[:10].replace(' ', '_'), MODE)     # 3D-Pose File (Kinect Format) 
    save_mp4_path = "./output/mp4/{}_{}.mp4".format(input_text[:10].replace(' ', '_'), MODE)

    generateGesture(
        input_text, 
        input_audio=input_audio,
        save_csv_path=save_csv_path)