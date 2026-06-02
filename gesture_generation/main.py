import os
import re
import socket
import subprocess
import pathlib
import warnings

import pandas as pd
import generate_gesture
from utils.convert_laban import convert_laban_format


csv_path = "./output/csv/{}.csv"
mp4_path = "./output/mp4/{}.mp4"
save_laban_path = "./output/laban/{}.json"
PORT = 50000
BUFFER_SIZE = 1024

warnings.simplefilter('ignore')


def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


def concat_csvs(csv_paths, output_path):
    dfs = []
    time_offset = 0
    for path in csv_paths:
        df = pd.read_csv(path, header=None)
        df[0] = (df[0] + time_offset).astype(int)
        frame_interval = int(df[0].iloc[1] - df[0].iloc[0]) if len(df) > 1 else 17
        time_offset = int(df[0].iloc[-1]) + frame_interval
        dfs.append(df)
    pd.concat(dfs, ignore_index=True).to_csv(output_path, header=False, index=False)


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('127.0.0.1', PORT))
    s.listen()
    print("Run python gestureBotDesignKit\src\Samples\gestureService_w2v\main.py")
    print("And access http://localhost:8002/ to enter text, http://localhost:8001/ to play gestures.")
    print('The gesture generation server is ready.')
    while True:
        (connection, client) = s.accept()
        try:
                print('Client connected', client)
                chunks = []
                while True:
                    chunk = connection.recv(BUFFER_SIZE)
                    if not chunk:
                        break
                    chunks.append(chunk)
                data = b"".join(chunks)
                input_text = data.decode()
                print("Input Text: ", input_text)
                filename = re.sub(r'[\\/:*?"<>|]+','',input_text[:30])

                # Gesture Generation (per sentence)
                sentences = split_sentences(input_text)
                print(f"Split into {len(sentences)} sentence(s): {sentences}")
                tmp_csv_paths = []
                prev_end_laban = None
                for i, sentence in enumerate(sentences):
                    tmp_csv = "./output/csv/tmp_{}_{}.csv".format(filename, i)
                    prev_end_laban = generate_gesture.generateGesture(
                        input_text=sentence,
                        save_csv_path=tmp_csv,
                        start_laban=prev_end_laban
                    )
                    tmp_csv_paths.append(tmp_csv)

                concat_csvs(tmp_csv_paths, csv_path.format(filename))
                for p in tmp_csv_paths:
                    os.remove(p)

                # Overwrite with full-text audio (sentence-1 audio was created during generation)
                full_audio_path = "./gestureBotDesignKit/src/Libraries/gestureBot/web/audio/{}.wav".format(filename)
                generate_gesture.text2speech(input_text, full_audio_path)
                
                # Convert to labanotation
                cmd = ["python", "./LabanSuiteBeta/GestureAuthoringTools/LabanEditor/src/main.py", "--alg", "parallel", 
                        "--inputfile", csv_path.format(filename), "--nogui", "--outputfolder", "./.tmp/"]
                subprocess.call(cmd)

                converted = convert_laban_format("./.tmp/{}.json".format(filename), save_laban_path.format(filename))
                abs_path = pathlib.Path(save_laban_path.format(filename)).resolve()

                print('Output to ', abs_path)

                connection.send(str(abs_path).encode())

        finally:
            connection.close()




