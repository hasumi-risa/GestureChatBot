import os
import re
import socket
import subprocess
import pathlib
import warnings

import generate_gesture
from utils.convert_laban import convert_laban_format


csv_path = "./output/csv/{}.csv"
mp4_path = "./output/mp4/{}.mp4"
save_laban_path = "./output/laban/{}.json"
PORT = 50000
BUFFER_SIZE = 1024

warnings.simplefilter('ignore')

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
                data = connection.recv(BUFFER_SIZE)
                input_text = data.decode()
                print("Input Text: ", input_text)
                filename = re.sub(r'[\\/:*?"<>|]+','',input_text[:30])

                # Gesture Generation
                generate_gesture.generateGesture(
                    input_text=input_text, 
                    save_csv_path=csv_path.format(filename)
                    # save_mp4_path=mp4_path.format(filename)
                    )
                
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




