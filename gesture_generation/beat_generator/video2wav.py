import subprocess

video_path = "./data/1L6l-FiV4xo_8/1L6l-FiV4xo_8.mp4"

wav_path = video_path[:-4] + '.wav'

command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(video_path, wav_path)

subprocess.call(command, shell=True)