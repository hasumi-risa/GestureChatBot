import os
import subprocess

audio_path = "./speech_data/He_rotated.wav"
video_path = "./src/data/He_rotated.mp4"
save_path = "./src/data/He_rotated_audio.mp4"

cmd = "ffmpeg -i {mp4} -i {wav} -c:v copy -c:a aac -strict experimental -map 0:v -map 1:a {output} -y".format(
    mp4=video_path, wav=audio_path, output=save_path)
subprocess.call(cmd)

print("Output to {}".format(save_path))