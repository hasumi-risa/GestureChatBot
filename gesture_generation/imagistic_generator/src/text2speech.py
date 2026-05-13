import os
from gtts import gTTS

def text2speech(input_text, output_audio_path):
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    tts = gTTS(text=input_text, lang="en")
    tts.save(output_audio_path)
    print("Text2Speech saved to", output_audio_path)