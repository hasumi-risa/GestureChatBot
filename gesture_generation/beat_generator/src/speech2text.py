import whisper

# import time
# import json 
# import azure.cognitiveservices.speech as speechsdk


def speech2text(inputPath):
    model = whisper.load_model("base")
    result = model.transcribe(inputPath, word_timestamps=True)
    
    word_list = []
    for segment in result["segments"]:
        for word in segment["words"]:
            word_list.append([word["word"].strip(), word["start"], word["end"]])
    
    return word_list

    # --- 元のコード（Azure） ---
    # speech_key, service_region = "f8a95505e6604cb9acaf8e8807cece33", "japaneast"
    # speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # speech_config.request_word_level_timestamps()
    # speech_config.output_format = speechsdk.OutputFormat(1)
    # audio_config = speechsdk.audio.AudioConfig(filename=inputPath)
    # speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    # done = False
    # words = []
    # def stop_callback(evt):
    #     speech_recognizer.stop_continuous_recognition()
    #     nonlocal done
    #     done = True
    # def add_to_res(evt):
    #     response = json.loads(evt.result.json)
    #     confidence_list_temp = [item.get('Confidence') for item in response['NBest']]
    #     max_confidence_index = confidence_list_temp.index(max(confidence_list_temp))
    #     words.extend(response['NBest'][max_confidence_index]['Words'])
    # speech_recognizer.recognized.connect(add_to_res)
    # speech_recognizer.session_stopped.connect(stop_callback)
    # speech_recognizer.canceled.connect(stop_callback)
    # speech_recognizer.start_continuous_recognition()
    # while not done:
    #     time.sleep(.5)
    # sr = 10000000
    # word_list = [[w['Word'], w['Offset']/sr, (w['Offset'] + w['Duration'])/sr] for w in words]
    # return word_list

if __name__ == "__main__":
    inputPath = "./.tmp/wav/tmp.wav"
    word_list = speech2text(inputPath)
    print(word_list)