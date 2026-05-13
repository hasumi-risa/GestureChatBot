import os
import subprocess
import requests
import xml.etree.ElementTree as ElementTree

def get_token(subscription_key):
    fetch_token_url = 'https://japaneast.api.cognitive.microsoft.com/sts/v1.0/issuetoken'
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key
    }
    response = requests.post(fetch_token_url, headers=headers)
    access_token = str(response.text)
    return access_token

def text2speech(input_text, output_audio_path):
    subscription_key = '36e2edd9a1f04eeaa2a9a9256b8b1d46'
    access_token = get_token(subscription_key)

    constructed_url = 'https://japaneast.tts.speech.microsoft.com/cognitiveservices/v1'

    headers = {
        'Authorization': 'Bearer ' + access_token,
        'Content-Type': 'application/ssml+xml',
        'X-Microsoft-OutputFormat': 'audio-16khz-128kbitrate-mono-mp3',
    }

    xml_body = ElementTree.Element('speak', version='1.0')
    xml_body.set('{http://www.w3.org/XML/1998/namespace}lang', 'en-US')
    voice = ElementTree.SubElement(xml_body, 'voice')
    voice.set('{http://www.w3.org/XML/1998/namespace}lang', 'en-US')
    voice.set('name', 'Microsoft Server Speech Text to Speech Voice (en-US, JennyNeural)')
    prosody = ElementTree.SubElement(voice, 'prosody')
    prosody.set('pitch','medium') # high
    prosody.set('rate','medium') # fast
    prosody.text = input_text
    body = ElementTree.tostring(xml_body)

    response = requests.post(constructed_url, headers=headers, data=body)
    if response.status_code == 200:
        with open(output_audio_path, 'wb') as audio:
            audio.write(response.content)
    else:
        print("\nText-to-Speech Status code: " + str(response.status_code) + "\nSomething went wrong. Check your subscription key and headers.\n")
    


if __name__ == '__main__':
    # input_text = "all I have to do is build a platform and all these people are going to put their stuff on top and I sit back and roll it in ?"
    # input_text = "I can dodge what I don't want and pull in what I want ."
    # input_text = "they keep spinning with the same axis, indefinitely. Hubble kind of rotates around them, and so it can orient itself."
    # input_text = "a doughnut or a half-moon shape with a large, central hole."
    # input_text = "you'd see that picture of dog poop high up in the search results"
    input_text = "He rotated the wheel slowly."

    
    output_audio = "./speech_data/He_rotated.wav"

    text2speech(input_text, output_audio)



    

    
