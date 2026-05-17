# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import sys
import os
import re
import threading
import time
import socket
import openai

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Libraries'))
import Common.settings as settings
import Common.webSocket as webSocket
import Controller.controller as controller
import gestureBot.gestureBot as gestureBot
import gestureService_w2v.gestureService as gestureService
import gestureService_w2v.text2speech as t2s

# -----------------------------------------------------------------------------
#
class application(webSocket.HttpServerWrapper):
    #------------------------------------------------------------------------------
    # Class initialization
    #
    def __init__(self):
        # set global application variable now so all other objects have access 
        # to the application object.
        settings.application = self

        print('Labanotation Sample: gesture Service ' + settings.appVersion + '\r\n')

        self.context = "application"
        self.controller = None
        self.gestureBot = None
        self.gestureService = None
        self.audio_path = None
        self.laban_path = None

        self.shutdownEvent = threading.Event()

        self.wordList = None

        #
        # Create labanotation controller
        self.controller = controller.Controller(
            fnStatusUpdate=self.onControllerStatusUpdate, 
            fnGestureUpdate=self.onControllerGestureUpdate, 
            fnRequestShutdown=self.requestShutdownApplication,
            httpPort=8000)

        #
        # Create MSRAbot 
        self.gestureBot = gestureBot.gestureBot(self.controller, httpPort=8001, context="gestureBot")

        #
        # Create gesture Service
        self.gestureService = gestureService.GestureService(context="gestureService")

        if (self.gestureService is not None):
            self.actionTags = self.gestureService.getActionTags()
            self.wordsInActionTags = self.gestureService.getWordsInActionTags()
            self.labans = self.gestureService.getLabans()

        #
        # if requested, create and initialize http server to access gestureBot
        httpPort = 8002

        path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web'))

        webSocket.HttpServerWrapper.__init__(self, httpPort, path, self.context)

        webSocket.startHttpServerIOLoop()

    # -----------------------------------------------------------------------------
    #
    def onWSConnect(self):
        if (self.fnSendMessage):
            msg = {
                'msgType': "initialization", 
                'initialization': { 
                    'actionTags': self.actionTags, 
                    'labans': self.labans, 
                    'actionTagsWords': self.wordsInActionTags, 
                    'audio': self.audio_path
                    }
            }
            self.fnSendMessage(msg)

        return None

    # -----------------------------------------------------------------------------
    #
    def onWSDisconnect(self):
        pass

    # -----------------------------------------------------------------------------
    #
    def onWSMessage(self, msg):
        msgType = msg['msgType']
        if msgType == "initialization":
            pass
        elif msgType == "processMsg":
            self.processMsg(msg['processMsg'])
        elif msgType == "loadGesture":
            self.loadGesture(msg['gesture'])
        elif msgType == "playGesture":
            self.playGesture(msg['gesture'])
        else:
            print("onWSMessage(): unhandled message: '" + str(msg) + "'")

    # -----------------------------------------------------------------------------
    #
    def processMsg(self, msg):
    if (not 'message' in msg):
        return

    user_input = msg['message']

    # ChatGPT
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Reply in 2 short sentences only."},
            {"role": "user", "content": user_input}
        ]
    )
    message = response.choices[0].message.content
    print("ChatGPT Response: ", message)

    # ↓↓↓ この2行を削除（古いw2vベースの処理）↓↓↓
    # self.wordList = self.gestureService.tokenizePhrase(message)
    # gestureName, scores, trigger_word_num = self.gestureService.findGesture(self.wordList)

    # Gesture Generation System（generate_gestureが2文分のCSV/labanを生成）
    PORT = 50000
    BUFFER_SIZE = 1024
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('127.0.0.1', PORT))
        s.send(message.encode())
        self.laban_path = s.recv(BUFFER_SIZE).decode()
        print("Gesture is generated!", self.laban_path)

    # 音声ファイル
    filename = re.sub(r'[\\/:*?"<>|]+','', message[:30])
    local_audio_path = os.path.join(os.path.dirname(__file__), 
        "../../Libraries/gestureBot/web/audio") + "/{}.wav".format(filename)
    if not os.path.exists(local_audio_path):
        t2s.text2speech(message, local_audio_path)
    self.audio_path = "./audio/{}.wav".format(filename)

    # ↓↓↓ フロントへの通知も簡略化（w2v情報は不要）↓↓↓
    if (self.fnSendMessage):
        msg = {
            'msgType': "w2v",
            'w2v': {
                'phrase': message,
                'wordList': [],        # 空でOK
                'gestureName': "",     # 空でOK
                'triggerWordNum': 0,
                'scores': [],
                'audio': self.audio_path
            }
        }
        self.fnSendMessage(msg)


        # ----------------------- Gesture Generation System ---------------------------
        PORT = 50000
        BUFFER_SIZE = 1024

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('127.0.0.1', PORT))
            s.send(message.encode())
            
            self.laban_path = s.recv(BUFFER_SIZE).decode()
            print("Gesture is generated!", self.laban_path)

        # -----------------------------------------------------------------------------

        # search or create audio file 
        filename = re.sub(r'[\\/:*?"<>|]+','', message[:30])
        local_audio_path = os.path.join(os.path.dirname(__file__), "../../Libraries/gestureBot/web/audio") +  "/{}.wav".format(filename)

        if not os.path.exists(local_audio_path):
            print('local audio file is not found ==> creating to ', local_audio_path)
            t2s.text2speech(message, local_audio_path)

        self.audio_path = "./audio/{}.wav".format(filename)

        self.currentWordIndex = 0
        self.triggerWordIndex = trigger_word_num
        self.selectedGesture = gestureName

        wordScores = []
        for score in scores:
            wordScores.append(score)

        if (self.fnSendMessage):
            msg = {
                'msgType': "w2v", 
                'w2v': { 
                    'phrase': message, 
                    'wordList': self.wordList, 
                    'gestureName': gestureName, 
                    'triggerWordNum': int(trigger_word_num), 
                    'scores': wordScores,
                    'audio': self.audio_path
                }
            }

            self.fnSendMessage(msg)

    # -----------------------------------------------------------------------------
    #
    def loadGesture(self, gesture):
        if (self.controller is not None):
            if self.laban_path:
                folder = os.path.dirname(self.laban_path) + '/'
                filename = os.path.basename(self.laban_path)
                self.controller.loadGesture(folder, filename, audio_path=self.audio_path)
            else:
                self.controller.loadGesture('./gestureLibrary/', gesture + '.json', audio_path=self.audio_path)

    # -----------------------------------------------------------------------------
    #
    def playGesture(self, gesture):
        if (self.controller is not None):
            self.controller.playGesture('./gestureLibrary/', gesture + '.json', self.audio_path)

    #------------------------------------------------------------------------------
    #
    def onControllerStatusUpdate(self, status):
        if (self.gestureBot is not None):
            self.gestureBot.onControllerStatusUpdate(status)

    #------------------------------------------------------------------------------
    #
    def onControllerGestureUpdate(self, info):
        if (self.gestureBot is not None):
            self.gestureBot.onGestureUpdate(info)

    # -----------------------------------------------------------------------------
    #
    def requestShutdownApplication(self):
        self.shutdownEvent.set()

    #------------------------------------------------------------------------------
    #
    def cleanup(self):
        # close http server
        super().close()

        webSocket.stopHttpServerIOLoop()

        if (self.gestureService is not None):
            self.gestureService.close()
            del self.gestureService
            self.gestureService = None

        if (self.gestureBot is not None):
            self.gestureBot.close()
            del self.gestureBot
            self.gestureBot = None

        if (self.controller is not None):
            self.controller.close()
            del self.controller
            self.controller = None

    #------------------------------------------------------------------------------
    #
    def run(self):
        # wait a few milliseconds for all previous print() calls to complete...
        time.sleep(0.03)

        print("Ready.")

        #
        # loop until shutdown request or Ctrl-C...
        try:
            while not (self.shutdownEvent.wait(timeout=0.5)):
                continue
        except KeyboardInterrupt:
            pass

        print("Shutting down...")
        self.cleanup()

#------------------------------------------------------------------------------
#
# main code
#
if __name__ == '__main__':
    # initialize global variables
    settings.initialize()

    # create main application object, then run it
    settings.application = application()
    settings.application.run()
