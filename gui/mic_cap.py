import numpy as np
import sys

from PyQt5 import QtWidgets, QtCore

import pyqtgraph as pg
from pyqtgraph import PlotWidget, plot

import pyaudio
import wave

sample_rate = 16000
frame_length = 1024


class PlotWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(PlotWindow, self).__init__(*args, **kwargs)
        
        self.graphwidget = pg.PlotWidget()
        self.setCentralWidget(self.graphwidget)
        
        self.savebutton = QtWidgets.QPushButton("save", self)
        self.savebutton.move(100,0)
        self.graphwidget.setTitle("mic")
        self.graphwidget.setYRange(-1.0, 1.0)
        self.curve = self.graphwidget.plot()  # プロットデータを入れる場所

        # マイク設定
        self.CHUNK = frame_length  # 1度に読み取る音声のデータ幅
        self.RATE = sample_rate  # サンプリング周波数
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=self.RATE,
                                      input=True,
                                      output=True,
                                      frames_per_buffer=self.CHUNK)

        # アップデート時間設定
        self.non_voice_cnt = -1
        self.reco_start = False
        self.save_frame_cut = []
        self.pre_frame = []
        self.save_id = 0
        self.sensitivity = 0.08
        
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update)
        self.savebutton.clicked.connect(self.SaveAudio)
        self.timer.start() 

        self.data = np.zeros(self.CHUNK)
        self.save_frame_ = []

    def update(self):
        self.data = np.append(self.data, self.AudioInput())
        
        if len(self.data)/1024>10:
            self.data = self.data[1024:]
            
        if self.non_voice_cnt == -1 and max(self.data)>self.sensitivity:
            self.non_voice_cnt = 0
            self.reco_start = True
            for i in range(1,10):
                self.save_frame_cut.append(self.pre_frame[-i])
            self.pre_frame = []
            self.save_frame_cut.append(self.ret_)
            
        elif max(self.data)>self.sensitivity:
            self.non_voice_cnt = 0
            self.reco_start = True
            self.save_frame_cut.append(self.ret_)
            
        elif self.reco_start and max(self.data)<=self.sensitivity:
            self.save_frame_cut.append(self.ret_)
            self.non_voice_cnt += 1
            
            
        if self.non_voice_cnt > 10 and self.reco_start:
            SaveAudio(self.audio,self.RATE,self.save_frame_cut,self.save_id)
            self.save_id += 1
            self.save_frame_cut = []
            self.reco_start = False
            self.non_voice_cnt = -1
            
        self.pre_frame.append(self.ret_)
        if len(self.pre_frame)>10:
            self.pre_frame = self.pre_frame[1::]  
        self.curve.setData(self.data)
        self.save_frame_.append(self.ret_)
        self.save_frame =  self.save_frame_

    def AudioInput(self):
        self.ret_ = self.stream.read(self.CHUNK)
        ret = np.frombuffer(self.ret_, dtype="int16") / 32768
        return ret
    
    def SaveAudio(self):
        # self.stream.stop_stream()
        # self.stream.close()
        # self.audio.terminate()
        wf = wave.open("./output.wav", 'wb')
        wf.setnchannels(1) #channel
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16)) #format(sample size byte数)
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.save_frame))
        wf.close()

def SaveAudio(audio,RATE,save_frame,save_id):
    wf = wave.open("./output_{}.wav".format(save_id), 'wb')
    wf.setnchannels(1) #channel
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16)) #format(sample size byte数)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(save_frame))
    wf.close()
        
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = PlotWindow()
    w.show()
    sys.exit(app.exec_())
    
    