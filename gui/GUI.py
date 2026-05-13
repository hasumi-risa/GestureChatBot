import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from communication_robot_001 import Ui_MainWindow
import cv2
import numpy as np
import pyaudio
import wave

sample_rate = 16000
frame_length = 1024
HAAR_FILE = "opencv/data/haarcascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(HAAR_FILE)


class Browser(QtWidgets.QMainWindow):
  msec = 1 # ms
  def __init__(self,parent=None):
    super(Browser, self).__init__(parent)
    self.ui = Ui_MainWindow()
    self.ui.setupUi(self)
    
    # conversation show
    self.ui.textBrowser.append("test")
    
    
    # camera show
    self.capture = cv2.VideoCapture(0)
    if self.capture.isOpened() is False:
        raise("IO Error")
    
    self.set()
    timer = QtCore.QTimer(self)
    timer.timeout.connect(self.set)
    timer.start(self.msec)
    

    # acudio plot
    self.ui.graphicsView.setTitle("mic")
    self.ui.graphicsView.setYRange(-1.0, 1.0)
    self.curve = self.ui.graphicsView.plot()
    self.CHUNK = frame_length 
    self.RATE = sample_rate
    self.audio = pyaudio.PyAudio()
    self.stream = self.audio.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=self.RATE,
                                    input=True,
                                    output=True,
                                    frames_per_buffer=self.CHUNK)

    self.non_voice_cnt = -1
    self.reco_start = False
    self.save_frame_cut = []
    self.pre_frame = []
    self.save_id = 0
    self.sensitivity = 0.08
    
    self.timer = QtCore.QTimer()
    self.timer.timeout.connect(self.update)
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

#   def SaveAudio(self):
#     wf = wave.open("./output.wav", 'wb')
#     wf.setnchannels(1) #channel
#     wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16)) #format(sample size byte数)
#     wf.setframerate(self.RATE)
#     wf.writeframes(b''.join(self.save_frame))
#     wf.close()


  def set(self):
    ret, cv_img = self.capture.read()
    
    face = cascade.detectMultiScale(cv_img)
    for x, y, w, h in face:
        cv2.rectangle(cv_img,(x,y),(x+w,y+h),(0,0,255),1)

    if ret == False:
        return
    cv_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
    cv_img = cv2.resize(cv_img,dsize=None,fx=0.8,fy=0.8)
    height, width, dim = cv_img.shape
    
    self.image = QtGui.QImage(cv_img.data, width, height, QtGui.QImage.Format_RGB888)
    self.item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(self.image))
    self.ui.graphicsView_2.sence.clear()
    self.ui.graphicsView_2.sence.addItem(self.item)
    self.ui.graphicsView_2.setScene(self.ui.graphicsView_2.sence)


def SaveAudio(audio,RATE,save_frame,save_id):
    wf = wave.open("./output_{}.wav".format(save_id), 'wb')
    wf.setnchannels(1) #channel
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16)) #format(sample size byte数)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(save_frame))
    wf.close()    
      
if __name__ == '__main__':
  app = QtWidgets.QApplication(sys.argv)
  window = Browser()
  window.show()
  sys.exit(app.exec_())