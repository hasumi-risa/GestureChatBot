from PyQt5 import QtCore, QtGui,QtWidgets
from video_image import Ui_Form

import sys
import cv2

class Movie(QtWidgets.QDialog):
    msec = 10 # ms
    def __init__(self,parent=None):
        super(Movie, self).__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        
        self.capture = cv2.VideoCapture(0)
        if self.capture.isOpened() is False:
            raise("IO Error")
            
        self.scene = QtWidgets.QGraphicsScene()
        self.set()

       
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.set)
        timer.start(self.msec)

    def set(self):

        ret, cv_img = self.capture.read()
        if ret == False:
            return
        cv_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
        height, width, dim = cv_img.shape

        self.image = QtGui.QImage(cv_img.data, width, height, QtGui.QImage.Format_RGB888)
        self.item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(self.image))
        self.scene.clear()
        self.scene.addItem(self.item)
        self.ui.graphicsView.setScene(self.scene)
   

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    window = Movie()
    window.show()
    sys.exit(app.exec_())