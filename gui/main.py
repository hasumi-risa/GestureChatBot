import sys
from PyQt5.QtWidgets import (QApplication, QWidget, 
                             QPushButton, QCheckBox, 
                             QSlider, QLineEdit, 
                             QCalendarWidget, QProgressBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
 
class MainWindow(QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.initUI()
    def initUI(self):
        self.setGeometry(400, 300, 600, 800) #self.move(400, 300), self.resize(400, 500)と一緒
        self.setWindowTitle('PyQt5 sample GUI')
        self.setWindowIcon(QIcon('simulog.gif'))
        # ボタン
        btn = QPushButton('Button', self)
        btn.move(100, 50)
        btn.resize(btn.sizeHint()) # sizeHintでいいかんじの大きさにしてくれる
        
        # チェックボックス
        cb = QCheckBox('Check Box', self)
        cb.move(100, 100)
 
        # スライダー
        sld = QSlider(Qt.Horizontal, self)
        sld.setGeometry(100, 150, 200, 20)
        
        # テキストボックス
        qle = QLineEdit(self)
        qle.setGeometry(100, 200, 200, 20)
        
        # プログレスバー
        pbar = QProgressBar(self)
        pbar.setGeometry(100, 250, 200, 25)
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv) #PyQtで必ず呼び出す必要のあるオブジェクト
    main_window = MainWindow() #ウィンドウクラスのオブジェクト生成
    main_window.show() #ウィンドウの表示
    sys.exit(app.exec_()) #プログラム終了