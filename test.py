import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import QCoreApplication

class MyApp(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        btn1 = QPushButton('Button1', self)
        btn2 = QPushButton('Button2', self)
        btn1.move(50,50)
        btn2.move(150,50)
        btn1.clicked.connect(QCoreApplication.instance().quit)
        btn2.clicked.connect(self.buttonClicked)

        self.setWindowTitle('PyQt5 GUI')
        self.setGeometry(300, 300, 300, 200)
        self.show()

    def buttonClicked(self):
        print('Button Clicked!')

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
