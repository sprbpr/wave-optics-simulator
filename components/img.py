import sys
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QFrame
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap


class Img:
    def __init__(self, label, img_path=None, size=(200, 200), scale=QSize(150, 150)):
        self.img_path = img_path
        self.layout = QVBoxLayout()

        self.label = QLabel(label)
        self.label.setAlignment(Qt.AlignCenter)

        self.image = QLabel("")
        if img_path:
            self.set_img(img_path, scale=scale)
        else:
            self.image.setFrameStyle(QFrame.Box)

        self.image.setFixedSize(size[0], size[1])
        self.image.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.image)
        self.layout.setAlignment(Qt.AlignCenter)

    def set_img(self, img_path, scale=None):
        self.img_path = img_path
        if img_path is None:
            self.image.setPixmap(QPixmap())
            self.image.setFrameStyle(QFrame.Box)
        else:
            if not scale:
                scale = self.image.size()
            pixmap = QPixmap(img_path)
            self.image.setPixmap(pixmap.scaled(scale, Qt.KeepAspectRatio))
            self.image.setAlignment(Qt.AlignCenter)
