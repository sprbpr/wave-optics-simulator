import sys
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QComboBox
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap

from components.img import Img
from components.inp import Inp


class BlackHole:
    def __init__(self):
        self.layout = QVBoxLayout()

        self.lens_comp = Img(
            "Mask", img_path="app-images/black-hole.png", scale=QSize(150, 150)
        )

        self.lens_distance_comp = Inp("Distance:", 100)

        # Add widgets to middle layout
        self.layout.addLayout(self.lens_comp.layout)
        self.layout.addSpacing(30)
        self.layout.addLayout(self.lens_distance_comp.layout)
        self.layout.addStretch()
