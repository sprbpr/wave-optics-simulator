import sys
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QComboBox
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap

from components.img import Img
from components.inp import Inp


class Mask:
    def __init__(self):
        self.layout = QVBoxLayout()

        self.lens_comp = Img(
            "Mask", img_path="app-images/circle.png", scale=QSize(150, 150)
        )
        self.lens_combo = QComboBox()
        self.lens_combo.addItems(["Circle", "Rectangle", "Bahtinov"])
        self.lens_combo.currentIndexChanged.connect(self.on_combo_change)

        self.lens_distance_comp = Inp("Distance:", 100)

        # Add widgets to middle layout
        self.layout.addLayout(self.lens_comp.layout)
        self.layout.addSpacing(30)
        self.layout.addWidget(self.lens_combo)
        self.layout.addLayout(self.lens_distance_comp.layout)
        self.layout.addStretch()

    def on_combo_change(self, index):
        text = self.lens_combo.currentText()
        self.lens_comp.set_img(f"app-images/{text.lower()}.png", scale=QSize(150, 150))
