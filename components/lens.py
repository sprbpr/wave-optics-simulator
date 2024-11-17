import sys
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QComboBox
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap

from components.img import Img
from components.inp import Inp


class Lens:
    def __init__(self):
        self.layout = QVBoxLayout()

        self.lens_comp = Img(
            "Lens", img_path="app-images/convex.png", scale=QSize(175, 175)
        )
        self.lens_combo = QComboBox()
        self.lens_combo.addItems(["Convex", "Concave"])
        self.lens_combo.currentIndexChanged.connect(self.on_combo_change)

        # Distance section
        self.f_comp = Inp("F:", 50)
        self.d_comp = Inp("D:", 20)

        lens_att_layout = QHBoxLayout()
        lens_att_layout.addLayout(self.f_comp.layout)
        lens_att_layout.addLayout(self.d_comp.layout)

        self.lens_distance_comp = Inp("Distance:", 100)

        # Add widgets to middle layout
        self.layout.addLayout(self.lens_comp.layout)
        self.layout.addSpacing(30)
        self.layout.addWidget(self.lens_combo)
        self.layout.addLayout(lens_att_layout)
        self.layout.addLayout(self.lens_distance_comp.layout)
        self.layout.addStretch()

    def on_combo_change(self, index):
        text = self.lens_combo.currentText()
        self.lens_comp.set_img(f"app-images/{text.lower()}.png", scale=QSize(175, 175))
        self.f_comp.inp.setText(str(-int(self.f_comp.inp.text())))
