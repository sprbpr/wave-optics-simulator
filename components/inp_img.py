import sys
from PyQt5.QtWidgets import QVBoxLayout, QFileDialog, QPushButton
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap

import numpy as np
import cv2

from components.img import Img
from components.inp import Inp


class InpImg:
    def __init__(self, window_obj, count):
        self.window_obj = window_obj
        self.count = count

        self.layout = QVBoxLayout()

        self.lens_comp = Img("Image")

        self.lens_distance_comp = Inp("Distance:", 100)
        upload_btn = QPushButton("Upload Image")

        # Add widgets to middle layout
        self.layout.addLayout(self.lens_comp.layout)
        self.layout.addSpacing(30)
        self.layout.addWidget(upload_btn)
        self.layout.addLayout(self.lens_distance_comp.layout)
        self.layout.addStretch()

        upload_btn.clicked.connect(self.upload_image)

    def upload_image(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self.window_obj, "Open File", "", "Image Files (*.png *.jpg *.bmp)"
        )
        if image_path:
            self.preprocess_image(image_path)
            # self.lens_comp.set_img(image_path)

    def preprocess_image(self, image_path):
        self.image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        h, w = gray_image.shape[:2]
        if h > w:
            padding = ((0, 0), ((h - w) // 2, (h - w) - (h - w) // 2))
        else:
            padding = (((w - h) // 2, (w - h) - (w - h) // 2), (0, 0))
        square_image = np.pad(gray_image, padding, mode="constant", constant_values=0)
        self.resized_image = cv2.resize(square_image, (201, 201))

        cv2.imwrite(f"preprocessed-img-{self.count}.png", self.resized_image)

        self.lens_comp.set_img(f"preprocessed-img-{self.count}.png")
