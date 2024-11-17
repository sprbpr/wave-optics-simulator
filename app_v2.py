import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QComboBox,
    QFileDialog,
    QSlider,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap

import numpy as np
import os
import cv2

from components.img import Img
from components.inp import Inp
from components.slider import Slider
from components.lens import Lens
from components.inp_img import InpImg
from components.mask import Mask
from components.black_hole import BlackHole

from matlab import *
from quantum import *
from components.img_worker import ImageProcessingWorker

# from iqa import assess_image_quality
from iqa_ai import assess_image_quality, calculate_similarity_score


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optical Simulator")
        self.window_size = (1200, 600)
        self.setFixedSize(self.window_size[0], self.window_size[1])

        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        self.setCentralWidget(main_widget)
        main_widget.setLayout(main_layout)

        # Create the three main vertical layouts
        left_layout = QVBoxLayout()
        self.middle_layout = QHBoxLayout()
        right_layout = QVBoxLayout()

        # Create horizontal layout for images
        images_layout = QHBoxLayout()

        # Raw Image section
        self.raw_img_comp = Img("Raw Image")
        images_layout.addLayout(self.raw_img_comp.layout)

        # Arrow Image section
        arrow_image_comp = Img("", img_path="app-images/arrow.png")
        images_layout.addLayout(arrow_image_comp.layout)

        # Single Tone Image section
        self.single_tone_comp = Img("Single Tone Image")
        images_layout.addLayout(self.single_tone_comp.layout)

        # Physical Size
        self.physical_size_comp = Inp("Physical Size:", 10)

        # Buttons
        upload_btn = QPushButton("Upload Image")
        lens_btn = QPushButton("Add Lens")
        imgs_btn = QPushButton("Add Images")
        masks_btn = QPushButton("Add Masks")
        black_hole_btn = QPushButton("Add Black Holes")
        remove_btn = QPushButton("Remove Object")
        reset_btn = QPushButton("Reset")

        # Add widgets to left layout
        left_layout.addLayout(images_layout)
        left_layout.addSpacing(30)
        left_layout.addLayout(self.physical_size_comp.layout)
        left_layout.addWidget(upload_btn)

        btns_layout = QHBoxLayout()
        btns_layout.addWidget(lens_btn)
        btns_layout.addWidget(imgs_btn)
        btns_layout.addWidget(masks_btn)
        btns_layout.addWidget(black_hole_btn)

        left_layout.addLayout(btns_layout)
        left_layout.addWidget(remove_btn)
        left_layout.addStretch()
        left_layout.addWidget(reset_btn)

        # Lens section
        self.lens_comp = Lens()
        self.middle_layout.addLayout(self.lens_comp.layout)
        # self.setMinimumSize(1300, 600)
        # self.lens_comp_2 = Lens()
        # middle_layout.addLayout(self.lens_comp_2.layout)

        # Right section
        self.out_img_comp = Img("Screen")

        # Distance input
        self.screen_distance_comp = Inp("Distance:", 100)
        self.scrn_dist_slider_comp = Slider(self.on_slider_value_changed)

        # Buttons
        show_btn = QPushButton("Show")
        set_screen_btn = QPushButton("Set to Screen")
        find_spot_btn = QPushButton("Find Finest Spot")

        # Add widgets to right layout
        right_layout.addLayout(self.out_img_comp.layout)
        right_layout.addSpacing(30)
        right_layout.addLayout(self.screen_distance_comp.layout)
        right_layout.addLayout(self.scrn_dist_slider_comp.layout)
        right_layout.addStretch()
        right_layout.addWidget(show_btn)
        right_layout.addWidget(set_screen_btn)

        quantum_btn_layout = QHBoxLayout()
        self.n_tries = Inp("# Tries:", 10)
        quantum_btn_layout.addLayout(self.n_tries.layout)
        self.n_bar = Inp("N Bar:", 10)
        quantum_btn_layout.addLayout(self.n_bar.layout)
        quantum_btn = QPushButton("Go Quantum")
        quantum_btn_layout.addWidget(quantum_btn)

        right_layout.addLayout(quantum_btn_layout)
        right_layout.addWidget(find_spot_btn)

        # Add frames to main layout
        main_layout.addLayout(left_layout)
        main_layout.addLayout(self.middle_layout)
        main_layout.addLayout(right_layout)

        # Connect signals
        upload_btn.clicked.connect(self.upload_image)
        lens_btn.clicked.connect(self.add_lens)
        imgs_btn.clicked.connect(self.add_imgs)
        black_hole_btn.clicked.connect(self.add_black_hole)
        masks_btn.clicked.connect(self.add_masks)
        remove_btn.clicked.connect(self.remove)
        reset_btn.clicked.connect(self.reset)
        show_btn.clicked.connect(self.show_output)
        set_screen_btn.clicked.connect(self.set_to_screen)
        quantum_btn.clicked.connect(self.go_quantum)
        find_spot_btn.clicked.connect(self.find_finest_spot)

        self.reset()

    def go_quantum(self):
        L = 10
        N = 301
        x1 = np.linspace(-L / 2, L / 2, N)
        y1 = np.linspace(-L / 2, L / 2, N)
        X1, Y1 = np.meshgrid(x1, y1)

        u1 = self.resized_image
        n_bar = int(self.n_tries.inp.text())
        test_num = int(self.n_tries.inp.text())

        z1 = 25 * L
        C = 5e3
        lad0 = z1 / C
        D = 10 * L

        u2, x2, y2 = screen(u1, x1, y1, lad0, z1, D, 3)

        photon_intensity = quantum_sampling(u2, x2, y2, n_bar, test_num)
        self.quantum_image = cv2.resize(photon_intensity, (201, 201))
        # cv2.imwrite(
        #     "quantum-img.png", (255 * np.abs(self.quantum_image)).astype(np.uint8)
        # )
        # cv2.imwrite("quantum-img.png", self.quantum_image)
        plt.imsave(f"quantum-img.png", photon_intensity, cmap="gray", format="png")
        self.out_img_comp.set_img(f"quantum-img.png")

    def upload_image(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "Image Files (*.png *.jpg *.bmp)"
        )
        if image_path:
            self.set_image(self.raw_img_comp.image, image_path)
            self.preprocess_image(image_path)

    def add_lens(self):
        self.window_size = (self.window_size[0] + 240, self.window_size[1])
        self.setFixedSize(self.window_size[0], self.window_size[1])
        self.lens_comp_2 = Lens()
        self.middle_layout.addLayout(self.lens_comp_2.layout)

    def add_imgs(self):
        self.window_size = (self.window_size[0] + 240, self.window_size[1])
        self.setFixedSize(self.window_size[0], self.window_size[1])
        self.lens_comp_2 = InpImg(self, self.middle_layout.count())
        self.middle_layout.addLayout(self.lens_comp_2.layout)

    def add_masks(self):
        self.window_size = (self.window_size[0] + 240, self.window_size[1])
        self.setFixedSize(self.window_size[0], self.window_size[1])
        self.lens_comp_2 = Mask()
        self.middle_layout.addLayout(self.lens_comp_2.layout)

    def add_black_hole(self):
        self.window_size = (self.window_size[0] + 240, self.window_size[1])
        self.setFixedSize(self.window_size[0], self.window_size[1])
        self.lens_comp_2 = BlackHole()
        self.middle_layout.addLayout(self.lens_comp_2.layout)

    def remove(self):
        count = self.middle_layout.count()
        last_layout_item = self.middle_layout.takeAt(count - 1)
        last_layout = last_layout_item.layout()
        if last_layout:
            self.clear_layout(last_layout)
            last_layout.deleteLater()
            self.window_size = (self.window_size[0] - 240, self.window_size[1])
            self.setFixedSize(self.window_size[0], self.window_size[1])

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())
                item.layout().deleteLater()

    def reset(self):
        for f in os.listdir("output-imgs"):
            os.remove(os.path.join("output-imgs", f))
        self.raw_img_comp.set_img(None)
        self.single_tone_comp.set_img(None)
        self.out_img_comp.set_img(None)

        self.physical_size_comp.set_value(10)
        self.lens_comp.lens_combo.setCurrentIndex(0)
        self.lens_comp.f_comp.set_value(50)
        self.lens_comp.d_comp.set_value(20)
        self.lens_comp.lens_distance_comp.set_value(100)

        self.screen_distance_comp.set_value(100)
        self.scrn_dist_slider_comp.slider.setValue(100)
        self.scrn_dist_slider_comp.min_input.setText(str(80))
        self.scrn_dist_slider_comp.n_bin_input.setText(str(4))
        self.scrn_dist_slider_comp.max_input.setText(str(120))

        self.n_tries.set_value(10)
        self.n_bar.set_value(10)

    # def on_combo_change(self, index):
    #     text = self.lens_combo.currentText()
    #     self.lens_comp.set_img(f"app-images/{text.lower()}.png", scale=QSize(175, 175))

    def gen_img(self, z2):
        l = int(self.physical_size_comp.inp.text())
        n = 201
        x1 = np.linspace(-l // 2, l // 2, n)
        y1 = np.linspace(-l // 2, l // 2, n)
        lad0 = l / 1000
        z1 = int(self.lens_comp.lens_distance_comp.inp.text())
        f = int(self.lens_comp.f_comp.inp.text())
        d = int(self.lens_comp.d_comp.inp.text())
        pad_factor = 1

        u1 = self.resized_image
        u2, x2, y2 = free_space(lad0, z1, u1, x1, y1, pad_factor=pad_factor)
        u3, x3, y3 = ring_lens(u2, 0, 0, d / 2, lad0, f, x2, y2)
        u4, x4, y4 = free_space(lad0, z2, u3, x3, y3, pad_factor=pad_factor)

        mask_x = np.abs(x4) <= l / 2
        mask_y = np.abs(y4) <= l / 2
        u4_cropped = u4[np.ix_(mask_y, mask_x)]

        out_path = f"output-imgs/{str(z2)}.png"
        cv2.imwrite(out_path, (255 * np.abs(u4_cropped)).astype(np.uint8))

        return out_path

    def show_output(self):
        # l = int(self.physical_size_comp.inp.text())
        # n = 201
        # x1 = np.linspace(-l // 2, l // 2, n)
        # y1 = np.linspace(-l // 2, l // 2, n)
        # z1 = int(self.lens_comp.lens_distance_comp.inp.text())
        # C = 1e3
        # lad0 = l / 1000
        # R = int(self.lens_comp.d_comp.inp.text()) // 2
        # f = int(self.lens_comp.f_comp.inp.text())
        # u1 = self.resized_image

        # u2, x2, y2 = ring_lens(u1, x1, y1, R, lad0, f, z1, pad_factor=2)
        # z2 = int(self.screen_distance_comp.inp.text())
        # u3, x3, y3 = screen(u2, x2, y2, lad0, z2, l, pad_factor=2)

        # mask_x = np.abs(x3) <= l / 2
        # mask_y = np.abs(y3) <= l / 2
        # u3_cropped = u3[np.ix_(mask_y, mask_x)]

        # cv2.imwrite(
        #     f"output-imgs/{int(z2)}.png",
        #     (255 * np.abs(u3)).astype(np.uint8),
        # )
        # self.out_img_comp.set_img(f"output-imgs/{float(z2)}.png")

        selected_z2 = int(self.screen_distance_comp.inp.text())
        path = self.gen_img(selected_z2)
        self.out_img_comp.set_img(path)

        min_value = int(self.scrn_dist_slider_comp.min_input.text())
        max_value = int(self.scrn_dist_slider_comp.max_input.text())
        bin_count = int(self.scrn_dist_slider_comp.n_bin_input.text())
        # z2s = np.linspace(min_value, max_value, num=bin_count)

        if bin_count > 1:
            interval = (max_value - min_value) // (bin_count)
        else:
            interval = max_value - min_value
        z2s = np.arange(min_value, max_value, interval)
        z2s = np.append(z2s, max_value)
        # z2s.append(max_value)
        # print(z2s)

        for z2 in z2s:
            z2 = self.round_slider(z2)
            # print(z2)
            self.gen_img(z2)

        # self.out_img_comp.set_img(f"output-imgs/{str(selected_z2)}.png")

    def set_to_screen(self):
        print("Set to screen clicked")

    def find_finest_spot(self):
        step = 10
        z2 = int(self.screen_distance_comp.inp.text())

        self.worker = ImageProcessingWorker(
            self,
            self.gen_img,
            calculate_similarity_score,
            z2,
            step,
            self.single_tone_comp.img_path,
        )
        self.worker.update_image_signal.connect(self.update_image)
        # self.worker.finished_signal.connect(self.on_worker_finished)
        self.worker.start()  # Start the worker thread

        # path = self.gen_img(z2)
        # best_score = 0
        # for _ in range(10):
        #     path = self.gen_img(z2)
        #     score = assess_image_quality(path)["overall_score"]
        #     self.out_img_comp.set_img(path)
        #     if score > best_score:
        #         z2 += step
        #         best_score = score
        #     elif score < best_score:
        #         step = -(step // 2)
        #         z2 += step
        #     else:
        #         break

    def update_image(self, path):
        # print("##", path)
        self.out_img_comp.set_img(path)

    def set_image(self, label, image_path, scale=None):
        if not scale:
            scale = label.size()
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap.scaled(scale, Qt.KeepAspectRatio))

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

        cv2.imwrite("preprocessed-img.png", self.resized_image)

        self.single_tone_comp.set_img("preprocessed-img.png")

    def round_slider(self, value):
        rounded_value = (
            round(value / self.scrn_dist_slider_comp.slider.tickInterval())
            * self.scrn_dist_slider_comp.slider.tickInterval()
        )
        return rounded_value

    def on_slider_value_changed(self, value):
        rounded_value = self.round_slider(value)
        self.scrn_dist_slider_comp.slider.setValue(rounded_value)
        self.screen_distance_comp.inp.setText(str(rounded_value))

        selected_z2 = int(self.screen_distance_comp.inp.text())
        self.out_img_comp.set_img(f"output-imgs/{selected_z2}.png")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
