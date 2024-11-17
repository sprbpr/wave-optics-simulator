import sys
from PyQt5.QtWidgets import QGridLayout, QVBoxLayout, QSlider, QLabel, QLineEdit
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap


class Slider:
    def __init__(self, on_val_changed):
        self.layout = QVBoxLayout()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(80)
        self.slider.setMaximum(120)
        self.slider.setValue(100)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(on_val_changed)

        ctrl_layout = QGridLayout()
        min_label = QLabel("Min")
        n_bin_label = QLabel("# Bins")
        max_label = QLabel("Max")
        self.min_input = QLineEdit(str(self.slider.minimum()))
        self.min_input.editingFinished.connect(self.on_slider_controller_change)
        self.n_bin_input = QLineEdit(
            str(
                (self.slider.maximum() - self.slider.minimum())
                // self.slider.tickInterval()
            )
        )
        self.n_bin_input.editingFinished.connect(self.on_slider_controller_change)
        self.max_input = QLineEdit(str(self.slider.maximum()))
        self.max_input.editingFinished.connect(self.on_slider_controller_change)
        ctrl_layout.addWidget(min_label, 0, 0)
        ctrl_layout.addWidget(n_bin_label, 0, 1)
        ctrl_layout.addWidget(max_label, 0, 2)
        ctrl_layout.addWidget(self.min_input, 1, 0)
        ctrl_layout.addWidget(self.n_bin_input, 1, 1)
        ctrl_layout.addWidget(self.max_input, 1, 2)
        ctrl_layout.setAlignment(min_label, Qt.AlignCenter)
        ctrl_layout.setAlignment(n_bin_label, Qt.AlignCenter)
        ctrl_layout.setAlignment(max_label, Qt.AlignCenter)

        self.layout.addWidget(self.slider)
        self.layout.addLayout(ctrl_layout)

    def on_slider_controller_change(self):
        min_value = int(self.min_input.text())
        max_value = int(self.max_input.text())
        bin_count = int(self.n_bin_input.text())

        if bin_count > 1:
            interval = (max_value - min_value) // (bin_count)
        else:
            interval = max_value - min_value

        # Update slider settings
        self.slider.setMinimum(min_value)
        self.slider.setMaximum(max_value)
        self.slider.setTickInterval(interval)
        self.slider.setSingleStep(interval)

        # Reset slider value and display
        self.slider.setValue(min_value)
        # self.value_line_edit.setText(str(min_value))
