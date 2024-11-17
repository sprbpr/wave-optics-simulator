from PyQt5.QtWidgets import QHBoxLayout, QLabel, QLineEdit


class Inp:
    def __init__(self, label, init_val, layout=QHBoxLayout):
        self.layout = layout()

        self.label = QLabel(label)
        self.inp = QLineEdit(str(init_val))

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.inp)

    def set_value(self, val):
        self.inp.setText(str(val))
