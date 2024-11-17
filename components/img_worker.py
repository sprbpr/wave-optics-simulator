from PyQt5.QtCore import QThread, pyqtSignal


class ImageProcessingWorker(QThread):
    update_image_signal = pyqtSignal(str)  # Signal to update image path in UI
    finished_signal = pyqtSignal()  # Signal when processing is complete

    def __init__(
        self,
        parent_window_obj,
        gen_img_func,
        assess_quality_func,
        initial_z2,
        step,
        single_tone_img_path=None,
    ):
        super().__init__()
        self.parent_window_obj = parent_window_obj
        self.gen_img_func = gen_img_func
        self.assess_quality_func = assess_quality_func
        self.z2 = initial_z2
        self.step = step
        self.last_score = 0
        self.best_score = 0
        self.best_path = ""
        self.best_z2 = 0

        self.single_tone_img_path = single_tone_img_path

    def run(self):
        for _ in range(10):
            path = self.gen_img_func(self.z2)
            score = self.assess_quality_func(path, self.single_tone_img_path)
            self.update_image_signal.emit(path)  # Emit signal to update UI image
            self.parent_window_obj.screen_distance_comp.inp.setText(str(self.z2))
            self.parent_window_obj.scrn_dist_slider_comp.slider.setValue(self.z2)

            if score > self.best_score:
                self.best_score = score
                self.best_path = path
                self.best_z2 = self.z2

            if score > self.last_score:
                self.z2 += self.step
                self.z2 = self.parent_window_obj.round_slider(self.z2)
            elif score < self.last_score:
                self.step = -(self.step // 2)
                self.z2 += self.step
                self.z2 = self.parent_window_obj.round_slider(self.z2)
            else:
                break

            self.last_score = score

        self.update_image_signal.emit(self.best_path)
        self.parent_window_obj.screen_distance_comp.inp.setText(str(self.best_z2))
        self.parent_window_obj.scrn_dist_slider_comp.slider.setValue(self.best_z2)
        self.finished_signal.emit()  # Emit finished signal when loop is complete
