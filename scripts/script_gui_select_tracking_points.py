"""
Use right click for initial global point selection. Can always be overwritten by selecting new points.
Use left click to add tracking points. New left clicks will add more points. To remove previous ones, use middle click.
Pressing next button will automatically save current progress and will go to the next video in the same directory. 
"""

import os
import sys
import cv2
import json
import numpy as np
import pandas as pd
from natsort import natsorted
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from PyQt5 import QtCore, QtWidgets, QtGui

# INPUT_FRAME_DIR = "/path/to/Cataract-1K/video_frames_jpg"
# OUTPUT_ANN_DIR = "/path/to/ann/Cataract-1K/ann_glob_pretracking_points"

INPUT_FRAME_DIR = "/path/to/Cataracts-50/video_frames_jpg"
OUTPUT_ANN_DIR = "/path/to/ann/Cataracts-50/ann_glob_pretracking_points"

# Switch this to None if normal annotation
POST_REFINEMENT_ANN = None
# POST_REFINEMENT_ANN = "/path/to/ann/Cataracts-50/ann_glob_posttracking_refinement.csv"

@dataclass
class TrackingPoints:
    points: List[Tuple[int, int]] = field(default_factory=list)
    def undo(self):
        self.points = [p for p in self.points[:-1]]
    
    def clear(self):
        self.points = []


@dataclass
class PointObject:
    name: str
    frame_idx: int = 0
    zoom_level: str = None
    global_point: Tuple[int, int] = None
    tracking_points: TrackingPoints = field(default_factory=TrackingPoints)


class Point_Selector(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Global and Tracking Points")
        self.setGeometry(50, 50, 650, 385)
        self.input_frame_dir = INPUT_FRAME_DIR
        self.output_ann_path = OUTPUT_ANN_DIR
        self.post_refinement_ann_path = POST_REFINEMENT_ANN
        self.frame_size = (512, 288)
        self.video_path = None
        self.init_ui()
    
    def init_obj(self):
        video_name = os.path.basename(self.video_path)
        self.obj = PointObject(name=video_name)
        if self.btn_zoom_level.checkedButton() is not None:
            zoom_text = self.btn_zoom_level.checkedButton().text()
            self.select_zoom_level(zoom_text)

    def init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        display_layout = QtWidgets.QVBoxLayout()
        button_layout = QtWidgets.QVBoxLayout()

        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(self.frame_size[0], self.frame_size[1])
        self.video_label.setScaledContents(True)
        self.video_label.installEventFilter(self)
        self.current_qimage = None

        title_layout = QtWidgets.QHBoxLayout()
        self.vid_title_status = QtWidgets.QLabel("")
        title_layout.addWidget(self.vid_title_status)
        title_layout.addSpacing(25)
        self.text_box = QtWidgets.QLineEdit()
        self.text_box.setPlaceholderText("Enter Frame Index...")
        self.text_box.returnPressed.connect(self.on_textbox_entered)
        title_layout.addWidget(self.text_box)

        display_layout.addLayout(title_layout)
        display_layout.addWidget(self.video_label, alignment=QtCore.Qt.AlignCenter)

        self.frame_status = QtWidgets.QLabel("")
        display_layout.addWidget(self.frame_status)
        self.status = QtWidgets.QLabel("")
        display_layout.addWidget(self.status)

        button_layout.addStretch()
        button_layout.addSpacing(-25)
        btn_load = QtWidgets.QPushButton("Load Video")
        btn_load.clicked.connect(self.select_video)
        button_layout.addWidget(btn_load)
        self.btn_load_next = QtWidgets.QPushButton("Load Next")
        self.btn_load_next.clicked.connect(self.next_video)
        self.btn_load_next.setEnabled(False)
        button_layout.addWidget(self.btn_load_next)
        self.btn_load_prev = QtWidgets.QPushButton("Load Previous")
        self.btn_load_prev.clicked.connect(self.previous_video)
        self.btn_load_prev.setEnabled(False)
        button_layout.addWidget(self.btn_load_prev)
        button_layout.addSpacing(20)
        self.btn_clear_all = QtWidgets.QPushButton("Clear All")
        self.btn_clear_all.clicked.connect(self.clear_annotations)
        self.btn_clear_all.setEnabled(False)
        button_layout.addWidget(self.btn_clear_all)
        button_layout.addSpacing(20)
        
        self.btn_zoom_level = QtWidgets.QButtonGroup(self)
        self.btn_zoom_level.setExclusive(True)
        btn_zoom_low = QtWidgets.QPushButton("Zoom: Low")
        btn_zoom_low.clicked.connect(lambda checked: self.select_zoom_level(btn_zoom_low.text()))
        btn_zoom_low.setCheckable(True)
        self.btn_zoom_level.addButton(btn_zoom_low)
        button_layout.addWidget(btn_zoom_low)
        btn_zoom_high = QtWidgets.QPushButton("Zoom: High")
        btn_zoom_high.clicked.connect(lambda checked: self.select_zoom_level(btn_zoom_high.text()))
        btn_zoom_high.setCheckable(True)
        self.btn_zoom_level.addButton(btn_zoom_high)
        button_layout.addWidget(btn_zoom_high)
        button_layout.addSpacing(20)
        
        self.btn_next_frame = QtWidgets.QPushButton("Frame: Next")
        self.btn_next_frame.clicked.connect(lambda: self.change_frame(+4))
        self.btn_next_frame.setEnabled(False)
        button_layout.addWidget(self.btn_next_frame)
        self.btn_prev_frame = QtWidgets.QPushButton("Frame: Previous")
        self.btn_prev_frame.clicked.connect(lambda: self.change_frame(-4))
        self.btn_prev_frame.setEnabled(False) 
        button_layout.addWidget(self.btn_prev_frame)
        self.btn_first_frame = QtWidgets.QPushButton("Frame: First")
        self.btn_first_frame.clicked.connect(lambda: self.change_frame(-100000))
        self.btn_first_frame.setEnabled(False) 
        button_layout.addWidget(self.btn_first_frame)
        button_layout.addStretch()

        layout.addLayout(display_layout)
        layout.addLayout(button_layout) 


    def select_video(self):
        if self.post_refinement_ann_path is None:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", "")
            if not path:
                QtWidgets.QMessageBox.critical(self, "Error", "No folder selected")
                return
            self.video_path = path
            self.init_obj()
            self.goto_frame()
        
        else:
            self.post_refinement_df = pd.read_csv(self.post_refinement_ann_path, skipinitialspace=True)
            self.video_path = os.path.join(self.input_frame_dir, self.post_refinement_df.iloc[0]['video_id'])
            self.status.setText("Total Frames:" + str(len(os.listdir(self.video_path))))
            self.init_obj()
            self.obj.frame_idx = int(self.post_refinement_df.iloc[0]['frame_idx'])
            self.goto_frame()
        
        self.btn_next_frame.setEnabled(True)
        self.btn_prev_frame.setEnabled(True)
        self.btn_first_frame.setEnabled(True)
        self.btn_clear_all.setEnabled(True)
        self.btn_load_next.setEnabled(True)
        self.btn_load_prev.setEnabled(True)
        

    def find_video(self, go_previous=False):
        parent_path = os.path.dirname(self.video_path)
        video_name = os.path.basename(self.video_path)
        
        if self.post_refinement_ann_path is None:
            video_list = natsorted(os.listdir(parent_path))
        else: video_list = self.post_refinement_df['video_id'].unique().tolist()
        
        video_idx = video_list.index(video_name)
        video_idx = video_idx - 1 if go_previous else video_idx + 1

        if video_idx >= len(video_list) or video_idx < 0:
            QtWidgets.QMessageBox.information(self, "Info", "No more videos in the directory")
            return
            
        if self.post_refinement_ann_path is None:    
            self.video_path = os.path.join(parent_path, video_list[video_idx])
            self.init_obj()
        else: 
            self.video_path = os.path.join(self.input_frame_dir, self.post_refinement_df.iloc[video_idx]['video_id'])
            self.status.setText("Total Frames:" + str(len(os.listdir(self.video_path))))
            self.init_obj()
            self.obj.frame_idx = int(self.post_refinement_df.iloc[video_idx]['frame_idx'])

        self.frame_status.setText(f"<b>Total Tracking Points:</b> {len(self.obj.tracking_points.points)} / 40")
        self.goto_frame()


    def next_video(self):
        if not self.export_annotations():
            return
        self.find_video(go_previous=False)

    def previous_video(self):
        self.find_video(go_previous=True)


    def on_textbox_entered(self):
        text = self.text_box.text()
        if text.isdigit():
            frame_idx = int(text)
            self.obj.frame_idx = frame_idx
            self.goto_frame()
        else:
            self.status.setText(f"Please enter a valid frame index (integer).")


    def change_frame(self, delta: int):
        new_idx = self.obj.frame_idx + delta
        if new_idx >= 0:
            self.obj.frame_idx = new_idx
        else:
            self.obj.frame_idx = 0
        self.goto_frame()


    def goto_frame(self):
        frame_idx = str(self.obj.frame_idx).zfill(10)
        frame_path = os.path.join(self.video_path, frame_idx+".jpg")
        frame_rgb = cv2.imread(frame_path)
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, self.frame_size, interpolation=cv2.INTER_AREA)
        frame_rgb = self.render_overlays(frame_rgb)
        self.video_label_set_frame(frame_rgb)
        self.vid_title_status.setText(f"<b>Selected Video:</b> {os.path.basename(self.video_path)} | <b>Frame Index:</b> {frame_idx}")

            
    def clear_annotations(self):
        self.init_obj()
        self.goto_frame()
        self.frame_status.setText(f"<b>Total Tracking Points:</b> {len(self.obj.tracking_points.points)} / 40")
        self.status.setText(f"Cleared All Annotations")


    def export_annotations(self):
        if self.obj.global_point is not None and len(self.obj.tracking_points.points) > 40 and self.obj.zoom_level is not None:
            ann_data = {
                "name": self.obj.name,
                "zoom_level": self.obj.zoom_level,
                "frames": 
                    [{
                      "frame_idx": self.obj.frame_idx,
                      "global_point": self.obj.global_point,
                      "tracking_points": self.obj.tracking_points.points
                    }]}
            
            ann_file_path = os.path.join(self.output_ann_path, f"{self.obj.name}.json")
            # Read first the json, if exists only add for new idx or replace existing idx
            if os.path.exists(ann_file_path):
                with open(ann_file_path, "r") as f:
                    old_ann_data = json.load(f)
                    for frame in old_ann_data["frames"]:
                        if frame["frame_idx"] != self.obj.frame_idx:
                            ann_data["frames"].append(frame)
                # Sort frames by frame idx
                ann_data["frames"] = sorted(ann_data["frames"], key=lambda x: x["frame_idx"])

            with open(ann_file_path, 'w') as f:
                json.dump(ann_data, f, indent=4)
            self.status.setText(f"Saving Annotations Before the Loading Next Video")
            return True
        else:
            self.status.setText(f"No Sufficient Annotations to Save")
            return False
                

    def select_zoom_level(self, zoom_level: str):
        if hasattr(self, "obj"):
            if zoom_level == "Zoom: Low": self.obj.zoom_level = "low"
            elif zoom_level == "Zoom: High": self.obj.zoom_level = "high"
        return
            

    def render_overlays(self, frame_rgb: np.ndarray) -> np.ndarray:
        canvas = frame_rgb.copy()
        if canvas.dtype != np.uint8:
            canvas = (canvas * 255).astype(np.uint8) if canvas.max() <= 1.0 else canvas.astype(np.uint8)
        h, w, _ = canvas.shape
        canvas_f = canvas.astype(np.float32)
        
        if self.obj.global_point is not None:
            x, y = self.obj.global_point
            cv2.circle(canvas_f, (int(x), int(y)), 3, (255,0,0), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas_f, (int(x), int(y)), 100, (255,0,0), 2, lineType=cv2.LINE_AA)
        for p in self.obj.tracking_points.points:
            x, y = p
            cv2.circle(canvas_f, (int(x), int(y)), 3, (20,150,255), -1, lineType=cv2.LINE_AA)
        canvas_out = np.clip(canvas_f, 0, 255).astype(np.uint8)
        return canvas_out
    

    def video_label_set_frame(self, rgb_frame: np.ndarray):
        if rgb_frame is None:
            return
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.current_qimage = qimg.copy()
        pix = QtGui.QPixmap.fromImage(self.current_qimage)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


    def eventFilter(self, obj, event):
        if obj == self.video_label:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                pos = event.pos()
                x, y = pos.x(), pos.y()
                if event.button() == QtCore.Qt.RightButton:
                    self.obj.global_point = (x, y)
                    self.status.setText(f"Global Point is Set")
                elif event.button() == QtCore.Qt.LeftButton:
                    self.obj.tracking_points.points.append((x, y))
                    self.status.setText(f"Added New Tracking Point")
                elif event.button() == QtCore.Qt.MiddleButton:
                    self.obj.tracking_points.undo()
                    self.status.setText(f"Removed Previous Tracking Point")
                
                self.frame_status.setText(f"<b>Total Tracking Points:</b> {len(self.obj.tracking_points.points)} / 40")
                self.goto_frame()
        return super().eventFilter(obj, event)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = Point_Selector()
    win.show()
    sys.exit(app.exec_())