from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
import numpy as np
from typing import List, Optional

from app.backend.pipeline.pipeline import PipelineStage


class AppView(QMainWindow):
    # Signals
    start_mission = Signal()
    emergency_stop = Signal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Mission Control")
        self.setGeometry(100, 100, 800, 800)
        self._init_ui()
        
    def _init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Stream display
        self.stream_label = QLabel()
        self.stream_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stream_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.stream_label)
        
        # Pipeline display
        pipeline_widget = QWidget()
        pipeline_layout = QHBoxLayout(pipeline_widget)
        self.pipeline_nodes: List[QLabel] = []
        for state in [PipelineStage.LAUNCH, PipelineStage.SCAN,
                      PipelineStage.IDENTIFY, PipelineStage.TRACK,
                      PipelineStage.RETURN]:
            node = QLabel(state.name)
            node.setAlignment(Qt.AlignmentFlag.AlignCenter)
            node.setStyleSheet("""
                QLabel {
                    background-color: #666666;
                    color: white;
                    padding: 10px;
                    border-radius: 15px;
                    min-width: 80px;
                }
            """)
            self.pipeline_nodes.append(node)
            pipeline_layout.addWidget(node)
        main_layout.addWidget(pipeline_widget)
        
        # Controls
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        
        self.start_button = QPushButton("Start Mission")
        self.emergency_button = QPushButton("Emergency Stop")
        self.emergency_button.setStyleSheet("background-color: #cc0000; color: white;")
        
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.emergency_button)
        main_layout.addWidget(controls_widget)
        
        # Mission log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        main_layout.addWidget(self.log_text)
        
        # Connect signals
        self.start_button.clicked.connect(self.start_mission.emit)
        self.emergency_button.clicked.connect(self.emergency_stop.emit)
        
    def update_frame(self, frame: np.ndarray) -> None:
        """Update the displayed frame."""
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.stream_label.setPixmap(QPixmap.fromImage(qimg))
        
    def update_pipeline_state(self, state: PipelineStage) -> None:
        """Update pipeline state visualization."""
        for node in self.pipeline_nodes:
            if node.text() == state.name:
                node.setStyleSheet("""
                    QLabel {
                        background-color: #2196f3;
                        color: white;
                        padding: 10px;
                        border-radius: 15px;
                        min-width: 80px;
                    }
                """)
            elif any(prev_node.text() == state.name for prev_node in self.pipeline_nodes[:self.pipeline_nodes.index(node)]):
                node.setStyleSheet("""
                    QLabel {
                        background-color: #4caf50;
                        color: white;
                        padding: 10px;
                        border-radius: 15px;
                        min-width: 80px;
                    }
                """)
                
    def log_message(self, message: str) -> None:
        """Add message to log."""
        self.log_text.append(message)
        
    def set_mission_running(self, running: bool) -> None:
        """Update UI state based on mission status."""
        self.start_button.setEnabled(not running)
        self.start_button.setText("Mission Running..." if running else "Start Mission")
