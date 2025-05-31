from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, Signal, QTimer, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor
import numpy as np
from typing import List, Optional, Dict, Any

from app.backend.container import MissionState
from app.backend.pipeline.pipeline import PipelineNodeType

class RippleLabel(QLabel):
    """Custom QLabel with water-like ripple animation around the label's contour."""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self._ripple_offset = 0
        self._max_ripple_offset = 15
        self._ripple_opacity = 255
        self._is_active = False

        self.offset_animation = QPropertyAnimation(self, b"ripple_offset")
        self.offset_animation.setDuration(1500)
        self.offset_animation.setEasingCurve(QEasingCurve.Type.OutQuad)
        self.offset_animation.finished.connect(self._restart_animation)

        self.opacity_animation = QPropertyAnimation(self, b"ripple_opacity")
        self.opacity_animation.setDuration(1500)
        self.opacity_animation.setEasingCurve(QEasingCurve.Type.Linear)
        self.opacity_animation.finished.connect(self._restart_animation)

        self.pulse_timer = QTimer(self)
        self.pulse_timer.timeout.connect(self.update)

    @Property(int)
    def ripple_offset(self):
        return self._ripple_offset

    @ripple_offset.setter
    def ripple_offset(self, value):
        self._ripple_offset = value
        self.update()

    @Property(int)
    def ripple_opacity(self):
        return self._ripple_opacity

    @ripple_opacity.setter
    def ripple_opacity(self, value):
        self._ripple_opacity = value
        self.update()

    def set_active(self, active: bool):
        if self._is_active == active:
            return
        self._is_active = active
        if active:
            self._start_ripple_animation()
            self.pulse_timer.start(16)
        else:
            self._stop_ripple_animation()
            self.pulse_timer.stop()

    def _start_ripple_animation(self):
        self.offset_animation.stop()
        self.opacity_animation.stop()
        self.offset_animation.setStartValue(0)
        self.offset_animation.setEndValue(self._max_ripple_offset)
        self.offset_animation.start()
        self.opacity_animation.setStartValue(255)
        self.opacity_animation.setEndValue(50)
        self.opacity_animation.start()

    def _stop_ripple_animation(self):
        self.offset_animation.stop()
        self.opacity_animation.stop()
        self._ripple_offset = 0
        self._ripple_opacity = 255
        self.update()

    def _restart_animation(self):
        if self._is_active:
            self._start_ripple_animation()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._is_active or self._ripple_offset <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        ripple_rect = self.rect().adjusted(
            -self._ripple_offset, -self._ripple_offset,
            self._ripple_offset, self._ripple_offset
        )
        pen = QPen(QColor(33, 150, 243, self._ripple_opacity), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(ripple_rect, 18, 18)

class PipelineNode(QWidget):
    """Enhanced pipeline node with connection lines and status indicators."""

    def __init__(self, text: str, is_last: bool = False):
        super().__init__()
        self.is_last = is_last
        self.status = "pending"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.label = RippleLabel(text)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_style()

        layout.addWidget(self.label)

    def set_status(self, status: str):
        self.status = status
        self._update_style()
        self.label.set_active(status == "active")

    def _update_style(self):
        styles = {
            "pending": """
                QLabel {
                    background-color: #455a64;
                    color: #b0bec5;
                    border: 2px solid #37474f;
                    padding: 12px 20px;
                    border-radius: 18px;
                    min-width: 100px;
                    font-weight: bold;
                    font-size: 12px;
                }
            """,
            "active": """
                QLabel {
                    background-color: #2196f3;
                    color: white;
                    border: 2px solid #1976d2;
                    padding: 12px 20px;
                    border-radius: 18px;
                    min-width: 100px;
                    font-weight: bold;
                    font-size: 12px;
                }
            """,
            "completed": """
                QLabel {
                    background-color: #4caf50;
                    color: white;
                    border: 2px solid #388e3c;
                    padding: 12px 20px;
                    border-radius: 18px;
                    min-width: 100px;
                    font-weight: bold;
                    font-size: 12px;
                }
            """,
            "error": """
                QLabel {
                    background-color: #f44336;
                    color: white;
                    border: 2px solid #d32f2f;
                    padding: 12px 20px;
                    border-radius: 18px;
                    min-width: 100px;
                    font-weight: bold;
                    font-size: 12px;
                }
            """
        }
        self.label.setStyleSheet(styles.get(self.status, styles["pending"]))
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setOffset(2, 2)
        if self.status == "active":
            shadow.setColor(QColor(33, 150, 243, 100))
            shadow.setBlurRadius(15)
        else:
            shadow.setColor(QColor(0, 0, 0, 50))
        self.label.setGraphicsEffect(shadow)

class ConnectionLine(QWidget):
    """Widget to draw connection lines between pipeline nodes."""

    def __init__(self, is_active: bool = False):
        super().__init__()
        self.is_active = is_active
        self.setFixedSize(30, 4)

    def set_active(self, active: bool):
        self.is_active = active
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(76, 175, 80) if self.is_active else QColor(69, 90, 100), 3 if self.is_active else 2)
        painter.setPen(pen)
        painter.drawLine(0, 2, 30, 2)

class AppView(QMainWindow):
    # Signals
    start_mission = Signal()
    emergency_stop = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Mission Control - Enhanced Pipeline")
        self.setGeometry(100, 100, 1200, 800)
        self.current_node_index = -1
        self._init_ui()

    def _init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(20)

        # Horizontal layout for stream and status panel
        display_layout = QHBoxLayout()

        # Stream display
        self.stream_label = QLabel()
        self.stream_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stream_label.setMinimumSize(640, 480)
        self.stream_label.setStyleSheet("""
            QLabel {
                background-color: #263238;
                border: 2px solid #37474f;
                border-radius: 8px;
            }
        """)
        display_layout.addWidget(self.stream_label)

        # Status panel
        self.status_panel = QWidget()
        self.status_panel.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                border: 1px solid #37474f;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        status_layout = QVBoxLayout(self.status_panel)
        status_layout.setSpacing(10)
        status_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Status labels
        self.status_labels = {}
        status_fields = [
            ("Mission Start", "mission_start_time"),
            ("Mission Time", "mission_duration"),
            ("Pipeline State", "pipeline_state"),
            ("Frames Processed", "frames_processed"),
            ("FPS", "fps"),
            ("Detected Faces", "detected_faces_count"),
            ("Battery Critical", "battery_critical"),
            ("Initialization Complete", "initialization_complete"),
            ("Battery (%)", "drone_data.battery"),
            ("Height (cm)", "drone_data.height"),
            ("Temperature (°C)", "drone_data.temperature"),
            ("Connection Lost", "connection_lost"),
            ("Error", "error"),
        ]
        for label_text, field in status_fields:
            label = QLabel(f"{label_text}: ")
            label.setStyleSheet("""
                QLabel {
                    color: #b0bec5;
                    font-size: 12px;
                    font-family: 'Courier New', monospace;
                }
            """)
            value_label = QLabel("N/A")
            value_label.setStyleSheet("""
                QLabel {
                    color: #ffffff;
                    font-size: 12px;
                    font-family: 'Courier New', monospace;
                }
            """)
            self.status_labels[field] = value_label
            status_row = QHBoxLayout()
            status_row.addWidget(label)
            status_row.addWidget(value_label)
            status_layout.addLayout(status_row)
        display_layout.addWidget(self.status_panel)
        main_layout.addLayout(display_layout)

        # Pipeline display
        pipeline_container = QWidget()
        pipeline_container.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                border-radius: 12px;
                padding: 20px;
            }
        """)
        pipeline_layout = QHBoxLayout(pipeline_container)
        pipeline_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pipeline_layout.setSpacing(0)

        self.pipeline_nodes: List[PipelineNode] = []
        self.connection_lines: List[ConnectionLine] = []

        node_names = [
            PipelineNodeType.IDLE, PipelineNodeType.LAUNCH, PipelineNodeType.SCAN,
            PipelineNodeType.IDENTIFY, PipelineNodeType.TRACK, PipelineNodeType.RETURN
        ]
        for i, state in enumerate(node_names):
            is_last = i == len(node_names) - 1
            node = PipelineNode(state.name, is_last)
            self.pipeline_nodes.append(node)
            pipeline_layout.addWidget(node)
            if not is_last:
                line = ConnectionLine()
                self.connection_lines.append(line)
                pipeline_layout.addWidget(line)
        main_layout.addWidget(pipeline_container)

        # Controls
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        self.start_button = QPushButton("🚀 Start Mission")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
        self.emergency_button = QPushButton("🛑 Emergency Stop")
        self.emergency_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.emergency_button)
        controls_layout.addStretch()
        main_layout.addWidget(controls_widget)

        # Mission log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #2e2e2e;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 8px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        main_layout.addWidget(self.log_text)

        # Connect signals
        self.start_button.clicked.connect(self.start_mission.emit)
        self.emergency_button.clicked.connect(self.emergency_stop.emit)

        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
                color: white;
            }
        """)

    def update_frame(self, frame: np.ndarray) -> None:
        """Update the displayed frame."""
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.stream_label.setPixmap(QPixmap.fromImage(qimg))

    def update_pipeline_state(self, state: PipelineNodeType) -> None:
        """Update pipeline state visualization with ripple effects."""
        for i, node in enumerate(self.pipeline_nodes):
            node_name = node.label.text()
            if node_name == state.pipeline_current_node.name.upper():
                self.current_node_index = i
                break
        for i, node in enumerate(self.pipeline_nodes):
            if i < self.current_node_index:
                node.set_status("completed")
            elif i == self.current_node_index:
                node.set_status("active")
            else:
                node.set_status("pending")
        for i, line in enumerate(self.connection_lines):
            line.set_active(i < self.current_node_index)

    def update_telemetry_panel(self, mission_state: Dict[str, Any]) -> None:
        """Update the status panel with mission status data."""
        for field, label in self.status_labels.items():
            if field.startswith("drone_data."):
                subfield = field.split(".")[1]
                value = mission_state.get("drone_data", {}).get(subfield, "N/A") if mission_state.get("drone_data") else "N/A"
            else:
                value = mission_state.get(field, "N/A")
            label.setText(str(value))

    def set_node_error(self, state: PipelineNodeType) -> None:
        """Set a specific node to error state."""
        for node in self.pipeline_nodes:
            if node.label.text() == state.name:
                node.set_status("error")
                break

    def reset_pipeline(self) -> None:
        """Reset all pipeline nodes to pending state."""
        self.current_node_index = -1
        for node in self.pipeline_nodes:
            node.set_status("pending")
        for line in self.connection_lines:
            line.set_active(False)

    def log_message(self, message: str) -> None:
        """Add message to log."""
        self.log_text.append(message)

    def set_mission_running(self, running: bool) -> None:
        """Update UI state based on mission status."""
        self.start_button.setEnabled(not running)
        self.start_button.setText("🔄 Mission Running..." if running else "🚀 Start Mission")