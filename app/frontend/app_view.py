from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, Signal, QTimer, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor
import numpy as np
from typing import List, Optional

from app.backend.pipeline.pipeline import PipelineNodeType

class RippleLabel(QLabel):
    """Custom QLabel with water-like ripple animation around the label's contour."""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self._ripple_offset = 0  # Distance from label's contour
        self._max_ripple_offset = 15  # Minimum 15px expansion
        self._ripple_opacity = 255  # Opacity of the ripple
        self._is_active = False

        # Setup ripple animation for offset
        self.offset_animation = QPropertyAnimation(self, b"ripple_offset")
        self.offset_animation.setDuration(1500)
        self.offset_animation.setEasingCurve(QEasingCurve.Type.OutQuad)  # Smooth water-like expansion
        self.offset_animation.finished.connect(self._restart_animation)

        # Setup ripple animation for opacity
        self.opacity_animation = QPropertyAnimation(self, b"ripple_opacity")
        self.opacity_animation.setDuration(1500)
        self.opacity_animation.setEasingCurve(QEasingCurve.Type.Linear)
        self.opacity_animation.finished.connect(self._restart_animation)

        # Pulser timer for continuous repaints
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
        """Set the active state and start/stop ripple animation."""
        if self._is_active == active:
            return
        self._is_active = active
        if active:
            self._start_ripple_animation()
            self.pulse_timer.start(16)  # ~60 FPS for smooth animation
        else:
            self._stop_ripple_animation()
            self.pulse_timer.stop()

    def _start_ripple_animation(self):
        """Start the ripple animation for offset and opacity."""
        self.offset_animation.stop()
        self.opacity_animation.stop()

        # Animate offset from 0 to max_ripple_offset
        self.offset_animation.setStartValue(0)
        self.offset_animation.setEndValue(self._max_ripple_offset)
        self.offset_animation.start()

        # Animate opacity from 255 to 50
        self.opacity_animation.setStartValue(255)
        self.opacity_animation.setEndValue(50)
        self.opacity_animation.start()

    def _stop_ripple_animation(self):
        """Stop the ripple animation and reset effect."""
        self.offset_animation.stop()
        self.opacity_animation.stop()
        self._ripple_offset = 0
        self._ripple_opacity = 255
        self.update()

    def _restart_animation(self):
        """Restart the ripple animation if active."""
        if self._is_active:
            self._start_ripple_animation()

    def paintEvent(self, event):
        """Draw a rectangular ripple around the label's contour."""
        super().paintEvent(event)  # Draw the label's text and background

        if not self._is_active or self._ripple_offset <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Define the ripple rectangle, expanding from the label's contour
        ripple_rect = self.rect().adjusted(
            -self._ripple_offset, -self._ripple_offset,
            self._ripple_offset, self._ripple_offset
        )

        # Set up the pen for a sharp, water-like ripple
        pen = QPen(QColor(33, 150, 243, self._ripple_opacity), 2)  # Thin, sharp ring
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)  # No fill to keep text visible

        # Draw the rectangular ripple with rounded corners to match label
        painter.drawRoundedRect(ripple_rect, 18, 18)  # Match label's border-radius

class PipelineNode(QWidget):
    """Enhanced pipeline node with connection lines and status indicators."""

    def __init__(self, text: str, is_last: bool = False):
        super().__init__()
        self.is_last = is_last
        self.status = "pending"  # pending, active, completed, error

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)  # Margins for ripple visibility

        # Create ripple label
        self.label = RippleLabel(text)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_style()

        layout.addWidget(self.label)

    def set_status(self, status: str):
        """Set node status: pending, active, completed, error."""
        self.status = status
        self._update_style()
        self.label.set_active(status == "active")

    def _update_style(self):
        """Update node styling based on status."""
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

        # Adjust shadow effect based on status
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setOffset(2, 2)
        if self.status == "active":
            shadow.setColor(QColor(33, 150, 243, 100))  # Blue shadow for active state
            shadow.setBlurRadius(15)  # Slightly larger blur for emphasis
        else:
            shadow.setColor(QColor(0, 0, 0, 50))  # Default shadow
        self.label.setGraphicsEffect(shadow)

class ConnectionLine(QWidget):
    """Widget to draw connection lines between pipeline nodes."""

    def __init__(self, is_active: bool = False):
        super().__init__()
        self.is_active = is_active
        self.setFixedSize(30, 4)

    def set_active(self, active: bool):
        """Set line active state."""
        self.is_active = active
        self.update()

    def paintEvent(self, event):
        """Draw connection line."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.is_active:
            pen = QPen(QColor(76, 175, 80), 3)  # Green for completed connection
        else:
            pen = QPen(QColor(69, 90, 100), 2)  # Gray for pending connection

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
        main_layout.addWidget(self.stream_label)

        # Pipeline display with enhanced styling
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

        # Create pipeline nodes and connections
        self.pipeline_nodes: List[PipelineNode] = []
        self.connection_lines: List[ConnectionLine] = []

        node_names = [PipelineNodeType.IDLE,
                      PipelineNodeType.LAUNCH, PipelineNodeType.SCAN,
                      PipelineNodeType.IDENTIFY, PipelineNodeType.TRACK,
                      PipelineNodeType.RETURN]

        for i, state in enumerate(node_names):
            # Create node
            is_last = i == len(node_names) - 1
            node = PipelineNode(state.name, is_last)
            self.pipeline_nodes.append(node)
            pipeline_layout.addWidget(node)

            # Add connection line if not last node
            if not is_last:
                line = ConnectionLine()
                self.connection_lines.append(line)
                pipeline_layout.addWidget(line)

        main_layout.addWidget(pipeline_container)

        # Controls with enhanced styling
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

        # Mission log with enhanced styling
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

        # Set dark theme for main window
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
        # Find the current node index
        for i, node in enumerate(self.pipeline_nodes):
            if node.label.text() == state.pipeline_current_node.node.name:
                self.current_node_index = i
                break

        # Update all nodes based on current state
        for i, node in enumerate(self.pipeline_nodes):
            if i < self.current_node_index:
                node.set_status("completed")
            elif i == self.current_node_index:
                node.set_status("active")
            else:
                node.set_status("pending")

        # Update connection lines
        for i, line in enumerate(self.connection_lines):
            line.set_active(i < self.current_node_index)

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