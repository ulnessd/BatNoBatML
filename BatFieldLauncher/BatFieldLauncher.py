#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BatFieldLauncher1.py - A central GUI to launch various bat analysis tools.
"""

import sys
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame
)

class BatLauncherWindow(QMainWindow):
    """
    The main window for the Bat Field Tools launcher application.
    Displays an image and provides buttons to run separate analysis scripts.
    """
    def __init__(self):
        super().__init__()

        # --- Basic Window Configuration ---
        self.setWindowTitle("Bat Field Tools Launcher")
        self.setGeometry(100, 100, 500, 600) # x, y, width, height

        # --- Central Widget and Layout ---
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # --- Determine script's directory to find assets ---
        # This makes the launcher runnable from anywhere.
        self.base_path = Path(__file__).parent
        self.apps_path = self.base_path / "apps"

        # --- Set Window Icon ---
        icon_path = self.base_path / "BatLookingAtBatCalls.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # --- Main Image Display ---
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap(str(icon_path))
        if not pixmap.isNull():
            # Scale image to a nice width while keeping its aspect ratio
            image_label.setPixmap(pixmap.scaledToWidth(400, Qt.TransformationMode.SmoothTransformation))
        else:
            image_label.setText("Launcher Image Not Found\n(Expecting BatLookingAtBatCalls.png)")
        main_layout.addWidget(image_label)

        # --- Separator Line ---
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(separator)

        # --- App Buttons ---
        # A dictionary mapping button text to the script filename.
        self.apps = {
            "Launch Bat/No-Bat Sorter": "BatNoBat1.py",
            "Launch Spectrogram Inspector": "BatInspector1.py",
            "Launch Audio Compressor": "BatCompressor1.py"
        }

        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()
        for btn_text, script_name in self.apps.items():
            button = QPushButton(btn_text)
            button.setMinimumHeight(45) # Make buttons taller
            # Use a lambda to capture the script_name for the click handler
            button.clicked.connect(lambda checked=False, s=script_name: self._launch_app(s))
            button_layout.addWidget(button)

        main_layout.addLayout(button_layout)
        main_layout.addStretch() # Pushes everything up

    def _launch_app(self, script_name: str):
        """
        Launches an external Python script as a separate process.
        """
        script_path = self.apps_path / script_name
        if not script_path.exists():
            print(f"Error: Could not find script at {script_path}")
            # Optionally, show a QMessageBox to the user here.
            return

        print(f"Launching: {script_path}...")
        try:
            # Use subprocess.Popen to run the script in a non-blocking way.
            # This allows the launcher to remain responsive.
            subprocess.Popen([sys.executable, str(script_path)])
        except Exception as e:
            print(f"Failed to launch {script_name}: {e}")

def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    window = BatLauncherWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
