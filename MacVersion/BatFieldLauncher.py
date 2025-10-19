#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BatFieldLauncher1.py - A central GUI to launch various bat analysis tools.
"""

import sys, subprocess, os, shutil

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame
)

from pathlib import Path
import sys, subprocess, os, shutil

from pathlib import Path
import sys


def _dist_dir() -> Path:
    """Directory where the other tools live."""
    if getattr(sys, "frozen", False):
        # In a one-folder build, all EXEs live next to the launcher binary.
        return Path(sys.executable).parent
    else:
        # Running from source: assume sibling scripts or per-project /dist.
        return Path(__file__).parent

def _platform_exe_name(base: str) -> str:
    """Map logical tool name to platform-specific binary name."""
    if sys.platform.startswith("win"):
        return f"{base}.exe"
    return base  # macOS/Linux bare name

def _candidate_paths(tool_basename: str) -> list[Path]:
    d = _dist_dir()
    exe_name = _platform_exe_name(tool_basename)
    return [
        d / exe_name,                 # bare exe in dist folder
        d / "apps" / exe_name,
        d / f"{tool_basename}.app",   # macOS .app bundle in dist folder
        # Fallbacks (dev runs)
        Path.cwd() / exe_name,
        Path.cwd() / f"{tool_basename}.app",
    ]

def _find_tool(tool_basename: str) -> tuple[str, bool]:
    """
    Return (path, is_app_bundle). Raises FileNotFoundError if not found.
    """
    for p in _candidate_paths(tool_basename):
        if p.exists():
            return (str(p), p.suffix == ".app")
    # Last resort: PATH search (useful during dev)
    which = shutil.which(tool_basename)
    if which:
        return (which, False)
    raise FileNotFoundError(f"Could not locate tool '{tool_basename}'")

def run_tool(tool_basename: str, *args, detach: bool = True) -> subprocess.Popen | int:
    """
    Launch a sibling tool. Returns a Popen on success (or 0 if detached via 'open -a').
    Set detach=False to inherit stdio and wait (mostly for debugging).
    """
    tool_path, is_app = _find_tool(tool_basename)

    if sys.platform == "darwin":
        if is_app:
            # Launch a .app bundle
            cmd = ["open", "-a", tool_path]
            if args:
                cmd += ["--args", *map(str, args)]
            # 'open' returns immediately; no useful Popen to manage
            subprocess.Popen(cmd)  # intentionally not waited
            return 0
        else:
            # Bare Mach-O binary
            if detach:
                # Start detached so your launcher GUI stays responsive
                return subprocess.Popen([tool_path, *map(str, args)],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                        start_new_session=True)
            else:
                # Foreground run (debugging)
                return subprocess.call([tool_path, *map(str, args)])

    else:
        # Windows / Linux: launch the binary directly
        if detach:
            creationflags = 0
            if sys.platform.startswith("win"):
                creationflags = 0x00000008 | 0x00000010  # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
            return subprocess.Popen([tool_path, *map(str, args)],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    start_new_session=True if not sys.platform.startswith("win") else False,
                                    creationflags=creationflags)
        else:
            return subprocess.call([tool_path, *map(str, args)])


class BatLauncherWindow(QMainWindow):
    """
    The main window for the Bat Field Tools launcher application.
    Displays an image and provides buttons to run separate analysis scripts.
    """
    def __init__(self):
        super().__init__()

        # --- Basic Window Configuration ---
        self.setWindowTitle("Bat Field Tools Launcher")
        self.setGeometry(100, 100, 500, 600)

        # --- Central Widget and Layout ---
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # --- Determine script's directory to find assets ---
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
        self.apps = {
            "Launch Bat/No-Bat Sorter": "BatNoBat1.py",
            "Launch Spectrogram Inspector": "BatInspector1.py",
            "Launch Audio Compressor": "BatCompressor1.py"
        }

        button_layout = QHBoxLayout()
        for btn_text, script_name in self.apps.items():
            button = QPushButton(btn_text)
            button.setMinimumHeight(45)
            button.clicked.connect(lambda checked=False, s=script_name: self._launch_app(s))
            button_layout.addWidget(button)

        main_layout.addLayout(button_layout)
        main_layout.addStretch()

    def _launch_app(self, script_name: str):
        """
        Launches a bundled executable, handling paths correctly for both
        development and the final packaged application.
        """
        # In a packaged app, the base path is a temporary folder.
        # In development, it's just the script's directory.
        if getattr(sys, 'frozen', False):
            # We are running in a PyInstaller bundle.
            # The executable is in the same directory as the launcher.
            base_path = Path(sys.executable).parent
        else:
            # We are running in a normal Python environment.
            # The executable is our standard Python interpreter.
            base_path = self.base_path

        if getattr(sys, 'frozen', False):
            # --- Packaged Mode ---
            exe_name = Path(script_name).stem
            if sys.platform == "win32":
                exe_name += ".exe"
            exe_path = base_path / exe_name
            command = [str(exe_path)]
        else:
            # --- Development Mode ---
            script_path = self.apps_path / script_name
            command = [sys.executable, str(script_path)]

        if not (getattr(sys, 'frozen', False) and exe_path.exists()) and not (not getattr(sys, 'frozen', False) and script_path.exists()):
            print(f"Error: Could not find target to launch.")
            return

        print(f"Executing command: {' '.join(command)}")
        try:
            subprocess.Popen(command)
        except Exception as e:
            print(f"Failed to launch: {e}")

def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    window = BatLauncherWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
