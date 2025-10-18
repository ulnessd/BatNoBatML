#!/usr/bin/env python3
# snippet_sorter.py — A high-performance tool for rapidly reviewing and sorting audio snippets.

import sys
import os
import shutil
import time
from pathlib import Path
from collections import deque
from typing import Optional, List, Dict

# --- Core Scientific Libraries (Backend) ---
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

# --- PySide6 Libraries (Frontend) ---
from PySide6.QtCore import (Qt, QThread, QObject, Signal, QSize, QTimer, QMutex, QMutexLocker)
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                               QListWidget, QLabel, QPushButton, QFileDialog, QSplitter,
                               QListWidgetItem)

# Use matplotlib just for its colormaps, not for plotting
import matplotlib.cm as cm

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# --- Performance Tuning ---
# Max number of spectrogram images to hold in memory.
CACHE_SIZE = 200
# The "rolling window" for pre-loading spectrograms around the current selection.
ROLLING_WINDOW_BEFORE = 25
ROLLING_WINDOW_AFTER = 50

# --- Spectrogram Generation ---
NPERSEG = 1024  # Increased from 512 for better frequency resolution
CMAP = "magma"  # Colormap for the spectrogram
IMG_WIDTH = 800
IMG_HEIGHT = 600


# =============================================================================
# 2. WORKER THREAD FOR SPECTROGRAM GENERATION
# =============================================================================

class SpectrogramWorker(QThread):
    """
    A dedicated background thread to generate spectrograms without freezing the UI.
    It processes jobs from a prioritized queue.
    """
    spectrogram_ready = Signal(str, QPixmap)

    def __init__(self, colormap_lut: np.ndarray, parent=None):
        super().__init__(parent)
        self.job_queue = deque()
        self.priority_queue = deque()
        self.mutex = QMutex()
        self.is_running = True
        self.colormap_lut = colormap_lut

    def stop(self):
        """Signals the thread to stop processing and exit gracefully."""
        self.is_running = False
        self.wait()

    def request_job(self, filepath: str, is_priority: bool):
        """Add a file to the processing queue."""
        with QMutexLocker(self.mutex):
            if is_priority:
                # High-priority jobs (current selection) go to the front
                if filepath not in self.priority_queue:
                    self.priority_queue.appendleft(filepath)
            else:
                # Normal-priority jobs (rolling window) go to the back
                if filepath not in self.job_queue:
                    self.job_queue.append(filepath)

    def clear_normal_jobs(self):
        """Clear the non-priority queue, used when the user jumps to a new location."""
        with QMutexLocker(self.mutex):
            self.job_queue.clear()

    def run(self):
        """The main processing loop of the thread."""
        while self.is_running:
            filepath = None
            with QMutexLocker(self.mutex):
                if self.priority_queue:
                    filepath = self.priority_queue.pop()
                elif self.job_queue:
                    filepath = self.job_queue.popleft()

            if filepath:
                try:
                    pixmap = self._generate_spectrogram(filepath)
                    if pixmap:
                        self.spectrogram_ready.emit(filepath, pixmap)
                except Exception as e:
                    print(f"Error generating spectrogram for {filepath}: {e}")
            else:
                # If no jobs, sleep briefly to avoid busy-waiting
                self.msleep(50)

    def _generate_spectrogram(self, filepath: str) -> Optional[QPixmap]:
        """Reads a WAV file, computes the spectrogram, and converts it to a QPixmap."""
        try:
            with sf.SoundFile(filepath, 'r') as f:
                samplerate = f.samplerate
                audio = f.read(dtype='float32')
        except Exception:
            return self._create_error_pixmap("Invalid File")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size < NPERSEG:
            return self._create_error_pixmap("Too Short")

        f, t, Sxx = spectrogram(audio, samplerate, nperseg=NPERSEG)

        Sxx_db = 10 * np.log10(Sxx + 1e-9)

        min_db = np.percentile(Sxx_db, 5)
        max_db = np.percentile(Sxx_db, 95)
        if max_db <= min_db: max_db = min_db + 1.0

        Sxx_norm = (Sxx_db - min_db) / (max_db - min_db)
        Sxx_norm = np.clip(Sxx_norm, 0, 1)

        # Convert normalized data to 8-bit indices
        indices = (Sxx_norm * 255).astype(np.uint8)

        # Apply the pre-computed colormap lookup table
        img_data = self.colormap_lut[indices]

        h, w, _ = img_data.shape
        q_image = QImage(img_data.data, w, h, QImage.Format_RGBA8888)
        q_image = q_image.mirrored(False, True)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap.scaled(IMG_WIDTH, IMG_HEIGHT, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    def _create_error_pixmap(self, text: str) -> QPixmap:
        pixmap = QPixmap(IMG_WIDTH, IMG_HEIGHT)
        pixmap.fill(QColor("black"))
        from PySide6.QtGui import QPainter
        painter = QPainter(pixmap)
        painter.setPen(QColor("red"))
        font = QFont();
        font.setPointSize(24)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, text)
        painter.end()
        return pixmap


# =============================================================================
# 3. MAIN APPLICATION WINDOW
# =============================================================================

class SnippetSorterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snippet Sorter (Phase 1)")
        self.resize(1200, 800)

        self.current_folder: Optional[Path] = None
        self.file_paths: List[Path] = []
        self.pixmap_cache: Dict[str, QPixmap] = {}
        self.rejected_count = 0
        self.worker: Optional[SpectrogramWorker] = None

        self._setup_ui()
        self._create_shortcuts()

        # Defer worker creation until after the event loop has started
        QTimer.singleShot(0, self._initialize_worker)

    def _initialize_worker(self):
        """Creates and starts the background worker thread safely."""
        # --- Pre-compute colormap for thread safety ---
        cmap = cm.get_cmap(CMAP)
        colormap_lut = (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)

        # --- Worker Thread ---
        self.worker = SpectrogramWorker(colormap_lut)
        self.worker.spectrogram_ready.connect(self._on_spectrogram_ready)
        self.worker.start()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.open_folder_btn = QPushButton("Choose Folder...")
        self.open_folder_btn.clicked.connect(self._choose_folder)
        left_layout.addWidget(self.open_folder_btn)
        self.file_list_widget = QListWidget()
        self.file_list_widget.currentRowChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.file_list_widget)
        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.spectrogram_viewer = QLabel("Open a folder to begin.")
        self.spectrogram_viewer.setAlignment(Qt.AlignCenter)
        self.spectrogram_viewer.setMinimumSize(IMG_WIDTH, IMG_HEIGHT)
        self.spectrogram_viewer.setStyleSheet("background-color: black; color: white;")
        right_layout.addWidget(self.spectrogram_viewer)
        splitter.addWidget(right_panel)

        splitter.setSizes([300, 900])

        self.statusBar().showMessage("Ready.")

    def _create_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, self._reject_current_file)
        QShortcut(QKeySequence(Qt.Key_Delete), self, self._reject_current_file)

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Snippet Folder")
        if folder:
            self.current_folder = Path(folder)
            self._load_folder()

    def _load_folder(self):
        if not self.current_folder:
            return

        self.statusBar().showMessage(f"Loading files from {self.current_folder.name}...")
        self.file_list_widget.clear()
        self.pixmap_cache.clear()
        if self.worker:
            self.worker.clear_normal_jobs()
        self.rejected_count = 0

        self.file_paths = sorted(
            [p for p in self.current_folder.glob("*.wav")] +
            [p for p in self.current_folder.glob("*.WAV")]
        )

        self.file_list_widget.addItems([p.name for p in self.file_paths])

        if self.file_paths:
            self.file_list_widget.setCurrentRow(0)
            self._update_status()
        else:
            self.statusBar().showMessage("No .wav files found in this folder.")
            self.spectrogram_viewer.setText("No .wav files found.")

    def _on_selection_changed(self, index: int):
        if 0 <= index < len(self.file_paths) and self.worker:
            filepath = str(self.file_paths[index])

            if filepath in self.pixmap_cache:
                self.spectrogram_viewer.setPixmap(self.pixmap_cache[filepath])
            else:
                self.spectrogram_viewer.setText("Loading spectrogram...")
                self.worker.request_job(filepath, is_priority=True)

            self.worker.clear_normal_jobs()
            start = max(0, index - ROLLING_WINDOW_BEFORE)
            end = min(len(self.file_paths), index + ROLLING_WINDOW_AFTER)
            for i in range(start, end):
                if i != index:
                    path_to_preload = str(self.file_paths[i])
                    if path_to_preload not in self.pixmap_cache:
                        self.worker.request_job(path_to_preload, is_priority=False)

    def _on_spectrogram_ready(self, filepath: str, pixmap: QPixmap):
        if len(self.pixmap_cache) >= CACHE_SIZE:
            oldest_key = next(iter(self.pixmap_cache))
            self.pixmap_cache.pop(oldest_key, None)
        self.pixmap_cache[filepath] = pixmap

        current_index = self.file_list_widget.currentRow()
        if 0 <= current_index < len(self.file_paths):
            current_path = str(self.file_paths[current_index])
            if current_path == filepath:
                self.spectrogram_viewer.setPixmap(pixmap)

    def _reject_current_file(self):
        index = self.file_list_widget.currentRow()
        if not (0 <= index < len(self.file_paths)):
            return

        filepath = self.file_paths[index]

        rejected_dir = self.current_folder / "_rejected_"
        rejected_dir.mkdir(exist_ok=True)
        try:
            shutil.move(str(filepath), str(rejected_dir / filepath.name))
        except Exception as e:
            self.statusBar().showMessage(f"Error moving file: {e}")
            return

        self.file_paths.pop(index)
        self.pixmap_cache.pop(str(filepath), None)

        self.file_list_widget.blockSignals(True)
        self.file_list_widget.takeItem(index)
        self.file_list_widget.blockSignals(False)

        if self.file_list_widget.count() == 0:
            self.spectrogram_viewer.setText("All files sorted.")
        else:
            new_index = min(index, self.file_list_widget.count() - 1)
            self.file_list_widget.setCurrentRow(new_index)
            self._on_selection_changed(new_index)

        self.rejected_count += 1
        self._update_status()

    def _update_status(self):
        total = len(self.file_paths) + self.rejected_count
        remaining = len(self.file_paths)
        self.statusBar().showMessage(
            f"Folder: {self.current_folder.name} | "
            f"Remaining: {remaining}/{total} | "
            f"Rejected: {self.rejected_count}"
        )

    def closeEvent(self, event):
        """Ensure the worker thread is stopped cleanly on exit."""
        if self.worker:
            self.worker.stop()
        event.accept()


# =============================================================================
# 4. APPLICATION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SnippetSorterWindow()
    window.show()
    sys.exit(app.exec())

