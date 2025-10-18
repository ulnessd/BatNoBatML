#!/usr/bin/env python3
# find_normalization_constants.py (GUI Version)
#
# A PySide6 GUI to analyze a sample of the curated dataset and determine
# the global normalization constants (min_db and max_db).

import sys
import random
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

# --- PySide6 Libraries (Frontend) ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QGridLayout, QPushButton, QLineEdit, QLabel, QFileDialog,
                               QMessageBox, QProgressBar, QGroupBox)
from PySide6.QtCore import Qt, QThread, QObject, Signal

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
NPERSEG = 1024
SAMPLE_SIZE = 2000  # Number of files to sample (1000 bat, 1000 noise)


# =============================================================================
# 2. WORKER LOGIC
# =============================================================================

def calculate_percentiles(filepath: Path):
    """Calculates the 5th and 95th percentile dB values for a single WAV file."""
    try:
        with sf.SoundFile(str(filepath), 'r') as f:
            audio = f.read(dtype='float32')
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if audio.size < NPERSEG: return None

        _, _, Sxx = spectrogram(audio, f.samplerate, nperseg=NPERSEG)
        Sxx_db = 10 * np.log10(Sxx + 1e-9)

        return np.percentile(Sxx_db, [5, 95])
    except Exception as e:
        # This will run in a thread, so print errors for debugging.
        print(f"Warning: Could not process {filepath.name}: {e}", file=sys.stderr)
        return None


class AnalysisWorker(QObject):
    """Runs the analysis in a background thread."""
    progress = Signal(int, int)
    finished = Signal(float, float)
    error = Signal(str)

    def __init__(self, source_dir: Path):
        super().__init__()
        self.source_dir = source_dir
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            bat_dir = self.source_dir / "EcholocationCalls"
            noise_dir = self.source_dir / "NoBatNoise"

            if not (bat_dir.is_dir() and noise_dir.is_dir()):
                self.error.emit(f"Could not find 'EcholocationCalls' and/or 'NoBatNoise' in '{self.source_dir}'")
                return

            bat_files = list(bat_dir.glob("*.wav"))
            noise_files = list(noise_dir.glob("*.wav"))

            bat_sample = random.sample(bat_files, min(len(bat_files), SAMPLE_SIZE // 2))
            noise_sample = random.sample(noise_files, min(len(noise_files), SAMPLE_SIZE // 2))
            all_samples = bat_sample + noise_sample

            if not all_samples:
                self.error.emit("No WAV files found to sample.")
                return

            all_percentiles = []
            total_samples = len(all_samples)
            for i, filepath in enumerate(all_samples):
                if not self.is_running:
                    self.error.emit("Operation cancelled.")
                    return

                self.progress.emit(i + 1, total_samples)
                percentiles = calculate_percentiles(filepath)
                if percentiles is not None:
                    all_percentiles.append(percentiles)

            if not all_percentiles:
                self.error.emit("Could not calculate statistics from any files.")
                return

            all_percentiles_np = np.array(all_percentiles)
            median_min_db = np.median(all_percentiles_np[:, 0])
            median_max_db = np.median(all_percentiles_np[:, 1])

            self.finished.emit(median_min_db, median_max_db)

        except Exception as e:
            self.error.emit(f"An unexpected error occurred: {e}")


# =============================================================================
# 3. PYSIDE6 GUI APPLICATION
# =============================================================================

class NormalizationFinderWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Global Normalization Constant Finder")
        self.setGeometry(100, 100, 600, 300)
        self.thread = None
        self.worker = None
        self._setup_ui()
        self._apply_stylesheet()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        io_group = QGroupBox("Source Directory")
        io_layout = QGridLayout(io_group)
        self.source_dir_edit = QLineEdit()
        self.source_dir_edit.setPlaceholderText("Select your 'MLRound1' folder...")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_folder)

        io_layout.addWidget(QLabel("Curated Data Folder:"), 0, 0)
        io_layout.addWidget(self.source_dir_edit, 0, 1)
        io_layout.addWidget(browse_btn, 0, 2)
        main_layout.addWidget(io_group)

        self.run_button = QPushButton("Find Normalization Constants")
        self.run_button.setObjectName("runButton")
        self.run_button.clicked.connect(self._run_analysis)
        main_layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Select a folder to begin.")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)
        main_layout.addStretch()

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder")
        if folder: self.source_dir_edit.setText(folder)

    def _run_analysis(self):
        source_dir = Path(self.source_dir_edit.text())
        if not source_dir.is_dir():
            QMessageBox.warning(self, "Input Error", "Please select a valid folder.");
            return

        self.run_button.setEnabled(False);
        self.run_button.setText("Analyzing...")
        self.status_label.setText("Starting analysis...")

        self.thread = QThread()
        self.worker = AnalysisWorker(source_dir)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        self.thread.start()

    def _on_progress(self, current, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"Processing {current} / {total} files")

    def _on_finished(self, min_db, max_db):
        self._cleanup_thread()
        self.status_label.setText("✅ Analysis Complete!")

        result_text = "Analysis Complete!\n\n"
        result_text += "These are your global normalization constants.\n"
        result_text += "Copy these values into your 'dataset_preprocessor.py' script.\n\n"
        result_text += f"GLOBAL_MIN_DB = {min_db:.4f}\n"
        result_text += f"GLOBAL_MAX_DB = {max_db:.4f}"

        QMessageBox.information(self, "Results", result_text)

    def _on_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self._cleanup_thread()

    def _cleanup_thread(self):
        self.run_button.setEnabled(True);
        self.run_button.setText("Find Normalization Constants")
        self.status_label.setText("Ready to start a new analysis.")
        self.progress_bar.setRange(0, 1);
        self.progress_bar.setValue(0);
        self.progress_bar.setFormat("")
        if self.thread:
            self.thread.quit();
            self.thread.wait();
            self.thread.deleteLater()
            self.worker.deleteLater();
            self.thread = None;
            self.worker = None

    def closeEvent(self, event):
        if self.worker: self.worker.stop()
        event.accept()

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QWidget { background-color: #2E3440; color: #ECEFF4; }
            QGroupBox { font-size: 14px; font-weight: bold; color: #88C0D0; border: 1px solid #4C566A; border-radius: 5px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 10px; }
            QLabel { font-size: 12px; } #statusLabel { font-size: 14px; color: #A3BE8C; }
            QLineEdit { background-color: #434C5E; border: 1px solid #4C566A; border-radius: 4px; padding: 6px; }
            QPushButton { background-color: #5E81AC; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #81A1C1; }
            #runButton { background-color: #A3BE8C; color: #2E3440; } #runButton:hover { background-color: #B48EAD; }
            #runButton:disabled { background-color: #4C566A; color: #D8DEE9; }
            QProgressBar { border: 1px solid #4C566A; border-radius: 5px; text-align: center; color: #ECEFF4; }
            QProgressBar::chunk { background-color: #A3BE8C; border-radius: 4px; }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NormalizationFinderWindow()
    window.show()
    sys.exit(app.exec())

