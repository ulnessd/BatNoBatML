#!/usr/bin/env python3
# bootstrap_sorter.py — A PySide6 GUI to apply a trained model to a large set of
# unverified snippets and sort them by prediction confidence into fine-grained bins.

import multiprocessing
import os
import sys
import time
import shutil
from pathlib import Path
from typing import List, Optional


# --- Environment setup for multiprocessing reliability ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# --- Core Scientific Libraries (Backend) ---
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram
from PIL import Image

# --- TensorFlow ---
# Set TensorFlow log level to suppress informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# --- PySide6 Libraries (Frontend) ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QGridLayout, QPushButton, QLineEdit, QLabel, QFileDialog,
                               QMessageBox, QProgressBar, QGroupBox)
from PySide6.QtCore import Qt, QThread, QObject, Signal

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
NPERSEG = 1024
IMAGE_SIZE = (224, 224)

# --- Fixed, dataset-wide dB scaling (must match training/preprocessor) ---
GLOBAL_MIN_DB = -95.0   # <- replace with your locked p5 value (optionally -2 dB padding)
GLOBAL_MAX_DB = -35.0   # <- replace with your locked p95 value (optionally +2 dB padding)
EPS_DB = 1e-6

# --- Fixed frequency band (match training) ---
FMIN_HZ = 18000
FMAX_HZ = 80000



# =============================================================================
# 2. WORKER LOGIC
# =============================================================================

# Global variables for the worker processes
worker_model = None
worker_output_dir = None


def init_worker(model_path: Path, output_dir: Path):
    """Initializer for each process in the multiprocessing pool."""
    global worker_model, worker_output_dir
    worker_model = tf.keras.models.load_model(str(model_path))
    worker_output_dir = output_dir


def get_category_from_prediction(prediction: float) -> str:
    """Sorts a prediction score into one of the seven fine-grained bins."""
    if prediction >= 0.95: return "95_100_bat"
    if prediction >= 0.80: return "80_95_bat"
    if prediction >= 0.60: return "60_80_bat"
    if prediction >= 0.40: return "40_60_ambiguous"
    if prediction >= 0.20: return "20_40_noise"
    if prediction >= 0.05: return "05_20_noise"
    return "00_05_noise"


def process_single_file(filepath: Path) -> str:
    """
    Processes one WAV file, gets a prediction, and moves the file to the correct bin.
    """
    try:
        # --- Generate Spectrogram ---
        with sf.SoundFile(str(filepath), 'r') as f:
            audio = f.read(dtype='float32')
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if audio.size < NPERSEG: return "skipped_short"

        # Capture samplerate while file is open
        with sf.SoundFile(str(filepath), 'r') as f:
            audio = f.read(dtype='float32')
            samplerate = f.samplerate
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size < NPERSEG:
            return "skipped_short"

        # Spectrogram
        f_vals, _, Sxx = spectrogram(audio, samplerate, nperseg=NPERSEG)

        # Fixed 18–80 kHz band crop (stationary)
        band = (f_vals >= FMIN_HZ) & (f_vals <= FMAX_HZ)
        Sxx = Sxx[band, :]

        # Fixed, dataset-wide dB mapping (NO per-snippet autoscale)
        Sxx_db = 10 * np.log10(Sxx + 1e-9)
        Sxx_db = np.clip(Sxx_db, GLOBAL_MIN_DB, GLOBAL_MAX_DB)
        Sxx_norm = (Sxx_db - GLOBAL_MIN_DB) / max((GLOBAL_MAX_DB - GLOBAL_MIN_DB), EPS_DB)
        Sxx_norm = np.clip(Sxx_norm, 0.0, 1.0)

        img_data = (np.flipud(Sxx_norm) * 255).astype(np.uint8)
        img = Image.fromarray(img_data, 'L').resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

        # --- Prepare for Model ---
        img_array = np.array(img, dtype=np.float32)
        img_array = (img_array / 127.5) - 1.0
        img_array = np.stack([img_array] * 3, axis=-1)
        img_tensor = np.expand_dims(img_array, axis=0)

        # --- Get Prediction and Sort ---
        prediction = worker_model.predict(img_tensor, verbose=0)[0][0]
        category = get_category_from_prediction(prediction)

        dest_dir = worker_output_dir / category
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / filepath.name
        shutil.copy2(str(filepath), str(dest_path))
        # os.remove(str(filepath)) # FIX: Commented out to prevent deleting original file

        return category

    except Exception as e:
        error_message = f"File: {filepath.name}, Error: {e}"
        print(f"WORKER ERROR: {error_message}", file=sys.stderr)
        error_dir = worker_output_dir / "error_files"
        error_dir.mkdir(parents=True, exist_ok=True)
        try:
            dest_path = error_dir / filepath.name
            shutil.copy2(str(filepath), str(dest_path))
            # os.remove(str(filepath)) # FIX: Commented out to prevent deleting original file
        except Exception as move_error:
            print(f"WORKER ERROR: Could not move error file {filepath.name}: {move_error}", file=sys.stderr)
        return f"error: {error_message}"


class BootstrapWorker(QObject):
    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, input_dir: Path, output_dir: Path, model_path: Path, cores: int):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_path = model_path
        self.cores = cores
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            self.status.emit("Finding WAV files...")
            wav_files = sorted(list(self.input_dir.glob("**/*.wav")))
            if not wav_files:
                self.error.emit("No .wav files found in the selected input folder (or its subdirectories).");
                return

            total_files = len(wav_files)
            self.status.emit(f"Found {total_files} files. Initializing worker processes...")

            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=self.cores, initializer=init_worker,
                          initargs=(self.model_path, self.output_dir)) as pool:

                self.status.emit(f"Processing {total_files} files across {self.cores} cores...")
                results = []
                for i, result in enumerate(pool.imap_unordered(process_single_file, wav_files)):
                    if not self.is_running:
                        pool.terminate();
                        break
                    results.append(result)
                    self.progress.emit(i + 1, total_files)

            if not self.is_running:
                self.finished.emit({"status": "Cancelled"});
                return

            # Tally results
            counts = {
                "95_100_bat": 0, "80_95_bat": 0, "60_80_bat": 0,
                "40_60_ambiguous": 0, "20_40_noise": 0, "05_20_noise": 0,
                "00_05_noise": 0, "errors": 0, "skipped_short": 0
            }
            for res in results:
                if res.startswith("error"):
                    counts['errors'] += 1
                elif res in counts:
                    counts[res] += 1

            self.finished.emit(counts)
        except Exception as e:
            self.error.emit(str(e))


# =============================================================================
# 3. PYSIDE6 GUI APPLICATION
# =============================================================================

class BootstrapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bootstrap Sorter (7 Bins)")
        self.setGeometry(100, 100, 700, 400)
        self.thread: Optional[QThread] = None
        self.worker: Optional[BootstrapWorker] = None
        self._setup_ui()
        self._apply_stylesheet()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        io_group = QGroupBox("Directories & Model")
        io_layout = QGridLayout(io_group)
        self.input_dir_edit = QLineEdit()
        self.output_dir_edit = QLineEdit()
        self.model_path_edit = QLineEdit("BatNoBat_Alpha.keras")
        browse_input_btn = QPushButton("Browse...");
        browse_input_btn.clicked.connect(self._browse_input)
        browse_output_btn = QPushButton("Browse...");
        browse_output_btn.clicked.connect(self._browse_output)
        browse_model_btn = QPushButton("Browse...");
        browse_model_btn.clicked.connect(self._browse_model)
        io_layout.addWidget(QLabel("Input Snippets Folder:"), 0, 0);
        io_layout.addWidget(self.input_dir_edit, 0, 1);
        io_layout.addWidget(browse_input_btn, 0, 2)
        io_layout.addWidget(QLabel("Output Parent Folder:"), 1, 0);
        io_layout.addWidget(self.output_dir_edit, 1, 1);
        io_layout.addWidget(browse_output_btn, 1, 2)
        io_layout.addWidget(QLabel("Trained Model File:"), 2, 0);
        io_layout.addWidget(self.model_path_edit, 2, 1);
        io_layout.addWidget(browse_model_btn, 2, 2)
        main_layout.addWidget(io_group)

        settings_group = QGroupBox("Settings")
        settings_layout = QGridLayout(settings_group)
        self.cores_edit = QLineEdit(str(max(1, os.cpu_count() - 1)))
        settings_layout.addWidget(QLabel("CPU Cores to use:"), 0, 0);
        settings_layout.addWidget(self.cores_edit, 0, 1)
        main_layout.addWidget(settings_group)

        self.run_button = QPushButton("Start Sorting");
        self.run_button.setObjectName("runButton")
        self.run_button.clicked.connect(self._start_processing)
        main_layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar();
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Select folders and model to begin.");
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)
        main_layout.addStretch()

    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Snippet Folder")
        if folder: self.input_dir_edit.setText(folder)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Parent Folder")
        if folder: self.output_dir_edit.setText(folder)

    def _browse_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", ".", "Keras Models (*.keras)")
        if model_path: self.model_path_edit.setText(model_path)

    def _start_processing(self):
        try:
            input_dir = Path(self.input_dir_edit.text())
            output_dir = Path(self.output_dir_edit.text())
            model_path = Path(self.model_path_edit.text())
            cores = int(self.cores_edit.text())
            if not (input_dir.is_dir() and output_dir.is_dir() and model_path.is_file()):
                raise ValueError("Please select valid paths for all inputs.")
            if not (0 < cores <= os.cpu_count()):
                raise ValueError(f"CPU cores must be between 1 and {os.cpu_count()}.")
        except Exception as e:
            QMessageBox.warning(self, "Input Error", str(e));
            return

        self.run_button.setEnabled(False);
        self.run_button.setText("Processing...")

        self.thread = QThread()
        self.worker = BootstrapWorker(input_dir, output_dir, model_path, cores)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.status.connect(self.status_label.setText)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        self.thread.start()

    def _on_progress(self, current, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"{current} / {total} files")

    def _on_finished(self, counts):
        if counts.get("status") == "Cancelled":
            msg = "Operation cancelled by user."
        else:
            msg = "Bootstrapping complete!\n\nSorting Summary:\n"
            for category, count in sorted(counts.items(), reverse=True):
                if count > 0:
                    msg += f"  - {category.replace('_', ' ').replace('bat', ' Bat').replace('noise', ' Noise').title()}: {count}\n"
        QMessageBox.information(self, "Finished", msg)
        self._cleanup_thread()

    def _on_error(self, message):
        QMessageBox.critical(self, "Error", f"An error occurred:\n{message}")
        self._cleanup_thread()

    def _cleanup_thread(self):
        self.run_button.setEnabled(True);
        self.run_button.setText("Start Sorting")
        self.status_label.setText("Ready to start a new task.")
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
            #runButton:disabled, QPushButton:disabled { background-color: #4C566A; color: #D8DEE9; }
            QProgressBar { border: 1px solid #4C566A; border-radius: 5px; text-align: center; color: #ECEFF4; }
            QProgressBar::chunk { background-color: #A3BE8C; border-radius: 4px; }
        """)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = BootstrapWindow()
    window.show()
    sys.exit(app.exec())

