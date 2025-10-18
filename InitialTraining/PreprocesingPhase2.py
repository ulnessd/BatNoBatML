#!/usr/bin/env python3
# dataset_preprocessor.py — A PySide6 GUI to prepare a curated audio dataset for TensorFlow.

import sys
import os
import shutil
import time
import random
import csv
from pathlib import Path
from typing import List, Tuple, Optional


# --- Core Scientific Libraries (Backend) ---
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram
from PIL import Image

# --- PySide6 Libraries (Frontend) ---
from PySide6.QtCore import (Qt, QThread, QObject, Signal, QSize)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QGridLayout, QPushButton, QLineEdit, QLabel, QFileDialog,
                               QMessageBox, QProgressBar, QGroupBox)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# --- Spectrogram Generation ---
NPERSEG = 1024
IMG_SIZE = (224, 224)  # Standard size for many CNNs (width, height)
# --- Global, fixed dB scaling (replace with your computed p5/p95) ---

# --- Fixed frequency band for all spectrograms ---
FMIN_HZ = 18000
FMAX_HZ = 80000

GLOBAL_MIN_DB = -95.0   # TODO: set to dataset-wide 5th percentile (dB)
GLOBAL_MAX_DB = -35.0   # TODO: set to dataset-wide 95th percentile (dB)
EPS_DB = 1e-6           # guard for zero division



# --- Dataset Split ---
VALIDATION_SPLIT = 0.2  # 20% of the data will be used for validation


# =============================================================================
# 2. WORKER THREAD FOR DATASET GENERATION
# =============================================================================

class PreprocessingWorker(QObject):
    """
    A QObject worker that runs the entire preprocessing pipeline in a separate thread.
    """
    progress = Signal(int, int, str)  # current, total, message
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, source_dir: str, output_dir: str):
        super().__init__()
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            # --- 1. Find and validate input folders ---
            self.progress.emit(0, 100, "Finding source WAV files...")
            bat_dir = self.source_dir / "EcholocationCalls"
            noise_dir = self.source_dir / "NoBatNoise"

            if not bat_dir.is_dir() or not noise_dir.is_dir():
                self.error.emit("Could not find 'EcholocationCalls' and/or 'NoBatNoise' subfolders.")
                return

            bat_files = sorted(list(bat_dir.glob("*.wav")))
            noise_files = sorted(list(noise_dir.glob("*.wav")))

            if not bat_files or not noise_files:
                self.error.emit("Source folders are empty. No .wav files found.")
                return

            # --- 2. Create Train/Validation Split ---
            self.progress.emit(10, 100, "Creating train/validation split...")
            random.shuffle(bat_files)
            random.shuffle(noise_files)

            bat_split_idx = int(len(bat_files) * (1 - VALIDATION_SPLIT))
            noise_split_idx = int(len(noise_files) * (1 - VALIDATION_SPLIT))

            train_files = [(p, 'bat') for p in bat_files[:bat_split_idx]] + \
                          [(p, 'noise') for p in noise_files[:noise_split_idx]]

            val_files = [(p, 'bat') for p in bat_files[bat_split_idx:]] + \
                        [(p, 'noise') for p in noise_files[noise_split_idx:]]

            random.shuffle(train_files)
            random.shuffle(val_files)

            # --- 3. Create Output Directory Structure ---
            self.progress.emit(20, 100, "Creating output directories...")
            train_bat_dir = self.output_dir / "train" / "bat"
            train_noise_dir = self.output_dir / "train" / "noise"
            val_bat_dir = self.output_dir / "validation" / "bat"
            val_noise_dir = self.output_dir / "validation" / "noise"

            for d in [train_bat_dir, train_noise_dir, val_bat_dir, val_noise_dir]:
                d.mkdir(parents=True, exist_ok=True)

            # --- 4. Process and Save Spectrograms ---
            all_tasks = [('train', p, l) for p, l in train_files] + \
                        [('validation', p, l) for p, l in val_files]

            total_files = len(all_tasks)
            manifests = {
                'train': csv.writer(open(self.output_dir / 'train_manifest.csv', 'w', newline='')),
                'validation': csv.writer(open(self.output_dir / 'validation_manifest.csv', 'w', newline=''))
            }
            for writer in manifests.values():
                writer.writerow(['filepath', 'label'])

            for i, (split, wav_path, label) in enumerate(all_tasks):
                if not self.is_running:
                    self.finished.emit("Operation cancelled.")
                    return

                self.progress.emit(i, total_files, f"Processing {wav_path.name}")

                try:
                    img = self._generate_spectrogram(wav_path)

                    # Determine output path
                    if split == 'train':
                        dest_dir = train_bat_dir if label == 'bat' else train_noise_dir
                    else:  # validation
                        dest_dir = val_bat_dir if label == 'bat' else val_noise_dir

                    out_path = dest_dir / (wav_path.stem + ".png")
                    img.save(out_path)

                    # Log to manifest
                    relative_path = out_path.relative_to(self.output_dir)
                    manifests[split].writerow([relative_path.as_posix(), label])

                except Exception as e:
                    print(f"Skipping file {wav_path.name} due to error: {e}")

            self.finished.emit(f"Dataset creation complete. Processed {total_files} files.")

        except Exception as e:
            self.error.emit(f"A critical error occurred: {e}")

    def _generate_spectrogram(self, filepath: Path) -> Image:
        """Reads a WAV, computes a spectrogram, and returns it as a PIL Image."""
        with sf.SoundFile(str(filepath), 'r') as f:
            samplerate = f.samplerate
            audio = f.read(dtype='float32')

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size < NPERSEG:
            raise ValueError("Audio signal too short")

        f, t, Sxx = spectrogram(audio, samplerate, nperseg=NPERSEG)
        band = (f >= FMIN_HZ) & (f <= FMAX_HZ)
        Sxx = Sxx[band, :]

        Sxx_db = 10 * np.log10(Sxx + 1e-9)

        # --- Fixed, dataset-wide normalization (NO per-snippet autoscale) ---
        Sxx_db = np.clip(Sxx_db, GLOBAL_MIN_DB, GLOBAL_MAX_DB)
        Sxx_norm = (Sxx_db - GLOBAL_MIN_DB) / max((GLOBAL_MAX_DB - GLOBAL_MIN_DB), EPS_DB)

        # Convert to 8-bit grayscale
        img_data = (np.flipud(Sxx_norm) * 255).astype(np.uint8)

        img = Image.fromarray(img_data, 'L')
        return img.resize(IMG_SIZE, Image.Resampling.LANCZOS)


# =============================================================================
# 3. MAIN APPLICATION WINDOW
# =============================================================================

class PreprocessorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Preprocessor")
        self.setGeometry(100, 100, 600, 400)
        self.thread: Optional[QThread] = None
        self.worker: Optional[PreprocessingWorker] = None
        self._setup_ui()
        self._apply_stylesheet()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        io_group = QGroupBox("Directories")
        io_layout = QGridLayout(io_group)
        self.source_dir_edit = QLineEdit()
        self.source_dir_edit.setPlaceholderText("Select folder containing 'EcholocationCalls' and 'NoBatNoise'...")
        browse_source_btn = QPushButton("Browse...")
        browse_source_btn.clicked.connect(self._browse_source)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select folder to save the final dataset...")
        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.clicked.connect(self._browse_output)

        io_layout.addWidget(QLabel("Source Folder:"), 0, 0);
        io_layout.addWidget(self.source_dir_edit, 0, 1);
        io_layout.addWidget(browse_source_btn, 0, 2)
        io_layout.addWidget(QLabel("Output Folder:"), 1, 0);
        io_layout.addWidget(self.output_dir_edit, 1, 1);
        io_layout.addWidget(browse_output_btn, 1, 2)
        main_layout.addWidget(io_group)

        self.run_button = QPushButton("Create Dataset");
        self.run_button.setObjectName("runButton")
        self.run_button.clicked.connect(self._start_processing)
        main_layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar();
        self.progress_bar.setTextVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Select directories and click 'Create Dataset'.");
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        main_layout.addStretch()

    def _browse_source(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder")
        if folder: self.source_dir_edit.setText(folder)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder: self.output_dir_edit.setText(folder)

    def _start_processing(self):
        source_dir = self.source_dir_edit.text()
        output_dir = self.output_dir_edit.text()
        if not (Path(source_dir).is_dir() and Path(output_dir).is_dir()):
            QMessageBox.warning(self, "Input Error", "Please select valid source and output directories.")
            return

        self.run_button.setEnabled(False);
        self.run_button.setText("Processing...")

        self.thread = QThread()
        self.worker = PreprocessingWorker(source_dir, output_dir)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.progress.connect(self._on_progress)
        self.thread.start()

    def _on_progress(self, current, total, message):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.status_label.setText(message)

    def _on_finished(self, message):
        QMessageBox.information(self, "Success", message);
        self._cleanup_thread()

    def _on_error(self, message):
        QMessageBox.critical(self, "Error", message);
        self._cleanup_thread()

    def _cleanup_thread(self):
        self.status_label.setText("Select directories and click 'Create Dataset'.")
        self.run_button.setEnabled(True);
        self.run_button.setText("Create Dataset")
        self.progress_bar.setValue(0)
        if self.thread and self.thread.isRunning():
            self.worker.stop();
            self.thread.quit();
            self.thread.wait()
        self.thread, self.worker = None, None

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning(): self.worker.stop()
        event.accept()

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QWidget { background-color: #2E3440; color: #ECEFF4; font-family: "Segoe UI", sans-serif; }
            QGroupBox { font-size: 14px; font-weight: bold; color: #88C0D0; border: 1px solid #4C566A; border-radius: 5px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 10px; }
            QLabel { font-size: 12px; } #statusLabel { font-size: 14px; color: #A3BE8C; }
            QLineEdit { background-color: #434C5E; border: 1px solid #4C566A; border-radius: 4px; padding: 6px; font-size: 12px; }
            QLineEdit:focus { border: 1px solid #88C0D0; }
            QPushButton { background-color: #5E81AC; color: #ECEFF4; border: none; padding: 8px 16px; border-radius: 4px; font-size: 14px; font-weight: bold; }
            QPushButton:hover { background-color: #81A1C1; }
            #runButton { background-color: #A3BE8C; color: #2E3440; } #runButton:hover { background-color: #B48EAD; } #runButton:disabled { background-color: #4C566A; color: #D8DEE9; }
            QProgressBar { border: 1px solid #4C566A; border-radius: 5px; text-align: center; color: #2E3440; font-weight: bold; }
            QProgressBar::chunk { background-color: #A3BE8C; border-radius: 4px; }
        """)


# =============================================================================
# 4. APPLICATION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PreprocessorWindow()
    window.show()
    sys.exit(app.exec())
