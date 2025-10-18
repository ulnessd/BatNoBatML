#!/usr/bin/env python3
# SorterPhase8.py — A PySide6 GUI for generating bat call training data.
# Integrates robust classification logic and new pre-screening guardrails.

import multiprocessing
import os
import sys
import time
import math
import random
from pathlib import Path

# --- Environment setup for multiprocessing reliability ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# --- Core Scientific Libraries (Backend) ---
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram, find_peaks, peak_widths
from scipy.stats import skew
import cv2
from sklearn.cluster import DBSCAN

# --- PySide6 Libraries (Frontend) ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QGridLayout, QPushButton, QLineEdit, QLabel, QFileDialog,
                               QMessageBox, QProgressBar, QGroupBox, QCheckBox)
from PySide6.QtCore import Qt, QThread, QObject, Signal

# Keep OpenCV from spinning extra threads
try:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# =============================================================================
# 1. BACKEND ANALYSIS LOGIC (WITH ROBUST FILTERS)
# =============================================================================

# --- Shared analysis parameters ---
NPERSEG = 1024
NOISE_WIN_S, NOISE_HOP_S, NOISE_KEEP_K, THRESH_C, THRESH_OFFSET_DB = 2.0, 1.0, 8, 3.0, 2.0
DBSCAN_MIN_SAMPLES_DEFAULT = 5
ECHO_MIN_NEG_SLOPE_HZ_PER_S = -120_000.0
MIN_RIDGE_DT_S, MIN_RIDGE_COLS = 0.003, 4
LR_TOP_PCT, LR_SMOOTH_WIN = 10.0, 3
MIN_LOW_FREQ_PIXELS = 20
CHUNK_TIMEOUT_S = 60
MIN_ELONGATION = 0.40
DEFAULT_PURITY_THRESHOLD = 4.5
DEFAULT_MAX_PEAK_FWHM_MS = 10.0
DEFAULT_MIN_SPECTRAL_SKEW = 0.5


def calculate_spectrogram_db(audio_chunk, samplerate, freq_min=0, freq_max=None):
    if freq_max is None: freq_max = samplerate / 2
    f, t, Sxx = spectrogram(audio_chunk, samplerate, nperseg=NPERSEG, noverlap=NPERSEG // 2, detrend=False,
                            scaling='density', mode='psd')
    freq_mask = (f >= freq_min) & (f <= freq_max)
    f, Sxx = f[freq_mask], Sxx[freq_mask, :]
    return f, t, 10 * np.log10(Sxx + 1e-12)


def threshold_and_clean(db_Sxx, global_threshold):
    binary_image_noisy = db_Sxx > global_threshold
    return cv2.morphologyEx(binary_image_noisy.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8),
                            iterations=1)


def _ridge_from_patch(patch_db, patch_binary, freq_axis, time_axis):
    _, T = patch_db.shape
    ridge_f = np.full(T, np.nan, dtype=np.float32)
    for j in range(T):
        col, mask_col = patch_db[:, j], patch_binary[:, j] > 0
        vals = col[mask_col] if np.any(mask_col) else col
        if vals.size == 0: continue
        top_idx = np.where(col >= np.percentile(vals, 100.0 - LR_TOP_PCT))[0]
        if top_idx.size == 0: top_idx = np.array([np.argmax(col)])
        w = np.maximum(col[top_idx] - np.min(col[top_idx]) + 1e-3, 1e-3)
        ridge_f[j] = float(np.sum(w * freq_axis[top_idx]) / np.sum(w))
    valid = ~np.isnan(ridge_f)
    if not np.any(valid): return None
    ridge_f, ridge_t = ridge_f[valid], time_axis[valid]
    if ridge_f.size < MIN_RIDGE_COLS: return None
    xp = np.pad(ridge_f, (LR_SMOOTH_WIN // 2, LR_SMOOTH_WIN // 2), mode='edge')
    ridge_f = np.convolve(xp, np.ones(LR_SMOOTH_WIN) / LR_SMOOTH_WIN, mode='valid')
    dt = ridge_t[-1] - ridge_t[0]
    return (ridge_f[-1] - ridge_f[0]) / dt if dt >= MIN_RIDGE_DT_S else None


def _component_elongation(patch_binary):
    ys, xs = np.nonzero(patch_binary)
    if len(xs) < 3: return 0.0, 0.0, 0.0
    pts = np.vstack([xs, ys]).astype(np.float32).T
    cov = np.cov((pts - np.mean(pts, axis=0)).T)
    vals, _ = np.linalg.eig(cov)
    vals = np.sort(np.real(vals))[::-1]
    return float(1.0 - (vals[1] / vals[0])), float(vals[0]), float(vals[1])


def calculate_coherence(patch):
    total_pixels = int(np.sum(patch))
    if total_pixels == 0: return 0.0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(patch.astype(np.uint8), connectivity=4)
    if num_labels <= 1: return 1.0
    largest = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    return largest / total_pixels


def calculate_tonal_purity(patch_db, patch_binary):
    active_pixels = patch_db[patch_binary.astype(bool)]
    if len(active_pixels) < 10: return 0.0
    overall_avg = float(np.mean(active_pixels))
    top_10_threshold = float(np.percentile(active_pixels, 90))
    top_pixels = active_pixels[active_pixels >= top_10_threshold]
    top_avg = float(np.mean(top_pixels)) if len(top_pixels) > 0 else overall_avg
    return top_avg - overall_avg


def classify_cluster(duration, bandwidth, min_freq, coeffs, coherence, purity, purity_threshold, *, simple_slope=None):
    if duration < 0.05:
        coh_thresh = 0.75
    elif duration < 0.15:
        coh_thresh = 0.70
    else:
        coh_thresh = 0.60
    if coherence < coh_thresh or duration < 0.01: return "Noise"
    purity_gate = purity_threshold + (1.0 if duration < 0.04 else 0.0)
    if purity < purity_gate: return "Noise"
    has_downward_sweep = (simple_slope is not None and simple_slope <= ECHO_MIN_NEG_SLOPE_HZ_PER_S) or (coeffs[1] < 0.0)
    if (min_freq > 17_000.0) and (bandwidth < 40_000.0) and (duration < 0.10) and has_downward_sweep:
        return "Echolocation"
    if (min_freq > 18_000.0) and (duration < 0.40) and (duration > 0.015):
        return "Social"
    return "Noise"


def pre_screen_snippet(db_Sxx, times, max_fwhm_ms, min_skew):
    """
    Applies high-level checks to the entire snippet before detailed analysis.
    Returns True if the snippet passes, False if it should be rejected as noise.
    """
    try:
        # 1. Time Profile Analysis: Must have at least one sharp peak
        time_profile = np.sum(db_Sxx, axis=0)
        peaks, _ = find_peaks(time_profile, height=np.percentile(time_profile, 80))
        if len(peaks) == 0: return False  # Must have prominent peaks

        widths, _, _, _ = peak_widths(time_profile, peaks, rel_height=0.5)
        time_per_sample_ms = (times[1] - times[0]) * 1000 if len(times) > 1 else 1.0
        widths_ms = widths * time_per_sample_ms
        if not np.any(widths_ms < max_fwhm_ms):
            return False  # No peak is sharp enough

        # 2. Frequency Profile Analysis: Must be skewed
        freq_profile = np.sum(db_Sxx, axis=1)
        if skew(freq_profile) < min_skew:
            return False  # Not enough skew, likely symmetric noise

        return True  # Passed all pre-screening checks
    except Exception:
        return False  # Fail safe on any calculation error


def has_low_freq_energy(audio_chunk, samplerate, threshold):
    try:
        _, _, db_Sxx_low = calculate_spectrogram_db(audio_chunk, samplerate, freq_min=0, freq_max=20000)
        binary_low = threshold_and_clean(db_Sxx_low, threshold)
        return np.sum(binary_low) > MIN_LOW_FREQ_PIXELS
    except Exception:
        return False


def noise_scan_global_threshold(filepath, samplerate, win_s, hop_s, keep_k, c):
    # ... (code is unchanged)
    medians = []
    win_frames, hop_frames = int(win_s * samplerate), int(hop_s * samplerate)
    if win_frames < NPERSEG: win_frames = max(NPERSEG, win_frames)
    for block in sf.blocks(filepath, blocksize=win_frames, overlap=win_frames - hop_frames, dtype='float32',
                           always_2d=True):
        if block.shape[0] < NPERSEG: continue
        audio_mono = block[:, 0]
        _, _, db_Sxx = calculate_spectrogram_db(audio_mono, samplerate)
        medians.append(float(np.median(db_Sxx)))
    if not medians: return -100.0
    medians = np.array(medians)
    k = int(max(1, min(keep_k, len(medians))))
    quiet_idx = np.argpartition(medians, k - 1)[:k]
    quiet_meds = medians[quiet_idx]
    median_med = float(np.median(quiet_meds))
    mad_med = float(np.median(np.abs(quiet_meds - median_med)))
    threshold = median_med + c * (1.4826 * mad_med)
    return float(threshold)


def process_and_classify_chunk(args):
    filepath, start_frame, num_frames, samplerate, global_threshold, use_harmonic_filter, use_elong_filter, purity_thresh, use_prescreen, max_fwhm, min_skew = args
    with sf.SoundFile(filepath, 'r') as f:
        f.seek(start_frame)
        audio_chunk = f.read(num_frames, dtype='float32', always_2d=True)
        if audio_chunk.shape[1] > 0:
            audio_chunk = audio_chunk[:, 0]
        else:
            return "None"
    if len(audio_chunk) < NPERSEG: return "None"

    frequencies, times, db_Sxx = calculate_spectrogram_db(audio_chunk, samplerate)

    # --- NEW PRE-SCREENING STEP ---
    if use_prescreen:
        if not pre_screen_snippet(db_Sxx, times, max_fwhm, min_skew):
            return "Noise"

    binary_image = threshold_and_clean(db_Sxx, global_threshold)
    points = np.argwhere(binary_image)
    if points.shape[0] < 10: return "None"

    db = DBSCAN(eps=3, min_samples=DBSCAN_MIN_SAMPLES_DEFAULT).fit(points)

    found_echo, found_social = False, False
    for k in set(db.labels_):
        if k == -1: continue
        cluster_points = points[db.labels_ == k]

        min_r, max_r, min_c, max_c = np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0]), np.min(
            cluster_points[:, 1]), np.max(cluster_points[:, 1])
        patch_binary = binary_image[min_r:max_r + 1, min_c:max_c + 1]
        patch_db = db_Sxx[min_r:max_r + 1, min_c:max_c + 1]
        patch_freqs, patch_times = frequencies[min_r:max_r + 1], times[min_c:max_c + 1]

        coherence = calculate_coherence(patch_binary)
        tonal_purity = calculate_tonal_purity(patch_db, patch_binary)

        c_times, c_freqs = times[cluster_points[:, 1]], frequencies[cluster_points[:, 0]]
        duration, bandwidth, min_freq = float(np.max(c_times) - np.min(c_times)), float(
            np.max(c_freqs) - np.min(c_freqs)), float(np.min(c_freqs))

        if use_elong_filter:
            elongation, _, _ = _component_elongation(patch_binary)
            e_gate = max(0.0, MIN_ELONGATION - (0.20 if duration < 0.05 else 0.10))
            if elongation < e_gate: continue

        try:
            coeffs = np.polyfit(c_times, c_freqs, 2)
        except:
            coeffs = (0.0, 0.0, 0.0)

        simple_slope = _ridge_from_patch(patch_db, patch_binary, patch_freqs, patch_times)

        label = classify_cluster(duration, bandwidth, min_freq, coeffs, coherence, tonal_purity, purity_thresh,
                                 simple_slope=simple_slope)

        if label == "Echolocation": found_echo = True
        if label == "Social": found_social = True

    if use_harmonic_filter and found_social and has_low_freq_energy(audio_chunk, samplerate, global_threshold):
        found_social = False

    if found_echo and found_social: return "Both"
    if found_echo: return "Echolocation"
    if found_social: return "Social"
    return "None"


class AnalysisWorker(QObject):
    progress_update = Signal(str);
    progress_value = Signal(int);
    finished = Signal(str);
    error = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.is_running = True

    def run(self):
        try:
            # Unpack params for clarity
            input_dir = Path(self.params['input_dir'])
            output_dir = Path(self.params['output_dir'])
            is_recursive = self.params['is_recursive']
            cores = self.params['cores']

            self.progress_update.emit(f"Searching for .wav files in '{input_dir}'...")
            glob_pattern = "**/*.wav" if is_recursive else "*.wav";
            glob_pattern_upper = "**/*.WAV" if is_recursive else "*.WAV"
            wav_files = sorted(list(input_dir.glob(glob_pattern))) + sorted(list(input_dir.glob(glob_pattern_upper)))
            if not wav_files: self.finished.emit("No .wav files found."); return
            self.progress_update.emit(f"Found {len(wav_files)} files to process.")

            total_snippets_generated = 0
            for i, filepath in enumerate(wav_files):
                if not self.is_running: self.finished.emit("Operation cancelled."); return
                self.progress_update.emit(f"Processing file {i + 1}/{len(wav_files)}: {filepath.name}")
                try:
                    info = sf.info(str(filepath)); sr, total_frames = info.samplerate, info.frames
                except Exception as e:
                    self.progress_update.emit(f"[WARNING] Skipping {filepath.name}: {e}"); continue

                noise_params = {'win_s': NOISE_WIN_S, 'hop_s': NOISE_HOP_S, 'keep_k': NOISE_KEEP_K, 'c': THRESH_C}
                global_threshold = noise_scan_global_threshold(str(filepath), sr, **noise_params) + THRESH_OFFSET_DB
                chunk_duration_s, step_duration_s = 0.5, 0.4
                chunk_frames, step_frames = int(chunk_duration_s * sr), int(step_duration_s * sr)

                tasks = [(str(filepath), start, chunk_frames, sr, global_threshold,
                          self.params['use_harmonic_filter'], self.params['use_elong_filter'],
                          self.params['purity_thresh'],
                          self.params['use_prescreen'], self.params['max_fwhm'], self.params['min_skew'])
                         for start in range(0, total_frames, step_frames) if start + chunk_frames <= total_frames]
                if not tasks: continue

                results = ["None"] * len(tasks)
                with multiprocessing.get_context("spawn").Pool(processes=cores) as pool:
                    pool_results = {i_task: pool.apply_async(process_and_classify_chunk, (task,)) for i_task, task in
                                    enumerate(tasks)}
                    for i_task, res in pool_results.items():
                        try:
                            results[i_task] = res.get(timeout=CHUNK_TIMEOUT_S)
                        except Exception as e:
                            self.progress_update.emit(f"[WARNING] Error in chunk for {filepath.name}: {e}")

                call_chunks_saved, none_chunk_indices = 0, []
                for j, verdict in enumerate(results):
                    if verdict and verdict != "None":
                        start_frame = tasks[j][1]
                        with sf.SoundFile(str(filepath), 'r') as f:
                            f.seek(start_frame)
                            audio_chunk = f.read(chunk_frames, dtype='float32', always_2d=True)[:, 0]
                        start_time_ms = int((j * step_duration_s) * 1000)
                        out_filename = f"{filepath.stem}__{start_time_ms}ms.wav"
                        out_path = output_dir / verdict.lower() / out_filename
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(out_path, audio_chunk, sr);
                        call_chunks_saved += 1
                    else:
                        none_chunk_indices.append(j)
                total_snippets_generated += call_chunks_saved

                num_noise_to_save = max(4 * call_chunks_saved, 100) if call_chunks_saved > 0 else self.params[
                    'noise_cap_per_file']
                if num_noise_to_save > 0 and none_chunk_indices:
                    indices_to_save = random.sample(none_chunk_indices, min(num_noise_to_save, len(none_chunk_indices)))
                    for j in indices_to_save:
                        start_frame = tasks[j][1]
                        with sf.SoundFile(str(filepath), 'r') as f:
                            f.seek(start_frame)
                            audio_chunk = f.read(chunk_frames, dtype='float32', always_2d=True)[:, 0]
                        start_time_ms = int((j * step_duration_s) * 1000)
                        out_filename = f"{filepath.stem}__{start_time_ms}ms.wav"
                        out_path = output_dir / "noise" / out_filename
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(out_path, audio_chunk, sr);
                        total_snippets_generated += 1
                self.progress_value.emit(i + 1)
            self.finished.emit(f"Processing complete. Generated {total_snippets_generated} total snippets.")
        except Exception as e:
            import traceback;
            print(traceback.format_exc());
            self.error.emit(f"A critical error occurred: {e}")

    def stop(self):
        self.is_running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__();
        self.setWindowTitle("Bat Snippet Generator (Phase 8)");
        self.setGeometry(100, 100, 600, 650)
        self.thread, self.worker = None, None;
        self.central_widget = QWidget();
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget);
        self._create_widgets();
        self._apply_stylesheet()

    def _create_widgets(self):
        io_group = QGroupBox("Directories");
        io_layout = QGridLayout(io_group)
        self.input_dir_edit = QLineEdit();
        self.input_dir_edit.setPlaceholderText("Select directory with raw .wav files...")
        browse_input_btn = QPushButton("Browse...");
        browse_input_btn.clicked.connect(lambda: self._browse_dir(self.input_dir_edit, "Select Input Folder"))
        self.output_dir_edit = QLineEdit();
        self.output_dir_edit.setPlaceholderText("Select directory to save snippets...")
        browse_output_btn = QPushButton("Browse...");
        browse_output_btn.clicked.connect(lambda: self._browse_dir(self.output_dir_edit, "Select Output Folder"))
        io_layout.addWidget(QLabel("Input Folder:"), 0, 0);
        io_layout.addWidget(self.input_dir_edit, 0, 1);
        io_layout.addWidget(browse_input_btn, 0, 2)
        io_layout.addWidget(QLabel("Output Folder:"), 1, 0);
        io_layout.addWidget(self.output_dir_edit, 1, 1);
        io_layout.addWidget(browse_output_btn, 1, 2)
        self.main_layout.addWidget(io_group)

        # New Pre-Screening Group
        prescreen_group = QGroupBox("Pre-Screening Filters");
        prescreen_layout = QGridLayout(prescreen_group)
        self.prescreen_check = QCheckBox("Enable Pre-Screening");
        self.prescreen_check.setChecked(True)
        self.max_fwhm_edit = QLineEdit(str(DEFAULT_MAX_PEAK_FWHM_MS))
        self.min_skew_edit = QLineEdit(str(DEFAULT_MIN_SPECTRAL_SKEW))
        prescreen_layout.addWidget(self.prescreen_check, 0, 0, 1, 2)
        prescreen_layout.addWidget(QLabel("Max Peak FWHM (ms):"), 1, 0);
        prescreen_layout.addWidget(self.max_fwhm_edit, 1, 1)
        prescreen_layout.addWidget(QLabel("Min Spectral Skew:"), 2, 0);
        prescreen_layout.addWidget(self.min_skew_edit, 2, 1)
        self.main_layout.addWidget(prescreen_group)

        settings_group = QGroupBox("Detailed Cluster Settings");
        settings_layout = QGridLayout(settings_group)
        self.cores_edit = QLineEdit(str(max(1, int(os.cpu_count() * 0.75))))
        self.noise_cap_edit = QLineEdit("500")
        self.purity_thresh_edit = QLineEdit(str(DEFAULT_PURITY_THRESHOLD))
        self.harmonic_filter_check = QCheckBox("Enable Social Call Harmonic Filter");
        self.harmonic_filter_check.setChecked(True)
        self.elongation_filter_check = QCheckBox("Enable Elongation Filter");
        self.elongation_filter_check.setChecked(True)
        self.recursive_check = QCheckBox("Search Subdirectories Recursively");
        self.recursive_check.setChecked(True)
        settings_layout.addWidget(QLabel("CPU Cores:"), 0, 0);
        settings_layout.addWidget(self.cores_edit, 0, 1)
        settings_layout.addWidget(QLabel("Noise Snippets from Bat-Free File:"), 1, 0);
        settings_layout.addWidget(self.noise_cap_edit, 1, 1)
        settings_layout.addWidget(QLabel("Tonal Purity Threshold:"), 2, 0);
        settings_layout.addWidget(self.purity_thresh_edit, 2, 1)
        settings_layout.addWidget(self.harmonic_filter_check, 3, 0);
        settings_layout.addWidget(self.elongation_filter_check, 3, 1)
        settings_layout.addWidget(self.recursive_check, 4, 0, 1, 2);
        self.main_layout.addWidget(settings_group)

        self.run_button = QPushButton("Start Generation");
        self.run_button.setObjectName("runButton");
        self.run_button.clicked.connect(self._start_analysis)
        self.main_layout.addWidget(self.run_button);
        self.progress_bar = QProgressBar();
        self.progress_bar.setTextVisible(False);
        self.main_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Select directories and click 'Start Generation'.");
        self.status_label.setObjectName("statusLabel");
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter);
        self.main_layout.addWidget(self.status_label);
        self.main_layout.addStretch()

    def _browse_dir(self, line_edit, title):
        folder = QFileDialog.getExistingDirectory(self, title);
        if folder: line_edit.setText(folder)

    def _get_params(self):
        try:
            params = {
                'input_dir': self.input_dir_edit.text(),
                'output_dir': self.output_dir_edit.text(),
                'cores': int(self.cores_edit.text()),
                'noise_cap_per_file': int(self.noise_cap_edit.text()),
                'purity_thresh': float(self.purity_thresh_edit.text()),
                'use_harmonic_filter': self.harmonic_filter_check.isChecked(),
                'use_elong_filter': self.elongation_filter_check.isChecked(),
                'is_recursive': self.recursive_check.isChecked(),
                'use_prescreen': self.prescreen_check.isChecked(),
                'max_fwhm': float(self.max_fwhm_edit.text()),
                'min_skew': float(self.min_skew_edit.text())
            }
            # Basic validation
            if not (1 <= params['cores'] <= os.cpu_count() and params['noise_cap_per_file'] > 0 and params[
                'purity_thresh'] >= 0):
                raise ValueError("Core parameter out of range.")
            if not (params['max_fwhm'] > 0 and params['min_skew'] >= 0):
                raise ValueError("Pre-screening parameter out of range.")
            return params
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Enter valid numbers for parameters. Error: {e}");
            return None

    def _start_analysis(self):
        params = self._get_params()
        if not params: return
        if not (Path(params['input_dir']).is_dir() and Path(params['output_dir']).is_dir()):
            QMessageBox.warning(self, "Input Error", "Please select valid directories.");
            return

        self.run_button.setEnabled(False);
        self.run_button.setText("Processing...")

        glob_pattern = "**/*.wav" if params['is_recursive'] else "*.wav";
        glob_pattern_upper = "**/*.WAV" if params['is_recursive'] else "*.WAV"
        wav_files = list(Path(params['input_dir']).glob(glob_pattern)) + list(
            Path(params['input_dir']).glob(glob_pattern_upper))

        self.progress_bar.setRange(0, len(wav_files));
        self.progress_bar.setValue(0)
        self.thread = QThread()
        self.worker = AnalysisWorker(params)
        self.worker.moveToThread(self.thread);
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_finished);
        self.worker.error.connect(self._on_error)
        self.worker.progress_update.connect(self.status_label.setText);
        self.worker.progress_value.connect(self.progress_bar.setValue)
        self.thread.start()

    def _on_finished(self, message):
        QMessageBox.information(self, "Success", message); self._cleanup_thread()

    def _on_error(self, message):
        QMessageBox.critical(self, "Error", message); self._cleanup_thread()

    def _cleanup_thread(self):
        self.status_label.setText("Select directories and click 'Start Generation'.");
        self.run_button.setEnabled(True);
        self.run_button.setText("Start Generation")
        self.progress_bar.setValue(0)
        if self.thread and self.thread.isRunning(): self.worker.stop(); self.thread.quit(); self.thread.wait()
        self.thread, self.worker = None, None

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning(): self.worker.stop()
        event.accept()

    def _apply_stylesheet(self):
        self.setStyleSheet(
            """QWidget { background-color: #2E3440; color: #ECEFF4; font-family: "Segoe UI", sans-serif; } QGroupBox { font-size: 14px; font-weight: bold; color: #88C0D0; border: 1px solid #4C566A; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 10px; } QLabel { font-size: 12px; } #statusLabel { font-size: 14px; color: #A3BE8C; } QLineEdit { background-color: #434C5E; border: 1px solid #4C566A; border-radius: 4px; padding: 6px; font-size: 12px; } QLineEdit:focus { border: 1px solid #88C0D0; } QPushButton { background-color: #5E81AC; color: #ECEFF4; border: none; padding: 8px 16px; border-radius: 4px; font-size: 14px; font-weight: bold; } QPushButton:hover { background-color: #81A1C1; } #runButton { background-color: #A3BE8C; color: #2E3440; } #runButton:hover { background-color: #B48EAD; } #runButton:disabled { background-color: #4C566A; color: #D8DEE9; } QProgressBar { border: 1px solid #4C566A; border-radius: 5px; text-align: center; color: #2E3440; font-weight: bold; } QProgressBar::chunk { background-color: #A3BE8C; border-radius: 4px; } QCheckBox { font-size: 12px; } QCheckBox::indicator { width: 18px; height: 18px; }""")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
