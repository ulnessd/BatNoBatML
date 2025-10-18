#!/usr/bin/env python3
"""
BatSpectrogramInspector v5 (Phase 2):
- Thumbnails locked to v4 behavior (entire spectrogram, fixed noise-aware dB scaling,
  letterboxed, Agg workers). No coupling to canvas controls.
- Canvas on the right with **publication controls** and **mouse drag-zoom**:
  • Colormap dropdown
  • Dynamic range (dB) & Visual gain (dB)
  • Title toggle + Title/X/Y labels
  • Font sizes: title, labels, ticks
  • Numeric ranges: tmin/tmax, fmin/fmax with two-way sync
  • Zoom mode (rectangle drag) + Reset View + Fit to Data
- No Matplotlib toolbar (clean UI). GUI drawing stays in main thread; STFT stays in canvas worker.
- Thread lifecycle hardened (cancel+wait; cleanup on finished; parented workers).

Dependencies: PySide6, numpy, soundfile, scipy (signal), matplotlib
"""
from __future__ import annotations

import io
import sys
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import json

import numpy as np
import soundfile as sf
from scipy.signal import stft, get_window
from scipy.signal import istft
from scipy.ndimage import gaussian_filter
#from scipy.signal import hilbert, resample_poly

import tempfile
import subprocess
from PIL import Image, ImageDraw
import imageio.v2 as imageio


# --- Matplotlib: Agg for workers; QtAgg lazily for GUI canvas ---
import matplotlib
matplotlib.use("Agg")  # worker/offscreen rendering
import matplotlib.pyplot as plt  # noqa: E402

from PySide6.QtCore import Qt, QSize, QThread, Signal, QObject, QSettings, QByteArray
from PySide6.QtGui import QImage, QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QWidget, QFileDialog, QListWidget, QListWidgetItem,
    QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSplitter,
    QMessageBox, QLineEdit, QFrame, QGroupBox, QGridLayout, QComboBox,
    QDoubleSpinBox, QCheckBox, QSpinBox
)

# -----------------------------
# Parameters / defaults
# -----------------------------
N_FFT = 2048
HOP = 1024
WINDOW = "hann"

# -----------------------------
# Canvas STFT defaults (user adjustable)
# -----------------------------
CANVAS_N_FFT_DEFAULT = 1024    # ~2.67 ms window @ 384 kHz
CANVAS_HOP_DEFAULT   = 256     # ~0.67 ms step (75% overlap)


# Noise band to estimate baseline power; keep away from bat band if desired
NOISE_BAND = (8000, 12000)

# Thumbnail rendering box and decimation caps
THUMB_W = 336
THUMB_H = 252  # 4:3
MAX_T_COLS = 1200
MAX_F_ROWS = 512

# Fixed thumbnail normalization (hard-coded; never changes at runtime)
THUMB_DYN_RANGE_DB = 70.0
THUMB_VGAIN_DB = 0.0
THUMB_FMIN = 0.0
THUMB_FMAX = 52000.0
THUMB_COLORMAP = "magma"

# Canvas decimation caps (larger than thumbnails so it looks nicer)
CANVAS_MAX_T_COLS = 4000
CANVAS_MAX_F_ROWS = 1024

# Canvas defaults (start equal to thumbnail defaults)
CANVAS_DYN_RANGE_DB_DEFAULT = THUMB_DYN_RANGE_DB
CANVAS_VGAIN_DB_DEFAULT = THUMB_VGAIN_DB
COLORMAPS = [
    "magma", "inferno", "viridis", "plasma", "cividis",
    "gray", "Greys", "turbo"
]


# =============================
# Thumbnail pipeline (Agg workers)
# =============================
@dataclass
class ThumbJob:
    wav_path: Path

class ThumbSignals(QObject):
    done = Signal(str, QPixmap, float, float)  # (path, pixmap, duration, sr)
    error = Signal(str, str)

class ThumbWorker(QThread):
    def __init__(self, parent: QObject, job: ThumbJob):
        super().__init__(parent)
        self.job = job
        self.signals = ThumbSignals()
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            pixmap, dur, sr = self._compute_thumbnail(self.job)
            if self._cancel:
                return
            self.signals.done.emit(str(self.job.wav_path), pixmap, dur, sr)
        except Exception as e:
            tb = traceback.format_exc()
            msg = f"{e}\n{traceback.format_exc()}"

    def _compute_thumbnail(self, job: ThumbJob) -> Tuple[QPixmap, float, float]:
        wav_path = job.wav_path
        with sf.SoundFile(str(wav_path)) as f:
            sr = float(f.samplerate)
        y, _ = sf.read(str(wav_path), always_2d=True)
        y = y.mean(axis=1).astype(np.float32, copy=False)
        dur = len(y) / sr if sr > 0 else 0.0

        win = get_window(WINDOW, N_FFT, fftbins=True)
        noverlap = N_FFT - HOP

        # Chunked STFT for memory-friendliness on long files
        chunk_sec = 8.0
        chunk_len = int(chunk_sec * sr)
        stride = max(chunk_len - int(0.5 * sr), chunk_len)
        if len(y) <= chunk_len:
            chunks = [(0, len(y))]
        else:
            chunks = []
            start = 0
            while start < len(y):
                end = min(start + chunk_len, len(y))
                chunks.append((start, end))
                if end == len(y):
                    break
                start += stride

        all_S = []
        for a, b in chunks:
            if self._cancel:
                break
            yf = y[a:b]
            f_hz, t_sec, Z = stft(yf, fs=sr, window=win, nperseg=N_FFT,
                                  noverlap=noverlap, boundary=None, padded=False)
            S = (np.abs(Z) ** 2).astype(np.float32)
            all_S.append((f_hz, t_sec + (a / sr), S))
        if self._cancel or not all_S:
            return QPixmap(), dur, sr

        # Frequency band mask (fixed for thumbnails)
        f_full = all_S[0][0]
        if THUMB_FMIN > 0 or THUMB_FMAX > 0:
            fmask = (f_full >= THUMB_FMIN) & (f_full <= (THUMB_FMAX if THUMB_FMAX > 0 else sr/2))
            if not np.any(fmask):
                fmask = np.ones_like(f_full, dtype=bool)
        else:
            fmask = np.ones_like(f_full, dtype=bool)
        f = f_full[fmask]

        S_list, t_list = [], []
        for _, t_chunk, S in all_S:
            S_list.append(S[fmask, :])
            t_list.append(t_chunk)
        S_full = np.concatenate(S_list, axis=1) if len(S_list) > 1 else S_list[0]
        t = np.concatenate(t_list) if len(t_list) > 1 else t_list[0]

        # Decimate to cap size
        S_ds, f_ds, t_ds = self._decimate(S_full, f, t, MAX_F_ROWS, MAX_T_COLS)

        # Noise-aware dB scaling
        ref_power = self._estimate_ref_power(S_ds, f_ds)
        eps = 1e-12
        S_db = 10.0 * np.log10((S_ds + eps) / (ref_power + eps))
        vmax = np.percentile(S_db, 99.5) + THUMB_VGAIN_DB
        vmin = vmax - THUMB_DYN_RANGE_DB
        S_db = np.clip(S_db, vmin, vmax)

        # Render full spectrogram (no axes), then letterbox into target box
        fig = plt.figure(figsize=(6.0, 4.0), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        extent = [float(t_ds[0]), float(t_ds[-1]), float(f_ds[0]), float(f_ds[-1])]
        ax.imshow(S_db, origin="lower", aspect="auto", cmap=THUMB_COLORMAP, extent=extent)
        buf = io.BytesIO(); fig.canvas.print_png(buf); plt.close(fig)
        qimg = QImage.fromData(buf.getvalue(), "PNG"); buf.close()
        if qimg.isNull():
            raise RuntimeError("Null thumbnail image")
        thumb = self._letterbox_qimage(qimg, QSize(THUMB_W, THUMB_H))
        return QPixmap.fromImage(thumb), dur, sr

    @staticmethod
    def _estimate_ref_power(S: np.ndarray, f: np.ndarray) -> float:
        nb_lo, nb_hi = NOISE_BAND
        nb_mask = (f >= nb_lo) & (f <= nb_hi)
        if np.any(nb_mask):
            ref = np.median(S[nb_mask, :])
            if np.isfinite(ref) and ref > 0:
                return float(ref)
        pos = S[S > 0]
        return float(np.median(pos) if pos.size else 1e-12)

    @staticmethod
    def _decimate(S: np.ndarray, f: np.ndarray, t: np.ndarray, max_rows: int, max_cols: int):
        # Frequency decimation
        fr = S.shape[0]
        if fr > max_rows:
            step_f = math.ceil(fr / max_rows)
            S = S[::step_f, :]
            f = f[::step_f]
        # Time decimation (avg pooling)
        tc = S.shape[1]
        if tc > max_cols:
            step_t = math.ceil(tc / max_cols)
            keep = (tc // step_t) * step_t
            S_trim = S[:, :keep]
            S_rs = S_trim.reshape(S.shape[0], -1, step_t)
            S = S_rs.mean(axis=2)
            t_trim = t[:keep]
            t_rs = t_trim.reshape(-1, step_t)
            t = t_rs[:, step_t // 2]
        return S, f, t

    @staticmethod
    def _letterbox_qimage(img: QImage, target: QSize) -> QImage:
        tw, th = target.width(), target.height()
        iw, ih = img.width(), img.height()
        scale = min(tw / iw, th / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        scaled = img.scaled(nw, nh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        out = QImage(tw, th, QImage.Format.Format_RGBA8888)
        out.fill(Qt.black)
        from PySide6.QtGui import QPainter
        p = QPainter(out)
        x, y = (tw - nw) // 2, (th - nh) // 2
        p.drawImage(x, y, scaled)
        p.end()
        return out


# =============================
# Canvas pipeline (DSP worker; GUI draws in main thread)
# =============================
@dataclass
class CanvasJob:
    wav_path: Path
    n_fft: int
    hop: int


class CanvasSignals(QObject):
    done = Signal(str, object, object, object, float)  # path, S(np.ndarray), f(np.ndarray), t(np.ndarray), sr
    error = Signal(str, str)

class CanvasWorker(QThread):
    def __init__(self, parent: QObject, job: CanvasJob):
        super().__init__(parent)
        self.job = job
        self.signals = CanvasSignals()

    def run(self):
        try:
            path = self.job.wav_path
            with sf.SoundFile(str(path)) as f:
                sr = float(f.samplerate)
            y, _ = sf.read(str(path), always_2d=True)
            y = y.mean(axis=1).astype(np.float32, copy=False)
            n_fft = self.job.n_fft
            hop = self.job.hop
            win = get_window(WINDOW, n_fft, fftbins=True)
            noverlap = n_fft - hop
            f_hz, t_sec, Z = stft(y, fs=sr, window=win, nperseg=n_fft,
                                  noverlap=noverlap, boundary=None, padded=False)
            S = (np.abs(Z) ** 2).astype(np.float32)
            # Emit full-resolution STFT; view-aware downsampling happens in GUI
            self.signals.done.emit(str(path), S, f_hz, t_sec, sr)

        except Exception as e:
            msg = f"{e}\n{traceback.format_exc()}"

            self.signals.error.emit(str(self.job.wav_path), msg)

    @staticmethod
    def _decimate_for_canvas(S: np.ndarray, f: np.ndarray, t: np.ndarray):
        fr = S.shape[0]
        if fr > CANVAS_MAX_F_ROWS:
            step_f = math.ceil(fr / CANVAS_MAX_F_ROWS)
            S = S[::step_f, :]
            f = f[::step_f]
        tc = S.shape[1]
        if tc > CANVAS_MAX_T_COLS:
            step_t = math.ceil(tc / CANVAS_MAX_T_COLS)
            keep = (tc // step_t) * step_t
            S_trim = S[:, :keep]
            S_rs = S_trim.reshape(S.shape[0], -1, step_t)
            S = S_rs.mean(axis=2)
            t_trim = t[:keep]
            t_rs = t_trim.reshape(-1, step_t)
            t = t_rs[:, step_t // 2]
        return S, f, t


# =============================
# Widgets
# =============================
class ThumbListItem(QWidget):
    def __init__(self, wav_path: Path, pix: QPixmap, duration_sr: str):
        super().__init__()
        self.wav_path = wav_path
        self.img = QLabel(); self.img.setPixmap(pix)
        self.img.setFixedSize(QSize(THUMB_W, THUMB_H))
        self.img.setScaledContents(True)

        self.title = QLabel(f"<b>{wav_path.name}</b>")
        self.subtitle = QLabel(duration_sr)
        self.subtitle.setStyleSheet("color:#aaa;")

        self.btn_send = QPushButton("Move to Canvas")

        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)
        v.addWidget(self.img)
        v.addWidget(self.title)
        v.addWidget(self.subtitle)
        v.addWidget(self.btn_send)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bat Spectrogram Inspector – Version 1")
        self.resize(2200, 900)

        # LEFT: folder + thumbnail list
        self.btn_choose = QPushButton("Choose Folder…")
        self.edit_folder = QLineEdit(); self.edit_folder.setPlaceholderText("Select a folder of WAVs…")
        self.list = QListWidget(); self.list.setIconSize(QSize(THUMB_W, THUMB_H))
        self.list.setUniformItemSizes(True); self.list.setSpacing(8)

        left_col = QVBoxLayout()
        left_col.addWidget(self.btn_choose)
        left_col.addWidget(self.edit_folder)
        left_col.addWidget(self.list, 1)
        left_widget = QWidget(); left_widget.setLayout(left_col)

        # RIGHT: canvas + controls
        # Canvas holder (we'll create the figure lazily)
        self.figure = None
        self.ax = None
        self.canvas = None
        self.canvas_holder = QFrame(); self.canvas_holder.setStyleSheet("border:1px solid #555;")
        cv = QVBoxLayout(self.canvas_holder); cv.setContentsMargins(0,0,0,0)

        # placeholder state
        self._placeholder_label = None
        self._show_startup_placeholder()

        # Controls
        self.combo_cmap = QComboBox(); self.combo_cmap.addItems(COLORMAPS); self.combo_cmap.setCurrentText(THUMB_COLORMAP)
        self.spin_dyn = QDoubleSpinBox(); self._cfg_spin(self.spin_dyn, 20, 120, 5, CANVAS_DYN_RANGE_DB_DEFAULT, " dB")
        self.spin_vgain = QDoubleSpinBox(); self._cfg_spin(self.spin_vgain, -30, 30, 1, CANVAS_VGAIN_DB_DEFAULT, " dB gain")

        self.chk_title = QCheckBox("Show title"); self.chk_title.setChecked(True)
        self.edit_title = QLineEdit("Spectrogram")
        self.edit_xlabel = QLineEdit("Time (s)")
        self.edit_ylabel = QLineEdit("Frequency (kHz)")
        self.spin_title_fs = QSpinBox(); self.spin_title_fs.setRange(6, 48); self.spin_title_fs.setValue(14)
        self.spin_label_fs = QSpinBox(); self.spin_label_fs.setRange(6, 36); self.spin_label_fs.setValue(12)
        self.spin_tick_fs = QSpinBox(); self.spin_tick_fs.setRange(6, 36); self.spin_tick_fs.setValue(10)

        self.spin_tmin = QDoubleSpinBox(); self._cfg_spin(self.spin_tmin, 0, 1e9, 0.01, 0, " s")
        self.spin_tmax = QDoubleSpinBox(); self._cfg_spin(self.spin_tmax, 0, 1e9, 0.01, 0, " s")
        self.spin_fmin = QDoubleSpinBox();
        self._cfg_spin(self.spin_fmin, 0, 200, 0.1, THUMB_FMIN / 1000.0, " kHz")
        self.spin_fmax = QDoubleSpinBox();
        self._cfg_spin(self.spin_fmax, 0, 200, 0.1, THUMB_FMAX / 1000.0, " kHz")

        self.chk_zoommode = QCheckBox("Zoom mode (drag to zoom)")
        self.chk_zoommode.setChecked(True)


        self.btn_reset = QPushButton("Reset View")
        self.btn_fit = QPushButton("Fit to Data")

        # Lay out controls
        g1 = QGroupBox("Color & Scaling")
        grid1 = QGridLayout(g1)
        grid1.addWidget(QLabel("Colormap:"), 0, 0); grid1.addWidget(self.combo_cmap, 0, 1)
        grid1.addWidget(QLabel("Dyn range:"), 1, 0); grid1.addWidget(self.spin_dyn, 1, 1)
        grid1.addWidget(QLabel("Visual gain:"), 1, 2); grid1.addWidget(self.spin_vgain, 1, 3)

        self.spin_fft = QSpinBox();
        self.spin_fft.setRange(128, 8192);
        self.spin_fft.setValue(CANVAS_N_FFT_DEFAULT);
        self.spin_fft.setSingleStep(128)
        self.spin_hop = QSpinBox();
        self.spin_hop.setRange(32, 4096);
        self.spin_hop.setValue(CANVAS_HOP_DEFAULT);
        self.spin_hop.setSingleStep(32)
        grid1.addWidget(QLabel("FFT size:"), 2, 0);
        grid1.addWidget(self.spin_fft, 2, 1)
        grid1.addWidget(QLabel("Hop size:"), 2, 2);
        grid1.addWidget(self.spin_hop, 2, 3)

        g2 = QGroupBox("Labels")
        grid2 = QGridLayout(g2)
        grid2.addWidget(self.chk_title, 0, 0)
        grid2.addWidget(QLabel("Title:"), 1, 0); grid2.addWidget(self.edit_title, 1, 1, 1, 3)
        grid2.addWidget(QLabel("X label:"), 2, 0); grid2.addWidget(self.edit_xlabel, 2, 1, 1, 3)
        grid2.addWidget(QLabel("Y label:"), 3, 0); grid2.addWidget(self.edit_ylabel, 3, 1, 1, 3)
        grid2.addWidget(QLabel("Title fs:"), 4, 0); grid2.addWidget(self.spin_title_fs, 4, 1)
        grid2.addWidget(QLabel("Label fs:"), 4, 2); grid2.addWidget(self.spin_label_fs, 4, 3)
        grid2.addWidget(QLabel("Tick fs:"), 5, 0); grid2.addWidget(self.spin_tick_fs, 5, 1)

        g3 = QGroupBox("Ranges & Zoom")
        grid3 = QGridLayout(g3)
        grid3.addWidget(QLabel("tmin:"), 0, 0); grid3.addWidget(self.spin_tmin, 0, 1)
        grid3.addWidget(QLabel("tmax:"), 0, 2); grid3.addWidget(self.spin_tmax, 0, 3)
        grid3.addWidget(QLabel("fmin:"), 1, 0); grid3.addWidget(self.spin_fmin, 1, 1)
        grid3.addWidget(QLabel("fmax:"), 1, 2); grid3.addWidget(self.spin_fmax, 1, 3)
        # Let the checkbox span all 4 columns
        grid3.addWidget(self.chk_zoommode, 2, 0, 1, 4)
        # Move the buttons to the next row
        grid3.addWidget(self.btn_reset, 3, 2)
        grid3.addWidget(self.btn_fit, 3, 3)

        # RIGHT: canvas on the left, controls in a narrow right column
        right_widget = QWidget()
        right_layout = QHBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # canvas gets the majority of space
        right_layout.addWidget(self.canvas_holder, 1)

        # controls column (fixed width)
        controls_panel = QWidget()
        controls_col = QVBoxLayout(controls_panel)
        controls_col.setContentsMargins(0, 0, 0, 0)
        controls_col.setSpacing(8)

        controls_col.addWidget(g3)  # Ranges & Zoom
        controls_col.addWidget(g1)  # Color & Scaling
        controls_col.addWidget(g2)  # Labels

        # --- Filters ---
        g_filters = QGroupBox("Filters")
        gridF = QGridLayout(g_filters)

        self.chk_hp = QCheckBox("Enable high-pass mask")
        self.spin_hp_fc_khz = QDoubleSpinBox(); self._cfg_spin(self.spin_hp_fc_khz, 0.0, 200.0, 0.1, 16.0, " kHz")
        self.spin_hp_width_khz = QDoubleSpinBox(); self._cfg_spin(self.spin_hp_width_khz, 0.05, 50.0, 0.05, 2.5, " kHz")

        gridF.addWidget(self.chk_hp, 0, 0, 1, 2)
        gridF.addWidget(QLabel("Cutoff:"), 1, 0); gridF.addWidget(self.spin_hp_fc_khz, 1, 1)
        gridF.addWidget(QLabel("Width:"), 2, 0); gridF.addWidget(self.spin_hp_width_khz, 2, 1)

        # Smoothing (optional, separable Gaussian in power domain)
        self.chk_smooth = QCheckBox("Smooth")
        self.spin_smooth_t_frames = QDoubleSpinBox();
        self._cfg_spin(self.spin_smooth_t_frames, 0.0, 20.0, 0.5, 0.0, " frames")
        self.spin_smooth_f_khz = QDoubleSpinBox();
        self._cfg_spin(self.spin_smooth_f_khz, 0.0, 20.0, 0.1, 0.0, " kHz")

        # Put smoothing controls to the right of the HP controls
        gridF.addWidget(self.chk_smooth, 0, 2, 1, 2)
        gridF.addWidget(QLabel("Time σ:"), 1, 2);
        gridF.addWidget(self.spin_smooth_t_frames, 1, 3)
        gridF.addWidget(QLabel("Freq σ:"), 2, 2);
        gridF.addWidget(self.spin_smooth_f_khz, 2, 3)

        # Insert the Filters group above Export & Presets
        controls_col.addWidget(g_filters)


        # --- Export & Presets group ---
        g4 = QGroupBox("Export & Presets")
        grid4 = QGridLayout(g4)
        # Figure export opts
        self.spin_dpi = QSpinBox();
        self.spin_dpi.setRange(72, 1200);
        self.spin_dpi.setValue(300)
        self.spin_w_in = QDoubleSpinBox();
        self._cfg_spin(self.spin_w_in, 2.0, 20.0, 0.1, 8.0, " in")
        self.spin_h_in = QDoubleSpinBox();
        self._cfg_spin(self.spin_h_in, 2.0, 20.0, 0.1, 4.5, " in")
        self.chk_transparent = QCheckBox("Transparent background")

        self.btn_export_fig = QPushButton("Export Figure…")
        self.btn_export_wav = QPushButton("Export WAV…")
        self.btn_preset_save = QPushButton("Save Preset…")
        self.btn_preset_load = QPushButton("Load Preset…")
        self.btn_export_video = QPushButton("Export Video (MOV)…")

        grid4.addWidget(QLabel("DPI:"), 0, 0);
        grid4.addWidget(self.spin_dpi, 0, 1)
        grid4.addWidget(QLabel("Size:"), 0, 2);
        grid4.addWidget(self.spin_w_in, 0, 3)
        grid4.addWidget(QLabel("×"), 0, 4);
        grid4.addWidget(self.spin_h_in, 0, 5)
        grid4.addWidget(self.chk_transparent, 1, 0, 1, 6)
        grid4.addWidget(self.btn_export_fig, 2, 0, 1, 3)
        grid4.addWidget(self.btn_export_wav, 2, 3, 1, 3)
        grid4.addWidget(self.btn_preset_save, 3, 0, 1, 3)
        grid4.addWidget(self.btn_preset_load, 3, 3, 1, 3)
        grid4.addWidget(self.btn_export_video, 4, 0, 1, 6)

        controls_col.addWidget(g4)
        controls_col.addStretch(1)

        controls_panel.setFixedWidth(360)
        right_layout.addWidget(controls_panel, 0)

        # Splitter with left thumbnails and right (canvas+controls)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setSizes([420, 1720])  # give more room to the canvas side
        root = QHBoxLayout(self)
        root.addWidget(self.splitter)

        # State
        self.current_workers: Dict[str, ThumbWorker] = {}
        self.canvas_worker: Optional[CanvasWorker] = None
        self._t_full = None  # type: Optional[np.ndarray]
        self._f_full = None  # type: Optional[np.ndarray]
        self._S_power = None  # type: Optional[np.ndarray]
        self._selector = None  # RectangleSelector
        self._current_wav_path: Optional[Path] = None

        # Hooks
        self.btn_choose.clicked.connect(self._choose_folder)
        self.edit_folder.returnPressed.connect(self._refresh_folder)

        # Canvas control hooks
        self.combo_cmap.currentTextChanged.connect(self._render_canvas)
        self.spin_dyn.valueChanged.connect(self._render_canvas)
        self.spin_vgain.valueChanged.connect(self._render_canvas)
        self.spin_fft.valueChanged.connect(self._recompute_canvas)
        self.spin_hop.valueChanged.connect(self._recompute_canvas)
        self.chk_title.toggled.connect(self._render_canvas)
        self.edit_title.editingFinished.connect(self._render_canvas)
        self.edit_xlabel.editingFinished.connect(self._render_canvas)
        self.edit_ylabel.editingFinished.connect(self._render_canvas)
        self.spin_title_fs.valueChanged.connect(self._render_canvas)
        self.spin_label_fs.valueChanged.connect(self._render_canvas)
        self.spin_tick_fs.valueChanged.connect(self._render_canvas)

        self.spin_tmin.valueChanged.connect(self._render_canvas)
        self.spin_tmax.valueChanged.connect(self._render_canvas)
        self.spin_fmin.valueChanged.connect(self._render_canvas)
        self.spin_fmax.valueChanged.connect(self._render_canvas)

        self.chk_zoommode.toggled.connect(self._toggle_selector)
        self.btn_reset.clicked.connect(self._reset_view)
        self.btn_fit.clicked.connect(self._fit_to_data)

        # Filters → live update
        self.chk_hp.toggled.connect(self._render_canvas)
        self.spin_hp_fc_khz.valueChanged.connect(self._render_canvas)
        self.spin_hp_width_khz.valueChanged.connect(self._render_canvas)
        self.chk_smooth.toggled.connect(self._render_canvas)
        self.spin_smooth_t_frames.valueChanged.connect(self._render_canvas)
        self.spin_smooth_f_khz.valueChanged.connect(self._render_canvas)
        self.spin_smooth_t_frames.setToolTip("Gaussian σ along time (frames). 0 = off.")
        self.spin_smooth_f_khz.setToolTip("Gaussian σ along frequency (kHz). 0 = off.")

        # Export & presets
        self.btn_export_fig.clicked.connect(self._export_figure)
        self.btn_export_wav.clicked.connect(self._export_wav)
        self.btn_preset_save.clicked.connect(self._save_preset)
        self.btn_preset_load.clicked.connect(self._load_preset)
        self.btn_export_video.clicked.connect(self._export_video_mapped)
        QShortcut(QKeySequence("V"), self, activated=self._export_video_mapped)

        # Shortcuts
        QShortcut(QKeySequence("Z"), self, activated=lambda: self.chk_zoommode.toggle())
        QShortcut(QKeySequence("R"), self, activated=self._reset_view)
        QShortcut(QKeySequence("F"), self, activated=self._fit_to_data)
        QShortcut(QKeySequence("E"), self, activated=self._export_figure)
        QShortcut(QKeySequence("W"), self, activated=self._export_wav)
        QShortcut(QKeySequence("S"), self, activated=self._save_preset)
        QShortcut(QKeySequence("L"), self, activated=self._load_preset)

        # Restore session (geometry, splitter, last folder, export defaults)
        self._restore_settings()

    # --- Helpers ---
    @staticmethod
    def _cfg_spin(spin: QDoubleSpinBox, lo, hi, step, val, suffix=""):
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(val)
        if suffix:
            spin.setSuffix(suffix)

    def _show_startup_placeholder(self):
        """Show a one-time image in the canvas area until the first spectrogram is loaded."""
        try:
            from PySide6.QtGui import QPixmap
            from PySide6.QtCore import Qt
            img_path = Path(__file__).parent / "BatSherlockHolms.png"
            if img_path.exists() and self._placeholder_label is None:
                self._placeholder_label = QLabel()
                self._placeholder_label.setAlignment(Qt.AlignCenter)
                self._placeholder_label.setStyleSheet("background: #111;")
                pm = QPixmap(str(img_path))
                if not pm.isNull():
                    self._placeholder_label.setPixmap(pm)
                    self._placeholder_label.setScaledContents(False)  # keep aspect ratio
                    self.canvas_holder.layout().addWidget(self._placeholder_label)
        except Exception:
            # Fail silently if image can't be shown
            pass

    def _remove_startup_placeholder(self):
        if getattr(self, "_placeholder_label", None) is not None:
            lay = self.canvas_holder.layout()
            lay.removeWidget(self._placeholder_label)
            try:
                self._placeholder_label.deleteLater()
            except Exception:
                pass
            self._placeholder_label = None

    # =============================
    # Folder + thumbnails
    # =============================
    def _choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select folder of WAVs")
        if d:
            self.edit_folder.setText(d)
            self._refresh_folder()
            QSettings("ConcordiaChem", "BatSpectrogramInspector").setValue("paths/last_folder", d)

    def _refresh_folder(self):
        # Stop previous workers BEFORE clearing state/UI
        for w in list(self.current_workers.values()):
            try:
                w.cancel()
                w.wait()
            except Exception:
                pass
        self.current_workers.clear()

        self.list.clear()
        folder = Path(self.edit_folder.text().strip())
        if not folder.is_dir():
            return
        wavs = sorted([p for p in folder.iterdir() if p.suffix.lower() in (".wav", ".wave")])
        if not wavs:
            QMessageBox.information(self, "No WAVs", "No .wav/.wave files in this folder.")
            return

        for wav in wavs:
            item = QListWidgetItem(); item.setSizeHint(QSize(THUMB_W + 16, THUMB_H + 90))
            placeholder = self._placeholder_pixmap()
            widget = ThumbListItem(wav, placeholder, "rendering…")
            widget.btn_send.clicked.connect(lambda _, p=wav: self._move_to_canvas(p))
            self.list.addItem(item)
            self.list.setItemWidget(item, widget)

            job = ThumbJob(wav)
            worker = ThumbWorker(self, job)
            worker.signals.done.connect(self._on_thumb_done)
            worker.signals.error.connect(self._on_thumb_error)
            worker.finished.connect(lambda p=str(wav), wkr=worker: self._on_thumb_finished(p, wkr))
            self.current_workers[str(wav)] = worker
            worker.start()

    def _on_thumb_done(self, wav_path_str: str, pix: QPixmap, dur: float, sr: float):
        # Update UI only; cleanup happens on finished
        for i in range(self.list.count()):
            it = self.list.item(i)
            widget = self.list.itemWidget(it)
            if widget and str(widget.wav_path) == wav_path_str:
                widget.img.setPixmap(pix)
                widget.subtitle.setText(f"{dur:.2f} s @ {sr:.0f} Hz")
                break

    def _on_thumb_error(self, wav_path_str: str, msg: str):
        for i in range(self.list.count()):
            it = self.list.item(i)
            widget = self.list.itemWidget(it)
            if widget and str(widget.wav_path) == wav_path_str:
                widget.subtitle.setText("[error]")
                break
        # (keep worker until finished signal)

    def _on_thumb_finished(self, wav_path_str: str, worker: ThumbWorker):
        try:
            worker.deleteLater()
        except Exception:
            pass
        self.current_workers.pop(wav_path_str, None)

    @staticmethod
    def _placeholder_pixmap() -> QPixmap:
        img = QImage(THUMB_W, THUMB_H, QImage.Format.Format_RGB32)
        img.fill(0x202020)
        return QPixmap.fromImage(img)

    # =============================
    # Canvas (controls + zoom)
    # =============================
    def _ensure_canvas(self):
        if self.figure is not None:
            return
        import matplotlib
        matplotlib.use("QtAgg", force=True)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        import matplotlib.pyplot as mplt
        from matplotlib.widgets import RectangleSelector

        # remove the startup image forever
        self._remove_startup_placeholder()

        self._RectangleSelector = RectangleSelector
        self.figure = mplt.figure(figsize=(7.0, 4.8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        lay = self.canvas_holder.layout()
        lay.addWidget(self.canvas)
        self.canvas.mpl_connect('resize_event', lambda evt: self._render_canvas())

    def _move_to_canvas(self, wav_path: Path):
        self._ensure_canvas()
        self._current_wav_path = Path(wav_path)
        # Stop previous canvas worker if any
        if self.canvas_worker is not None:
            try:
                self.canvas_worker.signals.done.disconnect()
                self.canvas_worker.signals.error.disconnect()
            except Exception:
                pass
            self.canvas_worker.quit(); self.canvas_worker.wait()
        job = CanvasJob(wav_path,
                        n_fft=int(self.spin_fft.value()),
                        hop=int(self.spin_hop.value()))

        self.canvas_worker = CanvasWorker(self, job)
        self.canvas_worker.signals.done.connect(self._on_canvas_ready)
        self.canvas_worker.signals.error.connect(lambda p, m: QMessageBox.warning(self, "Canvas error", m))
        self.canvas_worker.start()

    def _recompute_canvas(self):
        if self._S_power is None or self.canvas_worker is None:
            return
        # Rerun with same file but new FFT/Hop
        wav_path = Path(self.canvas_worker.job.wav_path)
        self._move_to_canvas(wav_path)

    def _on_canvas_ready(self, path_str: str, S: np.ndarray, f: np.ndarray, t: np.ndarray, sr: float):
        # Cache full data extents
        self._S_power, self._f_full, self._t_full = S, f, t
        # Initialize ranges to full extents
        self.spin_tmin.blockSignals(True); self.spin_tmax.blockSignals(True)
        self.spin_fmin.blockSignals(True); self.spin_fmax.blockSignals(True)
        self.spin_tmin.setRange(float(t[0]), float(t[-1]))
        self.spin_tmax.setRange(float(t[0]), float(t[-1]))
        self.spin_tmin.setValue(float(t[0]))
        self.spin_tmax.setValue(float(t[-1]))
        self.spin_fmin.setRange(float(f[0] / 1000.0), float(f[-1] / 1000.0))
        self.spin_fmax.setRange(float(f[0] / 1000.0), float(f[-1] / 1000.0))
        self.spin_fmin.setValue(float(f[0] / 1000.0))
        self.spin_fmax.setValue(float(f[-1] / 1000.0))
        self.spin_tmin.blockSignals(False); self.spin_tmax.blockSignals(False)
        self.spin_fmin.blockSignals(False); self.spin_fmax.blockSignals(False)
        # Render
        self._render_canvas()
        # Ensure selector state according to checkbox
        self._toggle_selector(self.chk_zoommode.isChecked())

    def _db_scale(self, S: np.ndarray, f: np.ndarray) -> np.ndarray:
        nb_lo, nb_hi = NOISE_BAND
        nb_mask = (f >= nb_lo) & (f <= nb_hi)
        if np.any(nb_mask):
            ref = float(np.median(S[nb_mask, :]))
            if not (np.isfinite(ref) and ref > 0):
                pos = S[S > 0]; ref = float(np.median(pos) if pos.size else 1e-12)
        else:
            pos = S[S > 0]; ref = float(np.median(pos) if pos.size else 1e-12)
        eps = 1e-12
        S_db = 10.0 * np.log10((S + eps) / (ref + eps))
        vmax = np.percentile(S_db, 99.5) + float(self.spin_vgain.value())
        vmin = vmax - float(self.spin_dyn.value())
        return np.clip(S_db, vmin, vmax)

    def _canvas_pixel_budget(self) -> Tuple[int, int]:
        """Approximate pixel budget for the current canvas area (cols, rows)."""
        if self.canvas is None:
            return (1600, 900)  # sane default
        try:
            dpr = float(self.canvas.devicePixelRatioF())
        except Exception:
            dpr = 1.0
        w = max(1, int(self.canvas.width() * dpr))
        h = max(1, int(self.canvas.height() * dpr))
        return (w, h)

    def _view_crop(self,
                   S: np.ndarray, f: np.ndarray, t: np.ndarray,
                   tmin: float, tmax: float, fmin: float, fmax: float
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crop S,f,t to the current view ranges (no resampling)."""
        if tmin > tmax: tmin, tmax = tmax, tmin
        if fmin > fmax: fmin, fmax = fmax, fmin
        tmask = (t >= tmin) & (t <= tmax)
        fmask = (f >= fmin) & (f <= fmax)
        if not np.any(tmask):
            tmask[:] = True
        if not np.any(fmask):
            fmask[:] = True
        S_slice = S[fmask, :][:, tmask]
        f_slice = f[fmask]
        t_slice = t[tmask]
        # Ensure at least 2 points in each axis to avoid extent/imshow issues
        if t_slice.size < 2 and t.size >= 2:
            idx = np.clip(np.searchsorted(t, (tmin + tmax) / 2), 1, t.size - 1)
            t_slice = t[idx-1:idx+1]
            S_slice = S[fmask, :][:, idx-1:idx+1]
        if f_slice.size < 2 and f.size >= 2:
            idx = np.clip(np.searchsorted(f, (fmin + fmax) / 2), 1, f.size - 1)
            f_slice = f[idx-1:idx+1]
            S_slice = S[idx-1:idx+1, :][:, tmask]
        return S_slice, f_slice, t_slice

    def _display_downsample(self,
                            S: np.ndarray, f: np.ndarray, t: np.ndarray,
                            max_rows: int, max_cols: int
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Downsample only if needed to match display pixel budget.
        Mean-pool along time and frequency to avoid aliasing.
        """
        # --- Time (columns) ---
        cols = S.shape[1]
        if cols > max_cols:
            step_t = int(math.ceil(cols / max_cols))
            keep = (cols // step_t) * step_t
            S_trim = S[:, :keep]
            S_rs = S_trim.reshape(S.shape[0], -1, step_t)
            S = S_rs.max(axis=2)
            t_trim = t[:keep]
            t_rs = t_trim.reshape(-1, step_t)
            t = t_rs.mean(axis=1)  # center-of-bin is fine too

        # --- Frequency (rows) ---
        rows = S.shape[0]
        if rows > max_rows:
            step_f = int(math.ceil(rows / max_rows))
            keep = (rows // step_f) * step_f
            S_trim = S[:keep, :]
            S_rs = S_trim.reshape(-1, step_f, S.shape[1])
            S = S_rs.mean(axis=1)
            f_trim = f[:keep]
            f = f_trim.reshape(-1, step_f).mean(axis=1)

        return S, f, t


    def _render_canvas(self):
        if self.canvas is None or self._S_power is None:
            return

        # 1) Read current view ranges from the spin boxes
        tmin, tmax = float(self.spin_tmin.value()), float(self.spin_tmax.value())
        fmin, fmax = float(self.spin_fmin.value()) * 1000.0, float(self.spin_fmax.value()) * 1000.0
        if tmin > tmax: tmin, tmax = tmax, tmin
        if fmin > fmax: fmin, fmax = fmax, fmin

        # 2) Crop to view (no resampling) -- NOTE: fmin/fmax are in Hz here
        S_view, f_view_hz, t_view = self._view_crop(self._S_power, self._f_full, self._t_full,
                                                    tmin, tmax, fmin, fmax)

        # 3) Downsample only if needed to match canvas pixels
        px_x, px_y = self._canvas_pixel_budget()
        S_ds, f_ds_hz, t_ds = self._display_downsample(S_view, f_view_hz, t_view,
                                                       max_rows=max(1, int(px_y * 0.9)),
                                                       max_cols=max(1, int(px_x * 0.9)))

        # --- Optional global high-pass mask (power domain) ---
        if self.chk_hp.isChecked():
            fc_hz = float(self.spin_hp_fc_khz.value()) * 1000.0
            w_hz  = float(self.spin_hp_width_khz.value()) * 1000.0
            m = self._hp_mask(f_ds_hz, fc_hz, w_hz)         # shape: (F,)
            S_ds = S_ds * m[:, None]                        # power × mask

        # --- Optional separable Gaussian smoothing (power domain) ---
        if self.chk_smooth.isChecked():
            # Convert UI widths to bin units on the downsampled grid
            # Time axis: columns → frames; user inputs σ in "frames" directly
            sigma_t = float(self.spin_smooth_t_frames.value())
            # Frequency axis: user inputs σ in kHz → convert to bins using df
            if f_ds_hz.size >= 2:
                df = float(np.median(np.diff(f_ds_hz)))
                sigma_f_bins = (float(self.spin_smooth_f_khz.value()) * 1000.0) / max(df, 1e-9)
            else:
                sigma_f_bins = 0.0
            if sigma_t > 0.0 or sigma_f_bins > 0.0:
                # gaussian_filter expects (rows, cols) = (freq, time)
                S_ds = gaussian_filter(S_ds, sigma=(max(0.0, sigma_f_bins), max(0.0, sigma_t)))

        # 4) Convert to dB with current scaling controls (expects Hz)
        S_db = self._db_scale(S_ds, f_ds_hz)

        # 5) Draw (plot in kHz)
        self.ax.clear()
        f_ds_khz = f_ds_hz / 1000.0
        extent = [float(t_ds[0]), float(t_ds[-1]), float(f_ds_khz[0]), float(f_ds_khz[-1])]
        self.ax.imshow(
            S_db,
            origin='lower',
            aspect='auto',
            cmap=self.combo_cmap.currentText(),
            extent=extent,
            vmin=S_db.min(),
            vmax=S_db.max(),
            interpolation='nearest'
        )

        # Labels & fonts
        if self.chk_title.isChecked() and self.edit_title.text().strip():
            self.ax.set_title(self.edit_title.text().strip(), fontsize=self.spin_title_fs.value())
        self.ax.set_xlabel(self.edit_xlabel.text().strip(), fontsize=self.spin_label_fs.value())
        self.ax.set_ylabel(self.edit_ylabel.text().strip(), fontsize=self.spin_label_fs.value())
        self.ax.tick_params(labelsize=self.spin_tick_fs.value())

        # 6) Axes limits also in kHz
        self.ax.set_xlim(tmin, tmax)
        self.ax.set_ylim(fmin / 1000.0, fmax / 1000.0)

        self.canvas.draw_idle()

    def _hp_mask(self, f_hz: np.ndarray, fc_hz: float, width_hz: float) -> np.ndarray:
        """
        Smooth, time-invariant high-pass mask along frequency axis.
        tanh step from ~0 (below fc) to ~1 (above fc).
        """
        width_hz = max(1.0, float(width_hz))  # avoid divide-by-zero
        x = (f_hz.astype(np.float64) - float(fc_hz)) / width_hz
        m = 0.5 * (1.0 + np.tanh(x))
        return m.astype(np.float32)


    def _apply_ranges_from_spins(self):
        if self.canvas is None or self._S_power is None:
            return
        xmin, xmax = float(self.spin_tmin.value()), float(self.spin_tmax.value())
        if xmin > xmax: xmin, xmax = xmax, xmin
        ymin, ymax = float(self.spin_fmin.value()), float(self.spin_fmax.value())
        if ymin > ymax: ymin, ymax = ymax, ymin
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.canvas.draw_idle()

    def _toggle_selector(self, on: bool):
        if self.canvas is None:
            return
        from matplotlib.widgets import RectangleSelector
        if on:
            if self._selector is None:
                def on_select(eclick, erelease):
                    x0, y0 = eclick.xdata, eclick.ydata
                    x1, y1 = erelease.xdata, erelease.ydata
                    if x0 is None or x1 is None or y0 is None or y1 is None:
                        return
                    xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
                    ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
                    # Update spin boxes (which will update axes as well)
                    self.spin_tmin.blockSignals(True);
                    self.spin_tmax.blockSignals(True)
                    self.spin_fmin.blockSignals(True);
                    self.spin_fmax.blockSignals(True)
                    self.spin_tmin.setValue(float(xmin));
                    self.spin_tmax.setValue(float(xmax))
                    self.spin_fmin.setValue(float(ymin));
                    self.spin_fmax.setValue(float(ymax))
                    self.spin_tmin.blockSignals(False);
                    self.spin_tmax.blockSignals(False)
                    self.spin_fmin.blockSignals(False);
                    self.spin_fmax.blockSignals(False)
                    self._render_canvas()

                # High-contrast outline while dragging; blitting keeps it snappy
                self._selector = RectangleSelector(
                    self.ax, on_select,
                    useblit=True,  # draw the outline during drag
                    button=[1],
                    interactive=False,
                    spancoords='data',
                    drag_from_anywhere=False,
                    minspanx=0.001,  # seconds
                    minspany=2.0,  # kHz (since your y-axis is now kHz)
                    props=dict(facecolor='none', edgecolor='lime', linewidth=1.8, alpha=0.9)
                )
            self._selector.set_active(True)
        else:
            if self._selector is not None:
                self._selector.set_active(False)

    def _reset_view(self):
        if self._t_full is None or self._f_full is None:
            return
        self._set_view_full()

    def _fit_to_data(self):
        # Same as reset for now; placeholder if we later add margins
        self._reset_view()

    def _export_figure(self):
        if self.canvas is None or self._S_power is None:
            QMessageBox.information(self, "Export Figure", "Nothing to export yet.")
            return
        # Ask for filename
        path, selected = QFileDialog.getSaveFileName(
            self, "Export Figure",
            str(Path(self.edit_folder.text().strip() or ".") / "spectrogram.png"),
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )
        if not path:
            return
        # Temporarily set figure size / dpi (restore after)
        old_size = self.figure.get_size_inches()
        old_dpi = self.figure.dpi
        try:
            self.figure.set_size_inches(float(self.spin_w_in.value()), float(self.spin_h_in.value()))
            self.figure.set_dpi(int(self.spin_dpi.value()))
            # Re-render at export DPI to get crisp text layout
            self._render_canvas()
            self.figure.savefig(
                path,
                dpi=int(self.spin_dpi.value()),
                bbox_inches="tight",
                transparent=self.chk_transparent.isChecked()
            )
            # Persist last export opts
            self._save_settings(partial_only=True)
        except Exception as e:
            QMessageBox.warning(self, "Export Figure", f"Failed to export:\n{e}")
        finally:
            # Restore size/dpi and re-render for UI
            self.figure.set_size_inches(*old_size)
            self.figure.set_dpi(old_dpi)
            self._render_canvas()

    def _export_wav(self):
        if not self._current_wav_path:
            QMessageBox.information(self, "Export WAV", "No audio loaded.")
            return
        tmin = float(self.spin_tmin.value())
        tmax = float(self.spin_tmax.value())
        if tmax <= tmin:
            QMessageBox.information(self, "Export WAV", "Invalid time window.")
            return
        # Ask for filename
        def_suggest = Path(self._current_wav_path).with_suffix("")
        def_suggest = def_suggest.parent / (def_suggest.name + f"_{tmin:.2f}-{tmax:.2f}s.wav")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export WAV", str(def_suggest), "WAV (*.wav)"
        )
        if not path:
            return
        try:
            with sf.SoundFile(str(self._current_wav_path)) as f:
                sr = int(f.samplerate)
            # Read only needed segment (mono)
            with sf.SoundFile(str(self._current_wav_path)) as f:
                start = int(tmin * f.samplerate)
                stop = int(tmax * f.samplerate)
                start = max(0, min(start, len(f)))
                stop = max(start + 1, min(stop, len(f)))
                f.seek(start)
                seg = f.read(stop - start, dtype="float32", always_2d=True)
            y = seg.mean(axis=1)  # mono
            sf.write(path, y, sr)
        except Exception as e:
            QMessageBox.warning(self, "Export WAV", f"Failed to export:\n{e}")

    def _transform_via_stft_map(self, y: np.ndarray, sr: float,
                                m: float = 0.4, b_hz: float = -6000.0) -> np.ndarray:
        """
        STFT-domain linear frequency remap:
            f' = m * f + b_hz
        Implemented by: Z(f', t) = Z_src(f=(f'-b)/m, t) with frequency-axis interpolation,
        then iSTFT back to time.
        """
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32, copy=False)
        if y.size < 16:
            return y

        # ~20 ms Hann, 50% overlap (robust & not too smeary)
        nper = max(64, int(0.02 * sr))
        nover = nper // 2
        win = get_window("hann", nper, fftbins=True)

        # One-sided STFT
        f, t, Z = stft(y, fs=sr, window=win, nperseg=nper, noverlap=nover,
                       boundary="zeros", padded=True, return_onesided=True)

        # Target grid keeps same bins f' == f; compute source freq for each f'
        df = float(f[1] - f[0]) if len(f) > 1 else (sr / 2.0)
        fprime = f.astype(np.float64)
        f_src = (fprime - float(b_hz)) / float(m)

        # Map source frequency to fractional source index on the original grid
        x_src = np.arange(len(f), dtype=np.float64)
        x_of_f = (f_src - float(f[0])) / df

        # Interpolate complex STFT along frequency axis for all time frames
        from scipy.interpolate import interp1d
        interp = interp1d(
            x_src,
            Z,  # complex array; interp1d handles complex dtype
            kind="linear",
            axis=0,
            bounds_error=False,
            fill_value=0.0,
            assume_sorted=True,
        )
        Zm = interp(x_of_f)  # shape (n_freqs, n_frames)

        # --- Optional global high-pass mask on the output STFT (amplitude domain) ---
        if getattr(self, "chk_hp", None) and self.chk_hp.isChecked():
            fc_hz = float(self.spin_hp_fc_khz.value()) * 1000.0
            w_hz  = float(self.spin_hp_width_khz.value()) * 1000.0
            # fprime is the output frequency grid you computed above
            m = self._hp_mask(fprime, fc_hz, w_hz).astype(np.float64, copy=False)  # (F,)
            Zm *= np.sqrt(m)[:, None]   # amplitude × sqrt(mask) → power × mask


        # Reconstruct time-domain (keep same length as input)
        _, y_rec = istft(Zm, fs=sr, window=win, nperseg=nper, noverlap=nover, input_onesided=True)
        y_rec = y_rec[:y.size]

        # Gentle normalize (avoid clipping)
        peak = float(np.max(np.abs(y_rec))) if y_rec.size else 1.0
        if peak > 1e-9:
            y_rec = 0.95 * y_rec / peak
        return y_rec.astype(np.float32)

    def _clim_for_transformed(self, x_t: np.ndarray, sr: int,
                              dyn_range_db: float, vmax_pct: float = 99.5) -> tuple[float, float]:
        # Quick STFT-based PSD estimate for robust vmin/vmax
        from scipy.signal import get_window, stft
        nper = max(64, int(0.02 * sr))  # ~20 ms
        win = get_window("hann", nper, fftbins=True)
        _, _, Z = stft(x_t + 1e-20, fs=sr, window=win, nperseg=nper, noverlap=nper // 2,
                       boundary="zeros", padded=True)
        P = (np.abs(Z) ** 2).astype(np.float64)
        P_db = 10.0 * np.log10(np.maximum(P, 1e-30))
        vmax = float(np.percentile(P_db, vmax_pct))
        vmin = vmax - float(dyn_range_db)
        return vmin, vmax

    def _render_base_png(self, x_t: np.ndarray, sr: int, vmin: float, vmax: float,
                         out_dir: Path) -> tuple[Path, tuple[float, float, float, float]]:
        """
        Draw a spectrogram styled like the canvas (cmap/labels/fonts) and save a base PNG.
        Returns:
          (png_path, (x0_px, x1_px, y_top_px, y_bottom_px)) for tracer placement in pixels.
        """
        import matplotlib.pyplot as mplt
        from matplotlib import mlab
        from matplotlib.ticker import FuncFormatter

        fig = mplt.figure(figsize=(12.8, 7.2), dpi=100)
        ax = fig.add_subplot(111)

        # --- Make spectrogram arrays (not via ax.specgram) ---
        # Use the same ~20 ms / 50% overlap as elsewhere so vmin/vmax match
        nper = max(64, int(0.02 * sr))
        nover = nper // 2
        Pxx, freqs, bins = mlab.specgram(
            x_t + 1e-20, NFFT=nper, Fs=sr, noverlap=nover
        )  # Pxx: (F, T), freqs in Hz, bins in s

        # Optional global high-pass (power-domain) before display
        if getattr(self, "chk_hp", None) and self.chk_hp.isChecked():
            fc_hz = float(self.spin_hp_fc_khz.value()) * 1000.0
            w_hz = float(self.spin_hp_width_khz.value()) * 1000.0
            m = self._hp_mask(freqs, fc_hz, w_hz).astype(Pxx.dtype, copy=False)  # (F,)
            Pxx = Pxx * m[:, None]

        # Convert to dB and draw with imshow so we control scaling explicitly
        Pxx_db = 10.0 * np.log10(np.maximum(Pxx, 1e-30))
        extent = [float(bins[0]), float(bins[-1]), float(freqs[0]), float(freqs[-1])]

        ax.imshow(
            Pxx_db,
            origin="lower",
            aspect="auto",
            extent=extent,  # <- true Hz grid of the *mapped* audio
            cmap=self.combo_cmap.currentText(),
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        # Y axis: show in kHz; cap the visible top near the expected mapped ceiling
        # For raw Nyquist = sr/2 (Hz), mapped ceiling is ~0.4*Nyquist(kHz) - 6 kHz.
        nyq_khz = (sr / 2.0) / 1000.0
        mapped_top_hz = max(1000.0, (0.4 * nyq_khz - 6.0) * 1000.0)  # ≥ 1 kHz
        # Respect the UI fmax (kHz) if set lower
        ui_top_hz = float(self.spin_fmax.value()) * 1000.0
        y_top_hz = min(ui_top_hz, mapped_top_hz)
        ax.set_ylim(0.0, y_top_hz)

        # Format ticks as kHz but keep axis units in Hz
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / 1000.0:.0f}"))
        ax.set_ylabel(self.edit_ylabel.text().strip() or "Frequency (kHz)",
                      fontsize=self.spin_label_fs.value())
        ax.set_xlabel(self.edit_xlabel.text().strip() or "Time (s)",
                      fontsize=self.spin_label_fs.value())
        if self.chk_title.isChecked() and self.edit_title.text().strip():
            ax.set_title(self.edit_title.text().strip(), fontsize=self.spin_title_fs.value())
        ax.tick_params(labelsize=self.spin_tick_fs.value())

        fig.tight_layout()
        fig.canvas.draw()

        # Axes bbox (figure fraction → PNG pixels) for tracer placement
        pos = ax.get_position()
        png_path = out_dir / "base.png"
        fig.savefig(png_path, dpi=100)
        mplt.close(fig)

        from PIL import Image
        base_img = Image.open(png_path)
        w, h = base_img.size
        x0_px = pos.x0 * w
        x1_px = pos.x1 * w
        y_top_px = (1.0 - pos.y1) * h
        y_bottom_px = (1.0 - pos.y0) * h
        return png_path, (x0_px, x1_px, y_top_px, y_bottom_px)

    def _export_video_mapped(self):
        if not self._current_wav_path:
            QMessageBox.information(self, "Export Video", "No audio loaded.")
            return

        # Time window in seconds (UI)
        tmin = float(self.spin_tmin.value())
        tmax = float(self.spin_tmax.value())
        if tmax <= tmin:
            QMessageBox.information(self, "Export Video", "Invalid time window.")
            return

        # Ask for output MOV path
        suggest = Path(self._current_wav_path).with_suffix("")
        suggest = suggest.parent / (suggest.name + f"_{tmin:.2f}-{tmax:.2f}s.mov")
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export Video (MOV)", str(suggest), "QuickTime MOV (*.mov)"
        )
        if not out_path:
            return
        out_path = Path(out_path)

        try:
            # Read selection (mono)
            with sf.SoundFile(str(self._current_wav_path)) as f:
                sr = int(f.samplerate)
                start = int(round(tmin * sr))
                stop = int(round(tmax * sr))
                start = max(0, min(start, len(f)))
                stop = max(start + 1, min(stop, len(f)))
                f.seek(start)
                seg = f.read(stop - start, dtype="float32", always_2d=True)
            y = seg.mean(axis=1).astype(np.float32, copy=False)
            if y.size < 16:
                QMessageBox.information(self, "Export Video", "Selection is too short.")
                return

            # Transform per f' = 0.4 f - 6000
            y_t = self._transform_via_stft_map(y, sr, m=0.4, b_hz=-6000.0)
            dur_t = float(len(y_t)) / float(sr)

            # Global color limits from transformed signal & current dyn range
            vmin, vmax = self._clim_for_transformed(
                y_t, sr, dyn_range_db=float(self.spin_dyn.value()), vmax_pct=99.5
            )

            # Make a temp dir for intermediates
            with tempfile.TemporaryDirectory() as tdir:
                tdir = Path(tdir)

                # 1) Base spectrogram PNG styled like canvas
                base_png, (x0_px, x1_px, y_top_px, y_bottom_px) = self._render_base_png(
                    y_t, sr, vmin, vmax, tdir
                )
                base_img = Image.open(base_png).convert("RGBA")
                W, H = base_img.size

                # 2) Draw tracer per frame → silent video
                fps = 30  # fixed for now (simple UX)
                nframes = max(1, int(round(dur_t * fps)))
                silent_video = tdir / "silent.mov"

                # use libx264 and yuv420p for compatibility; imageio will call ffmpeg
                writer = imageio.get_writer(
                    silent_video, fps=fps, codec="libx264", macro_block_size=None, quality=8
                )
                try:
                    for i in range(nframes):
                        prog = i / (nframes - 1) if nframes > 1 else 1.0
                        xline = x0_px + (x1_px - x0_px) * prog
                        frame = base_img.copy()
                        draw = ImageDraw.Draw(frame)
                        draw.line([(xline, y_top_px), (xline, y_bottom_px)],
                                  fill=(255, 255, 255, 255), width=5)
                        writer.append_data(np.array(frame.convert("RGB")))
                finally:
                    writer.close()

                # 3) Save transformed audio to WAV
                aud_wav = tdir / "audio.wav"
                sf.write(aud_wav, y_t, sr, subtype="PCM_16")

                # 4) Mux into MOV with ffmpeg (copy video, AAC audio, faststart)
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(silent_video),
                    "-i", str(aud_wav),
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    "-shortest",
                    str(out_path)
                ]
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    # Helpful message if ffmpeg missing
                    errmsg = e.stderr or e.stdout or str(e)
                    QMessageBox.critical(self, "Export Video", f"ffmpeg error:\n{errmsg}")
                    return

            QMessageBox.information(self, "Export Video", f"Saved:\n{out_path}")

        except Exception as e:
            QMessageBox.warning(self, "Export Video", f"Failed:\n{e}")

    def _collect_preset(self) -> dict:
        return {
            "cmap": self.combo_cmap.currentText(),
            "dyn_db": float(self.spin_dyn.value()),
            "vgain_db": float(self.spin_vgain.value()),
            "fft": int(self.spin_fft.value()),
            "hop": int(self.spin_hop.value()),
            "title_on": bool(self.chk_title.isChecked()),
            "title": self.edit_title.text(),
            "xlabel": self.edit_xlabel.text(),
            "ylabel": self.edit_ylabel.text(),
            "fs_title": int(self.spin_title_fs.value()),
            "fs_label": int(self.spin_label_fs.value()),
            "fs_tick": int(self.spin_tick_fs.value()),
            # ranges in kHz + seconds (UI units)
            "tmin": float(self.spin_tmin.value()),
            "tmax": float(self.spin_tmax.value()),
            "fmin_khz": float(self.spin_fmin.value()),
            "fmax_khz": float(self.spin_fmax.value()),
        }

    def _apply_preset(self, P: dict):
        # Basic controls
        if "cmap" in P: self.combo_cmap.setCurrentText(P["cmap"])
        if "dyn_db" in P: self.spin_dyn.setValue(float(P["dyn_db"]))
        if "vgain_db" in P: self.spin_vgain.setValue(float(P["vgain_db"]))
        if "fft" in P: self.spin_fft.setValue(int(P["fft"]))
        if "hop" in P: self.spin_hop.setValue(int(P["hop"]))
        if "title_on" in P: self.chk_title.setChecked(bool(P["title_on"]))
        if "title" in P: self.edit_title.setText(str(P["title"]))
        if "xlabel" in P: self.edit_xlabel.setText(str(P["xlabel"]))
        if "ylabel" in P: self.edit_ylabel.setText(str(P["ylabel"]))
        if "fs_title" in P: self.spin_title_fs.setValue(int(P["fs_title"]))
        if "fs_label" in P: self.spin_label_fs.setValue(int(P["fs_label"]))
        if "fs_tick" in P: self.spin_tick_fs.setValue(int(P["fs_tick"]))
        # Ranges (only if data is present)
        if self._t_full is not None and self._f_full is not None:
            t0, t1 = float(self._t_full[0]), float(self._t_full[-1])
            f0k, f1k = float(self._f_full[0] / 1000.0), float(self._f_full[-1] / 1000.0)
            if "tmin" in P and "tmax" in P:
                self.spin_tmin.setValue(float(np.clip(P["tmin"], t0, t1)))
                self.spin_tmax.setValue(float(np.clip(P["tmax"], t0, t1)))
            if "fmin_khz" in P and "fmax_khz" in P:
                self.spin_fmin.setValue(float(np.clip(P["fmin_khz"], f0k, f1k)))
                self.spin_fmax.setValue(float(np.clip(P["fmax_khz"], f0k, f1k)))
        self._render_canvas()

    def _save_preset(self):
        P = self._collect_preset()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Preset", str(Path(self.edit_folder.text().strip() or ".") / "preset.json"),
            "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(P, f, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Save Preset", f"Failed to save:\n{e}")

    def _load_preset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Preset", str(Path(self.edit_folder.text().strip() or ".")),
            "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                P = json.load(f)
            self._apply_preset(P)
        except Exception as e:
            QMessageBox.warning(self, "Load Preset", f"Failed to load:\n{e}")

    def _set_view_full(self):
        # Restore to full extents
        t, f = self._t_full, self._f_full
        self.spin_tmin.blockSignals(True); self.spin_tmax.blockSignals(True)
        self.spin_fmin.blockSignals(True); self.spin_fmax.blockSignals(True)
        self.spin_tmin.setValue(float(t[0])); self.spin_tmax.setValue(float(t[-1]))
        self.spin_fmin.setValue(float(f[0] / 1000.0));
        self.spin_fmax.setValue(float(f[-1] / 1000.0))

        self.spin_tmin.blockSignals(False); self.spin_tmax.blockSignals(False)
        self.spin_fmin.blockSignals(False); self.spin_fmax.blockSignals(False)
        self._render_canvas()

    def _restore_settings(self):
        s = QSettings("ConcordiaChem", "BatSpectrogramInspector")
        # Window & splitter
        geo = s.value("main/geometry", None)
        if isinstance(geo, QByteArray):
            self.restoreGeometry(geo)
        state = s.value("main/splitter", None)
        if isinstance(state, QByteArray):
            self.splitter.restoreState(state)
        # Last folder
        last = s.value("paths/last_folder", "", type=str)
        if last:
            self.edit_folder.setText(last)
        # Export defaults
        self.spin_dpi.setValue(s.value("export/dpi", self.spin_dpi.value(), type=int))
        self.spin_w_in.setValue(s.value("export/w", self.spin_w_in.value(), type=float))
        self.spin_h_in.setValue(s.value("export/h", self.spin_h_in.value(), type=float))
        self.chk_transparent.setChecked(s.value("export/transparent", False, type=bool))

    def _save_settings(self, partial_only: bool=False):
        s = QSettings("ConcordiaChem", "BatSpectrogramInspector")
        # Window & splitter
        if not partial_only:
            s.setValue("main/geometry", self.saveGeometry())
            s.setValue("main/splitter", self.splitter.saveState())
            s.setValue("paths/last_folder", self.edit_folder.text().strip())
        # Export defaults
        s.setValue("export/dpi", int(self.spin_dpi.value()))
        s.setValue("export/w", float(self.spin_w_in.value()))
        s.setValue("export/h", float(self.spin_h_in.value()))
        s.setValue("export/transparent", bool(self.chk_transparent.isChecked()))

    def closeEvent(self, event):
        try:
            self._save_settings()
        finally:
            super().closeEvent(event)


# =============================
# Entrypoint
# =============================

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
