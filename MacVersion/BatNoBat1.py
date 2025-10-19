#!/usr/bin/env python3
# BatNoBatMLPhase1.py — Field "Bat / No Bat / Noisy Bat" sorter (Full Metrics mode)
# Requirements: PySide6, numpy, scipy, soundfile, pillow (PIL), tensorflow
# Parity with training: fixed global dB clamp, 18–80 kHz band, 224×224 resize, [-1,1] scaling.

import os
import sys
import math
import json
import csv
import shutil
import time
import psutil
import traceback
import multiprocessing
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# --- Environment for stable multiprocessing / BLAS ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Core scientific stack ---
import numpy as np
import soundfile as sf
from PIL import Image
from scipy.signal import spectrogram

# --- TensorFlow ---
import tensorflow as tf

# --- GUI ---
from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QFileDialog, QProgressBar,
    QMessageBox, QCheckBox
)

# =============================================================================
# Configuration (match Round 2 training)
# =============================================================================
NPERSEG = 1024
IMAGE_SIZE = (224, 224)
FMIN_HZ = 18_000
FMAX_HZ = 80_000
GLOBAL_MIN_DB = -95.0
GLOBAL_MAX_DB = -35.0
EPS_DB = 1e-6
WINDOW_SEC = 0.5  # non-overlapping windows

# =============================================================================
# Worker globals (populated via init_worker)
# =============================================================================
_worker_model = None
_worker_params = {}

from pathlib import Path
import sys

# optional: quieter TF/oneDNN logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

def resource_path(*parts: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)                  # one-file temp dir
    elif getattr(sys, "frozen", False):
        base = Path(sys.executable).parent         # one-folder dist dir
    else:
        base = Path(__file__).parent               # source tree
    return base.joinpath(*parts)

# Import Keras with a robust fallback (Keras 3 vs tf.keras)
try:
    import keras                                   # Keras 3 package
except Exception:
    from tensorflow import keras                   # fallback

model_path = resource_path('BatNoBat_Beta.keras')
model = keras.models.load_model(str(model_path))


ICON_FILE = resource_path('batcrossingguard.png')
# Example uses:
# app.setWindowIcon(QIcon(str(ICON_FILE)))
# pixmap = QPixmap(str(ICON_FILE))


def init_worker(model_path: Path,
                confidence_tau: float,
                min_bat_time_s: float,
                drr_threshold: float,
                write_json: bool,
                sidecar_dir: Path,
                input_root: Path,
                retain_structure: bool):

    """Initializer for worker processes; loads the model once per process."""
    global _worker_model, _worker_params
    _worker_model = tf.keras.models.load_model(str(model_path))
    _worker_params = {
        "tau": confidence_tau,
        "min_bat_time_s": float(min_bat_time_s),
        "drr_threshold": float(drr_threshold),
        "write_json": bool(write_json),
        "sidecar_dir": sidecar_dir,
        "input_root": input_root,
        "retain_structure": bool(retain_structure),
    }


def _normalize_db(Sxx: np.ndarray) -> np.ndarray:
    """Fixed global dB clamp and [0,1] normalization."""
    Sxx_db = 10.0 * np.log10(Sxx + 1e-9)
    Sxx_db = np.clip(Sxx_db, GLOBAL_MIN_DB, GLOBAL_MAX_DB)
    Sxx01 = (Sxx_db - GLOBAL_MIN_DB) / max((GLOBAL_MAX_DB - GLOBAL_MIN_DB), EPS_DB)
    Sxx01 = np.clip(Sxx01, 0.0, 1.0)
    return Sxx_db, Sxx01


def _window_indices(times: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Return boolean mask for spectrogram columns with t in [t0, t1)."""
    return (times >= t0) & (times < t1)


def _img_from_window(S01_win: np.ndarray) -> np.ndarray:
    """Convert [freq x time] in [0,1] to 224×224 uint8 image (flipped freq)."""
    img_data = (np.flipud(S01_win) * 255.0).astype(np.uint8)
    img = Image.fromarray(img_data, mode="L").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    # replicate to 3 channels and rescale to [-1, 1] as in MobileNetV2 preprocessing
    arr = (arr / 127.5) - 1.0
    arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def _percentile_dr(S_db_win: np.ndarray) -> float:
    """Dynamic range p95 - p5 (dB) within a window, robust to outliers."""
    p5 = np.percentile(S_db_win, 5.0)
    p95 = np.percentile(S_db_win, 95.0)
    return float(max(0.0, p95 - p5))


def _label_from_metrics(bat_time_s: float, drr: Optional[float],
                        min_bat_time_s: float, drr_threshold: float) -> str:
    if bat_time_s < min_bat_time_s:
        return "No Bat"
    if drr is None:
        # If no noise windows to compute DRR, fall back to Bat
        return "Bat"
    return "Noisy Bat" if (drr < drr_threshold) else "Bat"


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def process_single_wav(wav_path: Path) -> Dict[str, Any]:
    """
    Process a single WAV file end-to-end (Full Metrics):
      - compute spectrogram once
      - iterate 0.5 s non-overlapping windows
      - predict p(bat), margin, window DR
      - aggregate: bat-time, % bat-time, avg±std margin, DRR
      - emit JSON sidecar (optional) + return CSV-ready summary
    """
    try:
        # 1) Load audio and metadata
        with sf.SoundFile(str(wav_path), "r") as f:
            sr = f.samplerate
            nframes = len(f)
            duration_s = nframes / float(sr)
            audio = f.read(dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Short-file rule
        if duration_s < WINDOW_SEC:
            result = {
                "file": str(wav_path),
                "duration_s": duration_s,
                "label": "No Bat",
                "bat_time_s": 0.0,
                "percent_bat_time": 0.0,
                "avg_margin": 0.0,
                "std_margin": 0.0,
                "drr": None,
                "n_bat_windows": 0,
                "n_noise_windows": 0,
                "notes": ["short_file_lt_0.5s"],
            }

            # --- Copy WAV to label folder for inspection/triage ---
            try:
                dest_root = _worker_params["sidecar_dir"].parent / "sorted"
                label_dir = result["label"].replace(" ", "_")  # "Bat", "Noisy_Bat", "No_Bat"
                if _worker_params.get("retain_structure", True):
                    # mirror the input subfolder structure *then* label
                    rel = wav_path.parent.relative_to(_worker_params["input_root"])
                    dest_dir = dest_root / rel / label_dir
                else:
                    # flat three folders
                    dest_dir = dest_root / label_dir
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(wav_path), str(dest_dir / wav_path.name))

            except Exception:
                # Non-fatal: skip copying if something odd happens (e.g., perms)
                pass

            if _worker_params["write_json"]:
                _write_json_sidecar(wav_path, result, _worker_params["sidecar_dir"])
            return {
                "file": result["file"],
                "duration_s": result["duration_s"],
                "label": result["label"],
                "bat_time_s": result["bat_time_s"],
                "percent_bat_time": result["percent_bat_time"],
                "avg_margin": result["avg_margin"],
                "std_margin": result["std_margin"],
                "drr": result["drr"],
                "n_bat_windows": result["n_bat_windows"],
                "n_noise_windows": result["n_noise_windows"],
            }

        # 2) Full-file spectrogram once (match training/apply)
        freqs, times, Sxx = spectrogram(audio, fs=sr, nperseg=NPERSEG)
        band = (freqs >= FMIN_HZ) & (freqs <= FMAX_HZ)
        if band.sum() == 0:
            # Edge case: sample rate too low for 18–80 kHz; bail cleanly
            result = {
                "file": str(wav_path),
                "duration_s": duration_s,
                "label": "No Bat",
                "bat_time_s": 0.0,
                "percent_bat_time": 0.0,
                "avg_margin": 0.0,
                "std_margin": 0.0,
                "drr": None,
                "n_bat_windows": 0,
                "n_noise_windows": 0,
                "notes": ["no_inband_freqs_18to80kHz"],
            }
            if _worker_params["write_json"]:
                _write_json_sidecar(wav_path, result, _worker_params["sidecar_dir"])
            return {
                "file": result["file"],
                "duration_s": result["duration_s"],
                "label": result["label"],
                "bat_time_s": result["bat_time_s"],
                "percent_bat_time": result["percent_bat_time"],
                "avg_margin": result["avg_margin"],
                "std_margin": result["std_margin"],
                "drr": result["drr"],
                "n_bat_windows": result["n_bat_windows"],
                "n_noise_windows": result["n_noise_windows"],
            }

        Sxx_band = Sxx[band, :]
        S_db, S01 = _normalize_db(Sxx_band)

        # 3) 0.5 s non-overlapping windows, ignore tail < 0.5 s
        n_windows = int(duration_s // WINDOW_SEC)
        tau = _worker_params["tau"]

        # Accumulators
        bat_flags: List[bool] = []
        margins_all: List[float] = []
        dr_bat: List[float] = []
        dr_noise: List[float] = []
        windows_compact: List[Tuple[float, float, float, float]] = []  # t0, p, margin, DR

        for w in range(n_windows):
            t0 = w * WINDOW_SEC
            t1 = t0 + WINDOW_SEC
            cols = _window_indices(times, t0, t1)
            if not np.any(cols):
                # No spectrogram columns in this interval (rare boundary effect)
                continue

            # Window slices
            S01_win = S01[:, cols]
            Sdb_win = S_db[:, cols]

            # Build model input and predict
            arr = _img_from_window(S01_win)
            logits = _worker_model.predict(np.expand_dims(arr, 0), verbose=0)
            p = float(logits[0][0])
            margin = abs(p - 0.5)

            # Window dynamic range (robust)
            dr = _percentile_dr(Sdb_win)

            # Accumulate
            is_bat = (p >= tau)
            bat_flags.append(is_bat)
            margins_all.append(margin)
            if is_bat:
                dr_bat.append(dr)
            else:
                dr_noise.append(dr)

            windows_compact.append((t0, p, margin, dr))

        # 4) Aggregation
        bat_windows = sum(1 for b in bat_flags if b)
        noise_windows = sum(1 for b in bat_flags if not b)
        bat_time_s = bat_windows * WINDOW_SEC
        percent_bat_time = (bat_time_s / duration_s) if duration_s > 0 else 0.0
        avg_margin = float(np.mean(margins_all)) if margins_all else 0.0
        std_margin = float(np.std(margins_all)) if margins_all else 0.0

        # DRR with small-N guard
        drr = None
        if len(dr_bat) >= 3 and len(dr_noise) >= 3:
            drr = float(np.mean(dr_bat) / max(np.mean(dr_noise), 1e-9))
        elif len(dr_bat) >= 1 and len(dr_noise) >= 1:
            # fallback to medians if tiny samples
            drr = float(np.median(dr_bat) / max(np.median(dr_noise), 1e-9))

        label = _label_from_metrics(
            bat_time_s=bat_time_s,
            drr=drr,
            min_bat_time_s=_worker_params["min_bat_time_s"],
            drr_threshold=_worker_params["drr_threshold"],
        )

        result = {
            "file": str(wav_path),
            "duration_s": float(duration_s),
            "label": label,
            "bat_time_s": float(bat_time_s),
            "percent_bat_time": float(percent_bat_time),
            "avg_margin": float(avg_margin),
            "std_margin": float(std_margin),
            "drr": _safe_float(drr),
            "n_bat_windows": int(bat_windows),
            "n_noise_windows": int(noise_windows),
            "notes": [],
            # Compact per-window record for sidecar only
            "windows_compact": windows_compact,
            "params": mode_params(),
        }

        # --- Copy WAV to label folder for inspection/triage ---
        try:
            dest_root = _worker_params["sidecar_dir"].parent / "sorted"
            label_dir = result["label"].replace(" ", "_")  # "Bat", "Noisy_Bat", "No_Bat"
            if _worker_params.get("retain_structure", True):
                # mirror the input subfolder structure *then* label
                rel = wav_path.parent.relative_to(_worker_params["input_root"])
                dest_dir = dest_root / rel / label_dir
            else:
                # flat three folders
                dest_dir = dest_root / label_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(wav_path), str(dest_dir / wav_path.name))

        except Exception:
            # Non-fatal: keep processing even if copy fails
            pass

        # 5) Sidecar JSON
        if _worker_params["write_json"]:
            _write_json_sidecar(wav_path, result, _worker_params["sidecar_dir"])

        # Return CSV-friendly summary (minus heavy window arrays)
        return {
            "file": result["file"],
            "duration_s": result["duration_s"],
            "label": result["label"],
            "bat_time_s": result["bat_time_s"],
            "percent_bat_time": result["percent_bat_time"],
            "avg_margin": result["avg_margin"],
            "std_margin": result["std_margin"],
            "drr": result["drr"],
            "n_bat_windows": result["n_bat_windows"],
            "n_noise_windows": result["n_noise_windows"],
        }

    except Exception as e:
        # Emit error row; GUI will report totals
        return {
            "file": str(wav_path),
            "duration_s": None,
            "label": "ERROR",
            "bat_time_s": None,
            "percent_bat_time": None,
            "avg_margin": None,
            "std_margin": None,
            "drr": None,
            "n_bat_windows": 0,
            "n_noise_windows": 0,
        }


def mode_params():
    return {
        "win_s": WINDOW_SEC,
        "tau": _worker_params.get("tau"),
        "min_bat_time_s": _worker_params.get("min_bat_time_s"),
        "drr_threshold": _worker_params.get("drr_threshold"),
        "global_db": [GLOBAL_MIN_DB, GLOBAL_MAX_DB],
        "band_hz": [FMIN_HZ, FMAX_HZ],
        "nperseg": NPERSEG,
        "image_size": IMAGE_SIZE,
        "scale": "[-1,1]",
        "metrics_mode": "FullMetrics",
    }


def _write_json_sidecar(wav_path: Path, result: Dict[str, Any], sidecar_dir: Path):
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    out_path = sidecar_dir / (wav_path.stem + ".json")
    # Strip heavy arrays if present in summary version (keep compact)
    doc = {
        "file": result["file"],
        "duration_s": result["duration_s"],
        "summary": {
            "label": result["label"],
            "bat_time_s": result["bat_time_s"],
            "percent_bat_time": result["percent_bat_time"],
            "avg_margin": result["avg_margin"],
            "std_margin": result["std_margin"],
            "drr": result["drr"],
            "n_bat_windows": result["n_bat_windows"],
            "n_noise_windows": result["n_noise_windows"],
            "notes": result.get("notes", []),
        },
        "params": result.get("params", mode_params()),
        "windows_compact": result.get("windows_compact", []),  # [t0, p, margin, DR]
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


# =============================================================================
# GUI + Orchestration
# =============================================================================

class Orchestrator(QObject):
    progress = Signal(int, int)          # current, total
    status = Signal(str)                 # status text
    finished = Signal(dict)              # summary dict
    error = Signal(str)

    def __init__(self, input_dir: Path, output_dir: Path, model_path: Path,
                 cores: int, tau: float, min_bat_time_s: float, drr_threshold: float,
                 write_json: bool,
                 retain_structure: bool,
                 delete_original: bool):

        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_path = model_path
        self.cores = cores
        self.tau = tau
        self.min_bat_time_s = min_bat_time_s
        self.drr_threshold = drr_threshold
        self.write_json = write_json
        self.retain_structure = retain_structure
        self.delete_original = delete_original

        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            self.status.emit("Scanning for WAV files...")
            wavs = sorted([p for p in self.input_dir.rglob("*") if p.suffix.lower() == ".wav"])

            # Map file -> size (bytes) and total bytes
            size_by_path = {str(p): p.stat().st_size for p in wavs}
            total_bytes = sum(size_by_path.values())
            start_time = time.perf_counter()

            if not wavs:
                self.error.emit("No .wav files found in the selected input folder (or subfolders).")
                return
            total = len(wavs)
            self.status.emit(f"Found {total} WAV files.")

            # Prepare outputs
            sidecar_dir = self.output_dir / "json_sidecars"
            csv_path = self.output_dir / "BatNoBatML1_summary.csv"

            # --- Multiprocessing pool with per-file timeout + recycle on stall ---
            self.status.emit(f"Initializing {self.cores} worker processes...")
            ctx = multiprocessing.get_context("spawn")

            def make_pool():
                return ctx.Pool(
                    processes=self.cores,
                    initializer=init_worker,
                    initargs=(
                        self.model_path,
                        self.tau,
                        self.min_bat_time_s,
                        self.drr_threshold,
                        self.write_json,
                        sidecar_dir,
                        self.input_dir,  # needed to rebuild relative paths
                        self.retain_structure  # copy-mode flag
                    ),
                    maxtasksperchild=50,  # recycle workers to avoid long-run resource buildup
                )

            pool = make_pool()
            results: List[Dict[str, Any]] = []
            timed_out: List[str] = []
            recovered_on_retry: List[str] = []

            # Per-file timeout (seconds). Adjust if you have extremely long/slow I/O.
            #TIMEOUT_S = 180.0

            # --- Adaptive timeout helpers (per file) ---
            def _estimate_duration_seconds(path: Path) -> float:
                """Try to read WAV header quickly; fall back to size-based estimate."""
                try:
                    import wave
                    with wave.open(str(path), "rb") as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        if rate > 0:
                            return frames / float(rate)
                except Exception:
                    pass
                # Fallback: estimate from size, assuming 384 kHz, mono, 16-bit PCM
                try:
                    size = path.stat().st_size
                    bytes_per_sec = 384000 * 2 * 1  # sr * bytes/sample * channels
                    return size / float(bytes_per_sec)
                except Exception:
                    return 0.0

            def _adaptive_timeout_seconds(path: Path) -> float:
                """
                Base 60 s + (duration in seconds) with a floor and a ceiling.
                Example: 10 min file -> 60 + 600 = 660 s.
                """
                dur_s = _estimate_duration_seconds(path)
                t = max(60.0, 60.0 + dur_s)  # base + duration
                return min(t, 660.0)  # cap at 20 minutes to avoid runaway waits

            try:
                self.status.emit("Processing files (Full Metrics, with timeouts)...")
                processed = 0
                for p in wavs:
                    if not self._running:
                        break

                    # Submit single file, wait with timeout
                    async_res = pool.apply_async(process_single_wav, (p,))
                    try:
                        timeout_s = _adaptive_timeout_seconds(p)
                        res = async_res.get(timeout=timeout_s)


                    except multiprocessing.context.TimeoutError:
                        # First timeout — recycle the pool and try the file ONCE more.
                        try:
                            pool.terminate()
                            pool.join()
                        except Exception:
                            pass
                        pool = make_pool()

                        # Inform the status bar / log (optional)
                        self.status.emit(f"Timeout on: {str(p)} — retrying once...")

                        # Retry with same adaptive timeout (or 1.5x if you prefer)
                        try:
                            timeout_s_retry = _adaptive_timeout_seconds(p)  # or: 1.5 * _adaptive_timeout_seconds(p)
                            async_res2 = pool.apply_async(process_single_wav, (p,))
                            res = async_res2.get(timeout=timeout_s_retry)
                            # Success on retry
                            recovered_on_retry.append(str(p))

                        except multiprocessing.context.TimeoutError:
                            # Final failure after retry — record as ERROR_TIMEOUT
                            timed_out.append(str(p))
                            res = {
                                "file": str(p),
                                "duration_s": None,
                                "label": "ERROR_TIMEOUT",
                                "bat_time_s": None,
                                "percent_bat_time": None,
                                "avg_margin": None,
                                "std_margin": None,
                                "drr": None,
                                "n_bat_windows": 0,
                                "n_noise_windows": 0,
                            }

                        except Exception:
                            # Any other per-file exception on retry
                            res = {
                                "file": str(p),
                                "duration_s": None,
                                "label": "ERROR",
                                "bat_time_s": None,
                                "percent_bat_time": None,
                                "avg_margin": None,
                                "std_margin": None,
                                "drr": None,
                                "n_bat_windows": 0,
                                "n_noise_windows": 0,
                            }

                    results.append(res)
                    processed += 1
                    self.progress.emit(processed, total)

                # Normal completion: close and join workers cleanly
                if self._running:
                    pool.close()
                else:
                    pool.terminate()
                pool.join()

            except Exception:
                # Any error during mapping — terminate workers and re-raise
                try:
                    pool.terminate()
                    pool.join()
                except Exception:
                    pass
                raise

            if not self._running:
                self.finished.emit({"status": "Cancelled"})
                return

            end_time = time.perf_counter()
            wall_s = end_time - start_time

            # Write CSV summary
            self.status.emit("Writing CSV summary...")
            fieldnames = [
                "file", "duration_s", "label", "bat_time_s", "percent_bat_time",
                "avg_margin", "std_margin", "drr", "n_bat_windows", "n_noise_windows"
            ]
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in results:
                    writer.writerow(row)

            # --- Batch metrics ---
            bat_labels = {"Bat", "Noisy Bat"}
            bat_bytes = 0
            for r in results:
                lbl = r.get("label", "")
                fp = r.get("file", "")
                if lbl in bat_labels and fp in size_by_path:
                    bat_bytes += size_by_path[fp]

            size_gb = total_bytes / (1024 ** 3) if total_bytes else 0.0
            files_per_s = (len(results) / wall_s) if wall_s > 0 else 0.0
            s_per_gb = (wall_s / size_gb) if size_gb > 0 else 0.0
            gb_per_min = ((size_gb * 60.0) / wall_s) if wall_s > 0 else 0.0
            compression = (bat_bytes / total_bytes) if total_bytes else 0.0

            # RAM snapshot (system-wide)
            vm = psutil.virtual_memory()
            used_gb = vm.used / (1024 ** 3)
            available_gb = vm.available / (1024 ** 3)

            # --- Easy aggregates for the report ---
            from collections import Counter
            import statistics

            label_counts = Counter(r.get("label", "ERROR") for r in results)
            n_bat = label_counts.get("Bat", 0)
            n_noisybat = label_counts.get("Noisy Bat", 0)
            n_nobat = label_counts.get("No Bat", 0)
            n_err = label_counts.get("ERROR", 0) + label_counts.get("ERROR_TIMEOUT", 0)

            # Totals across the whole batch
            total_duration_s = sum((r.get("duration_s") or 0.0) for r in results)
            total_bat_time_s = sum((r.get("bat_time_s") or 0.0) for r in results)
            overall_bat_pct = (100.0 * total_bat_time_s / total_duration_s) if total_duration_s > 0 else 0.0

            # Per-file duration stats (useful sanity checks)
            durations = [r.get("duration_s") for r in results if r.get("duration_s") is not None]
            mean_file_s = statistics.mean(durations) if durations else 0.0
            median_file_s = statistics.median(durations) if durations else 0.0

            # If you kept `timed_out` from the timeout-enabled run:
            n_timeouts = len(timed_out) if "timed_out" in locals() else 0

            # Process-level memory (current, not peak)
            proc = psutil.Process()
            try:
                rss_self = proc.memory_info().rss
            except Exception:
                rss_self = 0
            child_rss = 0
            for c in proc.children(recursive=True):
                try:
                    child_rss += c.memory_info().rss
                except Exception:
                    pass
            rss_self_gb = rss_self / (1024 ** 3)
            child_rss_gb = child_rss / (1024 ** 3)
            rss_total_gb = rss_self_gb + child_rss_gb

            # --- System snapshot (CPU / OS / Disk) ---
            import platform, os, shutil
            uname = platform.uname()
            cpu_name = uname.processor or platform.processor() or "Unknown CPU"
            logical_cores = os.cpu_count() or 0
            physical_cores = psutil.cpu_count(logical=False) or logical_cores
            used_cores = self.cores
            try:
                freq = psutil.cpu_freq()
                cpu_freq_str = f"{freq.current:.0f} MHz" if freq else "N/A"
            except Exception:
                cpu_freq_str = "N/A"
            os_string = f"{uname.system} {uname.release} ({uname.version.split()[0]})"
            try:
                drive = shutil.disk_usage(self.output_dir)
                total_out_gb = drive.total / (1024 ** 3)
                free_out_gb = drive.free / (1024 ** 3)
                used_percent = 100 * (drive.used / drive.total)
                disk_info = f"Total {total_out_gb:.1f} GiB, Free {free_out_gb:.1f} GiB ({100 - used_percent:.1f}% free)"
            except Exception:
                disk_info = "Unavailable"

            # Write a simple text report next to the CSV
            report_path = self.output_dir / "BatNoBatML_report.txt"
            with open(report_path, "w", encoding="utf-8") as rf:
                rf.write("Bat / No-Bat Classifier — Batch Report (Full Metrics)\n")
                rf.write("=====================================================\n\n")
                rf.write(f"Total files           : {len(results)} of {total}\n")
                rf.write(f"Total size (GiB)      : {size_gb:.3f}\n")
                rf.write(f"Total wall time (s)   : {wall_s:.2f}\n")
                rf.write(f"Throughput (files/s)  : {files_per_s:.2f}\n")
                rf.write(f"Seconds per GiB       : {s_per_gb:.2f}\n")
                rf.write(f"**GiB per minute**     : {gb_per_min:.2f}\n")
                rf.write(f"Compression (bat/total): {compression:.2%}\n")
                rf.write(f"RAM used now (GiB)    : {used_gb:.2f}\n")
                rf.write(f"RAM available (GiB)   : {available_gb:.2f}\n")
                rf.write(f"Process RSS (GiB)     : {rss_self_gb:.2f}\n")
                rf.write(f"Workers RSS (GiB)     : {child_rss_gb:.2f}\n")
                rf.write(f"Proc+Workers RSS (GiB): {rss_total_gb:.2f}\n")
                rf.write("\nCounts by Label\n")
                rf.write("----------------\n")
                rf.write(f"Bat               : {n_bat}\n")
                rf.write(f"Noisy Bat         : {n_noisybat}\n")
                rf.write(f"No Bat            : {n_nobat}\n")
                rf.write(f"Errors (all)      : {n_err}\n")
                if n_timeouts:
                    rf.write(f"Timed-out files   : {n_timeouts}\n")

                rf.write("\nAggregate Call-Time Metrics\n")
                rf.write("---------------------------\n")
                rf.write(f"Total duration (s): {total_duration_s:.2f}\n")
                rf.write(f"Total bat time (s): {total_bat_time_s:.2f}\n")
                rf.write(f"Overall bat time %%: {overall_bat_pct:.2f}\n")

                rf.write("\nPer-file Duration (s)\n")
                rf.write("---------------------\n")
                rf.write(f"Mean              : {mean_file_s:.2f}\n")
                rf.write(f"Median            : {median_file_s:.2f}\n")

                # Timed-out files section (if any)
                if timed_out:
                    rf.write("\nTimed-out files (skipped after per-file timeout):\n")
                    for t in timed_out:
                        rf.write(f"  - {t}\n")

                rf.write("\nSystem Snapshot\n")
                rf.write("---------------\n")
                rf.write(f"CPU model          : {cpu_name}\n")
                rf.write(f"CPU frequency      : {cpu_freq_str}\n")
                rf.write(f"Physical cores     : {physical_cores}\n")
                rf.write(f"Logical cores      : {logical_cores}\n")
                rf.write(f"Cores used         : {used_cores}\n")
                rf.write(f"Operating system   : {os_string}\n")
                rf.write(f"Disk info          : {disk_info}\n")

            # Optional deletion step
            if self.delete_original:
                # 1) delete ALL original input WAVs
                for p in self.input_dir.rglob("*"):
                    try:
                        if p.is_file() and p.suffix.lower() == ".wav":
                            p.unlink(missing_ok=True)
                    except Exception:
                        pass

                # 2) delete "No Bat" copies in the output "sorted" tree
                sorted_root = (self.output_dir / "json_sidecars").parent / "sorted"
                try:
                    if getattr(self, "retain_structure", True):
                        # mirrored layout: No_Bat can exist at multiple depths
                        for nb_dir in sorted_root.rglob("No_Bat"):
                            if nb_dir.is_dir():
                                for p in nb_dir.rglob("*"):
                                    try:
                                        if p.is_file():
                                            p.unlink(missing_ok=True)
                                    except Exception:
                                        pass
                    else:
                        # flat layout: one No_Bat folder
                        no_bat_dir = sorted_root / "No_Bat"
                        if no_bat_dir.exists():
                            for p in no_bat_dir.rglob("*"):
                                try:
                                    if p.is_file():
                                        p.unlink(missing_ok=True)
                                except Exception:
                                    pass
                except Exception:
                    # non-fatal; keep going
                    pass

            # Build quick tallies
            tallies = {"Bat": 0, "Noisy Bat": 0, "No Bat": 0, "ERROR": 0}
            for r in results:
                label = r.get("label", "ERROR")
                if label not in tallies:
                    tallies["ERROR"] += 1
                else:
                    tallies[label] += 1

            self.finished.emit({
                "status": "OK",
                "counts": tallies,
                "csv": str(csv_path),
                "json_sidecars": str(sidecar_dir),
                "total_files": total,
                "report": str(report_path),
                "metrics": {
                    "size_gb": size_gb,
                    "wall_s": wall_s,
                    "files_per_s": files_per_s,
                    "s_per_gb": s_per_gb,
                    "gb_per_min": gb_per_min,
                    "compression": compression,
                    "ram_used_gb": used_gb,
                    "ram_avail_gb": available_gb,
                }
            })

        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BatNoBatML1")
        self.setGeometry(80, 80, 820, 420)
        self.thread: Optional[QThread] = None
        self.worker: Optional[Orchestrator] = None
        self._build_ui()
        self._style()

    # In apps/BatNoBat1.py

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        layout = QVBoxLayout(cw)

        # --- Cute bat crossing guard image ---
        img_row = QHBoxLayout()
        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignCenter)
        try:
            # This part is already correct and robust!
            png_path = resource_path("batcrossingguard.png")
            if png_path.exists():
                pm = QPixmap(str(png_path))
                pm = pm.scaled(360, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.lbl_img.setPixmap(pm)
            else:
                self.lbl_img.setText("batcrossingguard.png not found")
        except Exception as _:
            self.lbl_img.setText("Image load error")

        img_row.addWidget(self.lbl_img)
        layout.addLayout(img_row)

        # I/O group
        io = QGroupBox("Directories & Model")
        grid = QGridLayout(io)
        self.edit_input = QLineEdit()
        self.edit_output = QLineEdit()

        # --- THIS IS THE CORRECTED PART ---
        # Build a robust path to the model, works in source and frozen builds
        model_default_path = resource_path("BatNoBat_Beta.keras")
        self.edit_model = QLineEdit(str(model_default_path))
        # --- END OF CORRECTION ---

        b_in = QPushButton("Browse…");
        b_in.clicked.connect(self._pick_input)
        b_out = QPushButton("Browse…");
        b_out.clicked.connect(self._pick_output)
        b_model = QPushButton("Browse…");
        b_model.clicked.connect(self._pick_model)
        grid.addWidget(QLabel("Input folder (WAVs):"), 0, 0);
        grid.addWidget(self.edit_input, 0, 1);
        grid.addWidget(b_in, 0, 2)
        grid.addWidget(QLabel("Output folder:"), 1, 0);
        grid.addWidget(self.edit_output, 1, 1);
        grid.addWidget(b_out, 1, 2)
        grid.addWidget(QLabel("Model (.keras):"), 2, 0);
        grid.addWidget(self.edit_model, 2, 1);
        grid.addWidget(b_model, 2, 2)
        layout.addWidget(io)

        # (The rest of the function remains exactly the same...)
        sg = QGroupBox("Settings (Full Metrics)")
        sg_grid = QGridLayout(sg)
        self.edit_cores = QLineEdit(str(max(1, (os.cpu_count() or 4) * 3 // 4)))
        self.edit_tau = QLineEdit("0.98")
        self.edit_minbat = QLineEdit("1.5")  # cumulative seconds
        self.edit_drr = QLineEdit("1.7")  # DRR threshold for Noisy Bat
        self.chk_json = QCheckBox("Write per-file JSON sidecars");
        self.chk_json.setChecked(True)
        self.chk_retain = QCheckBox("Retain folder structure");
        self.chk_retain.setChecked(True)
        self.chk_delete = QCheckBox("Delete original data files");
        self.chk_delete.setChecked(False)

        sg_grid.addWidget(QLabel("CPU cores:"), 0, 0);
        sg_grid.addWidget(self.edit_cores, 0, 1)
        sg_grid.addWidget(QLabel("Confidence τ:"), 1, 0);
        sg_grid.addWidget(self.edit_tau, 1, 1)
        sg_grid.addWidget(QLabel("Min bat time (s):"), 2, 0);
        sg_grid.addWidget(self.edit_minbat, 2, 1)
        sg_grid.addWidget(QLabel("DRR threshold:"), 3, 0);
        sg_grid.addWidget(self.edit_drr, 3, 1)
        sg_grid.addWidget(self.chk_json, 4, 0, 1, 2)
        sg_grid.addWidget(self.chk_retain, 5, 0, 1, 2)
        sg_grid.addWidget(self.chk_delete, 6, 0, 1, 2)

        layout.addWidget(sg)

        # Controls
        self.btn_run = QPushButton("Start (Full Metrics)")
        self.btn_run.clicked.connect(self._start)
        layout.addWidget(self.btn_run)

        self.prog = QProgressBar();
        self.prog.setTextVisible(True)
        layout.addWidget(self.prog)
        self.lbl_status = QLabel("Select folders + model, set parameters, then Start.")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_status)
        layout.addStretch(1)

    def _style(self):
        self.setStyleSheet("""
            QWidget { background-color: #1F2430; color: #E6E9EF; }
            QGroupBox { font-weight: 700; border: 1px solid #3A3F58; border-radius: 8px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 8px; color: #80CBC4; }
            QLabel { color: #B0BEC5; }
            QLineEdit { background: #2B3142; border: 1px solid #3A3F58; border-radius: 6px; padding: 6px; color: #E6E9EF; }
            QPushButton { background: #80CBC4; color: #0B0E14; border: none; padding: 10px 16px; border-radius: 6px; font-weight: 700; }
            QPushButton:hover { background: #A3E1DA; }
            QProgressBar { border: 1px solid #3A3F58; border-radius: 6px; text-align: center; }
            QProgressBar::chunk { background-color: #C3E88D; border-radius: 6px; }
        """)

    def _pick_input(self):
        d = QFileDialog.getExistingDirectory(self, "Select Input Folder (WAVs)")
        if d: self.edit_input.setText(d)

    def _pick_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d: self.edit_output.setText(d)

    def _pick_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Trained Model", ".", "Keras Model (*.keras)")
        if p: self.edit_model.setText(p)

    def _start(self):
        try:
            input_dir = Path(self.edit_input.text()); assert input_dir.is_dir()
            output_dir = Path(self.edit_output.text()); assert output_dir.is_dir()
            model_path = Path(self.edit_model.text()); assert model_path.is_file()
            cores = int(float(self.edit_cores.text()))
            tau = float(self.edit_tau.text())
            minbat = float(self.edit_minbat.text())
            drr = float(self.edit_drr.text())
            write_json = self.chk_json.isChecked()
            retain_structure = self.chk_retain.isChecked()
            delete_original = self.chk_delete.isChecked()

            # Safety confirmation
            if delete_original:
                reply = QMessageBox.question(
                    self,
                    "Confirm Deletion",
                    "You selected 'Delete original data files'.\n\n"
                    "This will permanently remove ALL original input WAV files\n"
                    "and any 'No Bat' copies in the output.\n\n"
                    "Are you sure you want to proceed?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

            if not (1 <= cores <= (os.cpu_count() or 4)):
                raise ValueError(f"CPU cores must be in [1, {os.cpu_count()}].")
            if not (0.5 <= tau <= 1.0):
                raise ValueError("Confidence τ should be in [0.5, 1.0].")
            if minbat < 0.0:
                raise ValueError("Min bat time must be ≥ 0.")
            if drr <= 0.0:
                raise ValueError("DRR threshold must be > 0.")

        except Exception as e:
            QMessageBox.warning(self, "Input Error", str(e))
            return

        self.btn_run.setEnabled(False)
        self.btn_run.setText("Running…")

        self.thread = QThread()
        self.worker = Orchestrator(
            input_dir, output_dir, model_path,
            cores, tau, minbat, drr,
            write_json,
            retain_structure,
            delete_original
        )

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.status.connect(self.lbl_status.setText)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        self.thread.start()

    def _on_progress(self, current: int, total: int):
        self.prog.setRange(0, total)
        self.prog.setValue(current)
        self.prog.setFormat(f"{current} / {total} files")

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)
        self._cleanup()

    def _on_finished(self, info: Dict[str, Any]):
        if info.get("status") == "Cancelled":
            QMessageBox.information(self, "Cancelled", "Operation cancelled by user.")
        else:
            counts = info.get("counts", {})
            csv_path = info.get("csv", "")
            sidecars = info.get("json_sidecars", "")
            m = info.get("metrics", {})
            msg = (
                "Finished (Full Metrics)\n\n"
                f"Bat: {counts.get('Bat', 0)}\n"
                f"Noisy Bat: {counts.get('Noisy Bat', 0)}\n"
                f"No Bat: {counts.get('No Bat', 0)}\n"
                f"Errors: {counts.get('ERROR', 0)}\n\n"
                f"CSV: {csv_path}\n"
                f"JSON Sidecars: {sidecars}\n"
                f"Report: {info.get('report', '')}\n\n"
                f"Throughput: {m.get('files_per_s', 0):.2f} files/s | "
                f"{m.get('gb_per_min', 0):.2f} GiB/min\n"
                f"Compression (bat/total): {m.get('compression', 0):.2%}\n"
            )

            QMessageBox.information(self, "Done", msg)
        self._cleanup()

    def _cleanup(self):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Start (Full Metrics)")
        self.lbl_status.setText("Ready.")
        self.prog.setRange(0, 1)
        self.prog.setValue(0)
        if self.thread:
            self.worker.stop()
            self.thread.quit()
            self.thread.wait()
            self.worker.deleteLater()
            self.thread.deleteLater()
            self.thread = None
            self.worker = None


def main():
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()