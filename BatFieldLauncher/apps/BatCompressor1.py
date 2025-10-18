#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BatCompressorGUI.py — PySide6 GUI to create "bat-only" audio from JSON sidecars.

Overview
--------
Given a folder of JSON sidecars produced by the Bat/NoBat sorter, this tool:
  • Reads each JSON to locate the original WAV and its per-window predictions
  • Keeps only 0.5 s windows whose bat probability ≥ user threshold (default 0.85)
  • Writes either:
      - Individual per-file compressed WAVs (concatenated bat segments), OR
      - A single "bat anthology" WAV per top-level folder (grouped by directory)
  • Produces a text report with throughput, compression, and a system snapshot
  • Optionally deletes the original raw WAVs after successful compression

Assumptions
-----------
• Sidecar JSON contains:
    {
      "file": "<absolute path to WAV>",
      "duration_s": <float>,
      "params": { "win_s": 0.5, ... },
      "windows_compact": [ [start_s, margin, prob, dyn_range], ... ]
    }
  Where `prob` is bat probability in [0,1], and each window spans win_s seconds.

• WAV files are 16-bit PCM, mono or stereo, typically 384 kHz (AudioMoth).

Notes
-----
• We stream segments in chunks; we do not load entire files into RAM.
• We apply short cosine fades (5 ms) at segment boundaries to avoid clicks.
• Missing WAVs are logged; JSON is skipped.

"""

from __future__ import annotations
import json
import os
import sys
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import psutil
import platform
import shutil
import wave

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QGroupBox, QHBoxLayout, QComboBox,
    QTextEdit, QProgressBar,
)
from PySide6.QtGui import QPixmap, QIcon # <-- This is the new line

# ---------------------------- Utilities ---------------------------- #

@dataclass
class WindowEntry:
    start_s: float
    prob: float
    margin: float
    dyn_range: float

@dataclass
class Sidecar:
    wav_path: Path
    duration_s: float
    win_s: float
    windows: List[WindowEntry]
    label: str

FADE_MS = 5.0  # fade length at segment boundaries (ms)


def load_sidecar(path: Path) -> Optional[Sidecar]:
    """Parse a sidecar JSON into a Sidecar object.
    Expects `windows_compact` rows like [start_s, margin, prob, dyn_range].
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        wav_path = Path(data["file"])  # absolute path expected
        duration_s = float(data.get("duration_s", 0.0))
        params = data.get("params", {})
        win_s = float(params.get("win_s", 0.5))
        summary = data.get("summary", {})
        label = str(summary.get("label", "")).strip()
        entries = []
        for row in data.get("windows_compact", []):
            # Defensive parse: allow length 3 or 4
            start_s = float(row[0])
            if len(row) >= 3:
                # Schema: [start_s, p_noise, margin, dyn]
                prob = float(row[1])
                margin = float(row[2])  # |p_bat - 0.5| == |0.5 - p_noise|
                dyn = float(row[3]) if len(row) >= 4 else 0.0
            else:
                # Minimal fallback: treat row[1] as p_noise
                prob = float(row[1]) if len(row) >= 2 else 0.5
                margin = abs(prob - 0.5)
                dyn = 0.0
            entries.append(WindowEntry(start_s=start_s, prob=prob, margin=margin, dyn_range=dyn))

        return Sidecar(wav_path=wav_path, duration_s=duration_s, win_s=win_s, windows=entries, label=label)
    except Exception:
        return None


def longest_common_prefix(paths: List[Path]) -> Path:
    if not paths:
        return Path("")
    split = [p.parts for p in paths]
    prefix = []
    for parts in zip(*split):
        if all(p == parts[0] for p in parts):
            prefix.append(parts[0])
        else:
            break
    return Path(*prefix)


def group_key_by_main_folder(wav_paths: List[Path]) -> Dict[Path, str]:
    """Group WAVs by the path segment immediately below their common prefix.
    Returns dict: wav_path -> group_name.
    """
    lcp = longest_common_prefix(wav_paths)
    mapping: Dict[Path, str] = {}
    for wp in wav_paths:
        rel = wp.relative_to(lcp) if lcp != Path("") else wp
        first = rel.parts[0] if rel.parts else wp.parent.name
        mapping[wp] = first
    return mapping


def cosine_fade(n: int) -> np.ndarray:
    # 0..1 fade window (half-cosine)
    t = np.arange(n, dtype=np.float32)
    return 0.5 * (1 - np.cos(np.pi * (t + 1) / (n + 1)))


def write_segments_to_wave(in_wav: Path,
                           segments: List[Tuple[int, int]],
                           out_wav: Path,
                           fade_ms: float = FADE_MS,
                           dtype='<i2') -> Dict[str, float]:
    """Concatenate segments from input WAV to output WAV with tiny fades.
    segments: list of (start_sample, end_sample), inclusive-exclusive.
    Returns stats dict with kept_samples, total_samples.
    """
    with wave.open(str(in_wav), 'rb') as r:
        nch = r.getnchannels()
        srate = r.getframerate()
        sampwidth = r.getsampwidth()
        if sampwidth != 2:
            raise RuntimeError(f"Only 16-bit PCM supported (found {sampwidth*8} bits)")

        # Prepare output file
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_wav), 'wb') as w:
            w.setnchannels(nch)
            w.setsampwidth(sampwidth)
            w.setframerate(srate)

            fade_len = int(max(1, round(fade_ms * 1e-3 * srate)))
            fade = cosine_fade(fade_len)
            kept_total = 0
            total_samples = r.getnframes()

            for (a, b) in segments:
                a = max(0, min(a, total_samples))
                b = max(0, min(b, total_samples))
                if b <= a:
                    continue
                n = b - a
                r.setpos(a)
                raw = r.readframes(n)
                # Convert to numpy for fades - AND MAKE A WRITABLE COPY
                arr = np.frombuffer(raw, dtype=np.int16).copy()
                if nch == 2:
                    arr = arr.reshape(-1, 2)
                else:
                    arr = arr.reshape(-1, 1)

                # Apply short fades on boundaries
                f = min(fade_len, len(arr))
                if f > 1:
                    # fade-in
                    arr[:f, :] = (arr[:f, :].astype(np.float32) * fade[:f, None]).astype(np.int16)
                    # fade-out
                    arr[-f:, :] = (arr[-f:, :].astype(np.float32) * fade[:f][::-1, None]).astype(np.int16)

                # Write back
                w.writeframes(arr.astype(np.int16).tobytes())
                kept_total += len(arr)

    return {"kept_samples": float(kept_total), "total_samples": float(total_samples)}


# ---------------------------- Worker thread ---------------------------- #

class CompressorWorker(QThread):
    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self,
                 sidecar_dir: Path,
                 output_dir: Path,
                 conf_thresh: float,
                 mode: str,  # "individual" or "anthology"
                 delete_originals: bool,
                 process_clean: bool,
                 process_noisy: bool):
        super().__init__()
        self.sidecar_dir = sidecar_dir
        self.output_dir = output_dir
        self.conf_thresh = conf_thresh
        self.mode = mode
        self.delete_originals = delete_originals
        self.process_clean = process_clean
        self.process_noisy = process_noisy
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            t0 = time.perf_counter()
            # Discover JSON sidecars
            sidecars = sorted([p for p in self.sidecar_dir.iterdir() if p.suffix.lower() == ".json"])
            if not sidecars:
                self.error.emit("No JSON files found in the selected folder.")
                return
            self.status.emit(f"Found {len(sidecars)} JSON sidecars. Parsing…")

            parsed: List[Sidecar] = []
            missing: List[str] = []
            for p in sidecars:
                sc = load_sidecar(p)
                if sc is None:
                    continue
                if not sc.wav_path.exists():
                    missing.append(str(sc.wav_path))
                    continue
                parsed.append(sc)

            if not parsed:
                self.error.emit("No valid sidecars with existing WAV paths.")
                return

            # Filter by label per user toggles: skip all 'No Bat'
            filtered: List[Sidecar] = []
            skipped_counts = {"Bat": 0, "Noisy Bat": 0, "No Bat": 0, "Other": 0}
            for sc in parsed:
                lab = (sc.label or "").strip()
                if lab == "Bat":
                    if self.process_clean:
                        filtered.append(sc)
                    else:
                        skipped_counts["Bat"] += 1
                elif lab == "Noisy Bat":
                    if self.process_noisy:
                        filtered.append(sc)
                    else:
                        skipped_counts["Noisy Bat"] += 1
                elif lab == "No Bat":
                    skipped_counts["No Bat"] += 1
                    continue
                else:
                    skipped_counts["Other"] += 1
                    continue

            if not filtered:
                self.error.emit("After label filtering, no files matched the selection (clean/noisy toggles).")
                return

            wav_paths = [s.wav_path for s in filtered]

            # Decide grouping for anthology mode
            group_map: Dict[Path, str] = {}
            if self.mode == "anthology":
                group_map = group_key_by_main_folder(wav_paths)

            kept_bytes = 0
            total_bytes = 0
            kept_seconds = 0.0
            total_seconds = 0.0

            # Output roots
            out_root = self.output_dir
            created_outputs: List[Path] = []

            # --- ANTHOLOGY MODE SETUP ---
            anth_writer: Optional[wave.Wave_write] = None
            anth_rate: Optional[int] = None
            if self.mode == "anthology":
                # Define a single path for the anthology file
                anth_path = self.output_dir / "bat_anthology.wav"
                created_outputs.append(anth_path)

            # Process
            for i, sc in enumerate(filtered, start=1):
                if not self._running:
                    break

                self.progress.emit(i, len(filtered))
                self.status.emit(f"{i}/{len(filtered)}: {sc.wav_path.name} [{sc.label}]")

                # Build keep segments in samples
                keep_segments: List[Tuple[int,int]] = []
                try:
                    with wave.open(str(sc.wav_path), 'rb') as r:
                        sr = r.getframerate()
                        nframes = r.getnframes()
                        nch = r.getnchannels()
                        sw = r.getsampwidth()
                        if sw != 2:
                            raise RuntimeError("Only 16-bit PCM supported.")

                        win_n = int(round(sc.win_s * sr))
                        # Using prob ≥ threshold
                        for w in sc.windows:
                            if w.prob >= self.conf_thresh:
                                a = int(round(w.start_s * sr))
                                b = min(a + win_n, nframes)
                                keep_segments.append((a, b))

                        # Merge adjacent segments if contiguous
                        merged: List[Tuple[int,int]] = []
                        for seg in sorted(keep_segments):
                            if not merged:
                                merged.append(seg)
                            else:
                                la, lb = merged[-1]
                                if seg[0] <= lb:  # contiguous or overlapping
                                    merged[-1] = (la, max(lb, seg[1]))
                                else:
                                    merged.append(seg)
                        keep_segments = merged

                except Exception as e:
                    self.status.emit(f"Skipping (read error): {sc.wav_path} — {e}")
                    continue

                # Write output according to mode
                try:
                    if self.mode == "individual":
                        # out path mirrors WAV folder structure under out_root
                        rel_parent = sc.wav_path.parent
                        # Use flat layout under out_root, preserving parent folder names
                        dest_dir = out_root / rel_parent.name
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        out_path = dest_dir / (sc.wav_path.stem + "__batonly.wav")
                        stats = write_segments_to_wave(sc.wav_path, keep_segments, out_path)
                        created_outputs.append(out_path)

                        total_seconds += sc.duration_s
                        total_bytes   += sc.wav_path.stat().st_size if sc.wav_path.exists() else 0
                        kept_seconds  += (stats["kept_samples"] / float(sr))
                        kept_bytes    += out_path.stat().st_size if out_path.exists() else 0

                        if self.delete_originals and out_path.exists():
                            try:
                                sc.wav_path.unlink(missing_ok=True)
                            except Exception:
                                pass

                    else:  # anthology
                        # If the single anthology writer isn't open yet, open it now
                        # using the audio parameters from the CURRENT file.
                        if anth_writer is None:
                            with wave.open(str(sc.wav_path), 'rb') as r_params:
                                anth_rate = r_params.getframerate()
                                nch = r_params.getnchannels()
                                sw = r_params.getsampwidth()

                            anth_writer = wave.open(str(anth_path), 'wb')
                            anth_writer.setnchannels(nch)
                            anth_writer.setsampwidth(sw)
                            anth_writer.setframerate(anth_rate)

                        # Create a temporary file for the current WAV's bat segments
                        tmp_out = self.output_dir / (sc.wav_path.stem + "__tmpbat.wav")
                        stats = write_segments_to_wave(sc.wav_path, keep_segments, tmp_out)

                        # Append the temporary file's content to the main anthology file
                        with wave.open(str(tmp_out), 'rb') as tr:
                            raw_frames = tr.readframes(tr.getnframes())
                            anth_writer.writeframes(raw_frames)

                        # Clean up the temporary file
                        try:
                            tmp_out.unlink(missing_ok=True)
                        except Exception:
                            pass

                        # Update running totals
                        total_seconds += sc.duration_s
                        total_bytes += sc.wav_path.stat().st_size if sc.wav_path.exists() else 0
                        if anth_rate:
                            kept_seconds += (stats["kept_samples"] / float(anth_rate))

                        if self.delete_originals:
                            try:
                                sc.wav_path.unlink(missing_ok=True)
                            except Exception:
                                pass

                except Exception as e:
                    self.status.emit(f"Write error: {e}")
                    continue

            # Close anthology writer
            if anth_writer:
                try:
                    anth_writer.close()
                    # Now that the file is closed, get its final size
                    kept_bytes = anth_path.stat().st_size
                except Exception:
                    pass

            # Build report
            t1 = time.perf_counter()
            wall_s = t1 - t0
            size_gb = total_bytes / (1024**3) if total_bytes else 0.0
            files_per_s = (len(filtered) / wall_s) if wall_s > 0 else 0.0
            gb_per_min = ((size_gb * 60.0) / wall_s) if wall_s > 0 else 0.0
            compress_size = (size_gb * kept_bytes / total_bytes) if total_bytes else 0.0
            compression = (100* kept_bytes / total_bytes) if total_bytes else 0.0
            kept_pct = (100.0 * kept_seconds / total_seconds) if total_seconds > 0 else 0.0

            # System snapshot
            uname = platform.uname()
            cpu_name = uname.processor or platform.processor() or "Unknown CPU"
            logical_cores = os.cpu_count() or 0
            physical_cores = psutil.cpu_count(logical=False) or logical_cores
            try:
                freq = psutil.cpu_freq(); cpu_freq_str = f"{freq.current:.0f} MHz" if freq else "N/A"
            except Exception:
                cpu_freq_str = "N/A"
            os_string = f"{uname.system} {uname.release} ({uname.version.split()[0]})"
            try:
                drive = shutil.disk_usage(self.output_dir)
                total_out_gb = drive.total / (1024**3)
                free_out_gb = drive.free / (1024**3)
                used_percent = 100 * (drive.used / drive.total)
                disk_info = f"Total {total_out_gb:.1f} GiB, Free {free_out_gb:.1f} GiB ({100 - used_percent:.1f}% free)"
            except Exception:
                disk_info = "Unavailable"

            vm = psutil.virtual_memory()
            used_gb = vm.used / (1024 ** 3)
            available_gb = vm.available / (1024 ** 3)

            report_path = self.output_dir / "BatCompressor_report.txt"
            with open(report_path, "w", encoding="utf-8") as rf:
                rf.write("Bat Compressor — Batch Report\n")
                rf.write("==============================\n\n")
                rf.write(f"Sidecars processed  : {len(filtered)}\n")
                rf.write(f"Wall time (s)       : {wall_s:.2f}\n")
                rf.write(f"Throughput (files/s): {files_per_s:.2f}\n")
                rf.write(f"GiB per minute      : {gb_per_min:.2f}\n")
                rf.write(f"Total size in (GiB) : {size_gb:.3f}\n")
                rf.write(f"Compressed size     : {compress_size:.2f}\n")
                rf.write(f"Compression (%)     : {compression:.2f}\n")
                rf.write(f"Kept time (%)       : {kept_pct:.2f}\n")
                rf.write(f"Skipped (No Bat)    : {skipped_counts.get('No Bat', 0)}\n")
                rf.write(f"Skipped (clean off) : {skipped_counts.get('Bat', 0)}\n")
                rf.write(f"Skipped (noisy off) : {skipped_counts.get('Noisy Bat', 0)}\n\n")

                rf.write("System Snapshot\n")
                rf.write("---------------\n")
                rf.write(f"CPU model          : {cpu_name}\n")
                rf.write(f"CPU frequency      : {cpu_freq_str}\n")
                rf.write(f"Physical cores     : {physical_cores}\n")
                rf.write(f"Logical cores      : {logical_cores}\n")
                rf.write(f"Operating system   : {os_string}\n")
                rf.write(f"Disk info          : {disk_info}\n")
                rf.write(f"RAM used now (GiB) : {used_gb:.2f}\n")
                rf.write(f"RAM avail (GiB)    : {available_gb:.2f}\n")

            self.finished.emit({
                "status": "OK",
                "report": str(report_path),
                "outputs": [str(p) for p in created_outputs],
                "metrics": {
                    "wall_s": wall_s,
                    "files_per_s": files_per_s,
                    "gb_per_min": gb_per_min,
                    "compression": compression,
                    "kept_pct": kept_pct,
                }
            })

        except Exception as e:
            import traceback
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ---------------------------- GUI ---------------------------- #

# ---------------------------- GUI ---------------------------- #

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bat Compressor (JSON → bat-only WAVs)")
        self.resize(900, 600)

        # --- Load and Set Application Image/Icon ---
        script_dir = Path(__file__).parent
        image_path = script_dir / "BatCuttingNewspaper.png"
        bat_cartoon_label = QLabel()  # Create the label for the image

        if image_path.exists():
            # Set the window icon first
            self.setWindowIcon(QIcon(str(image_path)))

            # Then, prepare and display the cartoon image in the layout
            pixmap = QPixmap(str(image_path))
            scaled_pixmap = pixmap.scaledToWidth(300, Qt.TransformationMode.SmoothTransformation)
            bat_cartoon_label.setPixmap(scaled_pixmap)
        else:
            # Fallback if the image is not found
            print(f"Warning: Image not found at {image_path}")
            bat_cartoon_label.setText("Bat Cartoon Placeholder")

        bat_cartoon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- Main Layout and Widgets ---
        central = QWidget(self)
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)

        # Inputs group
        grp = QGroupBox("Inputs")
        grid = QGridLayout(grp)

        self.edit_sidecar = QLineEdit()
        self.btn_sidecar = QPushButton("Select JSON sidecars folder…")
        self.btn_sidecar.clicked.connect(self._choose_sidecar)

        self.edit_output = QLineEdit()
        self.btn_output = QPushButton("Select output folder…")
        self.btn_output.clicked.connect(self._choose_output)

        self.edit_thresh = QLineEdit("0.85")
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Individual files", "Full bat anthology"])
        self.chk_clean = QCheckBox("Process clean Bat files")
        self.chk_clean.setChecked(True)
        self.chk_noisy = QCheckBox("Process Noisy Bat files")
        self.chk_noisy.setChecked(True)
        self.chk_delete = QCheckBox("Delete original files (after successful write)")
        self.chk_delete.setChecked(False)

        grid.addWidget(QLabel("JSON sidecars folder:"), 0, 0)
        grid.addWidget(self.edit_sidecar, 0, 1)
        grid.addWidget(self.btn_sidecar, 0, 2)
        grid.addWidget(QLabel("Output folder:"), 1, 0)
        grid.addWidget(self.edit_output, 1, 1)
        grid.addWidget(self.btn_output, 1, 2)
        grid.addWidget(QLabel("Confidence threshold:"), 2, 0)
        grid.addWidget(self.edit_thresh, 2, 1)
        grid.addWidget(QLabel("Mode:"), 3, 0)
        grid.addWidget(self.combo_mode, 3, 1)
        grid.addWidget(self.chk_clean, 4, 0, 1, 3)
        grid.addWidget(self.chk_noisy, 5, 0, 1, 3)
        grid.addWidget(self.chk_delete, 6, 0, 1, 3)

        vbox.addWidget(grp)
        vbox.addWidget(bat_cartoon_label)  # Add the cartoon to the layout

        # Run controls
        h = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_run.clicked.connect(self._start)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop)
        h.addWidget(self.btn_run)
        h.addWidget(self.btn_stop)
        vbox.addLayout(h)

        # Progress + log
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        vbox.addWidget(self.progress)
        vbox.addWidget(self.log, 1)

        # Worker holder
        self.worker: Optional[CompressorWorker] = None

    # ---- UI helpers ----
    def _choose_sidecar(self):
        d = QFileDialog.getExistingDirectory(self, "Select sidecars folder")
        if d:
            self.edit_sidecar.setText(d)

    def _choose_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.edit_output.setText(d)

    def _append_log(self, msg: str):
        self.log.append(msg)

    def _start(self):
        try:
            sc_dir = Path(self.edit_sidecar.text()).expanduser()
            out_dir = Path(self.edit_output.text()).expanduser()
            if not sc_dir.exists() or not sc_dir.is_dir():
                self._append_log("ERROR: Sidecars folder not found.")
                return
            if not out_dir.exists():
                out_dir.mkdir(parents=True, exist_ok=True)

            try:
                thr = float(self.edit_thresh.text())
                if not (0.0 <= thr <= 1.0):
                    raise ValueError
            except Exception:
                self._append_log("ERROR: Confidence threshold must be in [0,1].")
                return

            mode = "individual" if self.combo_mode.currentIndex() == 0 else "anthology"
            delete_originals = self.chk_delete.isChecked()

            self.worker = CompressorWorker(sc_dir, out_dir, thr, mode, delete_originals,
                                           self.chk_clean.isChecked(), self.chk_noisy.isChecked())
            self.worker.progress.connect(self._on_progress)
            self.worker.status.connect(lambda s: self._append_log(s))
            self.worker.error.connect(lambda e: self._append_log(f"ERROR: {e}"))
            self.worker.finished.connect(self._on_finished)
            self.worker.start()
            self._append_log("Started.")
        except Exception as e:
            self._append_log(f"ERROR: {e}")

    def _stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self._append_log("Stopping… (will finish current file)")

    def _on_progress(self, i: int, n: int):
        self.progress.setRange(0, n)
        self.progress.setValue(i)

    def _on_finished(self, payload: dict):
        self._append_log("Finished.")
        report = payload.get("report")
        if report:
            self._append_log(f"Report: {report}")


# ---------------------------- Main entry ---------------------------- #

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()