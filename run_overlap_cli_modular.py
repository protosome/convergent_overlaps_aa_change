#!/usr/bin/env python3
"""CLI wrapper for the notebook's 'Run Overlap' form workflow."""

from __future__ import annotations

import argparse
import ast
import builtins
import importlib.util
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import textwrap
import time
import traceback
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, TextIO
from pathlib import Path

import pandas as pd
from paths import ROOT_DIR
from overlap_runtime.notebook_api import create_runtime


S4PRED_WEIGHTS_URL = (
    "https://bioinf.cs.ucl.ac.uk/downloads/s4pred/weights.tar.gz"
)
ALLOWED_PLOT_METRICS = ("Combined", "ESM_avg", "SS_avg", "Align1", "Align2", "Sub1", "Sub2")
SUMMARY_RE = re.compile(r"Summary so far:\s*(\{.*\})")
WINDOW_RE = re.compile(r"Entering window\s+(\d+)/(\d+)")
ATTEMPT_RE = re.compile(r"=== Optimization attempt\s+(\d+)/(\d+)\s+===")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
WINDOW_AA = 10
STRIDE_AA = 8
MODEL_AA = 105


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run convergent overlap generation from CLI using modular runtime code."
    )
    parser.add_argument(
        "--working-dir",
        default="test_results",
        help="Directory where run outputs and archives are written.",
    )
    parser.add_argument(
        "--excel-path",
        default="test_results/aa_1_aa_2.xlsx",
        help="Path to input .xlsx/.xls/.csv with aa_seq_1 and aa_seq_2 columns.",
    )
    parser.add_argument(
        "--overlap-length-selected",
        default="310",
        help="Single overlap length or comma-separated list (e.g. 310 or 310,311).",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="1-based start row in input table.",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=0,
        help="1-based end row in input table; 0 means process to end.",
    )
    parser.add_argument(
        "--use-row-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use per-row ss/sub/aln/esm weights when available.",
    )
    parser.add_argument("--ss-w", type=float, default=0.15, help="Override SS weight.")
    parser.add_argument("--sub-w", type=float, default=0.15, help="Override substitution weight.")
    parser.add_argument("--aln-w", type=float, default=0.10, help="Override alignment weight.")
    parser.add_argument("--esm-w", type=float, default=0.60, help="Override ESM weight.")
    parser.add_argument(
        "--normalize-override-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize override weights to sum to 1.",
    )
    parser.add_argument(
        "--first-pass-iters",
        type=int,
        default=1,
        help="First-pass optimization iterations.",
    )
    parser.add_argument(
        "--second-pass-iters",
        type=int,
        default=75,
        help="Second-pass optimization iterations.",
    )
    parser.add_argument(
        "--reset-caches-per-row",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, clear internal caches per row before optimization.",
    )
    parser.add_argument(
        "--archive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a zip archive for this run.",
    )
    parser.add_argument(
        "--notebook-path",
        default="convergent_overlapping_gene_generation.ipynb",
        help="Notebook path to load function definitions from.",
    )
    parser.add_argument(
        "--check-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only validate dependencies and inputs, then exit.",
    )
    parser.add_argument(
        "--setup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Install dependencies and download/extract S4PRED weights before checks/running.",
    )
    parser.add_argument(
        "--setup-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run setup steps and exit without running optimization.",
    )
    parser.add_argument(
        "--weights-url",
        default=S4PRED_WEIGHTS_URL,
        help="URL for the S4PRED weights tarball.",
    )
    parser.add_argument(
        "--force-weights-download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force re-download and extract S4PRED weights even if present.",
    )
    parser.add_argument(
        "--debug-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print detailed model-loading and inference checkpoints.",
    )
    parser.add_argument(
        "--auto-setup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically run setup when required deps/weights are missing (default: enabled).",
    )
    parser.add_argument(
        "--esm-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for ESM features. 'auto' tries CUDA then falls back to CPU if needed.",
    )
    parser.add_argument(
        "--esm-batch-size",
        type=int,
        default=4,
        help="ESM batch size (lower values are safer for CUDA stability).",
    )
    parser.add_argument(
        "--esm-autocast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable fp16 autocast for ESM on CUDA (default: disabled for stability).",
    )
    parser.add_argument(
        "--require-esm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail fast if ESM features cannot be computed for a row (default: enabled).",
    )
    parser.add_argument(
        "--display-mode",
        choices=["plain", "tui", "radar"],
        default="plain",
        help="Output mode: plain logs (default), live dashboard (tui), or dual-window radar view (radar).",
    )
    parser.add_argument(
        "--display-style",
        choices=["plain", "boxed"],
        default="boxed",
        help="Live view style for tui/radar modes.",
    )
    parser.add_argument(
        "--plot-metric",
        type=_parse_plot_metrics_arg,
        default=["Combined"],
        help=(
            "Metric(s) to render as live trend plots in --display-mode tui/radar. "
            "Use a comma-separated list, e.g. Combined,SS_avg,ESM_avg."
        ),
    )
    return parser.parse_args()


def _parse_plot_metrics_arg(value: str) -> list[str]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("At least one plot metric is required.")
    bad = [p for p in parts if p not in ALLOWED_PLOT_METRICS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Invalid plot metric(s): {', '.join(bad)}. Allowed: {', '.join(ALLOWED_PLOT_METRICS)}"
        )
    # de-duplicate while preserving order
    return list(dict.fromkeys(parts))


@dataclass
class TuiState:
    overlap_length: int
    row_display_num: int
    plot_metrics: list[str]
    display_mode: str = "tui"
    display_style: str = "boxed"
    attempt: int = 0
    total_attempts: int = 0
    window: int = 0
    total_windows: int = 0
    summaries: list[dict] = field(default_factory=list)
    window_attempt_marks: dict[int, int] = field(default_factory=dict)
    recent_events: deque[str] = field(default_factory=lambda: deque(maxlen=8))
    ss_progress: str = ""
    run_start_ts: float = field(default_factory=time.time)
    last_summary_ts: float = field(default_factory=time.time)
    windows_done: int = 0
    prev_metrics: dict[str, float] = field(default_factory=dict)
    best_metrics: dict[str, float] = field(default_factory=dict)
    best_window: int = 0
    final_attempt_summary: str = ""

    def _color(self, text: str, code: str) -> str:
        if os.environ.get("NO_COLOR"):
            return text
        return f"\x1b[{code}m{text}\x1b[0m"

    def _trend(self, metric: str, width: int = 48) -> tuple[str, float, float, int]:
        values = [float(s.get(metric, 0.0)) for s in self.summaries if metric in s]
        if not values:
            return "(no data yet)", 0.0, 0.0, 0
        if len(values) > width:
            values = values[-width:]
        lo = min(values)
        hi = max(values)
        bins = "▁▂▃▄▅▆▇█"
        if hi - lo < 1e-12:
            return bins[-1] * len(values), lo, hi, len(values)
        out: list[str] = []
        span = hi - lo
        for v in values:
            idx = int((v - lo) / span * (len(bins) - 1))
            out.append(bins[idx])
        return "".join(out), lo, hi, len(values)

    def _latest_metrics(self) -> dict[str, float]:
        if not self.summaries:
            return {}
        latest = self.summaries[-1]
        keys = ("Combined", "SS_score1", "SS_score2", "SS_avg", "ESM1", "ESM2", "ESM_avg", "Align1", "Align2")
        return {k: float(latest[k]) for k in keys if k in latest}

    def _metric_color(self, metric: str) -> str:
        if self.attempt <= 1:
            base = {
                "Combined": "1;92",
                "SS_avg": "1;93",
                "ESM_avg": "1;94",
                "Align1": "1;90",
                "Align2": "1;90",
                "Sub1": "1;96",
                "Sub2": "1;96",
            }
        else:
            base = {
                "Combined": "1;91",
                "SS_avg": "1;33",
                "ESM_avg": "1;95",
                "Align1": "1;37",
                "Align2": "1;37",
                "Sub1": "1;36",
                "Sub2": "1;36",
            }
        return base.get(metric, "1;92")

    def _pass_color(self) -> str:
        return "1;96" if self.attempt <= 1 else "1;33"

    def _panel(self, title: str, lines: list[str], *, width: int = 84, height: int = 6) -> list[str]:
        if self.display_style != "boxed":
            return [title] + lines
        inner = max(10, width - 2)
        top = f"┌{self._fit(title, inner, fill='─')}┐"
        body_lines: list[str] = []
        for ln in lines:
            body_lines.extend(self._wrap_line(ln, inner))
        body_lines = body_lines[:height]
        while len(body_lines) < height:
            body_lines.append("")
        body = [f"│{self._fit(line, inner)}│" for line in body_lines]
        bot = f"└{'─' * inner}┘"
        return [top, *body, bot]

    def _fit(self, text: str, width: int, fill: str = " ") -> str:
        raw = str(text)
        visible = ANSI_RE.sub("", raw)
        if len(visible) > width:
            # Avoid breaking ANSI sequences by truncating plain-text fallback only.
            return visible[:max(0, width - 1)] + "…"
        return raw + (fill * (width - len(visible)))

    def _wrap_line(self, text: str, width: int) -> list[str]:
        raw = str(text)
        vis = ANSI_RE.sub("", raw)
        if len(vis) <= width:
            return [raw]
        # For colored lines, wrap plain fallback to avoid broken ANSI sequences.
        if vis != raw:
            wrapped = textwrap.wrap(vis, width=width, break_long_words=False, break_on_hyphens=False)
            return wrapped if wrapped else [vis]
        wrapped = textwrap.wrap(raw, width=width, break_long_words=False, break_on_hyphens=False)
        return wrapped if wrapped else [raw]

    def _fmt_delta(self, key: str, cur: float) -> str:
        if key not in self.prev_metrics:
            return "n/a"
        d = cur - self.prev_metrics[key]
        return f"{d:+.3f}"

    def _update_best(self, metrics: dict[str, float]) -> None:
        if not metrics:
            return
        cur = metrics.get("Combined", float("-inf"))
        best = self.best_metrics.get("Combined", float("-inf"))
        if cur > best:
            self.best_metrics = dict(metrics)
            self.best_window = self.window

    def _throughput_eta(self) -> tuple[float, float]:
        elapsed = max(1e-6, time.time() - self.run_start_ts)
        rate = self.windows_done / elapsed if self.windows_done > 0 else 0.0
        remaining = max(0, self.total_windows - self.window) if self.total_windows else 0
        eta = (remaining / rate) if rate > 0 else 0.0
        return rate, eta

    def _format_seconds(self, v: float) -> str:
        s = int(max(0, round(v)))
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _status_lines(self) -> list[str]:
        rate, eta = self._throughput_eta()
        initial_note = ""
        if self.attempt <= 1 and self.window <= 1:
            initial_note = (
                "Note: first window in pass 1 may be slower due to initial inference warm-up; "
                "later windows/passes are usually faster."
            )
        return [
            f"{self._color('Length', '1;36')}={self.overlap_length}  {self._color('Row', '1;36')}={self.row_display_num}",
            f"{self._color('Attempt', self._pass_color())}={self.attempt}/{self.total_attempts or '?'}  {self._color('Window', '1;36')}={self.window}/{self.total_windows or '?'}",
            f"{self._color('Elapsed', '1;36')}={self._format_seconds(time.time() - self.run_start_ts)}  {self._color('Win/s', '1;36')}={rate:.2f}  {self._color('ETA', '1;36')}={self._format_seconds(eta)}",
            self._color(initial_note, "1;33") if initial_note else "",
        ]

    def _metrics_lines(self, metrics: dict[str, float]) -> list[str]:
        if not metrics:
            return ["waiting for first summary"]
        return [
            (
                f"{self._color('Combined', self._metric_color('Combined'))}={metrics.get('Combined', float('nan')):.3f} ({self._fmt_delta('Combined', metrics.get('Combined', 0.0))})  "
                f"{self._color('SS1', '1;93')}={metrics.get('SS_score1', float('nan')):.2f}  "
                f"{self._color('SS2', '1;93')}={metrics.get('SS_score2', float('nan')):.2f}  "
                f"{self._color('SS_avg', self._metric_color('SS_avg'))}={metrics.get('SS_avg', float('nan')):.2f} ({self._fmt_delta('SS_avg', metrics.get('SS_avg', 0.0))})"
            ),
            (
                f"{self._color('ESM1', '1;94')}={metrics.get('ESM1', float('nan')):.2f}  "
                f"{self._color('ESM2', '1;94')}={metrics.get('ESM2', float('nan')):.2f}  "
                f"{self._color('ESM_avg', self._metric_color('ESM_avg'))}={metrics.get('ESM_avg', float('nan')):.2f} ({self._fmt_delta('ESM_avg', metrics.get('ESM_avg', 0.0))})  "
                f"{self._color('Align1', self._metric_color('Align1'))}={metrics.get('Align1', float('nan')):.2f}  {self._color('Align2', self._metric_color('Align2'))}={metrics.get('Align2', float('nan')):.2f}"
            ),
            (
                self._color("Best so far:", "1;95") + " "
                + (
                    f"W{self.best_window} Combined={self.best_metrics.get('Combined', float('nan')):.3f} "
                    f"SS_avg={self.best_metrics.get('SS_avg', float('nan')):.2f} "
                    f"ESM_avg={self.best_metrics.get('ESM_avg', float('nan')):.2f}"
                    if self.best_metrics else "n/a"
                )
            ),
        ]

    def _window_bounds(self, window_1based: int) -> tuple[int, int]:
        start = max(0, (window_1based - 1) * STRIDE_AA)
        end = min(start + WINDOW_AA, MODEL_AA)
        return start, end

    def _build_track(self, *, reverse: bool, width: int = 56) -> str:
        chars = ["."] * width
        for w in sorted(self.window_attempt_marks):
            s, e = self._window_bounds(w)
            i0 = int((s / MODEL_AA) * width)
            i1 = max(i0 + 1, int((e / MODEL_AA) * width))
            mark_attempt = self.window_attempt_marks.get(w, 1)
            mark_char = "@" if mark_attempt <= 1 else "#"
            for i in range(max(0, i0), min(width, i1)):
                chars[i] = mark_char

        if self.window > 0:
            s, e = self._window_bounds(self.window)
            i0 = int((s / MODEL_AA) * width)
            i1 = max(i0 + 1, int((e / MODEL_AA) * width))
            active_char = "<" if reverse else ">"
            for i in range(max(0, i0), min(width, i1)):
                chars[i] = active_char

        if reverse:
            chars.reverse()
        return "".join(chars)

    def _radar_lines(self) -> list[str]:
        if self.window > 0:
            s, e = self._window_bounds(self.window)
            aa_label = f"{s}-{e}"
        else:
            aa_label = "n/a"
        fwd = self._build_track(reverse=False)
        rev = self._build_track(reverse=True)
        return [
            f"{self._color('Active AA range', '1;36')}: {aa_label}",
            f"{self._color('Forward  0->104', '1;92')}: {self._color(fwd, '1;92')}",
            f"{self._color('Reverse  104->0', '1;94')}: {self._color(rev, '1;94')}",
            self._color("Legend: . initial  @ attempt1  # attempt2+  > active forward  < active reverse", "1;90"),
        ]

    def render(self, out: Callable[[str], None]) -> None:
        metrics = self._latest_metrics()
        self._update_best(metrics)
        title = self._color("Convergent Overlap Live View", self._pass_color())
        status_panel = self._panel(" Status ", self._status_lines(), height=5)
        metrics_panel = self._panel(" Metrics ", self._metrics_lines(metrics), height=4)
        radar_panel = self._panel(" Window Radar ", self._radar_lines(), height=4)
        trend_lines = []
        for metric in self.plot_metrics:
            trend, lo, hi, n_metric = self._trend(metric)
            c = self._metric_color(metric)
            trend_lines.append(f"{self._color(metric + ':', c)} {self._color(trend, c)}")
            trend_lines.append(f"  x=window (latest {n_metric})   y~[{lo:.3f}, {hi:.3f}]")
        trends_panel = self._panel(" Trends ", trend_lines or ["(no trend data)"], height=8)
        summary_panel = self._panel(
            " Attempt Summary ",
            [self.final_attempt_summary or "(awaiting attempt-final metrics)"],
            height=2,
        )
        ev_lines = list(self.recent_events) if self.recent_events else ["waiting for optimizer output..."]
        if self.ss_progress:
            ev_lines = [f"SS: {self.ss_progress}", *ev_lines]
        events_panel = self._panel(" Recent Events ", [f"- {e}" for e in ev_lines], height=8)

        lines = ["\x1b[2J\x1b[H", title]
        lines.extend(status_panel)
        if self.display_mode == "radar":
            lines.extend(radar_panel)
        lines.extend(metrics_panel)
        lines.extend(trends_panel)
        lines.extend(summary_panel)
        lines.extend(events_panel)
        out("\n".join(lines) + "\n")

    def handle_line(self, line: str, out: Callable[[str], None]) -> None:
        text = line.strip()
        if not text:
            return

        attempt_match = ATTEMPT_RE.search(text)
        if attempt_match:
            self.attempt = int(attempt_match.group(1))
            self.total_attempts = int(attempt_match.group(2))
            self.recent_events.append(f"Optimization attempt {self.attempt}/{self.total_attempts}")
            self.render(out)
            return

        window_match = WINDOW_RE.search(text)
        if window_match:
            self.window = int(window_match.group(1))
            self.total_windows = int(window_match.group(2))
            self.recent_events.append(f"Entering window {self.window}/{self.total_windows}")
            self.render(out)
            return

        summary_match = SUMMARY_RE.search(text)
        if summary_match:
            try:
                summary = ast.literal_eval(summary_match.group(1))
                if isinstance(summary, dict):
                    previous_metrics = self._latest_metrics()
                    self.prev_metrics = previous_metrics
                    self.summaries.append(summary)
                    self.windows_done += 1
                    try:
                        window_1based = int(summary.get("window", -1)) + 1
                        if window_1based > 0:
                            prev = self.window_attempt_marks.get(window_1based, 0)
                            cur = self.attempt if self.attempt > 0 else 1
                            self.window_attempt_marks[window_1based] = max(prev, cur)
                    except Exception:
                        pass
                    evt_metrics = []
                    for m in self.plot_metrics[:3]:
                        if m in summary:
                            evt_metrics.append(f"{m}={summary[m]}")
                    evt_tail = ", ".join(evt_metrics) if evt_metrics else "metrics updated"
                    self.recent_events.append(
                        f"Window {summary.get('window', '?')} {evt_tail}"
                    )
                    self.render(out)
                    return
            except Exception:
                pass

        if "Final metrics this attempt" in text or "Secondary structure below" in text:
            if "Final metrics this attempt" in text:
                self.final_attempt_summary = text
            self.recent_events.append(text)
            self.render(out)
            return
        if "Inference attempt failed" in text or "All retries failed" in text:
            self.recent_events.append(text)
            self.render(out)
            return

    def handle_stderr_line(self, line: str, out: Callable[[str], None]) -> None:
        text = line.replace("\r", "").strip()
        if not text:
            return
        if "Predicting SS:" in text:
            self.ss_progress = text
            self.render(out)
            return
        # keep unexpected stderr events visible in recent events panel
        self.recent_events.append(text)
        self.render(out)

    def final_summary_box(self, out_path: str | None) -> str:
        metrics = self._latest_metrics()
        lines = [
            f"Length={self.overlap_length} Row={self.row_display_num} Attempts={self.total_attempts or self.attempt}",
            (
                f"Final: Combined={metrics.get('Combined', float('nan')):.3f} "
                f"SS1={metrics.get('SS_score1', float('nan')):.2f} "
                f"SS2={metrics.get('SS_score2', float('nan')):.2f} "
                f"SS_avg={metrics.get('SS_avg', float('nan')):.2f} "
                f"ESM_avg={metrics.get('ESM_avg', float('nan')):.2f}"
                if metrics else "Final: n/a"
            ),
            (
                f"Best: W{self.best_window} Combined={self.best_metrics.get('Combined', float('nan')):.3f} "
                f"SS_avg={self.best_metrics.get('SS_avg', float('nan')):.2f}"
                if self.best_metrics else "Best: n/a"
            ),
            f"Output: {out_path or '(none)'}",
        ]
        return "\n".join(self._panel(" Row Summary ", lines, height=4))


class _TuiStderrProxy(TextIO):
    def __init__(self, sink: Callable[[str], None], passthrough: TextIO) -> None:
        self._sink = sink
        self._passthrough = passthrough
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._handle_line(line)
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self._handle_line(self._buf)
            self._buf = ""
        self._passthrough.flush()

    def _handle_line(self, line: str) -> None:
        text = line.replace("\r", "").strip()
        if not text:
            return
        if "Predicting SS:" in text:
            self._sink(text)
            return
        self._passthrough.write(line + "\n")

    def isatty(self) -> bool:
        return self._passthrough.isatty()


def _run_with_print_intercept(
    fn: Callable[[], str | None],
    on_line: Callable[[str], None],
    stderr_on_line: Callable[[str], None] | None = None,
) -> str | None:
    orig_print = builtins.print
    orig_stderr = sys.stderr

    def intercepted_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(str(a) for a in args)
        for part in text.splitlines():
            on_line(part)
        if end != "\n":
            on_line(text + end)

    builtins.print = intercepted_print
    if stderr_on_line is not None:
        sys.stderr = _TuiStderrProxy(stderr_on_line, orig_stderr)
    try:
        return fn()
    finally:
        builtins.print = orig_print
        if stderr_on_line is not None:
            try:
                sys.stderr.flush()
            except Exception:
                pass
            sys.stderr = orig_stderr


def _parse_overlap_lengths(overlap_length_selected: str) -> list[int]:
    try:
        model_numbers = [int(x.strip()) for x in overlap_length_selected.split(",") if x.strip()]
        if not model_numbers:
            model_numbers = [311]
    except Exception:
        model_numbers = [311]
    return model_numbers


def _sanitize_pairs_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["aa_seq_1", "aa_seq_2", "aa_seq_1_brackets", "aa_seq_2_brackets"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"\s+", "", regex=True)
        else:
            df[col] = ""
    return df


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total > 0:
        return {k: v / total for k, v in weights.items()}
    return weights


def _load_notebook_runtime(notebook_path: Path) -> dict:
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    if len(code_cells) < 4:
        raise RuntimeError("Unexpected notebook layout: missing code cells.")

    deps_cell = "".join(code_cells[1].get("source", []))
    main_cell = "".join(code_cells[2].get("source", []))

    # Guard this notebook call so CLI can run even when CUDA is unavailable.
    main_cell = main_cell.replace(
        "torch.cuda.get_device_name(0)",
        (
            "torch.cuda.get_device_name(0) if torch.cuda.is_available() "
            "else print('CUDA not available; using CPU')"
        ),
    )

    namespace: dict = {"__name__": "__main__"}
    exec(compile(deps_cell, str(notebook_path), "exec"), namespace)
    exec(compile(main_cell, str(notebook_path), "exec"), namespace)
    _register_notebook_classes_for_torch_unpickle(namespace)
    return namespace


def _register_notebook_classes_for_torch_unpickle(namespace: dict) -> None:
    """
    Notebook-trained .pth files may pickle model objects as __main__.TransformerModel.
    Register these classes on the current __main__ module so torch.load can resolve them.
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return

    for name in ("TransformerModel", "SinusoidalPositionalEncoding", "TransformerBlock"):
        cls = namespace.get(name)
        if cls is not None:
            setattr(main_mod, name, cls)


def _check_dependencies() -> tuple[list[str], list[str]]:
    required = {
        "torch": "torch",
        "tensorflow": "tensorflow",
        "numpy": "numpy",
        "pandas": "pandas",
        "tqdm": "tqdm",
        "Bio": "biopython",
        "openpyxl": "openpyxl",
        "skimage": "scikit-image",
    }
    optional = {
        "esm": "fair-esm",
    }

    missing_required: list[str] = []
    missing_optional: list[str] = []

    for module_name, pip_name in required.items():
        if importlib.util.find_spec(module_name) is None:
            missing_required.append(pip_name)
    for module_name, pip_name in optional.items():
        if importlib.util.find_spec(module_name) is None:
            missing_optional.append(pip_name)

    # De-duplicate while preserving order
    missing_required = list(dict.fromkeys(missing_required))
    missing_optional = list(dict.fromkeys(missing_optional))
    return missing_required, missing_optional


def _expected_weight_files(root_dir: Path) -> list[Path]:
    weights_dir = root_dir / "s4pred" / "weights"
    return [weights_dir / f"weights_{i}.pt" for i in range(1, 6)]


def _missing_weight_files(root_dir: Path) -> list[Path]:
    return [p for p in _expected_weight_files(root_dir) if not p.exists()]


def _download_and_extract_weights(root_dir: Path, weights_url: str) -> None:
    s4pred_dir = root_dir / "s4pred"
    if not s4pred_dir.exists():
        raise FileNotFoundError(f"s4pred directory not found at: {s4pred_dir}")

    tar_path = s4pred_dir / "weights.tar.gz"
    print(f"[SETUP] Downloading S4PRED weights from: {weights_url}")

    # The S4PRED endpoint currently presents an expired certificate.
    # This mirrors the notebook's wget --no-check-certificate setup.
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(weights_url, context=ssl_context) as response:
        with tar_path.open("wb") as output:
            shutil.copyfileobj(response, output)

    print(f"[SETUP] Extracting weights archive: {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=s4pred_dir)

    try:
        tar_path.unlink()
    except OSError:
        pass


def _run_setup(
    root_dir: Path,
    missing_required: list[str],
    missing_optional: list[str],
    *,
    weights_url: str,
    force_weights_download: bool,
) -> None:
    print(f"[SETUP] Root dir is: {root_dir}")

    # Mirror notebook setup packages and add anything currently missing.
    base_pkgs = ["biopython", "fair-esm", "pandas"]
    install_pkgs = list(dict.fromkeys(base_pkgs + missing_required + missing_optional))
    if install_pkgs:
        print("[SETUP] Installing Python packages:", ", ".join(install_pkgs))
        subprocess.run([os.sys.executable, "-m", "pip", "install", *install_pkgs], check=True)

    missing_weights = _missing_weight_files(root_dir)
    if force_weights_download or missing_weights:
        _download_and_extract_weights(root_dir, weights_url)
    else:
        print("[SETUP] S4PRED weights already present; skipping download.")

    missing_after = _missing_weight_files(root_dir)
    if missing_after:
        raise RuntimeError(
            "S4PRED weights are still missing after setup: "
            + ", ".join(str(p) for p in missing_after)
        )
    print("[SETUP] Setup complete.")


def _remap_model_paths(runtime: dict, root_dir: Path) -> None:
    """
    Remap Colab-style model paths in runtime['model_data'] to local aa_change_models.
    """
    if "model_data" not in runtime:
        return
    model_data = runtime["model_data"]
    if "model_pth_location" not in model_data.columns:
        return

    local_models_dir = root_dir / "aa_change_models"
    remapped = 0
    missing_after = []

    new_paths = []
    for p in model_data["model_pth_location"].astype(str):
        candidate = Path(p)
        if candidate.exists():
            new_paths.append(str(candidate))
            continue

        # Colab absolute path -> local ROOT_DIR + relative tail.
        colab_prefix = "/content/convergent_overlaps_aa_change/"
        if p.startswith(colab_prefix):
            rel = p[len(colab_prefix):]
            candidate = root_dir / rel
        else:
            # Fallback: keep filename and resolve under local aa_change_models.
            candidate = local_models_dir / Path(p).name

        if str(candidate) != p:
            remapped += 1
        if not candidate.exists():
            missing_after.append(str(candidate))
        new_paths.append(str(candidate))

    model_data = model_data.copy()
    model_data["model_pth_location"] = new_paths
    runtime["model_data"] = model_data

    print(f"[INFO] Model path remap count: {remapped}")
    if missing_after:
        preview = ", ".join(missing_after[:3])
        raise RuntimeError(
            "Some model files are missing after path remap. Example(s): "
            + preview
            + f"\nExpected under: {local_models_dir}"
        )


def _validate_s4pred_runtime(runtime: dict, root_dir: Path) -> None:
    s4pred_dir = root_dir / "s4pred"
    if not s4pred_dir.exists():
        raise RuntimeError(f"s4pred directory is missing: {s4pred_dir}")

    fn = runtime.get("predict_secondary_structure")
    if fn is None:
        return
    consts = {c for c in getattr(fn, "__code__", object()).co_consts or () if isinstance(c, str)}
    legacy = "/content/convergent_overlaps_aa_change/s4pred"
    if legacy in consts:
        raise RuntimeError(
            "Legacy Colab S4PRED path detected in running_s4pred.py. "
            "Expected local ROOT_DIR/s4pred. Update running_s4pred.py to use paths.ROOT_DIR."
        )


def _print_model_checkpoint(runtime: dict, length: int) -> None:
    model_data = runtime.get("model_data")
    if model_data is None:
        print("[CHKPT] model_data not available in runtime.", file=sys.__stderr__)
        return
    if "overlap_length" not in model_data.columns or "model_pth_location" not in model_data.columns:
        print("[CHKPT] model_data missing expected columns.", file=sys.__stderr__)
        return

    rows = model_data.loc[model_data["overlap_length"] == int(length), ["overlap_length", "model_pth_location"]]
    if rows.empty:
        print(f"[CHKPT] No model_data row for overlap_length={length}.", file=sys.__stderr__)
        return
    path_str = str(rows.iloc[0]["model_pth_location"])
    p = Path(path_str)
    exists = p.exists()
    size = p.stat().st_size if exists else -1
    print(
        f"[CHKPT] overlap_length={length} model_path={path_str} exists={exists} size={size}",
        file=sys.__stderr__,
    )


def _enable_debug_checkpoints(runtime: dict) -> None:
    torch_mod = runtime.get("torch")
    if torch_mod is None:
        print("[CHKPT] torch module missing in runtime; cannot hook torch.load", file=sys.__stderr__)
        return

    orig_torch_load = torch_mod.load

    def _debug_torch_load(*args, **kwargs):
        model_path = args[0] if args else kwargs.get("f", "<unknown>")
        p = Path(str(model_path))
        exists = p.exists()
        size = p.stat().st_size if exists else -1
        print(
            f"[CHKPT] torch.load path={p} exists={exists} size={size} kwargs={list(kwargs.keys())}",
            file=sys.__stderr__,
        )
        return orig_torch_load(*args, **kwargs)

    torch_mod.load = _debug_torch_load

    if "run_inference_for_models" in runtime:
        orig_run = runtime["run_inference_for_models"]

        def _debug_run_inference_for_models(*args, **kwargs):
            model_nums = args[0] if args else kwargs.get("model_numbers")
            print(
                f"[CHKPT] run_inference_for_models start model_numbers={model_nums}",
                file=sys.__stderr__,
            )
            try:
                out = orig_run(*args, **kwargs)
                shape = getattr(out, "shape", None)
                print(f"[CHKPT] run_inference_for_models success shape={shape}", file=sys.__stderr__)
                return out
            except Exception as exc:
                print(f"[CHKPT] run_inference_for_models EXCEPTION: {exc}", file=sys.__stderr__)
                traceback.print_exc(file=sys.__stderr__)
                raise

        runtime["run_inference_for_models"] = _debug_run_inference_for_models

    print("[CHKPT] Debug checkpoints enabled.", file=sys.__stderr__)


def _configure_and_validate_esm(runtime: dict, args: argparse.Namespace) -> None:
    """
    Configure ESM runtime knobs and validate ESM can produce features.
    Falls back from CUDA -> CPU in auto mode if CUDA ESM fails.
    """
    if not runtime.get("_ESM_AVAILABLE", False):
        raise RuntimeError(
            "ESM is not available in runtime (fair-esm import failed). "
            "Install with: pip install fair-esm"
        )

    torch_mod = runtime.get("torch")
    if torch_mod is None:
        raise RuntimeError("Torch not available in runtime; cannot configure ESM.")

    want_device = args.esm_device
    if want_device == "auto":
        device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    else:
        if want_device == "cuda" and not torch_mod.cuda.is_available():
            raise RuntimeError("Requested --esm-device cuda but CUDA is not available.")
        device = torch_mod.device(want_device)

    runtime["ESM_DEVICE"] = device
    runtime["ESM_BATCH_SIZE"] = max(1, int(args.esm_batch_size))
    runtime["USE_AUTOCast_FP16_IF_CUDA"] = bool(args.esm_autocast)
    runtime["USE_MODEL_FP16"] = False
    print(
        f"[INFO] ESM config: device={runtime['ESM_DEVICE']} "
        f"batch_size={runtime['ESM_BATCH_SIZE']} autocast={runtime['USE_AUTOCast_FP16_IF_CUDA']}"
    )

    test_seq = ["MSTNPKPQRITF"]
    try:
        mac = runtime["_load_esm_model"]()
        runtime["esm_features_cached"](test_seq, model_alphabet_converter=mac)
        print("[INFO] ESM preflight passed.")
        return
    except Exception as e:
        if args.esm_device == "auto" and device.type == "cuda":
            print(f"[WARN] ESM CUDA preflight failed, retrying on CPU. Error: {e}")
            runtime["ESM_DEVICE"] = torch_mod.device("cpu")
            runtime["USE_AUTOCast_FP16_IF_CUDA"] = False
            runtime["USE_MODEL_FP16"] = False
            runtime["ESM_BATCH_SIZE"] = 1
            try:
                mac = runtime["_load_esm_model"]()
                runtime["esm_features_cached"](test_seq, model_alphabet_converter=mac)
                print("[INFO] ESM preflight passed on CPU fallback.")
                return
            except Exception as e2:
                raise RuntimeError(f"ESM preflight failed on CUDA and CPU fallback. CUDA error: {e}; CPU error: {e2}") from e2
        raise RuntimeError(f"ESM preflight failed on device={device}. Error: {e}") from e


def _ensure_esm_for_row(runtime: dict, seq1: str, seq2: str) -> None:
    """
    Validate ESM features for the row's original split sequences.
    Raises if ESM cannot produce/cache features for either sequence.
    """
    s1 = seq1.strip().replace(" ", "").strip("*")
    s2 = seq2.strip().replace(" ", "").strip("*")
    concat = runtime["process_sequences"](s1, s2)
    aa1, aa2 = runtime["split_sequence"](concat)
    runtime["esm_features_cached"]([aa1, aa2])

    emb_cache = runtime.get("ESM_EMB_CACHE", {})
    cont_cache = runtime.get("ESM_CONT_CACHE", {})
    missing = []
    if aa1 not in emb_cache or aa1 not in cont_cache:
        missing.append("seq1")
    if aa2 not in emb_cache or aa2 not in cont_cache:
        missing.append("seq2")
    if missing:
        raise RuntimeError(f"ESM cache missing after feature pass for: {', '.join(missing)}")


def main() -> int:
    args = parse_args()

    root_dir = ROOT_DIR.resolve()
    working_dir = (root_dir / args.working_dir).resolve()
    excel_path = Path(args.excel_path)
    if not excel_path.is_absolute():
        excel_path = (root_dir / excel_path).resolve()

    if not excel_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {excel_path}")

    missing_required, missing_optional = _check_dependencies()
    missing_weights = _missing_weight_files(root_dir)

    needs_setup = bool(missing_required or missing_weights)
    if args.auto_setup and needs_setup and not args.setup_only and not args.setup:
        print("[INFO] Missing requirements detected; running automatic setup.")
        _run_setup(
            root_dir,
            missing_required,
            missing_optional,
            weights_url=args.weights_url,
            force_weights_download=bool(args.force_weights_download),
        )
        missing_required, missing_optional = _check_dependencies()
        missing_weights = _missing_weight_files(root_dir)

    if args.setup or args.setup_only:
        _run_setup(
            root_dir,
            missing_required,
            missing_optional,
            weights_url=args.weights_url,
            force_weights_download=bool(args.force_weights_download),
        )
        missing_required, missing_optional = _check_dependencies()
        missing_weights = _missing_weight_files(root_dir)

    if args.setup_only:
        return 0

    if missing_required:
        pkg_line = " ".join(missing_required)
        raise RuntimeError(
            "Missing required dependencies for notebook runtime: "
            + ", ".join(missing_required)
            + f"\nInstall with:\n  pip install {pkg_line}"
        )
    if missing_optional:
        print(
            "[WARN] Optional packages not found (needed only for ESM/contact-map paths): "
            + ", ".join(missing_optional)
        )
    if missing_weights:
        weights_dir = root_dir / "s4pred" / "weights"
        raise RuntimeError(
            "Missing S4PRED model weights in: "
            + str(weights_dir)
            + "\nRun setup with:\n  python run_overlap_cli.py --setup --check-only"
        )

    if args.check_only:
        print("[OK] Dependency and input checks passed.")
        return 0

    model_numbers = _parse_overlap_lengths(args.overlap_length_selected)
    print(f"[INFO] model_numbers -> {model_numbers}")
    if args.notebook_path:
        print(
            "[WARN] --notebook-path is accepted for compatibility but ignored in modular CLI."
        )
    print("[INFO] Loading modular runtime from overlap_runtime.notebook_api")

    os.makedirs(working_dir, exist_ok=True)

    runtime = create_runtime(
        root_dir,
        esm_device=args.esm_device,
        esm_batch_size=args.esm_batch_size,
        esm_autocast=args.esm_autocast,
        require_esm=args.require_esm,
        reset_caches_per_row=args.reset_caches_per_row,
        debug_checkpoints=args.debug_checkpoints,
    )

    if "load_pairs_table" not in runtime or "optimize_pair_and_save" not in runtime:
        raise RuntimeError(
            "Missing required runtime functions: load_pairs_table or optimize_pair_and_save."
        )

    _remap_model_paths(runtime, root_dir)
    _validate_s4pred_runtime(runtime, root_dir)
    _configure_and_validate_esm(runtime, args)
    if args.debug_checkpoints:
        _enable_debug_checkpoints(runtime)

    runtime["RESET_CACHES_PER_ROW"] = bool(args.reset_caches_per_row)
    print(f"[INFO] RESET_CACHES_PER_ROW -> {runtime['RESET_CACHES_PER_ROW']}")

    print("[INFO] Loading pairs table...")
    pairs_df = runtime["load_pairs_table"](str(excel_path))
    pairs_df = _sanitize_pairs_df(pairs_df)

    s_row = max(1, int(args.start_row))
    e_row = int(args.end_row) if args.end_row > 0 else None
    if e_row is not None and e_row < s_row:
        raise ValueError(
            f"Invalid row range: start_row={s_row}, end_row={e_row}. "
            "Use --end-row 0 to process to the end."
        )
    start_idx = s_row - 1
    df_slice = pairs_df.iloc[start_idx:e_row] if e_row is not None else pairs_df.iloc[start_idx:]
    print(f"[INFO] Will process {len(df_slice)} rows (rows {s_row} to {'end' if e_row is None else e_row}).")

    override_weights = {
        "ss": float(args.ss_w),
        "sub": float(args.sub_w),
        "aln": float(args.aln_w),
        "esm": float(args.esm_w),
    }
    if args.normalize_override_weights:
        override_weights = _normalize_weights(override_weights)
    print("[INFO] Override weights (normalized if requested):", override_weights)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    overall_base = working_dir / f"convergent_overlap_run_{timestamp}"
    os.makedirs(overall_base, exist_ok=True)

    all_out_files: list[str] = []

    for length in model_numbers:
        print("\n" + "#" * 80)
        print(f"[INFO] Starting processing for overlap_length = {length}")
        length_dir = overall_base / f"len_{length}"
        os.makedirs(length_dir, exist_ok=True)

        runtime["CURRENT_OVERLAP_LENGTH"] = int(length)
        runtime["model_numbers"] = [int(length)]
        runtime["CURRENT_OUTPUT_DIR"] = str(length_dir)
        runtime["working_directory"] = str(length_dir)
        runtime["working_dir"] = str(length_dir)
        print(f"[INFO] Active model_numbers for this length: {runtime['model_numbers']}")
        if args.debug_checkpoints:
            _print_model_checkpoint(runtime, int(length))

        for i, row in df_slice.iterrows():
            row_display_num = i + 1
            print("\n" + "-" * 60)
            print(f"[ROW] length={length} | Processing row {row_display_num} (index {i}) ...")

            seq1 = row.get("aa_seq_1")
            seq2 = row.get("aa_seq_2")
            seq1_bracket = row.get("aa_seq_1_brackets", "")
            seq2_bracket = row.get("aa_seq_2_brackets", "")

            if not seq1_bracket or str(seq1_bracket).strip() == "" or str(seq1_bracket).lower() in ("nan", "none"):
                seq1_bracket = seq1
            if not seq2_bracket or str(seq2_bracket).strip() == "" or str(seq2_bracket).lower() in ("nan", "none"):
                seq2_bracket = seq2

            if not isinstance(seq1, str) or not isinstance(seq2, str) or not seq1 or not seq2:
                print(f"[WARN] Row {row_display_num}: missing/invalid sequences, skipping.")
                continue

            if args.require_esm:
                try:
                    _ensure_esm_for_row(runtime, seq1, seq2)
                    print(f"[INFO] ESM row precheck passed for row {row_display_num}.")
                except Exception as e:
                    raise RuntimeError(
                        f"ESM is required but failed before optimization for row {row_display_num}: {e}"
                    ) from e

            if args.use_row_weights:
                try:
                    row_weights = {
                        "ss": float(row.get("ss", args.ss_w)),
                        "sub": float(row.get("sub", args.sub_w)),
                        "aln": float(row.get("aln", args.aln_w)),
                        "esm": float(row.get("esm", args.esm_w)),
                    }
                    total = sum(row_weights.values())
                    if total > 0:
                        row_weights = {k: v / total for k, v in row_weights.items()}
                    joined = ", ".join([f"{k}={v:.3f}" for k, v in row_weights.items()])
                    print(f"Row {row_display_num} weights: {joined}")
                except Exception as exc:
                    print(f"[WARN] Row {row_display_num}: invalid weights, falling back to override. {exc}")
                    row_weights = override_weights
            else:
                row_weights = override_weights

            try:
                if args.display_mode in ("tui", "radar"):
                    state = TuiState(
                        overlap_length=int(length),
                        row_display_num=int(row_display_num),
                        plot_metrics=args.plot_metric,
                        display_mode=args.display_mode,
                        display_style=args.display_style,
                    )
                    state.render(lambda s: sys.stdout.write(s) or sys.stdout.flush())

                    def run_optimize():
                        return runtime["optimize_pair_and_save"](
                            seq1,
                            seq2,
                            seq1_bracket,
                            seq2_bracket,
                            row_display_num,
                            row_weights,
                            first_pass_iterations=args.first_pass_iters,
                            second_pass_iterations=args.second_pass_iters,
                        )

                    out_path = _run_with_print_intercept(
                        run_optimize,
                        lambda line: state.handle_line(
                            line, lambda s: sys.stdout.write(s) or sys.stdout.flush()
                        ),
                        stderr_on_line=lambda line: state.handle_stderr_line(
                            line, lambda s: sys.stdout.write(s) or sys.stdout.flush()
                        ),
                    )
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    sys.stdout.write(state.final_summary_box(out_path) + "\n")
                    sys.stdout.flush()
                else:
                    out_path = runtime["optimize_pair_and_save"](
                        seq1,
                        seq2,
                        seq1_bracket,
                        seq2_bracket,
                        row_display_num,
                        row_weights,
                        first_pass_iterations=args.first_pass_iters,
                        second_pass_iterations=args.second_pass_iters,
                    )
                if out_path:
                    all_out_files.append(out_path)
                    print(f"[OK] Wrote: {out_path}")
            except Exception:
                print(f"[ERROR] Row {row_display_num}: exception during optimize_pair_and_save:")
                traceback.print_exc()

    if args.archive:
        try:
            archive_base = working_dir / f"convergent_overlap_outputs_{timestamp}"
            overall_archive = shutil.make_archive(str(archive_base), "zip", root_dir=str(overall_base))
            print("\n[DONE] Created overall archive:", overall_archive)
        except Exception:
            print("\n[WARN] Failed to create overall archive:")
            traceback.print_exc()

    print("\n[FINISHED] Run complete.")
    print(f"[INFO] Individual output files generated: {len(all_out_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
