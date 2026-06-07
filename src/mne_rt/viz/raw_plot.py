"""Real-time raw M/EEG channel viewer.

Dark-themed scrolling raw signal display built on PyQt6 + pyqtgraph.

Classes
-------
RawPlot
    Scrolling multi-channel raw M/EEG signal viewer.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Per-channel trace colours — 20 visually distinct hues for a dark background
# ---------------------------------------------------------------------------

_TRACE_COLORS = [
    "#4fc3f7",  # sky blue
    "#ef9a9a",  # salmon
    "#a5d6a7",  # mint green
    "#fff176",  # yellow
    "#ffab91",  # light orange
    "#ce93d8",  # lavender
    "#80cbc4",  # teal
    "#ffcc80",  # peach
    "#80deea",  # cyan
    "#b39ddb",  # purple
    "#f48fb1",  # pink
    "#c5e1a5",  # lime green
    "#ffd54f",  # amber
    "#81d4fa",  # light blue
    "#dce775",  # yellow-green
    "#ff8a65",  # deep orange
    "#90caf9",  # blue
    "#e6ee9c",  # lime
    "#bcaaa4",  # warm grey
    "#ffe082",  # light amber
]

_TIME_WINDOW_OPTIONS = [5, 10, 20, 30, 60]

_QSS = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: "Segoe UI", sans-serif;
}
QPushButton {
    background-color: #16213e;
    color: #d0d0e8;
    border: 1px solid #0f3460;
    border-radius: 5px;
    padding: 5px 10px;
    font-size: 12px;
}
QPushButton:hover  { background-color: #0f3460; }
QPushButton:pressed { background-color: #533483; }
QPushButton:checked {
    background-color: #533483;
    border-color: #a882dd;
    color: #ffffff;
}
QComboBox {
    background-color: #16213e;
    color: #d0d0e8;
    border: 1px solid #0f3460;
    border-radius: 4px;
    padding: 3px 6px;
}
QComboBox QAbstractItemView {
    background-color: #16213e;
    color: #d0d0e8;
    selection-background-color: #0f3460;
}
QDoubleSpinBox, QSpinBox {
    background-color: #16213e;
    color: #d0d0e8;
    border: 1px solid #0f3460;
    border-radius: 4px;
    padding: 2px 4px;
}
QGroupBox {
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 6px;
    font-weight: bold;
    font-size: 11px;
    color: #8888aa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QLabel  { color: #b0b0c8; font-size: 11px; }
QCheckBox { color: #b0b0c8; font-size: 11px; }
QScrollArea { border: none; }
QStatusBar { background-color: #0d0d1a; color: #606080; font-size: 10px; }
QScrollBar:vertical {
    background-color: #0d0d1a;
    width: 14px;
    border: none;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background-color: #2a2a4a;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover { background-color: #404060; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background-color: #0d0d1a;
}
"""


# ---------------------------------------------------------------------------
# Artifact-corrector adapter classes
# ---------------------------------------------------------------------------

class _GEDAIWrapper:
    """Makes GEDAIDenoiser present the .transform(data) interface."""

    def __init__(self, gedai, n_noise: int) -> None:
        self._g = gedai
        self._n = n_noise

    def transform(self, data: np.ndarray) -> np.ndarray:
        idx = self._g.find_noise_components(self._n)
        return self._g.denoise(data, idx)


class _ORICAWrapper:
    """Online ORICA: adapts W on every chunk, suppresses highest-power ICs."""

    def __init__(self, orica, n_remove: int) -> None:
        self._o = orica
        self._n = max(1, int(n_remove))

    def transform(self, data: np.ndarray) -> np.ndarray:
        self._o.partial_fit(data)
        S = self._o.transform(data)
        rms = np.sqrt(np.mean(S ** 2, axis=1))
        noise_idx = np.argsort(rms)[-self._n:].tolist()
        S_clean = S.copy()
        S_clean[noise_idx] = 0.0
        return self._o.inverse_transform(S_clean)


# ---------------------------------------------------------------------------
# Event filter: intercepts wheel events on the plot viewport
# ---------------------------------------------------------------------------

class _WheelFilter(QObject):
    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self._cb = callback

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            self._cb(1 if event.angleDelta().y() > 0 else -1)
            return True
        return False


# ---------------------------------------------------------------------------
# RawPlot
# ---------------------------------------------------------------------------

class RawPlot(QMainWindow):
    """Scrolling raw M/EEG channel viewer.

    Displays all channels stacked vertically in a dark-themed
    :class:`pyqtgraph.PlotWidget`, colour-coded by channel index.  Channels
    are scrollable via the mouse wheel on the plot or the scrollbar on the
    right edge of the signal area.  Right-clicking any channel name on the
    Y-axis immediately opens an MNE sensor-position plot for that channel.

    Parameters
    ----------
    ch_names : list of str
        Channel names.  One row is shown per channel.
    sfreq : float
        Sampling frequency in Hz.
    time_window : float, default 10.0
        Visible time range in seconds at startup.
    n_shown : int, default 20
        Number of channels visible simultaneously.
    scale_uv : float, default 100.0
        Amplitude scale in µV.  A signal of this peak amplitude occupies
        half the per-channel row height.  For MEG (Tesla) pass
        ``scale_uv=1e-6`` (i.e., 1 pT per half-row).
    info : mne.Info | None, default None
        If provided: used to resolve channel types, apply SSP projectors,
        and show sensor positions on right-click.
    verbose : bool | str | None, default None
        Verbosity level.  See :func:`~mne_rt._logging.set_log_level`.

    See Also
    --------
    mne_rt.viz.NFPlot : Scrolling NF feature monitor.
    mne_rt.RTStream.record_main : Drives the raw display from the NF loop.

    Notes
    -----
    The control panel (right sidebar) provides:

    * **Playback** — pause/resume, clear buffer, screenshot.
    * **Amplitude** — ÷2 / ×2 scale buttons with a live readout.
    * **Display** — time-window selector, grid toggle, DC-removal toggle.
    * **Filter** — online causal bandpass/highpass/lowpass/notch (scipy).
      Applied to new data from the moment "Apply filter" is clicked;
      data already in the buffer is not retroactively filtered.
    * **Artifact Correction** — LMS adaptive filter or ASR, applied from
      the moment "Apply from now" is clicked.  LMS requires a reference
      channel; ASR calibrates on the current buffer content.
    * **SSP** — shown when ``info`` contains projectors; applied to new
      data from the moment the checkbox is ticked.

    Use the mouse wheel on the signal area or the vertical scrollbar to the
    right of the traces to page through channels.  Right-click any channel
    label on the Y-axis to open its sensor-position diagram.

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        ch_names: list[str],
        sfreq: float,
        time_window: float = 10.0,
        n_shown: int = 20,
        scale_uv: float = 100.0,
        info=None,
        verbose=None,
    ) -> None:
        from mne_rt._logging import set_log_level
        set_log_level(verbose)
        super().__init__()

        self._ch_names = list(ch_names)
        self._n_ch = len(ch_names)
        self._sfreq = float(sfreq)
        self._time_window = float(time_window)
        self._n_shown = min(int(n_shown), self._n_ch)
        self._scale = float(scale_uv) * 1e-6
        self._info = info
        self._page_start = 0
        self._paused = False
        self._dc_remove = False

        # Online causal filter — SOS coefficients + per-channel state vector
        # Both reset to None when the filter is changed or the buffer is cleared.
        self._filter_sos = None      # ndarray (n_sections, 6) or None
        self._filter_zi  = None      # ndarray (n_ch, n_sections, 2) or None

        # SSP projector matrix (n_ch × n_ch), applied to new incoming chunks
        self._ssp_proj = None

        # Artifact corrector — object with a .transform(data) → data method
        self._corrector = None

        # Re-referencing — applied in push() after the corrector
        self._reref_type: str = "none"   # "none", "average", "mastoid", "channel"
        self._reref_idx: int = 0         # index of the single reference channel
        self._reref_idxs: list[int] = [] # indices for multi-channel references

        # Per-channel colours and types
        self._ch_types: list[str] = []
        self._ch_colors: list[str] = []
        self._resolve_colors()

        n_pts = max(int(sfreq * time_window), 30)
        self._time_axis = np.linspace(0.0, time_window, n_pts)
        self._buf = np.zeros((self._n_ch, n_pts))

        pg.setConfigOptions(antialias=True, foreground="#c0c0d8", background="#0d0d1a")
        self._build_ui()
        self.setWindowTitle("MNE-RT — Raw")
        self.resize(1500, 720)

    # ------------------------------------------------------------------
    # Colour resolution
    # ------------------------------------------------------------------

    def _resolve_colors(self) -> None:
        if self._info is not None:
            try:
                import mne
                self._ch_types = [
                    mne.channel_type(self._info, i) for i in range(self._n_ch)
                ]
            except Exception:
                self._ch_types = ["misc"] * self._n_ch
        else:
            self._ch_types = ["misc"] * self._n_ch

        # Distinct colour per channel by cycling the palette; ensures adjacent
        # channels are always distinguishable even when all share one type.
        self._ch_colors = [
            _TRACE_COLORS[i % len(_TRACE_COLORS)] for i in range(self._n_ch)
        ]

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.setStyleSheet(_QSS)
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 4, 8)
        root.setSpacing(0)

        root.addWidget(self._build_plot_widget(), stretch=5)

        # Channel scrollbar — between the plot canvas and the control panel
        self._ch_scroll = QScrollBar(Qt.Orientation.Vertical)
        self._ch_scroll.setRange(0, max(0, self._n_ch - self._n_shown))
        self._ch_scroll.setPageStep(self._n_shown)
        self._ch_scroll.setSingleStep(1)
        self._ch_scroll.setFixedWidth(14)
        self._ch_scroll.valueChanged.connect(
            lambda v: self._set_page_start(v, source="scrollbar")
        )
        root.addWidget(self._ch_scroll)

        root.addSpacing(4)
        root.addWidget(self._build_control_panel(), stretch=0)

        self._status = self.statusBar()
        self._status.showMessage("Waiting for data …")

    def _build_plot_widget(self) -> pg.GraphicsLayoutWidget:
        glw = pg.GraphicsLayoutWidget()
        glw.setBackground("#0d0d1a")

        self._pi = glw.addPlot(row=0, col=0)
        self._pi.setMouseEnabled(x=False, y=False)
        vb = self._pi.getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.setMenuEnabled(False)
        self._pi.showGrid(x=True, y=False, alpha=0.25)

        for ax_name in ("left", "bottom"):
            ax = self._pi.getAxis(ax_name)
            ax.setPen(pg.mkPen("#303050"))
            ax.setTextPen(pg.mkPen("#9090aa"))
        self._pi.getAxis("left").setWidth(80)
        self._pi.setLabel("bottom", "Time", units="s", color="#9090aa")
        self._pi.setXRange(0.0, self._time_window, padding=0.01)
        self._pi.setYRange(-0.5, self._n_shown - 0.5, padding=0)

        self._curves: list[pg.PlotCurveItem] = [
            pg.PlotCurveItem(pen=pg.mkPen(color=_TRACE_COLORS[0], width=1))
            for _ in range(self._n_shown)
        ]
        for c in self._curves:
            self._pi.addItem(c)

        sep_pen = pg.mkPen(color=(70, 70, 110, 55), width=1, style=Qt.PenStyle.DotLine)
        for i in range(self._n_shown):
            self._pi.addItem(pg.InfiniteLine(pos=i, angle=0, pen=sep_pen))

        self._update_tick_labels()

        self._wheel_filter = _WheelFilter(self._on_plot_wheel, self)
        glw.viewport().installEventFilter(self._wheel_filter)

        # Right-click on the Y-axis tick label → open sensor position plot
        glw.scene().sigMouseClicked.connect(self._on_scene_clicked)

        self._glw = glw
        return glw

    def _build_control_panel(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(230)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)

        layout.addWidget(self._grp_playback())
        layout.addWidget(self._grp_amplitude())
        layout.addWidget(self._grp_display())
        layout.addWidget(self._grp_filter())
        layout.addWidget(self._grp_reref())
        layout.addWidget(self._grp_correction())
        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    # --- sidebar groups ---

    def _grp_playback(self) -> QGroupBox:
        grp = QGroupBox("Playback")
        lay = QVBoxLayout(grp)
        self._btn_pause = QPushButton("⏸  Pause")
        self._btn_pause.setCheckable(True)
        self._btn_pause.clicked.connect(self._toggle_pause)
        btn_clear = QPushButton("⟳  Clear")
        btn_clear.clicked.connect(self._clear)
        btn_shot = QPushButton("📷  Screenshot")
        btn_shot.clicked.connect(self._screenshot)
        for w in (self._btn_pause, btn_clear, btn_shot):
            lay.addWidget(w)
        return grp

    def _grp_amplitude(self) -> QGroupBox:
        grp = QGroupBox("Amplitude")
        lay = QVBoxLayout(grp)
        self._scale_lbl = QLabel(self._fmt_scale())
        self._scale_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scale_lbl.setStyleSheet("color:#7ec8e3; font-size:12px; font-weight:bold;")
        row = QHBoxLayout()
        btn_dn = QPushButton("÷2")
        btn_up = QPushButton("×2")
        for btn in (btn_dn, btn_up):
            btn.setFixedSize(42, 26)
        btn_dn.clicked.connect(self._scale_down)
        btn_up.clicked.connect(self._scale_up)
        row.addStretch()
        row.addWidget(btn_dn)
        row.addWidget(btn_up)
        row.addStretch()
        lay.addWidget(self._scale_lbl)
        lay.addLayout(row)
        return grp

    def _grp_display(self) -> QGroupBox:
        grp = QGroupBox("Display")
        lay = QVBoxLayout(grp)
        row = QHBoxLayout()
        row.addWidget(QLabel("Time window:"))
        self._cmb_tw = QComboBox()
        for secs in _TIME_WINDOW_OPTIONS:
            self._cmb_tw.addItem(f"{secs} s", secs)
        best = min(_TIME_WINDOW_OPTIONS, key=lambda s: abs(s - self._time_window))
        self._cmb_tw.setCurrentIndex(_TIME_WINDOW_OPTIONS.index(best))
        self._cmb_tw.currentIndexChanged.connect(self._change_time_window)
        row.addWidget(self._cmb_tw)
        lay.addLayout(row)
        chk_grid = QCheckBox("Show grid")
        chk_grid.setChecked(True)
        chk_grid.toggled.connect(
            lambda on: self._pi.showGrid(x=on, y=False, alpha=0.25 if on else 0.0)
        )
        lay.addWidget(chk_grid)
        chk_dc = QCheckBox("Remove DC")
        chk_dc.setChecked(False)
        chk_dc.toggled.connect(self._set_dc_remove)
        lay.addWidget(chk_dc)
        return grp

    def _grp_filter(self) -> QGroupBox:
        grp = QGroupBox("Filter")
        lay = QVBoxLayout(grp)
        lay.setSpacing(5)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self._cmb_filter = QComboBox()
        self._cmb_filter.addItems(["None", "High-pass", "Low-pass", "Band-pass", "Notch"])
        self._cmb_filter.currentIndexChanged.connect(self._on_filter_type_changed)
        type_row.addWidget(self._cmb_filter)
        lay.addLayout(type_row)

        self._flo_row = QWidget()
        flo_lay = QHBoxLayout(self._flo_row)
        flo_lay.setContentsMargins(0, 0, 0, 0)
        flo_lbl = QLabel("Lo cut:")
        flo_lbl.setFixedWidth(44)
        self._flo_spin = QDoubleSpinBox()
        self._flo_spin.setRange(0.1, 500.0)
        self._flo_spin.setValue(1.0)
        self._flo_spin.setSuffix(" Hz")
        self._flo_spin.setDecimals(1)
        flo_lay.addWidget(flo_lbl)
        flo_lay.addWidget(self._flo_spin)
        self._flo_row.setVisible(False)
        lay.addWidget(self._flo_row)

        self._fhi_row = QWidget()
        fhi_lay = QHBoxLayout(self._fhi_row)
        fhi_lay.setContentsMargins(0, 0, 0, 0)
        fhi_lbl = QLabel("Hi cut:")
        fhi_lbl.setFixedWidth(44)
        self._fhi_spin = QDoubleSpinBox()
        self._fhi_spin.setRange(0.1, 500.0)
        self._fhi_spin.setValue(40.0)
        self._fhi_spin.setSuffix(" Hz")
        self._fhi_spin.setDecimals(1)
        fhi_lay.addWidget(fhi_lbl)
        fhi_lay.addWidget(self._fhi_spin)
        self._fhi_row.setVisible(False)
        lay.addWidget(self._fhi_row)

        self._fnotch_row = QWidget()
        fnotch_lay = QHBoxLayout(self._fnotch_row)
        fnotch_lay.setContentsMargins(0, 0, 0, 0)
        fnotch_lbl = QLabel("Freq:")
        fnotch_lbl.setFixedWidth(44)
        self._fnotch_spin = QDoubleSpinBox()
        self._fnotch_spin.setRange(1.0, 500.0)
        self._fnotch_spin.setValue(50.0)
        self._fnotch_spin.setSuffix(" Hz")
        self._fnotch_spin.setDecimals(1)
        fnotch_lay.addWidget(fnotch_lbl)
        fnotch_lay.addWidget(self._fnotch_spin)
        self._fnotch_row.setVisible(False)
        lay.addWidget(self._fnotch_row)

        btn_apply = QPushButton("Apply filter")
        btn_apply.setStyleSheet(
            "background:#132744; color:#80d8ff; border:1px solid #2a6090;"
            "border-radius:4px; padding:4px; font-size:11px;"
        )
        btn_apply.clicked.connect(self._apply_filter_settings)
        lay.addWidget(btn_apply)

        self._filter_status = QLabel("○  No filter")
        self._filter_status.setStyleSheet("color:#505070; font-size:10px;")
        self._filter_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._filter_status)

        # SSP — shown only when info has projectors
        if self._info is not None:
            try:
                projs = self._info.get("projs", [])
                if projs:
                    n = len(projs)
                    self._ssp_chk = QCheckBox(f"Apply SSP  ({n} proj.)")
                    self._ssp_chk.setChecked(False)
                    self._ssp_chk.toggled.connect(self._toggle_ssp)
                    lay.addWidget(self._ssp_chk)
            except Exception:
                pass

        return grp

    def _grp_reref(self) -> QGroupBox:
        grp = QGroupBox("Re-reference")
        lay = QVBoxLayout(grp)
        lay.setSpacing(5)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Ref:"))
        self._cmb_reref = QComboBox()
        self._cmb_reref.addItems(
            ["None", "Average", "Mastoid (TP9/TP10)", "Linked Mastoid", "Channel"]
        )
        self._cmb_reref.currentIndexChanged.connect(self._on_reref_type_changed)
        type_row.addWidget(self._cmb_reref)
        lay.addLayout(type_row)

        # Single-channel reference selector (shown only for "Channel")
        self._reref_ch_row = QWidget()
        reref_ch_lay = QHBoxLayout(self._reref_ch_row)
        reref_ch_lay.setContentsMargins(0, 0, 0, 0)
        reref_ch_lbl = QLabel("Channel:")
        reref_ch_lbl.setFixedWidth(54)
        self._reref_ch_cmb = QComboBox()
        self._reref_ch_cmb.addItems(self._ch_names)
        self._reref_ch_cmb.setMaxVisibleItems(12)
        reref_ch_lay.addWidget(reref_ch_lbl)
        reref_ch_lay.addWidget(self._reref_ch_cmb)
        self._reref_ch_row.setVisible(False)
        lay.addWidget(self._reref_ch_row)

        btn_reref = QPushButton("Apply from now")
        btn_reref.setStyleSheet(
            "background:#132744; color:#80d8ff; border:1px solid #2a6090;"
            "border-radius:4px; padding:4px; font-size:11px;"
        )
        btn_reref.clicked.connect(self._apply_reref_settings)
        lay.addWidget(btn_reref)

        self._reref_status = QLabel("○  No re-reference")
        self._reref_status.setStyleSheet("color:#505070; font-size:10px;")
        self._reref_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._reref_status)

        return grp

    def _grp_correction(self) -> QGroupBox:
        grp = QGroupBox("Artifact Correction")
        lay = QVBoxLayout(grp)
        lay.setSpacing(5)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Method:"))
        self._cmb_corr = QComboBox()
        self._cmb_corr.addItems(["None", "LMS", "ASR", "GEDAI", "ORICA", "Maxwell"])
        self._cmb_corr.currentIndexChanged.connect(self._on_corr_type_changed)
        type_row.addWidget(self._cmb_corr)
        lay.addLayout(type_row)

        # LMS — reference channel by name
        self._lms_ref_row = QWidget()
        lms_lay = QHBoxLayout(self._lms_ref_row)
        lms_lay.setContentsMargins(0, 0, 0, 0)
        lms_lbl = QLabel("Ref ch:")
        lms_lbl.setFixedWidth(44)
        self._lms_ref_cmb = QComboBox()
        self._lms_ref_cmb.addItems(self._ch_names)
        self._lms_ref_cmb.setMaxVisibleItems(12)
        self._lms_ref_cmb.setToolTip(
            "Reference (EOG/ECG) channel used by LMS. "
            "Pick the channel that best captures the artifact."
        )
        lms_lay.addWidget(lms_lbl)
        lms_lay.addWidget(self._lms_ref_cmb)
        self._lms_ref_row.setVisible(False)
        lay.addWidget(self._lms_ref_row)

        # ASR — cutoff threshold
        self._asr_cut_row = QWidget()
        asr_lay = QHBoxLayout(self._asr_cut_row)
        asr_lay.setContentsMargins(0, 0, 0, 0)
        asr_lbl = QLabel("Cutoff σ:")
        asr_lbl.setFixedWidth(52)
        self._asr_cut_spin = QDoubleSpinBox()
        self._asr_cut_spin.setRange(2.0, 20.0)
        self._asr_cut_spin.setValue(5.0)
        self._asr_cut_spin.setDecimals(1)
        self._asr_cut_spin.setToolTip(
            "ASR rejection threshold in multiples of clean-data RMS.\n"
            "Lower = more aggressive (3–4); higher = more conservative (8–10)."
        )
        asr_lay.addWidget(asr_lbl)
        asr_lay.addWidget(self._asr_cut_spin)
        self._asr_cut_row.setVisible(False)
        lay.addWidget(self._asr_cut_row)

        # GEDAI — target band + number of noise components to remove
        self._gedai_row = QWidget()
        gedai_v = QVBoxLayout(self._gedai_row)
        gedai_v.setContentsMargins(0, 0, 0, 0)
        gedai_v.setSpacing(3)

        band_row = QHBoxLayout()
        band_row.addWidget(QLabel("Band:"))
        self._gedai_lo_spin = QDoubleSpinBox()
        self._gedai_lo_spin.setRange(0.1, 200.0)
        self._gedai_lo_spin.setValue(1.0)
        self._gedai_lo_spin.setSuffix(" Hz")
        self._gedai_lo_spin.setDecimals(1)
        self._gedai_lo_spin.setFixedWidth(70)
        band_row.addWidget(self._gedai_lo_spin)
        band_row.addWidget(QLabel("–"))
        self._gedai_hi_spin = QDoubleSpinBox()
        self._gedai_hi_spin.setRange(1.0, 500.0)
        self._gedai_hi_spin.setValue(40.0)
        self._gedai_hi_spin.setSuffix(" Hz")
        self._gedai_hi_spin.setDecimals(1)
        self._gedai_hi_spin.setFixedWidth(70)
        band_row.addWidget(self._gedai_hi_spin)
        gedai_v.addLayout(band_row)

        noise_row = QHBoxLayout()
        noise_row.addWidget(QLabel("Remove:"))
        self._gedai_noise_spin = QSpinBox()
        self._gedai_noise_spin.setRange(1, max(1, self._n_ch - 1))
        self._gedai_noise_spin.setValue(1)
        self._gedai_noise_spin.setSuffix(" comps")
        self._gedai_noise_spin.setToolTip(
            "Number of lowest-eigenvalue GEDAI components to suppress.\n"
            "These capture the least band-specific (artifact) activity."
        )
        noise_row.addWidget(self._gedai_noise_spin)
        gedai_v.addLayout(noise_row)

        self._gedai_row.setVisible(False)
        lay.addWidget(self._gedai_row)

        # ORICA — number of ICs to suppress
        self._orica_row = QWidget()
        orica_lay = QHBoxLayout(self._orica_row)
        orica_lay.setContentsMargins(0, 0, 0, 0)
        orica_lbl = QLabel("Remove:")
        orica_lbl.setFixedWidth(50)
        self._orica_n_spin = QSpinBox()
        self._orica_n_spin.setRange(1, max(1, self._n_ch - 1))
        self._orica_n_spin.setValue(1)
        self._orica_n_spin.setSuffix(" ICs")
        self._orica_n_spin.setToolTip(
            "Number of highest-power ICs to suppress per chunk.\n"
            "Artifacts are typically the highest-power independent components."
        )
        orica_lay.addWidget(orica_lbl)
        orica_lay.addWidget(self._orica_n_spin)
        self._orica_row.setVisible(False)
        lay.addWidget(self._orica_row)

        btn_apply_corr = QPushButton("Apply from now")
        btn_apply_corr.setStyleSheet(
            "background:#132744; color:#80d8ff; border:1px solid #2a6090;"
            "border-radius:4px; padding:4px; font-size:11px;"
        )
        btn_apply_corr.clicked.connect(self._apply_correction)
        lay.addWidget(btn_apply_corr)

        self._corr_status = QLabel("○  No correction")
        self._corr_status.setStyleSheet("color:#505070; font-size:10px;")
        self._corr_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._corr_status)

        return grp

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fmt_scale(self) -> str:
        uv = self._scale * 1e6
        if uv >= 1000:
            return f"{uv / 1000:.4g} mV / row"
        if uv >= 1:
            return f"{uv:.4g} µV / row"
        return f"{uv * 1000:.4g} nV / row"

    def _update_tick_labels(self) -> None:
        end = min(self._page_start + self._n_shown, self._n_ch)
        n_actual = end - self._page_start
        ticks = [
            (n_actual - 1 - i, self._ch_names[self._page_start + i])
            for i in range(n_actual)
        ]
        self._pi.getAxis("left").setTicks([ticks, []])

    def _set_page_start(self, new_start: int, source: str = "other") -> None:
        new_start = max(0, min(new_start, max(0, self._n_ch - self._n_shown)))
        if new_start == self._page_start:
            return
        self._page_start = new_start
        self._update_tick_labels()
        if source != "scrollbar":
            self._ch_scroll.blockSignals(True)
            self._ch_scroll.setValue(new_start)
            self._ch_scroll.blockSignals(False)
        self._redraw()

    def _reset_online_state(self) -> None:
        """Reset filter and corrector state when the buffer is cleared."""
        self._filter_zi = None

    # ------------------------------------------------------------------
    # Callbacks — scroll / wheel
    # ------------------------------------------------------------------

    def _on_plot_wheel(self, direction: int) -> None:
        step = max(1, self._n_shown // 4)
        self._set_page_start(self._page_start - direction * step, source="wheel")

    # ------------------------------------------------------------------
    # Callbacks — Y-axis right-click → sensor location
    # ------------------------------------------------------------------

    def _on_scene_clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.RightButton:
            return
        pos = event.scenePos()
        axis = self._pi.getAxis("left")
        if not axis.sceneBoundingRect().contains(pos):
            return
        y_val = self._pi.getViewBox().mapSceneToView(pos).y()
        end = min(self._page_start + self._n_shown, self._n_ch)
        n_actual = end - self._page_start
        vis_idx = int(round(n_actual - 1 - y_val))
        if 0 <= vis_idx < n_actual:
            ch_idx = self._page_start + vis_idx
            self._show_channel_location(self._ch_names[ch_idx])

    def _show_channel_location(self, ch_name: str) -> None:
        if self._info is None:
            self._status.showMessage(f"No Info available — cannot show position for {ch_name}")
            return
        try:
            import mne
            import matplotlib.pyplot as plt
            fig = mne.viz.plot_sensors(
                self._info,
                show_names=True,
                title=f"Sensor position — {ch_name}",
                show=False,
            )
            # Highlight the right-clicked channel in red
            for ax in fig.axes:
                for txt in ax.texts:
                    if txt.get_text() == ch_name:
                        txt.set_color("#cc0000")
                        txt.set_fontsize(10)
                        txt.set_fontweight("bold")
            plt.show(block=False)
        except Exception as exc:
            self._status.showMessage(f"Could not show sensor position: {exc}")

    # ------------------------------------------------------------------
    # Callbacks — playback / display
    # ------------------------------------------------------------------

    def _toggle_pause(self, checked: bool) -> None:
        self._paused = checked
        self._btn_pause.setText("▶  Resume" if checked else "⏸  Pause")

    def _clear(self) -> None:
        self._buf[:] = 0.0
        self._reset_online_state()
        self._redraw()

    def _screenshot(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = str(Path.home() / f"raw_plot_{ts}.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", default, "PNG Image (*.png)"
        )
        if not path:
            return
        exp = pg.exporters.ImageExporter(self._glw.scene())
        exp.parameters()["width"] = 1920
        exp.export(path)

    def _scale_up(self) -> None:
        self._scale *= 2.0
        self._scale_lbl.setText(self._fmt_scale())
        self._redraw()

    def _scale_down(self) -> None:
        self._scale /= 2.0
        self._scale_lbl.setText(self._fmt_scale())
        self._redraw()

    def _set_dc_remove(self, checked: bool) -> None:
        self._dc_remove = checked
        self._redraw()

    def _change_time_window(self, idx: int) -> None:
        secs = float(self._cmb_tw.itemData(idx))
        self._time_window = secs
        n_pts = max(int(self._sfreq * secs), 30)
        self._time_axis = np.linspace(0.0, secs, n_pts)
        self._buf = np.zeros((self._n_ch, n_pts))
        self._reset_online_state()
        self._pi.setXRange(0.0, secs, padding=0.01)

    # ------------------------------------------------------------------
    # Callbacks — filter
    # ------------------------------------------------------------------

    def _on_filter_type_changed(self, idx: int) -> None:
        ftypes = ["none", "highpass", "lowpass", "bandpass", "notch"]
        ft = ftypes[idx]
        self._flo_row.setVisible(ft in ("highpass", "bandpass"))
        self._fhi_row.setVisible(ft in ("lowpass", "bandpass"))
        self._fnotch_row.setVisible(ft == "notch")

    def _apply_filter_settings(self) -> None:
        ftypes = ["none", "highpass", "lowpass", "bandpass", "notch"]
        ft = ftypes[self._cmb_filter.currentIndex()]

        if ft == "none":
            self._filter_sos = None
            self._filter_zi  = None
            self._filter_status.setText("○  No filter")
            self._filter_status.setStyleSheet("color:#505070; font-size:10px;")
            return

        try:
            from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos
            nyq = self._sfreq / 2.0
            flo = self._flo_spin.value()
            fhi = self._fhi_spin.value()
            fn  = self._fnotch_spin.value()

            if ft == "highpass":
                sos = butter(4, flo / nyq, btype="high", output="sos")
                label = f"●  HP  {flo:.1f} Hz"
            elif ft == "lowpass":
                sos = butter(4, fhi / nyq, btype="low", output="sos")
                label = f"●  LP  {fhi:.1f} Hz"
            elif ft == "bandpass":
                sos = butter(4, [flo / nyq, fhi / nyq], btype="band", output="sos")
                label = f"●  BP  {flo:.1f}–{fhi:.1f} Hz"
            else:  # notch
                b, a = iirnotch(fn, Q=30, fs=self._sfreq)
                sos = tf2sos(b, a)
                label = f"●  Notch  {fn:.1f} Hz"

            # Prime the per-channel state from the current buffer tail
            # (1 s worth of samples) to avoid a transient at the start.
            zi_base = sosfilt_zi(sos)                                   # (n_sections, 2)
            self._filter_zi = np.zeros(
                (self._n_ch, zi_base.shape[0], zi_base.shape[1])
            )
            n_prime = min(int(self._sfreq), self._buf.shape[1])
            if n_prime > 0:
                for ch in range(self._n_ch):
                    _, self._filter_zi[ch] = sosfilt(
                        sos, self._buf[ch, -n_prime:], zi=zi_base.copy()
                    )

            self._filter_sos = sos
            self._filter_status.setText(label)
            self._filter_status.setStyleSheet(
                "color:#80d8ff; font-size:10px; font-weight:bold;"
            )
        except Exception as exc:
            self._filter_sos = None
            self._filter_zi  = None
            self._filter_status.setText(f"Error: {exc}")
            self._filter_status.setStyleSheet("color:#ff8080; font-size:10px;")

    def _toggle_ssp(self, checked: bool) -> None:
        if not checked or self._info is None:
            self._ssp_proj = None
            return
        try:
            import mne
            active = list(self._info.get("projs", []))
            proj, _, _ = mne.make_projector(active, self._info["ch_names"])
            self._ssp_proj = proj
        except Exception:
            self._ssp_proj = None

    # ------------------------------------------------------------------
    # Callbacks — re-referencing
    # ------------------------------------------------------------------

    def _on_reref_type_changed(self, idx: int) -> None:
        methods = ["none", "average", "mastoid", "linked_mastoid", "channel"]
        self._reref_ch_row.setVisible(methods[idx] == "channel")

    def _apply_reref_settings(self) -> None:
        methods = ["none", "average", "mastoid", "linked_mastoid", "channel"]
        method = methods[self._cmb_reref.currentIndex()]

        if method == "none":
            self._reref_type = "none"
            self._reref_status.setText("○  No re-reference")
            self._reref_status.setStyleSheet("color:#505070; font-size:10px;")
            return

        try:
            if method == "average":
                self._reref_type = "average"
                label = "●  Average ref"

            elif method == "mastoid":
                idxs = [
                    self._ch_names.index(n)
                    for n in ("TP9", "TP10")
                    if n in self._ch_names
                ]
                if not idxs:
                    raise RuntimeError("No TP9 or TP10 channels found.")
                self._reref_type = "mastoid"
                self._reref_idxs = idxs
                found = [self._ch_names[i] for i in idxs]
                label = f"●  Mastoid  ({'+'.join(found)})"

            elif method == "linked_mastoid":
                candidates = ("TP9", "TP10", "M1", "M2", "A1", "A2")
                idxs = [
                    self._ch_names.index(n)
                    for n in candidates
                    if n in self._ch_names
                ]
                if not idxs:
                    raise RuntimeError(
                        "No mastoid channels found (tried TP9, TP10, M1, M2, A1, A2)."
                    )
                self._reref_type = "mastoid"
                self._reref_idxs = idxs
                found = [self._ch_names[i] for i in idxs]
                label = f"●  Linked mastoid  ({'+'.join(found)})"

            else:  # channel
                ref_name = self._reref_ch_cmb.currentText()
                self._reref_type = "channel"
                self._reref_idx = self._ch_names.index(ref_name)
                label = f"●  Ref: {ref_name}"

            self._reref_status.setText(label)
            self._reref_status.setStyleSheet(
                "color:#80d8ff; font-size:10px; font-weight:bold;"
            )
        except Exception as exc:
            self._reref_type = "none"
            self._reref_status.setText(f"Error: {exc}")
            self._reref_status.setStyleSheet("color:#ff8080; font-size:10px;")

    # ------------------------------------------------------------------
    # Callbacks — artifact correction
    # ------------------------------------------------------------------

    def _on_corr_type_changed(self, idx: int) -> None:
        methods = ["none", "lms", "asr", "gedai", "orica", "maxwell"]
        method = methods[idx]
        self._lms_ref_row.setVisible(method == "lms")
        self._asr_cut_row.setVisible(method == "asr")
        self._gedai_row.setVisible(method == "gedai")
        self._orica_row.setVisible(method == "orica")

    def _apply_correction(self) -> None:
        methods = ["none", "lms", "asr", "gedai", "orica", "maxwell"]
        method = methods[self._cmb_corr.currentIndex()]

        self._corrector = None

        if method == "none":
            self._corr_status.setText("○  No correction")
            self._corr_status.setStyleSheet("color:#505070; font-size:10px;")
            return

        try:
            if method == "lms":
                from mne_rt.tools.lms import AdaptiveLMSFilter
                ref_name = self._lms_ref_cmb.currentText()
                ref_idx = self._ch_names.index(ref_name)
                self._corrector = AdaptiveLMSFilter(ref_ch_idx=ref_idx)
                label = f"●  LMS  (ref: {ref_name})"

            elif method == "asr":
                from mne_rt.tools.asr import ASRDenoiser
                cutoff = self._asr_cut_spin.value()
                nonzero_cols = np.any(self._buf != 0.0, axis=0)
                n_valid = int(nonzero_cols.sum())
                min_needed = max(int(self._sfreq), 30)
                if n_valid < min_needed:
                    raise RuntimeError(
                        f"Not enough data in buffer ({n_valid} samples < {min_needed}). "
                        "Wait for more data and try again."
                    )
                asr = ASRDenoiser(cutoff=cutoff)
                asr.fit(self._buf[:, nonzero_cols], self._sfreq)
                self._corrector = asr
                label = f"●  ASR  (cutoff={cutoff:.1f}σ)"

            elif method == "gedai":
                from mne_rt.tools.gedai import GEDAIDenoiser
                nonzero_cols = np.any(self._buf != 0.0, axis=0)
                n_valid = int(nonzero_cols.sum())
                min_needed = max(int(self._sfreq), 30)
                if n_valid < min_needed:
                    raise RuntimeError(
                        f"Not enough data in buffer ({n_valid} samples < {min_needed}). "
                        "Wait for more data and try again."
                    )
                lo = self._gedai_lo_spin.value()
                hi = self._gedai_hi_spin.value()
                n_noise = self._gedai_noise_spin.value()
                gedai = GEDAIDenoiser(n_channels=self._n_ch)
                gedai.fit_from_raw(
                    self._buf[:, nonzero_cols], self._sfreq, band=(lo, hi)
                )
                self._corrector = _GEDAIWrapper(gedai, n_noise)
                label = f"●  GEDAI  ({lo:.0f}–{hi:.0f} Hz, rm {n_noise})"

            elif method == "orica":
                from mne_rt.tools.orica import ORICA
                n_remove = self._orica_n_spin.value()
                self._corrector = _ORICAWrapper(ORICA(n_channels=self._n_ch), n_remove)
                label = (
                    f"●  ORICA  (rm {n_remove} IC{'s' if n_remove > 1 else ''})"
                )

            elif method == "maxwell":
                if self._info is None:
                    raise RuntimeError(
                        "Maxwell filter requires mne.Info. "
                        "Pass info= when constructing RawPlot."
                    )
                from mne_rt.tools.maxwell import RTMaxwellFilter
                mf = RTMaxwellFilter()
                mf.fit(self._info)
                self._corrector = mf
                label = "●  Maxwell SSS"

            self._corr_status.setText(label)
            self._corr_status.setStyleSheet(
                "color:#80d8ff; font-size:10px; font-weight:bold;"
            )
        except Exception as exc:
            self._corrector = None
            self._corr_status.setText(f"Error: {exc}")
            self._corr_status.setStyleSheet("color:#ff8080; font-size:10px;")

    # ------------------------------------------------------------------
    # Redraw — purely displays whatever is in _buf
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        end = min(self._page_start + self._n_shown, self._n_ch)
        visible = list(range(self._page_start, end))
        n_actual = len(visible)
        gain = 1.0 / (self._scale + 1e-300)

        for vis_idx, ch_idx in enumerate(visible):
            raw = self._buf[ch_idx].copy()

            if self._dc_remove:
                nz = raw[raw != 0]
                if nz.size > 0:
                    raw -= nz.mean()

            color = self._ch_colors[ch_idx]
            self._curves[vis_idx].setPen(pg.mkPen(color=color, width=1))
            offset = float(n_actual - 1 - vis_idx)
            self._curves[vis_idx].setData(self._time_axis, offset + raw * gain)

        for vis_idx in range(n_actual, self._n_shown):
            self._curves[vis_idx].setData([], [])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def push(self, data: np.ndarray) -> None:
        """Append a chunk of raw data and refresh the display.

        All active online processing (filter, SSP, artifact correction) is
        applied to the incoming *chunk* before it enters the circular buffer.
        Data already in the buffer is never retroactively modified — the
        display transitions from unprocessed to processed as new data arrives.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            New raw data chunk.

        Notes
        -----
        The call is a no-op when the plot is paused.
        """
        if self._paused:
            return

        chunk = data.copy()

        # 1. Online causal filter (sosfilt with persistent state)
        if self._filter_sos is not None and self._filter_zi is not None:
            try:
                from scipy.signal import sosfilt
                for ch in range(self._n_ch):
                    chunk[ch], self._filter_zi[ch] = sosfilt(
                        self._filter_sos, chunk[ch], zi=self._filter_zi[ch]
                    )
            except Exception:
                pass

        # 2. SSP projection
        if self._ssp_proj is not None:
            try:
                chunk = self._ssp_proj @ chunk
            except Exception:
                pass

        # 3. Artifact corrector (.transform must preserve shape)
        if self._corrector is not None:
            try:
                chunk = self._corrector.transform(chunk)
            except Exception:
                pass

        # 4. Re-referencing (spatial; subtract reference signal from every channel)
        try:
            if self._reref_type == "average":
                chunk = chunk - chunk.mean(axis=0)
            elif self._reref_type == "mastoid" and self._reref_idxs:
                ref = chunk[self._reref_idxs].mean(axis=0)
                chunk = chunk - ref
            elif self._reref_type == "channel":
                ref = chunk[self._reref_idx].copy()
                chunk = chunk - ref
        except Exception:
            pass

        # Write into circular buffer
        n = chunk.shape[1]
        self._buf = np.roll(self._buf, -n, axis=1)
        self._buf[:, -n:] = chunk

        end = min(self._page_start + self._n_shown, self._n_ch)
        self._status.showMessage(
            f"Streaming  —  ch {self._page_start + 1}–{end} of {self._n_ch}"
        )
        self._redraw()

    def closeEvent(self, event) -> None:
        super().closeEvent(event)
