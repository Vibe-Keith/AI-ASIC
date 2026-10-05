"""AI-ASIC graphical interface.

A single Tkinter window that ties together every AI-ASIC capability exposed by
the CLI:

  * Detect  -- probe the attached ASIC miner and show its profile/status.
  * Profiles-- browse every supported miner model in a sortable table.
  * Infer   -- run recursive inference on text (software or cgminer ASIC backend).
  * Encode  -- build a BM1387 (S9) work frame from an 80-byte header.

Tkinter ships with the Python standard library, so this keeps the project's
zero-third-party-dependency promise and runs on a clean Windows/macOS/Linux
install.  Launch with::

    py -m ai_asic.gui        (Windows)
    python -m ai_asic.gui

Long-running work (detection and inference touch the network) runs on worker
threads so the window never freezes; results are marshalled back to the Tk main
loop with ``after``.
"""
from __future__ import annotations

import queue
import threading
import traceback
from typing import Callable

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception as exc:  # pragma: no cover - headless / no-Tk builds
    raise SystemExit(
        "This GUI needs Tkinter, which is missing from this Python install.\n"
        f"({exc})\n"
        "On Linux install it with your package manager (e.g. 'apt install python3-tk');\n"
        "on Windows/macOS reinstall Python with the Tcl/Tk option enabled."
    )

from ai_asic import __version__
from ai_asic.hardware import bm1387
from ai_asic.hardware.bitcoin_header import prepare_asic_job
from ai_asic.hardware.device_detector import detect_asic, detection_summary
from ai_asic.hardware.miner_profiles import all_profiles
from ai_asic.hashing.methods import ASICHashMethod, SoftwareHashMethod
from ai_asic.hashing.network import HashNetwork
from ai_asic.hashing.recursive import RecursiveEngine


class AIASICApp:
    """Top-level application window wiring every engine capability together."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"AI-ASIC  v{__version__}")
        self.root.geometry("760x560")
        self.root.minsize(640, 480)

        # Results from worker threads are delivered through this queue and
        # drained on the Tk main loop, so no widget is touched off-thread.
        self._jobs: "queue.Queue[Callable[[], None]]" = queue.Queue()

        self._build_menu()

        # Status bar lives at the bottom but is created first: tab builders
        # (e.g. the initial profile load) report into it.
        self.status = tk.StringVar(value="Ready.")
        bar = ttk.Label(root, textvariable=self.status, anchor="w",
                        relief="sunken", padding=(6, 2))
        bar.pack(fill="x", side="bottom")

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self._build_detect_tab(nb)
        self._build_profiles_tab(nb)
        self._build_infer_tab(nb)
        self._build_encode_tab(nb)

        self.root.after(100, self._drain_jobs)

    # ------------------------------------------------------------------ menu
    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Quit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self._about)
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.root.config(menu=menubar)

    def _about(self) -> None:
        messagebox.showinfo(
            "About AI-ASIC",
            f"AI-ASIC v{__version__}\n\n"
            "SHA-256 hash neural-network inference on repurposed Bitcoin\n"
            "mining hardware (Antminer S2/S3 through S9 and newer).\n\n"
            "GUI built on the Python standard library (Tkinter).",
        )

    # --------------------------------------------------------------- detect
    def _build_detect_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Detect")

        row = ttk.Frame(tab)
        row.pack(fill="x")
        ttk.Label(row, text="cgminer host:").pack(side="left")
        self.detect_host = tk.StringVar(value="127.0.0.1")
        ttk.Entry(row, textvariable=self.detect_host, width=22).pack(side="left", padx=6)
        self.detect_btn = ttk.Button(row, text="Detect miner", command=self._run_detect)
        self.detect_btn.pack(side="left")

        self.detect_out = _make_output(tab)

    def _run_detect(self) -> None:
        host = self.detect_host.get().strip() or None
        self._start("Detecting miner...", self.detect_btn,
                    lambda: detection_summary(host),
                    lambda text: _set_text(self.detect_out, text),
                    done="Detection complete.")

    # ------------------------------------------------------------- profiles
    def _build_profiles_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Profiles")

        ttk.Button(tab, text="Refresh", command=self._load_profiles).pack(anchor="w")

        cols = ("model", "chip", "protocol", "conn", "hashrate")
        tree = ttk.Treeview(tab, columns=cols, show="headings", height=14)
        headings = {
            "model": ("Model", 170),
            "chip": ("Chip", 90),
            "protocol": ("Protocol", 150),
            "conn": ("Conn", 90),
            "hashrate": ("Hashrate", 110),
        }
        for key, (label, width) in headings.items():
            tree.heading(key, text=label)
            anchor = "e" if key == "hashrate" else "w"
            tree.column(key, width=width, anchor=anchor)
        vsb = ttk.Scrollbar(tab, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True, pady=(8, 0))
        vsb.pack(side="right", fill="y", pady=(8, 0))
        self.profiles_tree = tree
        self._load_profiles()

    def _load_profiles(self) -> None:
        tree = self.profiles_tree
        tree.delete(*tree.get_children())
        for p in all_profiles():
            tree.insert("", "end", values=(
                p.model, p.chip, p.protocol.value, p.connection.value,
                f"{p.nominal_hashrate / 1e12:.2f} TH/s",
            ))
        self.status.set(f"Loaded {len(tree.get_children())} miner profiles.")

    # ---------------------------------------------------------------- infer
    def _build_infer_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Infer")

        top = ttk.Frame(tab)
        top.pack(fill="x")
        ttk.Label(top, text="Input text:").grid(row=0, column=0, sticky="w")
        self.infer_text = tk.StringVar(value="hello world")
        ttk.Entry(top, textvariable=self.infer_text, width=48).grid(
            row=0, column=1, columnspan=3, sticky="we", padx=6, pady=2)

        self.infer_passes = tk.IntVar(value=21)
        self.infer_jitter = tk.DoubleVar(value=0.01)
        self.infer_seed = tk.BooleanVar(value=False)
        _spin(top, "Passes:", self.infer_passes, 1, 1, 1, 999)
        _spin(top, "Jitter:", self.infer_jitter, 1, 3, 0.0, 1.0, increment=0.01)

        self.infer_asic = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Use cgminer ASIC backend",
                        variable=self.infer_asic,
                        command=self._toggle_asic).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(top, text="Seed rotation",
                        variable=self.infer_seed).grid(row=2, column=2, columnspan=2, sticky="w", pady=(4, 0))

        hostrow = ttk.Frame(tab)
        hostrow.pack(fill="x", pady=(4, 0))
        ttk.Label(hostrow, text="ASIC host:").pack(side="left")
        self.infer_host = tk.StringVar(value="127.0.0.1")
        self.infer_host_entry = ttk.Entry(hostrow, textvariable=self.infer_host, width=18)
        self.infer_host_entry.pack(side="left", padx=6)
        ttk.Label(hostrow, text="port:").pack(side="left")
        self.infer_port = tk.IntVar(value=4028)
        self.infer_port_entry = ttk.Entry(hostrow, textvariable=self.infer_port, width=8)
        self.infer_port_entry.pack(side="left", padx=6)

        self.infer_btn = ttk.Button(tab, text="Run inference", command=self._run_infer)
        self.infer_btn.pack(anchor="w", pady=(6, 0))

        self.infer_out = _make_output(tab)
        self._toggle_asic()

    def _toggle_asic(self) -> None:
        state = "normal" if self.infer_asic.get() else "disabled"
        self.infer_host_entry.configure(state=state)
        self.infer_port_entry.configure(state=state)

    def _run_infer(self) -> None:
        try:
            text = self.infer_text.get()
            passes = int(self.infer_passes.get())
            jitter = float(self.infer_jitter.get())
            seed = bool(self.infer_seed.get())
            use_asic = bool(self.infer_asic.get())
            host = self.infer_host.get().strip() or "127.0.0.1"
            port = int(self.infer_port.get())
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid input", f"Check the numeric fields.\n{exc}")
            return

        def work() -> str:
            net = HashNetwork(64, 128, 64, 10)
            if use_asic:
                method = ASICHashMethod(host, port)
            else:
                method = SoftwareHashMethod()
            engine = RecursiveEngine(net, passes=passes, jitter=jitter,
                                     seed_rotation=seed, hash_method=method)
            result = engine.infer(text.encode("utf-8"))
            c = result.consensus
            return (
                f"Hash method:       {method.name()}\n"
                f"Prediction:        class {c.prediction}\n"
                f"Consensus conf:    {c.confidence:.3f}\n"
                f"Avg per-pass conf: {c.average_confidence:.3f}\n"
                f"Valid passes:      {result.valid_passes}/{result.total_passes}\n"
                f"Latency:           {result.latency_s * 1000:.1f} ms\n"
                f"Using hardware:    {engine.is_using_hardware()}"
            )

        self._start("Running inference...", self.infer_btn, work,
                    lambda text: _set_text(self.infer_out, text),
                    done="Inference complete.")

    # --------------------------------------------------------------- encode
    def _build_encode_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Encode work")

        ttk.Label(tab, text="80-byte header (160 hex chars, blank = demo header):").pack(anchor="w")
        self.encode_header = tk.StringVar(value="")
        ttk.Entry(tab, textvariable=self.encode_header).pack(fill="x", pady=(2, 6))

        row = ttk.Frame(tab)
        row.pack(fill="x")
        ttk.Label(row, text="work id:").pack(side="left")
        self.encode_work_id = tk.IntVar(value=1)
        ttk.Spinbox(row, from_=0, to=65535, textvariable=self.encode_work_id,
                    width=8).pack(side="left", padx=6)
        self.encode_btn = ttk.Button(row, text="Encode frame", command=self._run_encode)
        self.encode_btn.pack(side="left")

        self.encode_out = _make_output(tab)

    def _run_encode(self) -> None:
        raw = self.encode_header.get().strip()
        try:
            work_id = int(self.encode_work_id.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid input", "work id must be an integer.")
            return

        def work() -> str:
            if raw:
                header = bytes.fromhex(raw)
                if len(header) != 80:
                    raise ValueError("header must be 80 bytes (160 hex chars)")
            else:
                header = prepare_asic_job([0] * 12, 0, timestamp=0)
            w = bm1387.new_work_from_header(header, work_id)
            frame = w.encode()
            return (
                f"work_id:   {work_id}\n"
                f"midstate:  {w.midstates[0].hex()}\n"
                f"data(12):  {w.data.hex()}\n"
                f"frame:     {frame.hex()}\n"
                f"frame len: {len(frame)} bytes"
            )

        self._start("Encoding work frame...", self.encode_btn, work,
                    lambda text: _set_text(self.encode_out, text),
                    done="Work frame encoded.")

    # ----------------------------------------------------------- threading
    def _start(self, busy_msg: str, button: ttk.Button,
               work: Callable[[], str], on_ok: Callable[[str], None],
               done: str) -> None:
        """Run ``work`` on a worker thread; marshal the result back to Tk."""
        self.status.set(busy_msg)
        button.configure(state="disabled")

        def runner() -> None:
            try:
                result = work()
            except Exception as exc:  # surface any engine/network error
                tb = traceback.format_exc()
                self._jobs.put(lambda: self._on_error(button, exc, tb))
            else:
                def finish() -> None:
                    on_ok(result)
                    button.configure(state="normal")
                    self.status.set(done)
                self._jobs.put(finish)

        threading.Thread(target=runner, daemon=True).start()

    def _on_error(self, button: ttk.Button, exc: Exception, tb: str) -> None:
        button.configure(state="normal")
        self.status.set(f"Error: {exc}")
        messagebox.showerror("Error", f"{type(exc).__name__}: {exc}\n\n{tb}")

    def _drain_jobs(self) -> None:
        try:
            while True:
                job = self._jobs.get_nowait()
                job()
        except queue.Empty:
            pass
        self.root.after(100, self._drain_jobs)


# --------------------------------------------------------------- helpers
def _make_output(parent: tk.Widget) -> tk.Text:
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True, pady=(8, 0))
    text = tk.Text(frame, wrap="word", height=10, font=("Courier New", 10))
    vsb = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=vsb.set, state="disabled")
    text.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")
    return text


def _set_text(widget: tk.Text, content: str) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", content)
    widget.configure(state="disabled")


def _spin(parent: tk.Widget, label: str, var, row: int, col: int,
          lo, hi, increment=1) -> None:
    ttk.Label(parent, text=label).grid(row=row, column=col, sticky="e", pady=2)
    ttk.Spinbox(parent, from_=lo, to=hi, increment=increment,
                textvariable=var, width=8).grid(row=row, column=col + 1, sticky="w", padx=6)


def main(argv=None) -> int:
    root = tk.Tk()
    AIASICApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
