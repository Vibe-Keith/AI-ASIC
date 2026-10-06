"""AI-ASIC graphical interface.

A single Tkinter window that ties together every AI-ASIC capability exposed by
the CLI:

  * Detect  -- probe the attached ASIC miner and show its profile/status.
  * Simulate-- run a virtual BM1387 miner that serves the real cgminer API, so the
               Detect tab and the Infer ASIC backend find it like real hardware.
  * Workload-- run a hash-based classifier split across the host CPU (encoder + head)
               and the ASIC miner (the nonce-search mining layer); load/save models.
  * Chat    -- chat with a local GGUF LLM (CPU) with speculative drafts and KV-block
               routing; the ASIC does the native nonce searches (LSH buckets, seals).
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
    from tkinter import ttk, messagebox, filedialog
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
from ai_asic.hardware.simulator import SimConfig, VirtualMinerServer, simulate_header
from ai_asic.hashing.methods import ASICHashMethod, SoftwareHashMethod
from ai_asic.hashing.network import HashNetwork
from ai_asic.hashing.recursive import RecursiveEngine
from ai_asic.chat.accelerator import HashAccelerator
from ai_asic.chat.models_dir import hasher_dir, list_llm_models, load_config, resolve_llm
from ai_asic.server import HasherServer, VirtualAsicDevice
from ai_asic.workloads.split_model import MiningBackend, SplitModel


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

        # Virtual miner servers (Simulate tab); None until started. _vserver is the cgminer
        # API emulator (detection); _hserver is the ported on-device hasher-server (compute).
        self._vserver: "VirtualMinerServer | None" = None
        self._hserver: "HasherServer | None" = None
        # Current split-model for the Workload tab; None until created or loaded.
        self._wl_model: "SplitModel | None" = None
        # Chat engine (Chat tab); built lazily on first message, rebuilt when the model or
        # draft setting changes (drafts are fixed when llama.cpp loads the model).
        self._chat_engine = None
        self._chat_engine_key = None
        self._chat_acc = None       # kept across turns so the ASIC path stays warm
        self._chat_acc_port = None

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
        self._build_simulate_tab(nb)
        self._build_workload_tab(nb)
        self._build_chat_tab(nb)
        self._build_profiles_tab(nb)
        self._build_infer_tab(nb)
        self._build_encode_tab(nb)

        # Stop the virtual miner cleanly when the window is closed.
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._drain_jobs)

    # ------------------------------------------------------------------ menu
    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Quit", command=self._on_close)
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
        ttk.Entry(row, textvariable=self.detect_host, width=18).pack(side="left", padx=6)
        ttk.Label(row, text="port:").pack(side="left")
        self.detect_port = tk.IntVar(value=4028)
        ttk.Entry(row, textvariable=self.detect_port, width=8).pack(side="left", padx=6)
        self.detect_btn = ttk.Button(row, text="Detect miner", command=self._run_detect)
        self.detect_btn.pack(side="left")

        ttk.Label(tab, text="Tip: start a miner on the Simulate tab, then detect it here "
                            "(127.0.0.1:4028).", foreground="#555").pack(anchor="w", pady=(4, 0))

        self.detect_out = _make_output(tab)

    def _run_detect(self) -> None:
        host = self.detect_host.get().strip() or None
        try:
            port = int(self.detect_port.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid input", "port must be an integer.")
            return
        self._start("Detecting miner...", self.detect_btn,
                    lambda: detection_summary(host, port),
                    lambda text: _set_text(self.detect_out, text),
                    done="Detection complete.")

    # ------------------------------------------------------------- simulate
    def _build_simulate_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Simulate")

        intro = ("Run a virtual BM1387 miner in-process. It serves the real cgminer/bmminer\n"
                 "API, so the Detect tab (and the Infer ASIC backend) find it like real hardware.")
        ttk.Label(tab, text=intro, foreground="#555").pack(anchor="w")

        row = ttk.Frame(tab)
        row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text="Model:").pack(side="left")
        models = [p.model for p in all_profiles()]
        self.sim_model = tk.StringVar(value="Antminer S9")
        ttk.Combobox(row, textvariable=self.sim_model, values=models, width=20,
                     state="readonly").pack(side="left", padx=6)
        ttk.Label(row, text="port:").pack(side="left")
        self.sim_port = tk.IntVar(value=4028)
        ttk.Entry(row, textvariable=self.sim_port, width=8).pack(side="left", padx=6)

        row2 = ttk.Frame(tab)
        row2.pack(fill="x", pady=(6, 0))
        self.sim_start_btn = ttk.Button(row2, text="Start virtual miner",
                                        command=self._toggle_vserver)
        self.sim_start_btn.pack(side="left")
        self.sim_detect_btn = ttk.Button(row2, text="Detect it", command=self._sim_detect,
                                         state="disabled")
        self.sim_detect_btn.pack(side="left", padx=(6, 0))

        self.sim_state = tk.StringVar(value="Virtual miner: stopped.")
        ttk.Label(tab, textvariable=self.sim_state, foreground="#036").pack(
            anchor="w", pady=(6, 0))

        sep = ttk.Separator(tab, orient="horizontal")
        sep.pack(fill="x", pady=8)

        row3 = ttk.Frame(tab)
        row3.pack(fill="x")
        ttk.Label(row3, text="Self-test mine - difficulty (zero bits):").pack(side="left")
        self.sim_difficulty = tk.IntVar(value=16)
        ttk.Spinbox(row3, from_=1, to=28, textvariable=self.sim_difficulty,
                    width=6).pack(side="left", padx=6)
        self.sim_mine_btn = ttk.Button(row3, text="Run self-test mine",
                                       command=self._sim_mine)
        self.sim_mine_btn.pack(side="left")

        self.sim_out = _make_output(tab)

    def _toggle_vserver(self) -> None:
        if self._vserver is not None:
            self._vserver.stop()
            self._vserver = None
            if self._hserver is not None:
                self._hserver.stop()
                self._hserver = None
            self.sim_start_btn.configure(text="Start virtual miner")
            self.sim_detect_btn.configure(state="disabled")
            self.sim_state.set("Virtual miner: stopped.")
            self.status.set("Virtual miner stopped.")
            return
        try:
            model = self.sim_model.get()
            port = int(self.sim_port.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid input", "port must be an integer.")
            return
        try:
            srv = VirtualMinerServer(model=model, port=port).start()
            # The on-device hasher-server (compute/mining) on its own ephemeral port, sharing
            # the same simulated chip - what the Workload tab offloads to.
            hsrv = HasherServer(device=VirtualAsicDevice(model), port=0).start()
        except OSError as exc:
            messagebox.showerror(
                "Could not start virtual miner",
                f"Port {port} may be in use by another process.\n\n{exc}")
            return
        self._vserver = srv
        self._hserver = hsrv
        self.sim_port.set(srv.port)
        self.sim_start_btn.configure(text="Stop virtual miner")
        self.sim_detect_btn.configure(state="normal")
        self.sim_state.set(
            f"Virtual miner: {model} ({srv.profile.chip}) - cgminer API on 127.0.0.1:{srv.port} "
            f"(detect), hasher-server on 127.0.0.1:{hsrv.port} (compute).")
        self.status.set(f"Virtual miner + hasher-server started "
                        f"(ports {srv.port} / {hsrv.port}).")

    def _sim_detect(self) -> None:
        if self._vserver is None:
            messagebox.showinfo("No miner", "Start the virtual miner first.")
            return
        port = self._vserver.port
        self._start("Detecting virtual miner...", self.sim_detect_btn,
                    lambda: detection_summary("127.0.0.1", port),
                    lambda text: _set_text(self.sim_out, text),
                    done="Detection complete.")

    def _sim_mine(self) -> None:
        try:
            model = self.sim_model.get()
            difficulty = int(self.sim_difficulty.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid input", "difficulty must be an integer.")
            return

        def work() -> str:
            header = prepare_asic_job([1] * 12, 0, timestamp=0)
            cfg = SimConfig(difficulty_bits=difficulty)
            sim = simulate_header(header, work_id=1, config=cfg)
            r = sim.result
            if not r.found:
                return (f"No golden nonce in {r.hashes_tried:,} rolls.\n"
                        f"Lower the difficulty and try again.")
            ok = lambda b: "OK" if b else "MISMATCH"
            return (
                f"Virtual chip:      {model}\n"
                f"Golden nonce:      0x{r.nonce:08x} ({r.nonce})\n"
                f"Chip hash:         {r.hash_hex}\n"
                f"Leading zero bits: {r.leading_zeros} (target {difficulty})\n"
                f"Nonces rolled:     {r.hashes_tried:,}\n"
                f"Response frame:    {r.response_frame.hex()} "
                f"-> parsed nonce 0x{sim.parsed.nonce:08x}\n"
                f"Sim run time:      {r.sim_seconds:.2f} s\n"
                f"\nVerification:\n"
                f"  midstate math == hashlib over full header : {ok(sim.midstate_matches_header)}\n"
                f"  response frame round-trips                : {ok(sim.response_roundtrips)}\n"
                f"  hash meets target                         : {ok(r.leading_zeros >= difficulty)}\n"
                f"\nRESULT: " + ("PASS - the work/mine/nonce pipeline is correct."
                                 if sim.ok else "FAIL - see mismatches above.")
            )

        self._start("Running self-test mine (pure-Python, may take a few seconds)...",
                    self.sim_mine_btn, work,
                    lambda text: _set_text(self.sim_out, text),
                    done="Self-test mine complete.")

    # -------------------------------------------------------------- workload
    def _build_workload_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Workload")

        intro = ("A hash-based classifier split across devices: the encoder and output head\n"
                 "run on the host CPU; the mining layer (nonce search) runs on the ASIC when\n"
                 "the Simulate tab's virtual miner is running, else on the host as a fallback.")
        ttk.Label(tab, text=intro, foreground="#555").pack(anchor="w")

        top = ttk.Frame(tab)
        top.pack(fill="x", pady=(8, 0))
        ttk.Label(top, text="Input text:").grid(row=0, column=0, sticky="w")
        self.wl_text = tk.StringVar(value="optical alignment with AI")
        ttk.Entry(top, textvariable=self.wl_text, width=46).grid(
            row=0, column=1, columnspan=5, sticky="we", padx=6, pady=2)

        self.wl_difficulty = tk.IntVar(value=12)
        self.wl_mining = tk.IntVar(value=4)
        self.wl_classes = tk.IntVar(value=10)
        _spin(top, "Difficulty:", self.wl_difficulty, 1, 0, 1, 24)
        _spin(top, "Mining neurons:", self.wl_mining, 1, 2, 1, 16)
        _spin(top, "Classes:", self.wl_classes, 1, 4, 2, 64)

        self.wl_use_asic = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text="Offload mining layer to the virtual ASIC (Simulate tab)",
                        variable=self.wl_use_asic).pack(anchor="w", pady=(6, 0))

        btns = ttk.Frame(tab)
        btns.pack(fill="x", pady=(6, 0))
        self.wl_run_btn = ttk.Button(btns, text="Run workload", command=self._wl_run)
        self.wl_run_btn.pack(side="left")
        ttk.Button(btns, text="New model", command=self._wl_new).pack(side="left", padx=(6, 0))
        ttk.Button(btns, text="Load model...", command=self._wl_load).pack(side="left", padx=(6, 0))
        ttk.Button(btns, text="Save model...", command=self._wl_save).pack(side="left", padx=(6, 0))

        self.wl_model_lbl = tk.StringVar(value="Model: none yet (a random one is created on run).")
        ttk.Label(tab, textvariable=self.wl_model_lbl, foreground="#036").pack(
            anchor="w", pady=(6, 0))

        self.wl_out = _make_output(tab)

    def _wl_build_model(self) -> "SplitModel":
        return SplitModel(
            input_size=64, mining_neurons=int(self.wl_mining.get()),
            output_size=int(self.wl_classes.get()),
            difficulty_bits=int(self.wl_difficulty.get()),
        )

    def _wl_new(self) -> None:
        try:
            self._wl_model = self._wl_build_model()
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid input", f"Check the numeric fields.\n{exc}")
            return
        self.wl_model_lbl.set(
            f"Model: new random split-model ({self._wl_model.mining_neurons} mining neurons, "
            f"{self._wl_model.output_size} classes, difficulty {self._wl_model.difficulty_bits}).")
        self.status.set("New split-model created.")

    def _wl_load(self) -> None:
        path = filedialog.askopenfilename(
            title="Load split-model", initialdir=str(hasher_dir()),
            filetypes=[("Split-model JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            self._wl_model = SplitModel.load(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Load failed", f"{exc}")
            return
        self.wl_difficulty.set(self._wl_model.difficulty_bits)
        self.wl_mining.set(self._wl_model.mining_neurons)
        self.wl_classes.set(self._wl_model.output_size)
        self.wl_model_lbl.set(f"Model: loaded from {path}")
        self.status.set("Model loaded.")

    def _wl_save(self) -> None:
        if self._wl_model is None:
            try:
                self._wl_model = self._wl_build_model()
            except (tk.TclError, ValueError) as exc:
                messagebox.showerror("Invalid input", f"Check the numeric fields.\n{exc}")
                return
        path = filedialog.asksaveasfilename(
            title="Save split-model", defaultextension=".json", initialdir=str(hasher_dir()),
            filetypes=[("Split-model JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            self._wl_model.save(path)
        except OSError as exc:
            messagebox.showerror("Save failed", f"{exc}")
            return
        self.wl_model_lbl.set(f"Model: saved to {path}")
        self.status.set("Model saved.")

    def _wl_run(self) -> None:
        try:
            text = self.wl_text.get()
            use_asic = bool(self.wl_use_asic.get())
            if self._wl_model is None:
                self._wl_model = self._wl_build_model()
                self.wl_model_lbl.set(
                    f"Model: new random split-model ({self._wl_model.mining_neurons} mining "
                    f"neurons, {self._wl_model.output_size} classes, "
                    f"difficulty {self._wl_model.difficulty_bits}).")
            model = self._wl_model
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid input", f"Check the fields.\n{exc}")
            return

        host = port = None
        if use_asic and self._hserver is not None:
            host, port = "127.0.0.1", self._hserver.port
        asic_requested = use_asic

        def work() -> str:
            backend = MiningBackend(
                host=host, port=port, difficulty_bits=model.difficulty_bits,
                max_nonces=model.nonce_range, prefer_asic=bool(host))
            result = model.infer(text.encode("utf-8"), backend)
            t = result.trace
            lines = [f"Mining device:     {backend.label}"]
            if asic_requested and host is None:
                lines.append("(Virtual miner not running - start it on the Simulate tab to "
                             "offload to the ASIC.)")
            lines.append("")
            lines.append(f"{'Stage':<10}{'Device':<10}{'Ops':>6}{'Time (ms)':>12}  Detail")
            lines.append("-" * 70)
            for s in t.stages:
                lines.append(f"{s.name:<10}{s.device:<10}{s.ops:>6}"
                             f"{s.seconds * 1000:>12.1f}  {s.detail}")
            lines.append("-" * 70)
            lines.append(f"Host ops: {t.host_ops}  |  ASIC ops: {t.asic_ops}  |  "
                         f"Ran on hardware: {t.asic_is_hardware}")
            lines.append("")
            lines.append(f"Mined nonces:  {[hex(n) for n in result.nonces]}")
            lines.append(f"Prediction:    class {result.prediction}  "
                         f"(score {result.confidence:.4f})")
            return "\n".join(lines)

        self._start("Running workload (host + ASIC)...", self.wl_run_btn, work,
                    lambda text: _set_text(self.wl_out, text),
                    done="Workload complete.")

    # ------------------------------------------------------------------ chat
    def _build_chat_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Chat")

        row = ttk.Frame(tab)
        row.pack(fill="x")
        ttk.Label(row, text="Model:").pack(side="left")
        self.chat_model = tk.StringVar()
        self.chat_model_box = ttk.Combobox(row, textvariable=self.chat_model, width=36,
                                           state="readonly")
        self.chat_model_box.pack(side="left", padx=6)
        ttk.Button(row, text="Refresh", command=self._chat_refresh_models).pack(side="left")
        ttk.Button(row, text="New chat", command=self._chat_new).pack(side="right")

        opts = ttk.Frame(tab)
        opts.pack(fill="x", pady=(4, 0))
        cfg = load_config()
        self.chat_asic = tk.BooleanVar(value=True)
        self.chat_draft = tk.BooleanVar(value=bool(cfg["use_draft"]))
        self.chat_cache = tk.BooleanVar(value=bool(cfg["use_cache"]))
        self.chat_route = tk.BooleanVar(value=bool(cfg["route_context"]))
        ttk.Checkbutton(opts, text="ASIC nonce search (hasher-server, Simulate tab)",
                        variable=self.chat_asic).pack(side="left")
        ttk.Checkbutton(opts, text="Speculative drafts",
                        variable=self.chat_draft).pack(side="left", padx=(10, 0))
        ttk.Checkbutton(opts, text="Response cache",
                        variable=self.chat_cache).pack(side="left", padx=(10, 0))
        ttk.Checkbutton(opts, text="KV routing",
                        variable=self.chat_route).pack(side="left", padx=(10, 0))

        frame = ttk.Frame(tab)
        frame.pack(fill="both", expand=True, pady=(6, 0))
        self.chat_log = tk.Text(frame, wrap="word", height=12, font=("Segoe UI", 10))
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.chat_log.yview)
        self.chat_log.configure(yscrollcommand=vsb.set, state="disabled")
        self.chat_log.tag_configure("you", foreground="#0b5394", font=("Segoe UI", 10, "bold"))
        self.chat_log.tag_configure("bot", foreground="#1a1a1a")
        self.chat_log.tag_configure("meta", foreground="#888888",
                                    font=("Segoe UI", 9, "italic"))
        self.chat_log.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        entry_row = ttk.Frame(tab)
        entry_row.pack(fill="x", pady=(6, 0))
        self.chat_input = tk.StringVar()
        entry = ttk.Entry(entry_row, textvariable=self.chat_input)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda _e: self._chat_send())
        self.chat_send_btn = ttk.Button(entry_row, text="Send", command=self._chat_send)
        self.chat_send_btn.pack(side="left", padx=(6, 0))

        self.chat_trace = tk.Text(tab, wrap="none", height=6, font=("Courier New", 9),
                                  state="disabled")
        self.chat_trace.pack(fill="x", pady=(6, 0))

        self._chat_refresh_models()

    def _chat_refresh_models(self) -> None:
        names = [p.name for p in list_llm_models()]
        self.chat_model_box.configure(values=names)
        default = resolve_llm()
        if default is not None and default.name in names:
            self.chat_model.set(default.name)
        elif names:
            self.chat_model.set(names[0])
        else:
            self.chat_model.set("")
            self._chat_write("No models in ai_models/llm/. Run 'python -m ai_asic.cli models "
                             "download qwen2.5-0.5b' or drop a .gguf file there.\n", "meta")

    def _chat_write(self, text: str, tag: str) -> None:
        self.chat_log.configure(state="normal")
        self.chat_log.insert("end", text, tag)
        self.chat_log.see("end")
        self.chat_log.configure(state="disabled")

    def _append_trace(self, text: str) -> None:
        self.chat_trace.configure(state="normal")
        self.chat_trace.insert("end", text)
        self.chat_trace.configure(state="disabled")

    def _chat_new(self) -> None:
        if self._chat_engine is not None:
            self._chat_engine.new_chat()
        self.chat_log.configure(state="normal")
        self.chat_log.delete("1.0", "end")
        self.chat_log.configure(state="disabled")
        _set_text(self.chat_trace, "")
        self.status.set("New chat.")

    def _chat_send(self) -> None:
        if str(self.chat_send_btn.cget("state")) == "disabled":
            return  # a reply is still streaming
        text = self.chat_input.get().strip()
        model = self.chat_model.get()
        if not text:
            return
        if not model:
            messagebox.showinfo("No model", "Put a .gguf model in ai_models/llm/ first.")
            return
        self.chat_input.set("")
        self._chat_write(f"you: {text}\n", "you")
        self._chat_write("bot: ", "meta")

        use_draft = bool(self.chat_draft.get())
        use_cache = bool(self.chat_cache.get())
        use_route = bool(self.chat_route.get())
        hport = self._hserver.port if (self.chat_asic.get() and self._hserver) else None
        asic_requested = bool(self.chat_asic.get())

        def work() -> str:
            from ai_asic.chat.engine import ChatEngine

            if self._chat_acc is None or self._chat_acc_port != hport:
                self._chat_acc = (HashAccelerator("127.0.0.1", hport) if hport
                                  else HashAccelerator())
                self._chat_acc_port = hport
            acc = self._chat_acc
            key = (model, use_draft)
            if self._chat_engine is None or self._chat_engine_key != key:
                history = self._chat_engine.messages if self._chat_engine else None
                self._jobs.put(lambda: self.status.set(f"Loading {model}..."))
                engine = ChatEngine(model, accelerator=acc, use_draft=use_draft)
                engine.load()
                if history:
                    engine.messages = history  # keep the conversation across a reload
                self._chat_engine, self._chat_engine_key = engine, key
            engine = self._chat_engine
            engine.set_accelerator(acc)
            engine.cfg["use_cache"] = use_cache
            engine.cfg["route_context"] = use_route
            self._jobs.put(lambda: self.status.set("Generating..."))
            result = engine.reply(
                text, on_token=lambda t: self._jobs.put(lambda t=t: self._chat_write(t, "bot")))
            lines = [f"Accelerator: {result.trace.asic_label}"
                     + ("   (start the virtual miner on the Simulate tab to offload)"
                        if asic_requested and not hport else "")]
            ev = result.eval
            if ev is not None:
                lines.append(f"Verification passes: {ev.decode_passes}   accepted tokens/pass: "
                             f"{ev.tokens_per_pass:.2f}   attention pairs: "
                             f"{ev.attention_pairs:,}   unused draft tokens: "
                             f"{ev.rejected_tokens}")
            lines.append(f"{'Stage':<14}{'Device':<10}{'Ops':>9}{'Time (ms)':>11}  Detail")
            for s in list(result.trace.stages):
                lines.append(f"{s.name:<14}{s.device:<10}{s.ops:>9}"
                             f"{s.seconds * 1000:>11.1f}  {s.detail}")
            fut = result.seal_future
            if fut is not None and not fut.done():
                lines.append(f"{'seal':<14}{'':<10}{'':>9}{'':>11}  mining in the background...")

                def sealed(_f, result=result) -> None:
                    result.wait_seal()
                    st = next((x for x in result.trace.stages if x.name == "seal"), None)
                    if st is not None:
                        line = (f"\n{st.name:<14}{st.device:<10}{st.ops:>9}"
                                f"{st.seconds * 1000:>11.1f}  {st.detail}")
                        self._jobs.put(lambda: self._append_trace(line))

                fut.add_done_callback(sealed)
            return "\n".join(lines)

        def done(trace_text: str) -> None:
            self._chat_write("\n\n", "bot")
            _set_text(self.chat_trace, trace_text)

        self._start("Chatting...", self.chat_send_btn, work, done, done="Reply complete.")

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
        self.root.after(40, self._drain_jobs)  # short interval keeps chat streaming smooth

    def _on_close(self) -> None:
        """Stop the virtual miner / hasher-server (if running) before tearing down."""
        for attr in ("_vserver", "_hserver"):
            srv = getattr(self, attr, None)
            if srv is not None:
                try:
                    srv.stop()
                except Exception:
                    pass
                setattr(self, attr, None)
        self.root.destroy()


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
