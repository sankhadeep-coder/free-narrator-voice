import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import time
import subprocess
import torch
import torchaudio as ta

# ── Audio playback (pygame preferred, fallback to playsound) ──
_pygame_ok = False
try:
    import pygame
    pygame.mixer.init()
    _pygame_ok = True
except Exception:
    pass

# ─────────────────────────────────────────────
#  Lazy-load models so GUI opens instantly
# ─────────────────────────────────────────────
tts_en  = None   # ChatterboxTTS
tts_hi  = None   # ChatterboxMultilingualTTS

SAMPLE_RATE = 22050   # fallback; overwritten once model loads

def load_models(log):
    global tts_en, tts_hi, SAMPLE_RATE
    try:
        from chatterbox.tts import ChatterboxTTS
        log("⏳ Loading English model…")
        tts_en = ChatterboxTTS.from_pretrained(device="cpu")
        SAMPLE_RATE = tts_en.sr
        log("✅ English model ready.")
    except Exception as e:
        log(f"❌ English model failed: {e}")

    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        log("⏳ Loading Hindi model…")
        tts_hi = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        log("✅ Hindi model ready.")
    except Exception as e:
        log(f"❌ Hindi model failed: {e}")

    log("🎙️ All models loaded. You can generate now!\n")


# ─────────────────────────────────────────────
#  Auto-detect language from Unicode ranges
# ─────────────────────────────────────────────
def is_hindi_char(ch):
    """True if character belongs to Devanagari Unicode block."""
    return '\u0900' <= ch <= '\u097F' or '\u0980' <= ch <= '\u09FF'

def is_neutral_char(ch):
    """
    Neutral chars: punctuation, digits, spaces — they belong to whichever
    language is currently active and should NOT trigger a language switch.
    """
    return ch in ' \t\n.,!?;:\'"()[]{}।—–-_@#%&*+=/\\|`~^<>' or ch.isdigit()

def auto_split_mixed(line):
    """
    Split a mixed Hindi+English line into labelled segments.
    Neutral chars (punctuation, digits) stick to the current language.
    e.g. "hello मेरा नाम शंखदीप है, मैं एक अच्छा लड़का हूँ। मुझे ORANGES पसन्द हैं."
    → [('en','hello'), ('hi','मेरा नाम शंखदीप है, मैं एक अच्छा लड़का हूँ। मुझे'),
       ('en','ORANGES'), ('hi','पसन्द हैं.')]
    """
    segments = []
    current_lang = None   # 'en' | 'hi' | None
    current_text = []

    for ch in line:
        if is_neutral_char(ch):
            # Stick to whatever language is active; if nothing yet, buffer it
            current_text.append(ch)
            continue

        lang = 'hi' if is_hindi_char(ch) else 'en'

        if lang != current_lang:
            # Flush previous segment (strip trailing neutral chars like spaces)
            if current_text:
                text = ''.join(current_text).strip()
                if text and current_lang is not None:
                    segments.append((current_lang, text))
                elif text and current_lang is None:
                    # Leading neutral chars before any language — discard silently
                    pass
            current_lang = lang
            current_text = [ch]
        else:
            current_text.append(ch)

    # Flush last segment
    if current_text:
        text = ''.join(current_text).strip()
        if text and current_lang is not None:
            segments.append((current_lang, text))

    return segments


def parse_lines(raw):
    """
    Parse the script text box.
    Supports:
      [EN] Hello world        -> English segment
      [HI] mera naam hai      -> Hindi segment
      hello mera naam hai     -> auto-split mixed line (no prefix needed)
    Returns list of (lang, text).
    """
    result = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue

        # [EN] prefix
        if ln.upper().startswith("[EN]"):
            text = ln[4:].strip()
            if text:
                result.append(("en", text))
            continue

        # [HI] prefix
        if ln.upper().startswith("[HI]"):
            text = ln[4:].strip()
            if text:
                result.append(("hi", text))
            continue

        # No prefix -> auto-detect / auto-split mixed line
        segs = auto_split_mixed(ln)
        result.extend(segs)

    return result


# ─────────────────────────────────────────────
#  Audio generation helpers
# ─────────────────────────────────────────────
def generate_segment(text, lang, voice_path, exaggeration=0.5):
    """Returns (wav_tensor, sample_rate)"""
    if lang == "hi":
        if tts_hi is None:
            raise RuntimeError("Hindi model not loaded")
        wav = tts_hi.generate(text, language_id="hi",
                               audio_prompt_path=voice_path if voice_path else None)
        return wav, tts_hi.sr
    else:
        if tts_en is None:
            raise RuntimeError("English model not loaded")
        kwargs = {"exaggeration": exaggeration}
        if voice_path:
            kwargs["audio_prompt_path"] = voice_path
        wav = tts_en.generate(text, **kwargs)
        return wav, tts_en.sr


def combine_wavs(wavs_sr):
    """Concatenate list of (wav, sr) — resample all to first sr."""
    if not wavs_sr:
        return None, None
    target_sr = wavs_sr[0][1]
    tensors = []
    for wav, sr in wavs_sr:
        if sr != target_sr:
            wav = ta.functional.resample(wav, sr, target_sr)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        tensors.append(wav)
    combined = torch.cat(tensors, dim=-1)
    return combined, target_sr


# ─────────────────────────────────────────────
#  Audio playback helper
# ─────────────────────────────────────────────
def _play_audio(filepath):
    """Play a WAV file. Uses pygame if available, otherwise os default."""
    if _pygame_ok:
        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            return
        except Exception:
            pass
    # Fallback: OS default player (non-blocking)
    try:
        if os.name == "nt":
            os.startfile(filepath)
        elif os.uname().sysname == "Darwin":
            subprocess.Popen(["afplay", filepath])
        else:
            subprocess.Popen(["aplay", filepath])
    except Exception:
        pass


def _stop_audio():
    if _pygame_ok:
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass


# ─────────────────────────────────────────────
#  Audio Preview Dialog
# ─────────────────────────────────────────────
class AudioPreviewDialog(tk.Toplevel):
    """
    Shown after generation. User can:
      - Auto-preview plays immediately on open
      - ▶ Play Again
      - ✅ Save  → saves output + runs copy.py
      - 🔁 Regenerate → deletes temp file, closes, triggers regenerate
      - 🗑 Delete → deletes temp file, closes
    """

    BG   = "#0f0f1a"
    CARD = "#1a1a2e"
    ACC  = "#7c3aed"
    ACC2 = "#06b6d4"
    TXT  = "#e2e8f0"
    DIM  = "#64748b"
    GRN  = "#22c55e"
    RED  = "#ef4444"
    YEL  = "#f59e0b"

    def __init__(self, parent, tmp_file, final_file, duration, on_save, on_regen, on_delete):
        super().__init__(parent)
        self.title("🎧 Audio Preview")
        self.configure(bg=self.BG)
        self.resizable(False, False)
        self.grab_set()          # modal

        self._tmp_file   = tmp_file
        self._final_file = final_file
        self._duration   = duration
        self._on_save    = on_save
        self._on_regen   = on_regen
        self._on_delete  = on_delete
        self._decision   = None  # 'save' | 'regen' | 'delete'

        self._build()
        self.update_idletasks()

        # Centre over parent
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        w, h   = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{px + (pw-w)//2}+{py + (ph-h)//2}")

        # Auto-play after the window has drawn
        self.after(300, self._play)

        self.protocol("WM_DELETE_WINDOW", self._do_delete)

    def _build(self):
        B, C, A, A2, T, D, G, R, Y = (
            self.BG, self.CARD, self.ACC, self.ACC2,
            self.TXT, self.DIM, self.GRN, self.RED, self.YEL
        )

        # ── Title bar area
        top = tk.Frame(self, bg="#16013e", pady=14)
        top.pack(fill="x")
        tk.Label(top, text="🎧  Audio Preview",
                 font=("Courier New", 15, "bold"),
                 bg="#16013e", fg="#a78bfa").pack()
        tk.Label(top, text="Listen before you save",
                 font=("Courier New", 9), bg="#16013e", fg=D).pack()

        # ── Info card
        info = tk.Frame(self, bg=C, padx=20, pady=12)
        info.pack(fill="x", padx=14, pady=(12, 6))

        tk.Label(info, text=f"📁  {os.path.basename(self._tmp_file)}",
                 bg=C, fg=T, font=("Consolas", 10, "bold")).pack(anchor="w")
        tk.Label(info, text=f"⏱   Duration: {self._duration:.2f}s",
                 bg=C, fg=A2, font=("Consolas", 10)).pack(anchor="w", pady=(4, 0))

        # ── Playback status label
        self._status_var = tk.StringVar(value="⏳ Starting playback…")
        tk.Label(self, textvariable=self._status_var,
                 bg=B, fg=Y, font=("Consolas", 10, "italic")).pack(pady=(4, 2))

        # ── Play Again button
        tk.Button(self, text="▶  Play Again",
                  font=("Consolas", 10, "bold"),
                  bg="#1e3a5f", fg=A2, activebackground="#1e4a7f",
                  relief="flat", bd=0, padx=16, pady=7, cursor="hand2",
                  command=self._play).pack(pady=(4, 10))

        # ── Separator
        tk.Frame(self, bg="#2d2d4e", height=1).pack(fill="x", padx=14)

        # ── Question label
        tk.Label(self, text="Are you satisfied with this audio?",
                 bg=B, fg=T, font=("Consolas", 11, "bold")).pack(pady=(12, 8))

        # ── Action buttons row
        btn_row = tk.Frame(self, bg=B)
        btn_row.pack(pady=(0, 18))

        tk.Button(btn_row, text="✅  Save",
                  font=("Consolas", 11, "bold"),
                  bg=G, fg="#052e16", activebackground="#16a34a",
                  relief="flat", bd=0, padx=20, pady=10, cursor="hand2",
                  command=self._do_save).pack(side="left", padx=8)

        tk.Button(btn_row, text="🔁  Regenerate",
                  font=("Consolas", 11, "bold"),
                  bg=Y, fg="#451a03", activebackground="#d97706",
                  relief="flat", bd=0, padx=20, pady=10, cursor="hand2",
                  command=self._do_regen).pack(side="left", padx=8)

        tk.Button(btn_row, text="🗑  Delete",
                  font=("Consolas", 11, "bold"),
                  bg=R, fg="white", activebackground="#dc2626",
                  relief="flat", bd=0, padx=20, pady=10, cursor="hand2",
                  command=self._do_delete).pack(side="left", padx=8)

    def _play(self):
        self._status_var.set("🔊 Playing…")
        _stop_audio()
        _play_audio(self._tmp_file)
        # Update status after estimated duration
        ms = max(500, int(self._duration * 1000) + 500)
        self.after(ms, lambda: self._status_var.set("✅ Playback finished"))

    def _do_save(self):
        _stop_audio()
        self._decision = "save"
        self.destroy()
        self._on_save()

    def _do_regen(self):
        _stop_audio()
        self._decision = "regen"
        self.destroy()
        self._on_regen()

    def _do_delete(self):
        _stop_audio()
        self._decision = "delete"
        self.destroy()
        self._on_delete()


# ─────────────────────────────────────────────
#  Main GUI
# ─────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎙️ ChatterboxMix – Hindi + English TTS")
        self.geometry("820x760")
        self.resizable(True, True)
        self.configure(bg="#0f0f1a")

        self._voice_path_en = tk.StringVar(value="")
        self._voice_path_hi = tk.StringVar(value="")
        self._out_path      = tk.StringVar(value="output_mixed.wav")
        self._exaggeration  = tk.DoubleVar(value=0.5)
        self._running       = False

        self._build_ui()

        # Load models in background
        threading.Thread(target=load_models,
                         args=(self._log,), daemon=True).start()

    # ── UI construction ──────────────────────
    def _build_ui(self):
        BG   = "#0f0f1a"
        CARD = "#1a1a2e"
        ACC  = "#7c3aed"   # violet
        ACC2 = "#06b6d4"   # cyan
        TXT  = "#e2e8f0"
        DIM  = "#64748b"

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TXT, font=("Consolas", 10))
        style.configure("TFrame", background=CARD)
        style.configure("TLabel", background=CARD, foreground=TXT)
        style.configure("TButton", background=ACC, foreground="white",
                        borderwidth=0, focuscolor="none", relief="flat")
        style.map("TButton",
                  background=[("active", "#6d28d9"), ("pressed", "#5b21b6")])
        style.configure("Accent2.TButton", background=ACC2, foreground="black")
        style.map("Accent2.TButton",
                  background=[("active", "#0891b2"), ("pressed", "#0e7490")])
        style.configure("TEntry", fieldbackground="#16213e",
                        foreground=TXT, insertcolor=TXT)
        style.configure("TScale", background=CARD, troughcolor="#16213e",
                        sliderthickness=16)

        # ── Header
        hdr = tk.Frame(self, bg="#16013e", pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🎙️  ChatterboxMix",
                 font=("Courier New", 18, "bold"),
                 bg="#16013e", fg="#a78bfa").pack()
        tk.Label(hdr, text="Hindi + English · auto-detect · voice cloning",
                 font=("Courier New", 9), bg="#16013e", fg=DIM).pack()

        # ── Main frame
        main = tk.Frame(self, bg=BG, padx=14, pady=10)
        main.pack(fill="both", expand=True)

        # ── Lines editor
        lines_card = tk.LabelFrame(main, text="  📝 Script Lines  ",
                                   bg=CARD, fg=ACC2,
                                   font=("Consolas", 10, "bold"),
                                   relief="flat", bd=2)
        lines_card.pack(fill="x", pady=(0, 8))

        # Instruction labels — updated for new format
        tk.Label(lines_card,
                 text="Prefix:  [EN] = English   [HI] = Hindi   (or paste mixed text — auto-detected!)",
                 bg=CARD, fg=DIM, font=("Consolas", 9)).pack(anchor="w", padx=8, pady=(4,0))
        tk.Label(lines_card,
                 text="Example:  [EN] Hello my name is Sankhadeep",
                 bg=CARD, fg=DIM, font=("Consolas", 9)).pack(anchor="w", padx=8)
        tk.Label(lines_card,
                 text="          [HI] mera naam Sankhadeep hai",
                 bg=CARD, fg=DIM, font=("Consolas", 9)).pack(anchor="w", padx=8)
        tk.Label(lines_card,
                 text="          hello mera naam hai ORANGES pasand hain  <- auto-split!",
                 bg=CARD, fg=DIM, font=("Consolas", 9)).pack(anchor="w", padx=8, pady=(0,6))

        # ── Auto-detect button
        btn_frame = tk.Frame(lines_card, bg=CARD)
        btn_frame.pack(fill="x", padx=8, pady=(0, 4))
        tk.Button(
            btn_frame, text="🔍  Preview Auto-Split",
            font=("Consolas", 9),
            bg="#1e293b", fg=ACC2, activebackground="#334155",
            relief="flat", bd=0, padx=10, pady=4, cursor="hand2",
            command=self._preview_split
        ).pack(side="left")

        self.txt_lines = scrolledtext.ScrolledText(
            lines_card, height=9, font=("Consolas", 11),
            bg="#0d0d1f", fg=TXT, insertbackground=TXT,
            selectbackground=ACC, relief="flat", bd=0,
            wrap="word"
        )
        self.txt_lines.pack(fill="x", padx=8, pady=(0, 8))

        # Sample text using new number format
        self.txt_lines.insert("end",
            "[EN] Hello, my name is Sankhadeep.\n"
            "[HI] mera naam Sankhadeep hai.\n"
            "[EN] I love building cool projects.\n"
            "[HI] aaj ka din bahut accha hai.\n"
            "hello mera naam Sankhadeep hai, main ek accha ladka hoon. Mujhe ORANGES pasand hain.\n"
        )

        # ── Voice files
        vf = tk.LabelFrame(main, text="  🔊 Voice Reference Files  ",
                            bg=CARD, fg=ACC2,
                            font=("Consolas", 10, "bold"),
                            relief="flat", bd=2)
        vf.pack(fill="x", pady=(0, 8))
        vf.columnconfigure(1, weight=1)

        self._file_row(vf, "English voice (.mp3/.wav):",
                       self._voice_path_en, 0)
        self._file_row(vf, "Hindi voice   (.mp3/.wav):",
                       self._voice_path_hi, 1)

        # ── Settings row
        cfg = tk.LabelFrame(main, text="  ⚙️  Settings  ",
                             bg=CARD, fg=ACC2,
                             font=("Consolas", 10, "bold"),
                             relief="flat", bd=2)
        cfg.pack(fill="x", pady=(0, 8))
        cfg.columnconfigure(2, weight=1)

        tk.Label(cfg, text="Exaggeration (EN):",
                 bg=CARD, fg=TXT).grid(row=0, column=0, padx=10, pady=6, sticky="w")
        tk.Scale(cfg, from_=0.0, to=1.0, resolution=0.05,
                 orient="horizontal", variable=self._exaggeration,
                 bg=CARD, fg=TXT, highlightthickness=0,
                 troughcolor="#16213e", activebackground=ACC,
                 length=200).grid(row=0, column=1, padx=6)
        self._lbl_exag = tk.Label(cfg, textvariable=self._exaggeration,
                                   bg=CARD, fg=ACC2,
                                   font=("Consolas", 10, "bold"))
        self._lbl_exag.grid(row=0, column=2, padx=4)

        tk.Label(cfg, text="Output file:", bg=CARD, fg=TXT)\
            .grid(row=1, column=0, padx=10, pady=6, sticky="w")
        tk.Entry(cfg, textvariable=self._out_path, width=34,
                 font=("Consolas", 10),
                 bg="#0d0d1f", fg=TXT, insertbackground=TXT,
                 relief="flat", bd=4)\
            .grid(row=1, column=1, columnspan=2,
                  padx=6, pady=6, sticky="w")

        # ── Buttons
        btn_row = tk.Frame(main, bg=BG)
        btn_row.pack(fill="x", pady=(2, 6))

        self.btn_gen = tk.Button(
            btn_row, text="▶  Generate & Combine",
            font=("Consolas", 12, "bold"),
            bg=ACC, fg="white", activebackground="#6d28d9",
            relief="flat", bd=0, padx=20, pady=10, cursor="hand2",
            command=self._on_generate
        )
        self.btn_gen.pack(side="left", padx=(0, 10))

        tk.Button(
            btn_row, text="🗑  Clear Log",
            font=("Consolas", 10),
            bg="#1e293b", fg=DIM, activebackground="#334155",
            relief="flat", bd=0, padx=14, pady=10, cursor="hand2",
            command=lambda: self.log_box.delete("1.0", "end")
        ).pack(side="left")

        # ── Log box
        log_card = tk.LabelFrame(main, text="  📋 Log  ",
                                  bg=CARD, fg=ACC2,
                                  font=("Consolas", 10, "bold"),
                                  relief="flat", bd=2)
        log_card.pack(fill="both", expand=True)

        self.log_box = scrolledtext.ScrolledText(
            log_card, height=10, state="disabled",
            font=("Consolas", 9),
            bg="#060612", fg="#94a3b8",
            insertbackground=TXT, relief="flat", bd=0
        )
        self.log_box.pack(fill="both", expand=True, padx=6, pady=6)

        # colour tags
        self.log_box.tag_config("ok",   foreground="#4ade80")
        self.log_box.tag_config("err",  foreground="#f87171")
        self.log_box.tag_config("info", foreground="#38bdf8")
        self.log_box.tag_config("dim",  foreground="#64748b")

    def _file_row(self, parent, label, var, row):
        tk.Label(parent, text=label, bg="#1a1a2e", fg="#e2e8f0")\
            .grid(row=row, column=0, padx=10, pady=5, sticky="w")
        tk.Entry(parent, textvariable=var, width=38,
                 font=("Consolas", 10),
                 bg="#0d0d1f", fg="#e2e8f0", insertbackground="#e2e8f0",
                 relief="flat", bd=4)\
            .grid(row=row, column=1, padx=4, pady=5, sticky="ew")
        tk.Button(parent, text="Browse",
                  font=("Consolas", 9),
                  bg="#334155", fg="#e2e8f0",
                  activebackground="#475569",
                  relief="flat", bd=0, padx=8, pady=4,
                  cursor="hand2",
                  command=lambda v=var: self._browse(v))\
            .grid(row=row, column=2, padx=(0, 10), pady=5)

    # ── helpers ──────────────────────────────
    def _browse(self, var):
        p = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.mp3 *.wav *.m4a"), ("All", "*.*")])
        if p:
            var.set(p)

    def _log(self, msg, tag="info"):
        def _do():
            self.log_box.configure(state="normal")
            t = tag
            if any(x in msg for x in ("✅", "Done", "ready")):
                t = "ok"
            elif any(x in msg for x in ("❌", "Error", "failed")):
                t = "err"
            elif any(x in msg for x in ("⏳", "Loading", "Generating")):
                t = "info"
            self.log_box.insert("end", msg + "\n", t)
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.after(0, _do)

    def _preview_split(self):
        """Show how the current text will be parsed into segments."""
        raw = self.txt_lines.get("1.0", "end").strip()
        if not raw:
            messagebox.showinfo("Preview", "Text box is empty.")
            return
        try:
            lines = parse_lines(raw)
        except ValueError as e:
            messagebox.showerror("Parse Error", str(e))
            return

        if not lines:
            messagebox.showinfo("Preview", "No segments detected.")
            return

        preview = ""
        for i, (lang, text) in enumerate(lines, 1):
            label = "🇬🇧 EN" if lang == "en" else "🇮🇳 HI"
            preview += f"{i:>2}. [{label}]  {text}\n"

        win = tk.Toplevel(self)
        win.title("Auto-Split Preview")
        win.configure(bg="#0f0f1a")
        win.geometry("640x400")
        tk.Label(win, text="How your text will be split into segments:",
                 bg="#0f0f1a", fg="#06b6d4",
                 font=("Consolas", 10, "bold")).pack(anchor="w", padx=10, pady=(10,4))
        box = scrolledtext.ScrolledText(win, font=("Consolas", 10),
                                         bg="#0d0d1f", fg="#e2e8f0",
                                         relief="flat", bd=0)
        box.pack(fill="both", expand=True, padx=10, pady=(0,10))
        box.insert("end", preview)
        box.configure(state="disabled")

    # ── generation logic ─────────────────────
    def _on_generate(self):
        if self._running:
            return
        raw = self.txt_lines.get("1.0", "end").strip()
        if not raw:
            messagebox.showwarning("Empty", "Please enter at least one line.")
            return

        try:
            lines = parse_lines(raw)
        except ValueError as e:
            messagebox.showerror("Format Error", str(e))
            return

        if not lines:
            messagebox.showwarning("Empty", "No valid lines found.")
            return

        self._running = True
        self.btn_gen.configure(state="disabled", text="⏳ Generating…")
        threading.Thread(
            target=self._generate_thread,
            args=(lines,), daemon=True
        ).start()

    def _generate_thread(self, lines):
        voice_en = self._voice_path_en.get().strip() or None
        voice_hi = self._voice_path_hi.get().strip() or None
        exag     = self._exaggeration.get()
        out_file = self._out_path.get().strip() or "output_mixed.wav"

        # Save to a temp file first; only rename to out_file if user approves
        tmp_file = out_file + ".preview_tmp.wav"

        segments = []
        ok = True

        for i, (lang, text) in enumerate(lines, 1):
            flag = "🇮🇳" if lang == "hi" else "🇬🇧"
            self._log(f"\n[{i}/{len(lines)}] {flag}  {text[:60]}{'…' if len(text)>60 else ''}")
            try:
                voice = voice_hi if lang == "hi" else voice_en
                wav, sr = generate_segment(text, lang, voice, exag)
                segments.append((wav, sr))
                self._log(f"  ✅ Segment {i} done  ({wav.shape[-1]/sr:.2f}s)")

                seg_name = f"segment_{i:02d}_{lang}.wav"
                ta.save(seg_name, wav if wav.dim()==2 else wav.unsqueeze(0), sr)
                self._log(f"  💾 Saved: {seg_name}", "dim")

            except Exception as e:
                self._log(f"  ❌ Error on segment {i}: {e}", "err")
                ok = False
                break

        if ok and segments:
            self._log("\n🔗 Combining all segments…")
            try:
                combined, sr = combine_wavs(segments)
                ta.save(tmp_file,
                        combined if combined.dim()==2 else combined.unsqueeze(0),
                        sr)
                dur = combined.shape[-1] / sr
                self._log(f"✅ Generation done! ({dur:.2f}s, {len(segments)} segments)", "ok")
                self._log("🎧 Opening audio preview…", "info")

                # ── Open preview dialog on the main thread
                self.after(0, lambda: self._show_preview(tmp_file, out_file, dur))

            except Exception as e:
                self._log(f"❌ Combine failed: {e}", "err")
                self._running = False
                self.after(0, lambda: self.btn_gen.configure(
                    state="normal", text="▶  Generate & Combine"))
        else:
            self._running = False
            self.after(0, lambda: self.btn_gen.configure(
                state="normal", text="▶  Generate & Combine"))

    def _show_preview(self, tmp_file, out_file, duration):
        """Open the AudioPreviewDialog (must be called on main thread)."""
        self.btn_gen.configure(state="normal", text="▶  Generate & Combine")
        self._running = False

        def on_save():
            self._save_output(tmp_file, out_file)

        def on_regen():
            self._delete_tmp(tmp_file)
            self._log("🔁 Regenerating…", "info")
            # Re-trigger generation with current text
            self._on_generate()

        def on_delete():
            self._delete_tmp(tmp_file)
            self._log("🗑  Audio deleted. Generation cancelled.", "err")

        AudioPreviewDialog(self, tmp_file, out_file, duration,
                           on_save=on_save,
                           on_regen=on_regen,
                           on_delete=on_delete)

    def _save_output(self, tmp_file, out_file):
        """Move temp file → final output, then run copy.py."""
        try:
            if os.path.exists(out_file):
                os.remove(out_file)
            os.rename(tmp_file, out_file)
            self._log(f"💾 Saved → {out_file}", "ok")
        except Exception as e:
            self._log(f"❌ Save failed: {e}", "err")
            return

        # 🔹 Run copy.py only after user confirms save
        try:
            subprocess.run(["python", "copy.py"], check=True)
            self._log("📂 copy.py executed successfully", "ok")
        except Exception as e:
            self._log(f"❌ Failed to run copy.py: {e}", "err")

    def _delete_tmp(self, tmp_file):
        """Remove the temporary preview file."""
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
