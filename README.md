# free-narrator-voice
# 🎙️ ChatterboxMix – Hindi + English AI Voice Generator

A powerful Python-based desktop Text-to-Speech (TTS) application that automatically detects and splits Hindi + English mixed text, generates realistic AI speech using Chatterbox TTS models, supports voice cloning, and combines everything into a single seamless audio output.

Perfect for:

* 🎬 YouTube automation
* 🎙️ AI voiceovers
* 🌍 Multilingual narration
* 🎧 Dubbing projects
* 🤖 Experimental AI TTS apps
* 📱 Content creation

---

# ✨ Features

* ✅ Auto-detect Hindi & English mixed sentences
* ✅ Automatic sentence splitting
* ✅ Voice cloning support
* ✅ Separate Hindi & English voice references
* ✅ Modern Tkinter GUI
* ✅ Audio preview before saving
* ✅ Regenerate audio instantly
* ✅ Real-time logs
* ✅ WAV combining
* ✅ Unicode-based language detection
* ✅ CPU compatible
* ✅ Segment-wise audio saving

---

# 🛠️ Built With

* Python
* Tkinter
* PyTorch
* Torchaudio
* Chatterbox TTS
* Pygame

---

# 📦 Requirements

Install Python 3.10+ first.

Then install dependencies:

```bash
pip install torch torchaudio pygame
```

Install Chatterbox TTS:

```bash
pip install chatterbox-tts
```

---

# ▶️ How To Run

Run the application using:

```bash
python B.py
```

The GUI will open automatically.

---

# 🧠 How It Works

The app:

1. Detects Hindi & English text automatically
2. Splits mixed-language sentences
3. Generates speech using AI models
4. Combines all audio segments
5. Lets you preview before saving

---

# 📝 Supported Input Formats

### Manual Language Tags

```text
[EN] Hello my name is Sankhadeep
[HI] mera naam Sankhadeep hai
```

### Auto Detection

```text
hello mera naam Sankhadeep hai, mujhe ORANGES pasand hain
```

The app automatically splits and processes both languages separately.

---

# 🎤 Voice Cloning

You can provide:

* English voice sample (.wav/.mp3)
* Hindi voice sample (.wav/.mp3)

The AI will try to clone the voice style from the reference audio.

---

# 📂 Output

Generated audio is saved as:

```text
output_mixed.wav
```

Each segment is also saved separately:

```text
segment_01_en.wav
segment_02_hi.wav
```

---

# 🎧 Audio Preview System

After generation:

* ▶️ Play generated audio
* 🔁 Regenerate instantly
* ✅ Save final output
* 🗑️ Delete preview audio

---

# ⚡ Notes

* First launch may take time because models load lazily
* CPU works fine but GPU is faster
* Better voice samples = better cloning quality
* Short clean reference audios work best

---

# 📸 GUI Preview

Modern dark-themed interface with:

* Script editor
* Voice file selection
* Exaggeration controls
* Live logs
* Audio preview system

---

# 🚀 Future Improvements

* GPU acceleration
* More language support
* MP3 export
* Emotion controls
* Batch generation
* Subtitle export

---

# 📜 License

This project is for educational and experimental purposes.

---

# 👨‍💻 Author

Made by SANKHADEEP
