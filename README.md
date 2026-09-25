# 🎧 AI DJ — Gesture Controlled Audio Mixer

> **Turn your hands into a DJ controller.** 🖐️🎵
> Control music, vocals, volume, bass & treble using real-time hand gestures.

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge\&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=for-the-badge\&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange?style=for-the-badge)
![SoundDevice](https://img.shields.io/badge/SoundDevice-Audio-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

## 🎛️ What is AI DJ?

**AI DJ** is an interactive computer-vision music controller that lets you manipulate audio using your hands.

**No keyboard. No mouse. No physical DJ controller.**

**Gesture → Action → Music 🎶**

```text
🖐️ Hand Gestures
       ↓
MediaPipe Hand Landmarker
       ↓
Gesture Detection
       ↓
┌──────────┬──────────┬──────────┐
│ 🎚️Volume │ 🎤Vocals │ 🎵Playback│
└──────────┴──────────┴──────────┘
       ↓
🔊 Audio Output
```

## 🖐️ Gesture Controls

| Gesture             | Action                   |
| ------------------- | ------------------------ |
| ☝️ Finger movement  | 🎚️ Control parameters   |
| 🤏 Pinch / distance | 🔊 Adjust volume         |
| ✋ Hand movement     | 🎤 Manipulate vocals     |
| ✊ Fist              | ⏭️ Next / Previous track |
| ✊✊ Both fists       | ▶️ Play / ⏸️ Pause       |
| 🎚️ Hand position   | 🎛️ Bass & Treble        |

> Make music interaction feel like using a physical DJ setup — without the hardware.

## ⚡ Core Features

* 🎵 **Real-Time Music Control** — Control playback and audio parameters while music is playing.
* 🖐️ **Computer Vision** — MediaPipe Hand Landmarker + OpenCV for real-time hand tracking.
* 🎤 **Vocal Control** — Manipulate separated vocal components.
* 🎚️ **Audio Effects** — Volume, vocals, bass and treble controls.
* ⚡ **Touch-Free Interaction** — Turn natural hand movements into music controls.

## 🧠 Tech Stack

```text
Python
├── OpenCV       → Computer Vision
├── MediaPipe    → Hand Landmark Detection
├── NumPy        → Audio/Data Processing
├── SoundDevice  → Real-Time Audio Output
└── Demucs       → Vocal Separation
```

## 🚀 Getting Started

### 1️⃣ Clone

```bash
git clone https://github.com/manavupadhyay1/DJ-Audio-Controller.git
cd DJ-Audio-Controller
```

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install opencv-python mediapipe numpy sounddevice
```

### 4️⃣ Model

Make sure `hand_landmarker.task` is in the project root:

```text
DJ-Audio-Controller/
├── hand_landmarker.task
├── king2_fixed.py
└── ...
```

### 5️⃣ Add Music

Place your audio files inside:

```text
music/
```

Large audio files are intentionally excluded from GitHub.

### 6️⃣ Run

```bash
python3 king2_fixed.py
```

Allow camera and audio permissions when macOS asks.

## 📂 Project Structure

```text
DJ-Audio-Controller/
├── 🎧 music/
│   └── .gitkeep
├── 🔊 separated/
│   └── .gitkeep
├── 🧠 hand_landmarker.task
├── 🎛️ king.py
├── 🎛️ king2.py
├── 🚀 king2_fixed.py
├── 📦 king2_old.py
├── 🔊 dj_audio_controller.py
└── 📄 README.md
```

## 🔥 Why is this interesting?

Traditional DJ setup:

```text
DJ Controller → Knobs → Faders → Buttons → Music
```

AI DJ:

```text
Your Hands
    ↓
Computer Vision
    ↓
Gesture Recognition
    ↓
Audio Controls
    ↓
🎵 Music
```

This combines **Computer Vision + AI + Signal Processing + Human-Computer Interaction** into a touch-free music interface.

## 🎯 Future Improvements

* [ ] 🎚️ Advanced DJ effects
* [ ] 🌀 Gesture-controlled filters
* [ ] 🎵 Automatic beat detection
* [ ] 🎼 BPM synchronization
* [ ] 🔥 Reverb & echo
* [ ] 🎤 Improved vocal isolation
* [ ] 📊 Real-time audio visualizer
* [ ] 🖥️ Custom DJ dashboard
* [ ] 🤖 Personalized gesture recognition
* [ ] 📱 Remote/mobile controller

## 🎥 Demo

**Coming soon...**

Add your demo GIF here:

```markdown
![AI DJ Demo](demo.gif)
```

## 💡 Concept

> **"What if your hands were the DJ controller?"**

AI DJ explores a new approach to human-computer interaction by transforming natural hand movements into real-time music controls.

## 👨‍💻 Author

**Manav Upadhyay**
BTech Information Technology

⭐ If you found this project interesting, consider giving the repository a star!

[🎧 View Repository](https://github.com/manavupadhyay1/DJ-Audio-Controller)

**🖐️ Move your hands. 🎵 Control the music.**

*Built with Python • OpenCV • MediaPipe • SoundDevice*
