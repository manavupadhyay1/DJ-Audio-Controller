import cv2
import mediapipe as mp
import math
import os
import sounddevice as sd
import numpy as np
from scipy.signal import butter, lfilter
import threading
import librosa
import soundfile as sf
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings
import urllib.request
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

warnings.filterwarnings('ignore')

# ============================================================================
# REAL-TIME AUDIO CONTROLLER WITH VOCAL SEPARATION (DEMUCS)
# ============================================================================

class RealtimeAudioController:
    def __init__(self, music_folder="music", separated_folder="separated"):
        """Initialize real-time audio controller with automatic vocal separation"""
        self.music_folder = music_folder
        self.separated_folder = separated_folder
        self.playlist = self._load_playlist()
        self.current_track_index = 0
        self.is_playing = False
        
        # Initialize Demucs model
        print("🔧 Loading Demucs model (this may take a moment)...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Using device: {self.device.upper()}")
        self.model = get_model('htdemucs')
        self.model.to(self.device)
        self.model.eval()
        print("✅ Demucs ready!")
        
        # Sounddevice setup
        self.stream = None
        
        # Audio data (separate tracks)
        self.vocals_data = None
        self.accompaniment_data = None
        self.audio_position = 0
        
        # Audio settings
        self.chunk = 1024
        self.sample_rate = 44100
        
        # Effect parameters
        self.volume = 0.5
        self.bass_gain = 0.0
        self.treble_gain = 0.0
        self.vocal_mix = 1.0  # 1.0 = full vocals, 0.0 = beats only
        
        # Thread control
        self.stop_thread = False
        self.audio_thread = None
        
        # Create separated folder if it doesn't exist
        if not os.path.exists(self.separated_folder):
            os.makedirs(self.separated_folder)
        
        print(f"✅ Audio Controller: {len(self.playlist)} tracks loaded")
        if not self.playlist:
            print("⚠️ Add MP3/WAV files to 'music' folder")
    
    def _load_playlist(self):
        """Load all audio files from music folder"""
        if not os.path.exists(self.music_folder):
            os.makedirs(self.music_folder)
            print(f"📁 Created '{self.music_folder}' folder")
            return []
        
        supported = ['.mp3', '.wav', '.ogg', '.flac', '.m4a']
        playlist = []
        for file in os.listdir(self.music_folder):
            if any(file.lower().endswith(fmt) for fmt in supported):
                full_path = os.path.join(self.music_folder, file)
                playlist.append(full_path)
        return sorted(playlist)
    
    def _separate_vocals(self, track_path):
        """Separate vocals and accompaniment using Demucs"""
        track_name = os.path.splitext(os.path.basename(track_path))[0]
        vocals_path = os.path.join(self.separated_folder, f"{track_name}_vocals.wav")
        accompaniment_path = os.path.join(self.separated_folder, f"{track_name}_accompaniment.wav")
        
        # Check if already separated
        if os.path.exists(vocals_path) and os.path.exists(accompaniment_path):
            print(f"✅ Using cached separated files")
            return vocals_path, accompaniment_path
        
        print(f"🎵 Separating vocals with Demucs (30-90 seconds)...")
        print(f"   Processing: {track_name}")
        
        try:
            # Load audio
            wav, sr = librosa.load(track_path, sr=44100, mono=False)
            
            # Convert to stereo if mono
            if len(wav.shape) == 1:
                wav = np.stack([wav, wav])
            
            # Convert to torch tensor
            wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
            
            # Apply separation
            with torch.no_grad():
                sources = apply_model(self.model, wav_tensor, device=self.device)
            
            # Extract vocals and other stems
            sources = sources.cpu().numpy()[0]
            
            vocals = sources[3]  # vocals
            drums = sources[0]   # drums
            bass = sources[1]    # bass
            other = sources[2]   # other instruments
            
            # Combine non-vocal stems for accompaniment
            accompaniment = drums + bass + other
            
            # Save separated files
            sf.write(vocals_path, vocals.T, sr)
            sf.write(accompaniment_path, accompaniment.T, sr)
            
            print(f"✅ Separation complete!")
            return vocals_path, accompaniment_path
            
        except Exception as e:
            print(f"❌ Separation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def butter_lowpass(self, cutoff, fs, order=5):
        """Create low-pass filter for bass"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def butter_highpass(self, cutoff, fs, order=5):
        """Create high-pass filter for treble"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a
    
    def apply_effects(self, vocals_chunk, accompaniment_chunk):
        """Apply real-time audio effects and mix vocals/accompaniment"""
        try:
            if len(vocals_chunk) == 0 or len(accompaniment_chunk) == 0:
                return np.zeros(self.chunk)
            
            # Mix vocals and accompaniment based on vocal_mix
            mixed = (vocals_chunk * self.vocal_mix) + accompaniment_chunk
            
            # Apply bass boost
            if abs(self.bass_gain) > 0.01:
                try:
                    b_low, a_low = self.butter_lowpass(250, self.sample_rate, order=4)
                    bass = lfilter(b_low, a_low, mixed)
                    mixed = mixed + (bass * self.bass_gain)
                except:
                    pass
            
            # Apply treble boost
            if abs(self.treble_gain) > 0.01:
                try:
                    b_high, a_high = self.butter_highpass(3000, self.sample_rate, order=4)
                    treble = lfilter(b_high, a_high, mixed)
                    mixed = mixed + (treble * self.treble_gain)
                except:
                    pass
            
            # Apply volume
            mixed = mixed * self.volume
            
            # Clip to prevent distortion
            mixed = np.clip(mixed, -1.0, 1.0)
            
            return mixed
        
        except Exception as e:
            return np.zeros(self.chunk)
    
    def load_track(self, index):
        """Load specific track by index and separate vocals"""
        if not self.playlist:
            return False
        
        self.current_track_index = index % len(self.playlist)
        track_path = self.playlist[self.current_track_index]
        
        try:
            print(f"\n{'='*60}")
            print(f"📂 Loading: {os.path.basename(track_path)}")
            print('='*60)
            
            # Separate vocals and accompaniment
            vocals_path, accompaniment_path = self._separate_vocals(track_path)
            
            if vocals_path is None or accompaniment_path is None:
                print(f"❌ Failed to load separated tracks")
                return False
            
            # Load both tracks with librosa
            print(f"📥 Loading separated tracks...")
            self.vocals_data, _ = librosa.load(vocals_path, sr=self.sample_rate, mono=True)
            self.accompaniment_data, _ = librosa.load(accompaniment_path, sr=self.sample_rate, mono=True)
            
            # Make sure both tracks are the same length
            min_length = min(len(self.vocals_data), len(self.accompaniment_data))
            self.vocals_data = self.vocals_data[:min_length]
            self.accompaniment_data = self.accompaniment_data[:min_length]
            
            self.audio_position = 0
            
            duration = len(self.vocals_data) / self.sample_rate
            print(f"✅ Ready to play! Duration: {duration:.1f}s")
            print('='*60 + "\n")
            return True
            
        except Exception as e:
            print(f"❌ Error loading track: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _audio_playback_thread(self):
        """Thread for audio playback with vocal mixing"""
        try:
            while not self.stop_thread and self.vocals_data is not None:
                if not self.is_playing:
                    continue
                
                # Get next chunk from both tracks
                start = self.audio_position
                end = start + self.chunk
                
                if end >= len(self.vocals_data):
                    # Loop back to start
                    vocals_chunk = self.vocals_data[start:]
                    accompaniment_chunk = self.accompaniment_data[start:]
                    self.audio_position = 0
                else:
                    vocals_chunk = self.vocals_data[start:end]
                    accompaniment_chunk = self.accompaniment_data[start:end]
                    self.audio_position = end
                
                # Apply effects and mix
                processed_chunk = self.apply_effects(vocals_chunk, accompaniment_chunk)
                
                # Convert to bytes for playback
                audio_bytes = (processed_chunk * 32767).astype(np.int16).tobytes()
                
                # Play audio
                if self.stream:
                    try:
                        self.stream.write(audio_bytes)
                    except:
                        pass
        
        except Exception as e:
            print(f"❌ Playback error: {e}")
    
    def play(self):
        """Start playback"""
        if not self.playlist:
            return
        
        if self.vocals_data is None:
            if not self.load_track(self.current_track_index):
                return
        
        if self.stream is None:
            # Open stream
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk
            )
            
            # Start playback thread
            self.stop_thread = False
            self.audio_thread = threading.Thread(target=self._audio_playback_thread)
            self.audio_thread.daemon = True
            self.audio_thread.start()
        
        self.is_playing = True
    
    def pause(self):
        """Pause playback"""
        self.is_playing = False
    
    def unpause(self):
        """Resume playback"""
        self.is_playing = True
    
    def toggle(self):
        """Toggle play/pause"""
        if self.is_playing:
            self.pause()
        else:
            self.unpause()
    
    def stop(self):
        """Stop playback completely"""
        self.stop_thread = True
        self.is_playing = False
        
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        self.vocals_data = None
        self.accompaniment_data = None
        self.audio_position = 0
    
    def next_track(self):
        """Skip to next track"""
        if not self.playlist:
            return
        
        was_playing = self.is_playing
        self.stop()
        
        self.current_track_index = (self.current_track_index + 1) % len(self.playlist)
        
        if was_playing:
            self.play()
    
    def previous_track(self):
        """Go to previous track"""
        if not self.playlist:
            return
        
        was_playing = self.is_playing
        self.stop()
        
        self.current_track_index = (self.current_track_index - 1) % len(self.playlist)
        
        if was_playing:
            self.play()
    
    def set_volume(self, value):
        """Set volume (0-100)"""
        value = max(0, min(100, value))
        self.volume = value / 100.0
    
    def set_bass(self, value):
        """Set bass level (0-100)"""
        value = max(0, min(100, value))
        self.bass_gain = (value - 50) / 25.0  # -2 to +2
    
    def set_treble(self, value):
        """Set treble level (0-100)"""
        value = max(0, min(100, value))
        self.treble_gain = (value - 50) / 25.0  # -2 to +2
    
    def set_vocals(self, value):
        """Set vocal level (0-100) - 0% = beats only, 100% = full vocals"""
        value = max(0, min(100, value))
        self.vocal_mix = value / 100.0
    
    def get_track_name(self):
        """Get current track name"""
        if not self.playlist:
            return "No tracks"
        return os.path.basename(self.playlist[self.current_track_index])
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()

# ============================================================================
# HAND GESTURE DETECTION FUNCTIONS
# ============================================================================

def calculate_rotation(hand_landmarks):
    """Calculate hand rotation angle"""
    wrist = hand_landmarks.landmark[0]
    middle_base = hand_landmarks.landmark[9]
    angle = math.degrees(math.atan2(middle_base.y - wrist.y, middle_base.x - wrist.x))
    return angle

def get_hand_center(hand_landmarks):
    """Get center point of hand"""
    center_x = hand_landmarks.landmark[9].x
    center_y = hand_landmarks.landmark[9].y
    return center_x, center_y

def is_hand_closed(hand_landmarks):
    """Check if hand is closed (fist)"""
    wrist = hand_landmarks.landmark[0]
    finger_tips = [4, 8, 12, 16, 20]
    finger_bases = [2, 5, 9, 13, 17]
    closed_count = 0
    
    for tip, base in zip(finger_tips, finger_bases):
        tip_pos = hand_landmarks.landmark[tip]
        base_pos = hand_landmarks.landmark[base]
        tip_to_wrist = math.sqrt((tip_pos.x - wrist.x)**2 + (tip_pos.y - wrist.y)**2)
        base_to_wrist = math.sqrt((base_pos.x - wrist.x)**2 + (base_pos.y - wrist.y)**2)
        if tip_to_wrist < base_to_wrist * 1.1:
            closed_count += 1
    
    return closed_count >= 4

def detect_dominant_gesture(rotation_diff, y_diff, rotation_threshold=2, vertical_threshold=0.03):
    """Detect the dominant gesture (rotation vs vertical movement)"""
    
    abs_rotation = abs(rotation_diff)
    abs_vertical = abs(y_diff)
    
    rotation_active = abs_rotation > rotation_threshold
    vertical_active = abs_vertical > vertical_threshold
    
    if not rotation_active and not vertical_active:
        return None
    
    if rotation_active and not vertical_active:
        return 'rotation'
    if vertical_active and not rotation_active:
        return 'vertical'
    
    rotation_strength = abs_rotation / rotation_threshold
    vertical_strength = abs_vertical / vertical_threshold
    
    if rotation_strength > vertical_strength * 1.3:
        return 'rotation'
    elif vertical_strength > rotation_strength * 1.3:
        return 'vertical'
    else:
        return None

# ============================================================================
# DJ TKINTER INTERFACE
# ============================================================================

class DJVocalInterface:
    def __init__(self, root, audio_controller):
        self.root = root
        self.audio = audio_controller
        self.running = True
        
        # Configure root window
        self.root.title("🎤 AI DJ - Vocal Separation Controller")
        self.root.geometry("1400x800")
        self.root.configure(bg="#0d0d0d")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # MediaPipe Tasks Hand Landmarker setup
        self.mp = mp
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
        if not os.path.exists(model_path):
            print("📥 Downloading MediaPipe hand landmark model...")
            model_url = (
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/"
                "hand_landmarker.task"
            )
            urllib.request.urlretrieve(model_url, model_path)
            print("✅ Hand landmark model downloaded!")

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = HandLandmarker.create_from_options(options)
        self.mp_hands = mp.tasks.vision
        
        # Effect values
        self.effects = {
            'volume': 50,
            'bass': 50,
            'vocals': 100,
            'treble': 50
        }
        
        # Feature locks
        self.locks = {
            'volume': False,
            'vocals': False,
            'bass': False,
            'treble': False
        }
        
        # Previous positions
        self.prev_positions = {
            'right': {'rotation': None, 'center_y': None},
            'left': {'rotation': None, 'center_y': None}
        }
        
        # State tracking
        self.both_hands_closed = False
        self.right_hand_was_closed = False
        self.left_hand_was_closed = False
        
        # Create UI
        self.create_ui()
        
        # Start camera
        self.cap = cv2.VideoCapture(0)
        self.camera_ok = self.cap.isOpened()
        self.camera_thread = threading.Thread(target=self.update_camera, daemon=True)
        self.camera_thread.start()
        
        # Start status update
        self.update_status()
        
        # Bind keyboard shortcuts
        self.root.bind('v', lambda e: self.toggle_lock('volume'))
        self.root.bind('V', lambda e: self.toggle_lock('volume'))
        self.root.bind('c', lambda e: self.toggle_lock('vocals'))
        self.root.bind('C', lambda e: self.toggle_lock('vocals'))
        self.root.bind('b', lambda e: self.toggle_lock('bass'))
        self.root.bind('B', lambda e: self.toggle_lock('bass'))
        self.root.bind('t', lambda e: self.toggle_lock('treble'))
        self.root.bind('T', lambda e: self.toggle_lock('treble'))
    
    def create_ui(self):
        # Top bar
        top_frame = tk.Frame(self.root, bg="#000000", height=70)
        top_frame.pack(fill=tk.X, side=tk.TOP)
        
        title = tk.Label(top_frame, text="🎤 AI DJ - VOCAL SEPARATION 🎛️", 
                        font=("Helvetica", 26, "bold"),
                        bg="#000000", fg="#ff00ff")
        title.pack(pady=18)
        
        # Main content
        content_frame = tk.Frame(self.root, bg="#0d0d0d")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left - Camera
        left_frame = tk.Frame(content_frame, bg="#1a1a1a", relief=tk.RIDGE, bd=3)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        camera_label = tk.Label(left_frame, text="📹 GESTURE CAMERA", 
                               font=("Helvetica", 14, "bold"),
                               bg="#1a1a1a", fg="#ff00ff")
        camera_label.pack(pady=10)
        
        self.video_label = tk.Label(left_frame, bg="#000000")
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Right - Controls
        right_frame = tk.Frame(content_frame, bg="#0d0d0d", width=450)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Track info
        self.create_track_info(right_frame)
        
        # Playback controls
        self.create_playback_controls(right_frame)
        
        # RIGHT HAND section
        self.create_control_section(right_frame, "🖐️ RIGHT HAND", 
                                    ["Volume", "Vocals"],
                                    ["#00ff88", "#ff0099"])
        
        # LEFT HAND section
        self.create_control_section(right_frame, "🤚 LEFT HAND",
                                    ["Bass", "Treble"],
                                    ["#ff3366", "#3399ff"])
        
        # Instructions (now includes lock info)
        self.create_instructions(right_frame)
        
        # Bottom status
        self.create_status_bar()
    
    def create_track_info(self, parent):
        frame = tk.LabelFrame(parent, text="🎵 NOW PLAYING",
                             font=("Helvetica", 12, "bold"),
                             bg="#1a1a1a", fg="#ff00ff",
                             relief=tk.RIDGE, bd=2)
        frame.pack(fill=tk.X, pady=10)
        
        self.track_label = tk.Label(frame, text="No Track Loaded",
                                    font=("Helvetica", 11),
                                    bg="#1a1a1a", fg="#ffffff",
                                    wraplength=400)
        self.track_label.pack(pady=10, padx=10)
    
    def create_playback_controls(self, parent):
        frame = tk.Frame(parent, bg="#0d0d0d")
        frame.pack(fill=tk.X, pady=5)
        
        btn_style = {
            'font': ("Helvetica", 12, "bold"),
            'bg': "#222222",
            'fg': "#00ff88",
            'activebackground': "#333333",
            'activeforeground': "#00ff88",
            'relief': tk.RAISED,
            'bd': 2,
            'width': 8
        }
        
        prev_btn = tk.Button(frame, text="⏮️ PREV", command=self.prev_track, **btn_style)
        prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = tk.Button(frame, text="▶️ PLAY", command=self.toggle_play, **btn_style)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        next_btn = tk.Button(frame, text="⏭️ NEXT", command=self.next_track, **btn_style)
        next_btn.pack(side=tk.LEFT, padx=5)
    
    def create_control_section(self, parent, title, controls, colors):
        section = tk.LabelFrame(parent, text=title,
                               font=("Helvetica", 12, "bold"),
                               bg="#1a1a1a", fg="#ff00ff",
                               relief=tk.RIDGE, bd=2)
        section.pack(fill=tk.X, pady=10)
        
        control_dict = {}
        for name, color in zip(controls, colors):
            self.create_control_widget(section, name, color, control_dict)
        
        if title == "🖐️ RIGHT HAND":
            self.right_controls = control_dict
        else:
            self.left_controls = control_dict
    
    def create_control_widget(self, parent, name, color, control_dict):
        frame = tk.Frame(parent, bg="#1a1a1a")
        frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Label with lock indicator
        label_frame = tk.Frame(frame, bg="#1a1a1a")
        label_frame.pack(side=tk.LEFT)
        
        label = tk.Label(label_frame, text=name.upper(),
                        font=("Helvetica", 11, "bold"),
                        bg="#1a1a1a", fg=color, width=8, anchor=tk.W)
        label.pack(side=tk.LEFT)
        
        lock_label = tk.Label(label_frame, text="",
                             font=("Helvetica", 10),
                             bg="#1a1a1a", fg="#888888")
        lock_label.pack(side=tk.LEFT, padx=5)
        
        # Canvas for level display
        canvas = tk.Canvas(frame, width=220, height=30,
                          bg="#000000", highlightthickness=0)
        canvas.pack(side=tk.LEFT, padx=10)
        
        # Value label
        value_label = tk.Label(frame, text="50",
                              font=("Helvetica", 11, "bold"),
                              bg="#1a1a1a", fg=color, width=5)
        value_label.pack(side=tk.LEFT)
        
        control_dict[name] = {
            "canvas": canvas,
            "label": value_label,
            "lock_label": lock_label,
            "color": color
        }
    
    def create_lock_controls(self, parent):
        # Create lock buttons but keep them for internal tracking
        self.lock_buttons = {
            'volume': None,
            'vocals': None,
            'bass': None,
            'treble': None
        }
    
    def create_instructions(self, parent):
        frame = tk.LabelFrame(parent, text="📋 GESTURE CONTROLS",
                             font=("Helvetica", 11, "bold"),
                             bg="#1a1a1a", fg="#ff00ff",
                             relief=tk.RIDGE, bd=2)
        frame.pack(fill=tk.X, pady=10)
        
        instructions = [
            "🖐️ RIGHT HAND:",
            "  • Rotate hand = Volume control",
            "  • Move Up/Down = Vocals control",
            "  • Close fist then open = Next track",
            "",
            "🤚 LEFT HAND:",
            "  • Rotate hand = Bass control",
            "  • Move Up/Down = Treble control",
            "  • Close fist then open = Previous track",
            "",
            "👊 BOTH HANDS:",
            "  • Close both fists = Play/Pause toggle",
            "",
            "⌨️ KEYBOARD LOCKS:",
            "  • Press V = Lock/Unlock Volume",
            "  • Press C = Lock/Unlock Vocals",
            "  • Press B = Lock/Unlock Bass",
            "  • Press T = Lock/Unlock Treble"
        ]
        
        for inst in instructions:
            is_header = inst.startswith("🖐️") or inst.startswith("🤚") or inst.startswith("👊") or inst.startswith("⌨️")
            font_style = ("Helvetica", 9, "bold") if is_header else ("Helvetica", 9)
            fg_color = "#00ff88" if is_header else "#cccccc"
            
            lbl = tk.Label(frame, text=inst,
                          font=font_style,
                          bg="#1a1a1a", fg=fg_color,
                          anchor=tk.W)
            lbl.pack(fill=tk.X, padx=10, pady=1)
    
    def create_status_bar(self):
        self.status_frame = tk.Frame(self.root, bg="#000000", height=45)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(self.status_frame,
                                     text="⚡ INITIALIZING...",
                                     font=("Helvetica", 11, "bold"),
                                     bg="#000000", fg="#00ff88")
        self.status_label.pack(pady=12)
    
    def update_camera(self):
        while self.running:
            if self.camera_ok:
                ret, frame = self.cap.read()
                if not ret:
                    self.camera_ok = False
                    continue
                
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                # Process gestures
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = self.hands.detect(mp_image)
                
                # Track hands
                hands_detected = {'right': False, 'left': False}
                hands_closed_state = {'right': False, 'left': False}
                
                if results.hand_landmarks and results.handedness:
                    for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                        hand_label = handedness[0].category_name.lower()
                        # HandLandmarker reports the image-space handedness; mirror the
                        # camera feed so the user's physical right/left hand matches the UI.
                        hand_label = 'left' if hand_label == 'right' else 'right'
                        hands_detected[hand_label] = True
                        
                        class LandmarkList:
                            def __init__(self, landmarks):
                                self.landmark = landmarks
                        hand_landmarks = LandmarkList(hand_landmarks)

                        # Draw the hand connections with OpenCV.
                        connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
                        for connection in connections: 
                            start = hand_landmarks.landmark[connection.start]
                            end = hand_landmarks.landmark[connection.end]
                            p1 = (int(start.x * w), int(start.y * h))
                            p2 = (int(end.x * w), int(end.y * h))
                            cv2.line(frame, p1, p2, (255, 255, 255), 2)
                        for lm in hand_landmarks.landmark:
                            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (255, 255, 255), -1)

                        hand_closed = is_hand_closed(hand_landmarks)
                        hands_closed_state[hand_label] = hand_closed
                        
                        rotation = calculate_rotation(hand_landmarks)
                        center_x, center_y = get_hand_center(hand_landmarks)
                        
                        # Draw center point
                        center_px = int(center_x * w)
                        center_py = int(center_y * h)
                        color = (0, 0, 255) if hand_closed else (0, 255, 255)
                        cv2.circle(frame, (center_px, center_py), 8, color, -1)
                        status = "CLOSED" if hand_closed else "OPEN"
                        cv2.putText(frame, f"{hand_label.upper()}-{status}", (center_px + 15, center_py),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # RIGHT HAND CONTROLS
                        if hand_label == 'right' and not hand_closed:
                            if self.prev_positions['right']['rotation'] is not None:
                                rotation_diff = rotation - self.prev_positions['right']['rotation']
                                y_diff = center_y - self.prev_positions['right']['center_y']
                                
                                dominant = detect_dominant_gesture(rotation_diff, y_diff, 3, 0.03)
                                
                                if dominant == 'rotation' and not self.locks['volume']:
                                    self.effects['volume'] = max(0, min(100, self.effects['volume'] + rotation_diff))
                                    self.audio.set_volume(self.effects['volume'])
                                
                                elif dominant == 'vertical' and not self.locks['vocals']:
                                    self.effects['vocals'] = max(0, min(100, self.effects['vocals'] - y_diff * 100))
                                    self.audio.set_vocals(self.effects['vocals'])
                            
                            self.prev_positions['right']['rotation'] = rotation
                            self.prev_positions['right']['center_y'] = center_y
                        
                        # LEFT HAND CONTROLS
                        elif hand_label == 'left' and not hand_closed:
                            if self.prev_positions['left']['rotation'] is not None:
                                rotation_diff = rotation - self.prev_positions['left']['rotation']
                                y_diff = center_y - self.prev_positions['left']['center_y']
                                
                                dominant = detect_dominant_gesture(rotation_diff, y_diff, 3, 0.03)
                                
                                if dominant == 'rotation' and not self.locks['bass']:
                                    self.effects['bass'] = max(0, min(100, self.effects['bass'] + rotation_diff))
                                    self.audio.set_bass(self.effects['bass'])
                                
                                elif dominant == 'vertical' and not self.locks['treble']:
                                    self.effects['treble'] = max(0, min(100, self.effects['treble'] - y_diff * 100))
                                    self.audio.set_treble(self.effects['treble'])
                            
                            self.prev_positions['left']['rotation'] = rotation
                            self.prev_positions['left']['center_y'] = center_y
                
                # BOTH HANDS CLOSED = PLAY/PAUSE
                if hands_detected['right'] and hands_detected['left']:
                    if hands_closed_state['right'] and hands_closed_state['left']:
                        if not self.both_hands_closed:
                            self.audio.toggle()
                            self.both_hands_closed = True
                    else:
                        self.both_hands_closed = False
                
                # RIGHT HAND CLOSE/OPEN = NEXT TRACK
                if hands_detected['right'] and not (hands_detected['left'] and hands_closed_state['left']):
                    if hands_closed_state['right'] and not self.right_hand_was_closed:
                        self.right_hand_was_closed = True
                    elif not hands_closed_state['right'] and self.right_hand_was_closed:
                        self.right_hand_was_closed = False
                        self.next_track()
                
                # LEFT HAND CLOSE/OPEN = PREVIOUS TRACK
                if hands_detected['left'] and not (hands_detected['right'] and hands_closed_state['right']):
                    if hands_closed_state['left'] and not self.left_hand_was_closed:
                        self.left_hand_was_closed = True
                    elif not hands_closed_state['left'] and self.left_hand_was_closed:
                        self.left_hand_was_closed = False
                        self.prev_track()
                
                # Convert to PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((720, 540), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image=img)
                
                self.video_label.configure(image=photo)
                self.video_label.image = photo
            else:
                # Placeholder
                placeholder = np.zeros((540, 720, 3), dtype=np.uint8)
                cv2.putText(placeholder, "NO CAMERA INPUT", (200, 270),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                img = Image.fromarray(placeholder)
                photo = ImageTk.PhotoImage(image=img)
                self.video_label.configure(image=photo)
                self.video_label.image = photo
            
            self.root.update_idletasks()
    
    def update_status(self):
        if not self.running:
            return
        
        # Update controls
        self.update_level_control(self.right_controls["Volume"], self.effects['volume'], 0, 100)
        self.update_level_control(self.right_controls["Vocals"], self.effects['vocals'], 0, 100)
        self.update_level_control(self.left_controls["Bass"], self.effects['bass'], 0, 100)
        self.update_level_control(self.left_controls["Treble"], self.effects['treble'], 0, 100)
        
        # Update lock indicators
        for name in ['Volume', 'Vocals', 'Bass', 'Treble']:
            lock_key = name.lower()
            control = self.right_controls.get(name) or self.left_controls.get(name)
            if control:
                lock_text = "🔒" if self.locks[lock_key] else ""
                control['lock_label'].config(text=lock_text)
        
        # Update track name
        self.track_label.config(text=self.audio.get_track_name())
        
        # Update play button
        if self.audio.is_playing:
            self.play_btn.config(text="⏸️ PAUSE", fg="#ff3366")
        else:
            self.play_btn.config(text="▶️ PLAY", fg="#00ff88")
        
        # Update status bar
        vocal_text = "BEATS ONLY" if self.effects['vocals'] < 5 else f"Vocals:{self.effects['vocals']:.0f}%"
        self.status_label.config(
            text=f"🎚️ Vol:{self.effects['volume']:.0f}% | {vocal_text} | "
                 f"Bass:{self.effects['bass']:.0f}% | Treble:{self.effects['treble']:.0f}% | "
                 f"{'▶️ PLAYING' if self.audio.is_playing else '⏸️ PAUSED'}"
        )
        
        self.root.after(50, self.update_status)
    
    def update_level_control(self, control, value, min_val, max_val):
        canvas = control["canvas"]
        label = control["label"]
        color = control["color"]
        
        canvas.delete("all")
        
        # Calculate bar width
        normalized = (value - min_val) / (max_val - min_val)
        bar_width = int(220 * normalized)
        
        # Draw background grid
        for i in range(0, 221, 44):
            canvas.create_line(i, 0, i, 30, fill="#222222")
        
        # Draw bar
        if bar_width > 0:
            canvas.create_rectangle(0, 6, bar_width, 24,
                                   fill=color, outline=color)
        
        # Draw center line if applicable
        if min_val < 0:
            center = int(220 * (-min_val / (max_val - min_val)))
            canvas.create_line(center, 0, center, 30, fill="#666666", width=2)
        
        # Update value label
        if value < 5 and control == self.right_controls.get("Vocals"):
            label.config(text="BEATS")
        else:
            label.config(text=f"{value:.0f}")
    
    def toggle_lock(self, lock_key):
        self.locks[lock_key] = not self.locks[lock_key]
        
        # Update visual feedback on the control itself
        lock_status = "LOCKED 🔒" if self.locks[lock_key] else "UNLOCKED 🔓"
        
        print(f"{'='*40}")
        print(f"{lock_status}: {lock_key.upper()}")
        print(f"{'='*40}")
    
    def toggle_play(self):
        self.audio.toggle()
    
    def next_track(self):
        self.audio.next_track()
        print(f"⏭️ NEXT: {self.audio.get_track_name()}")
    
    def prev_track(self):
        self.audio.previous_track()
        print(f"⏮️ PREV: {self.audio.get_track_name()}")
    
    def on_closing(self):
        self.running = False
        self.audio.cleanup()
        if self.camera_ok:
            self.hands.close()
            self.cap.release()
        self.root.destroy()

# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    print("\n" + "="*60)
    print("🎤 AI DJ - VOCAL SEPARATION CONTROLLER 🎤")
    print("="*60)
    print("\n⚠️  First-time separation takes 30-90s per track")
    print("   Cached files will load instantly on subsequent plays")
    print("\n✅ Starting Tkinter interface...")
    print("="*60 + "\n")
    
    # Initialize Audio Controller
    audio = RealtimeAudioController()
    
    # Auto-load first track
    if audio.playlist:
        print("🎵 Preparing first track...\n")
        threading.Thread(target=lambda: audio.load_track(0), daemon=True).start()
    
    # Create Tkinter interface
    root = tk.Tk()
    app = DJVocalInterface(root, audio)
    
    print("✅ Interface ready! Use gestures or click buttons to control.\n")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        app.on_closing()

if __name__ == "__main__":
    main()