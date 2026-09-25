import cv2
import mediapipe as mp
import math
import os
import pyaudio
import numpy as np
from scipy.signal import butter, lfilter
import threading
import librosa
import soundfile as sf
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
warnings.filterwarnings('ignore')

# ============================================================================
# REAL-TIME DUAL-TRACK AUDIO CONTROLLER WITH VOCAL SEPARATION (DEMUCS)
# ============================================================================
class RealtimeAudioController:
    def __init__(self, music_folder="music", separated_folder="separated"):
        """Initialize real-time dual-track audio controller"""
        self.music_folder = music_folder
        self.separated_folder = separated_folder
        self.playlist = self._load_playlist()
        
        # Initialize Demucs model
        print("🔧 Loading Demucs model (this may take a moment)...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Using device: {self.device.upper()}")
        self.model = get_model('htdemucs')
        self.model.to(self.device)
        self.model.eval()
        print("✅ Demucs ready!")
        
        # PyAudio setup
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # TWO TRACKS - Left hand and Right hand
        self.tracks = {
            'left': {
                'vocals_data': None,
                'accompaniment_data': None,
                'audio_position': 0,
                'track_index': 0,
                'volume': 0.5,
                'bass_gain': 0.0,
                'treble_gain': 0.0,
                'vocal_mix': 0.5,
                'is_playing': True
            },
            'right': {
                'vocals_data': None,
                'accompaniment_data': None,
                'audio_position': 0,
                'track_index': 1 if len(self._load_playlist()) > 1 else 0,
                'volume': 0.5,
                'bass_gain': 0.0,
                'treble_gain': 0.0,
                'vocal_mix': 0.5,
                'is_playing': True
            }
        }
        
        # Audio settings
        self.chunk = 1024
        self.sample_rate = 44100
        
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
        
        if os.path.exists(vocals_path) and os.path.exists(accompaniment_path):
            print(f"✅ Using cached separated files")
            return vocals_path, accompaniment_path
        
        print(f"🎵 Separating vocals with Demucs (30-90 seconds)...")
        print(f"   Processing: {track_name}")
        
        try:
            wav, sr = librosa.load(track_path, sr=44100, mono=False)
            if len(wav.shape) == 1:
                wav = np.stack([wav, wav])
            wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                sources = apply_model(self.model, wav_tensor, device=self.device)
            
            sources = sources.cpu().numpy()[0]
            vocals = sources[3]
            drums = sources[0]
            bass = sources[1]
            other = sources[2]
            accompaniment = drums + bass + other
            
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
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def butter_highpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a
    
    def apply_effects(self, vocals_chunk, accompaniment_chunk, track_config):
        try:
            if len(vocals_chunk) == 0 or len(accompaniment_chunk) == 0:
                return np.zeros(self.chunk)
            
            mixed = (vocals_chunk * track_config['vocal_mix']) + \
                    (accompaniment_chunk * (1.0 - track_config['vocal_mix']))
            
            if abs(track_config['bass_gain']) > 0.01:
                try:
                    b_low, a_low = self.butter_lowpass(250, self.sample_rate, order=4)
                    bass = lfilter(b_low, a_low, mixed)
                    mixed = mixed + (bass * track_config['bass_gain'])
                except:
                    pass
            
            if abs(track_config['treble_gain']) > 0.01:
                try:
                    b_high, a_high = self.butter_highpass(3000, self.sample_rate, order=4)
                    treble = lfilter(b_high, a_high, mixed)
                    mixed = mixed + (treble * track_config['treble_gain'])
                except:
                    pass
            
            mixed = mixed * track_config['volume']
            mixed = np.clip(mixed, -1.0, 1.0)
            return mixed
        except Exception as e:
            return np.zeros(self.chunk)
    
    def load_track(self, hand, index):
        if not self.playlist:
            return False
        
        index = index % len(self.playlist)
        track_path = self.playlist[index]
        
        try:
            print(f"\n{'='*60}")
            print(f"📂 Loading for {hand.upper()} hand: {os.path.basename(track_path)}")
            print('='*60)
            
            vocals_path, accompaniment_path = self._separate_vocals(track_path)
            
            if vocals_path is None or accompaniment_path is None:
                print(f"❌ Failed to load separated tracks")
                return False
            
            print(f"📥 Loading separated tracks...")
            vocals_data, _ = librosa.load(vocals_path, sr=self.sample_rate, mono=True)
            accompaniment_data, _ = librosa.load(accompaniment_path, sr=self.sample_rate, mono=True)
            
            min_length = min(len(vocals_data), len(accompaniment_data))
            vocals_data = vocals_data[:min_length]
            accompaniment_data = accompaniment_data[:min_length]
            
            self.tracks[hand]['vocals_data'] = vocals_data
            self.tracks[hand]['accompaniment_data'] = accompaniment_data
            self.tracks[hand]['audio_position'] = 0
            self.tracks[hand]['track_index'] = index
            
            duration = len(vocals_data) / self.sample_rate
            print(f"✅ Ready to play! Duration: {duration:.1f}s")
            print('='*60 + "\n")
            return True
            
        except Exception as e:
            print(f"❌ Error loading track: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _audio_playback_thread(self):
        try:
            while not self.stop_thread:
                final_mix = np.zeros(self.chunk)
                
                for hand in ['left', 'right']:
                    track = self.tracks[hand]
                    
                    if not track['is_playing'] or track['vocals_data'] is None:
                        continue
                    
                    start = track['audio_position']
                    end = start + self.chunk
                    
                    if end >= len(track['vocals_data']):
                        vocals_chunk = track['vocals_data'][start:]
                        accompaniment_chunk = track['accompaniment_data'][start:]
                        track['audio_position'] = 0
                    else:
                        vocals_chunk = track['vocals_data'][start:end]
                        accompaniment_chunk = track['accompaniment_data'][start:end]
                        track['audio_position'] = end
                    
                    processed_chunk = self.apply_effects(vocals_chunk, accompaniment_chunk, track)
                    
                    if len(processed_chunk) == self.chunk:
                        final_mix += processed_chunk
                
                final_mix = np.clip(final_mix * 0.5, -1.0, 1.0)
                audio_bytes = (final_mix * 32767).astype(np.int16).tobytes()
                
                if self.stream:
                    try:
                        self.stream.write(audio_bytes)
                    except:
                        pass
        except Exception as e:
            print(f"❌ Playback error: {e}")
    
    def start(self):
        if self.stream is None:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk
            )
            
            self.stop_thread = False
            self.audio_thread = threading.Thread(target=self._audio_playback_thread)
            self.audio_thread.daemon = True
            self.audio_thread.start()
    
    def set_effect(self, hand, effect_name, value):
        value = max(0, min(100, value))
        
        if effect_name == 'volume':
            self.tracks[hand]['volume'] = value / 100.0
        elif effect_name == 'bass':
            self.tracks[hand]['bass_gain'] = (value - 50) / 25.0
        elif effect_name == 'treble':
            self.tracks[hand]['treble_gain'] = (value - 50) / 25.0
        elif effect_name == 'vocals':
            self.tracks[hand]['vocal_mix'] = value / 100.0
    
    def get_track_name(self, hand):
        if not self.playlist:
            return "No tracks"
        index = self.tracks[hand]['track_index']
        return os.path.basename(self.playlist[index])
    
    def toggle_play(self, hand):
        self.tracks[hand]['is_playing'] = not self.tracks[hand]['is_playing']
    
    def next_track(self, hand):
        if not self.playlist:
            return
        new_index = (self.tracks[hand]['track_index'] + 1) % len(self.playlist)
        self.load_track(hand, new_index)
    
    def prev_track(self, hand):
        if not self.playlist:
            return
        new_index = (self.tracks[hand]['track_index'] - 1) % len(self.playlist)
        self.load_track(hand, new_index)
    
    def cleanup(self):
        self.stop_thread = True
        
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        self.p.terminate()

# ============================================================================
# HAND GESTURE DETECTION FUNCTIONS
# ============================================================================

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def is_finger_thumb_touching(hand_landmarks, finger_tip_idx, threshold=0.05):
    thumb_tip = hand_landmarks.landmark[4]
    finger_tip = hand_landmarks.landmark[finger_tip_idx]
    distance = calculate_distance(thumb_tip, finger_tip)
    return distance < threshold

def get_active_gesture(hand_landmarks):
    gestures = {
        'index': 8,
        'middle': 12,
        'ring': 16,
        'pinky': 20
    }
    
    for gesture_name, tip_idx in gestures.items():
        if is_finger_thumb_touching(hand_landmarks, tip_idx):
            return gesture_name
    
    return None

def is_hand_closed(hand_landmarks):
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

def calculate_hand_rotation(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    middle_base = hand_landmarks.landmark[9]
    angle = math.degrees(math.atan2(middle_base.y - wrist.y, middle_base.x - wrist.x))
    return angle

def normalize_angle_diff(angle_diff):
    while angle_diff > 180:
        angle_diff -= 360
    while angle_diff < -180:
        angle_diff += 360
    return angle_diff

# ============================================================================
# TKINTER GUI
# ============================================================================
class DJInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("AI DJ - Vocal Separation Controller")
        self.root.configure(bg='#0a0a0a')
        
        # Set window size
        self.root.geometry("1600x750")
        
        # Initialize audio controller
        self.audio = RealtimeAudioController()
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # Effect values
        self.effects = {
            'left': {'volume': 50, 'bass': 50, 'treble': 50, 'vocals': 50},
            'right': {'volume': 50, 'bass': 50, 'treble': 50, 'vocals': 50}
        }
        
        # Gesture tracking
        self.prev_rotations = {'left': None, 'right': None}
        self.prev_hand_closed = {'left': False, 'right': False}
        
        # Create GUI
        self.create_gui()
        
        # Auto-load tracks
        if len(self.audio.playlist) >= 2:
            self.audio.load_track('left', 0)
            self.audio.load_track('right', 1)
            self.audio.start()
        elif len(self.audio.playlist) == 1:
            self.audio.load_track('left', 0)
            self.audio.load_track('right', 0)
            self.audio.start()
        
        # Start video loop
        self.update_frame()
        self.update_ui()
    
    def create_gui(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#0a0a0a')
        title_frame.pack(fill=tk.X, pady=10)
        
        title = tk.Label(title_frame, text="🎤 AI DJ - VOCAL SEPARATION 🎚️", 
                        font=("Arial", 24, "bold"), fg="#ff00ff", bg='#0a0a0a')
        title.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#0a0a0a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel - LEFT HAND controls
        left_panel = tk.Frame(main_frame, bg='#1a1a1a', relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.create_track_panel(left_panel, 'left', '👈 LEFT HAND')
        
        # Center - Camera view
        center_frame = tk.Frame(main_frame, bg='#1a1a1a', relief=tk.RAISED, bd=2)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        cam_label = tk.Label(center_frame, text="📹 GESTURE CAMERA", 
                            font=("Arial", 14, "bold"), fg="#ff00ff", bg='#1a1a1a')
        cam_label.pack(pady=10)
        
        self.video_label = tk.Label(center_frame, bg='#000000')
        self.video_label.pack(padx=10, pady=10, expand=True)
        
        # Right panel - RIGHT HAND controls
        right_panel = tk.Frame(main_frame, bg='#1a1a1a', relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.create_track_panel(right_panel, 'right', '👉 RIGHT HAND')
        
        # Bottom status bar
        self.status_label = tk.Label(self.root, text="Ready", 
                                     font=("Arial", 10), fg="#00ff00", bg='#0a0a0a')
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    
    def create_track_panel(self, parent, hand, title):
        # Title
        title_label = tk.Label(parent, text=title, font=("Arial", 16, "bold"), 
                              fg="#ff00ff", bg='#1a1a1a')
        title_label.pack(pady=10)
        
        # Now playing
        now_playing_frame = tk.Frame(parent, bg='#2a2a2a', relief=tk.GROOVE, bd=2)
        now_playing_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(now_playing_frame, text="🎵 NOW PLAYING", 
                font=("Arial", 10, "bold"), fg="#00ff00", bg='#2a2a2a').pack(pady=5)
        
        track_name = self.audio.get_track_name(hand)
        setattr(self, f'{hand}_track_label', 
                tk.Label(now_playing_frame, text=track_name[:25], 
                        font=("Arial", 9), fg="#ffffff", bg='#2a2a2a', wraplength=200))
        getattr(self, f'{hand}_track_label').pack(pady=5)
        
        # Control buttons
        btn_frame = tk.Frame(now_playing_frame, bg='#2a2a2a')
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="⏮ PREV", command=lambda: self.prev_track(hand),
                 bg='#00ff00', fg='#000000', font=("Arial", 9, "bold"), width=8).pack(side=tk.LEFT, padx=2)
        
        play_pause_text = "⏸ PAUSE" if self.audio.tracks[hand]['is_playing'] else "▶ PLAY"
        setattr(self, f'{hand}_play_btn', 
                tk.Button(btn_frame, text=play_pause_text, command=lambda: self.toggle_play(hand),
                         bg='#ff00ff', fg='#000000', font=("Arial", 9, "bold"), width=8))
        getattr(self, f'{hand}_play_btn').pack(side=tk.LEFT, padx=2)
        
        tk.Button(btn_frame, text="⏭ NEXT", command=lambda: self.next_track(hand),
                 bg='#00ff00', fg='#000000', font=("Arial", 9, "bold"), width=8).pack(side=tk.LEFT, padx=2)
        
        # Effects
        effects_frame = tk.Frame(parent, bg='#1a1a1a')
        effects_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.create_effect_control(effects_frame, hand, 'volume', '🔊 VOLUME', '#00ff00')
        self.create_effect_control(effects_frame, hand, 'vocals', '🎤 VOCALS', '#ff00ff')
        self.create_effect_control(effects_frame, hand, 'bass', '🎸 BASS', '#ff0000')
        self.create_effect_control(effects_frame, hand, 'treble', '🎹 TREBLE', '#00ffff')
        
        # Gesture guide
        guide_frame = tk.Frame(parent, bg='#2a2a2a', relief=tk.GROOVE, bd=2)
        guide_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(guide_frame, text="🤚 GESTURE CONTROLS", 
                font=("Arial", 10, "bold"), fg="#ffff00", bg='#2a2a2a').pack(pady=5)
        
        controls = [
            "• Index+Thumb → Rotate = Volume",
            "• Middle+Thumb → Rotate = Bass",
            "• Ring+Thumb → Rotate = Treble",
            "• Pinky+Thumb → Rotate = Vocals",
            "• Close Fist → Play/Pause"
        ]
        
        for control in controls:
            tk.Label(guide_frame, text=control, font=("Arial", 8), 
                    fg="#cccccc", bg='#2a2a2a', anchor='w').pack(fill=tk.X, padx=10)
    
    def create_effect_control(self, parent, hand, effect, label, color):
        frame = tk.Frame(parent, bg='#2a2a2a', relief=tk.RAISED, bd=1)
        frame.pack(fill=tk.X, pady=5)
        
        # Label
        tk.Label(frame, text=label, font=("Arial", 11, "bold"), 
                fg=color, bg='#2a2a2a').pack(side=tk.LEFT, padx=10)
        
        # Progress bar (visual indicator)
        canvas = tk.Canvas(frame, width=180, height=20, bg='#1a1a1a', highlightthickness=0)
        canvas.pack(side=tk.LEFT, padx=5)
        setattr(self, f'{hand}_{effect}_bar', canvas)
        
        # Value label
        value_label = tk.Label(frame, text=f"{self.effects[hand][effect]:.0f}%", 
                              font=("Arial", 10, "bold"), fg=color, bg='#2a2a2a', width=5)
        value_label.pack(side=tk.LEFT, padx=5)
        setattr(self, f'{hand}_{effect}_label', value_label)
    
    def update_effect_display(self, hand, effect, value, color):
        # Update label
        label = getattr(self, f'{hand}_{effect}_label')
        label.config(text=f"{value:.0f}%")
        
        # Update bar
        canvas = getattr(self, f'{hand}_{effect}_bar')
        canvas.delete('all')
        bar_width = int((value / 100) * 180)
        canvas.create_rectangle(0, 0, bar_width, 20, fill=color, outline='')
    
    def toggle_play(self, hand):
        self.audio.toggle_play(hand)
        btn = getattr(self, f'{hand}_play_btn')
        if self.audio.tracks[hand]['is_playing']:
            btn.config(text="⏸ PAUSE")
        else:
            btn.config(text="▶ PLAY")
    
    def next_track(self, hand):
        self.audio.next_track(hand)
        label = getattr(self, f'{hand}_track_label')
        label.config(text=self.audio.get_track_name(hand)[:25])
    
    def prev_track(self, hand):
        self.audio.prev_track(hand)
        label = getattr(self, f'{hand}_track_label')
        label.config(text=self.audio.get_track_name(hand)[:25])
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    hand_label = handedness.classification[0].label.lower()
                    hand_closed = is_hand_closed(hand_landmarks)
                    
                    if hand_closed and not self.prev_hand_closed[hand_label]:
                        self.toggle_play(hand_label)
                    
                    self.prev_hand_closed[hand_label] = hand_closed
                    
                    if not hand_closed:
                        active_gesture = get_active_gesture(hand_landmarks)
                        
                        if active_gesture:
                            rotation = calculate_hand_rotation(hand_landmarks)
                            
                            if self.prev_rotations[hand_label] is not None:
                                rotation_diff = rotation - self.prev_rotations[hand_label]
                                rotation_diff = normalize_angle_diff(rotation_diff)
                                
                                if abs(rotation_diff) > 2:
                                    effect_map = {
                                        'index': 'volume',
                                        'middle': 'bass',
                                        'ring': 'treble',
                                        'pinky': 'vocals'
                                    }
                                    
                                    effect_name = effect_map.get(active_gesture)
                                    if effect_name:
                                        change = rotation_diff * 0.8
                                        self.effects[hand_label][effect_name] = max(0, min(100,
                                            self.effects[hand_label][effect_name] + change))
                                        self.audio.set_effect(hand_label, effect_name, 
                                                            self.effects[hand_label][effect_name])
                            
                            self.prev_rotations[hand_label] = rotation
                        else:
                            self.prev_rotations[hand_label] = None
            
            # Convert to PhotoImage
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.root.after(10, self.update_frame)
    
    def update_ui(self):
        # Update effect displays
        colors = {
            'volume': '#00ff00',
            'vocals': '#ff00ff',
            'bass': '#ff0000',
            'treble': '#00ffff'
        }
        
        for hand in ['left', 'right']:
            for effect in ['volume', 'vocals', 'bass', 'treble']:
                value = self.effects[hand][effect]
                self.update_effect_display(hand, effect, value, colors[effect])
        
        # Update status
        left_status = "▶️" if self.audio.tracks['left']['is_playing'] else "⏸️"
        right_status = "▶️" if self.audio.tracks['right']['is_playing'] else "⏸️"
        
        status_text = f"{left_status} LEFT: Vol:{self.effects['left']['volume']:.0f}% Voc:{self.effects['left']['vocals']:.0f}% Bass:{self.effects['left']['bass']:.0f}% Treb:{self.effects['left']['treble']:.0f}% | {right_status} RIGHT: Vol:{self.effects['right']['volume']:.0f}% Voc:{self.effects['right']['vocals']:.0f}% Bass:{self.effects['right']['bass']:.0f}% Treb:{self.effects['right']['treble']:.0f}%"
        self.status_label.config(text=status_text)
        
        self.root.after(50, self.update_ui)
    
    def cleanup(self):
        self.cap.release()
        self.hands.close()
        self.audio.cleanup()
        self.root.destroy()

# ============================================================================
# MAIN PROGRAM
# ============================================================================
def main():
    print("\n" + "="*60)
    print("🎧 DUAL DJ HAND CONTROL - TKINTER GUI 🎧")
    print("="*60)
    print("✋ Loading interface...")
    print("="*60 + "\n")
    
    root = tk.Tk()
    app = DJInterface(root)
    
    def on_closing():
        app.cleanup()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()