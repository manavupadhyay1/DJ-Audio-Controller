import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- Configuration Constants ---
# Angle thresholds (adjusted for better detection)
UP_THRESHOLD = 160       # Slightly lower for more leniency
DOWN_THRESHOLD = 100     # Slightly higher for easier detection
FLARE_THRESHOLD = 135    # More lenient elbow flare detection

# Form scoring weights
PERFECT_BONUS = 1
GOOD_BONUS = 1
IMPERFECT_PENALTY = -1

# Smoothing configuration
ANGLE_SMOOTHING_WINDOW = 5  # Frames to average for angle smoothing

# State variables
stage = "UP"
score = 0
feedback = "Setup camera (overhead view) and begin!"
form_angle_at_bottom = 0
consecutive_frames_up = 0
consecutive_frames_down = 0
FRAME_THRESHOLD = 3  # Frames needed to confirm state change

# Angle history for smoothing
left_angle_history = deque(maxlen=ANGLE_SMOOTHING_WINDOW)
right_angle_history = deque(maxlen=ANGLE_SMOOTHING_WINDOW)

# --- Helper Functions ---
def calculate_angle(a, b, c):
    """Calculates the angle between three 3D points (a, b, c) in degrees."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    # Add small epsilon to avoid division by zero
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0
    
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def smooth_angle(angle_history):
    """Returns smoothed angle using moving average."""
    if len(angle_history) == 0:
        return 0
    return np.mean(angle_history)

def get_landmark_coords(landmarks, landmark_type):
    """Safely extract landmark coordinates."""
    lm = landmarks[landmark_type.value]
    return [lm.x, lm.y, lm.z]

def check_form_quality(elbow_angle, left_angle, right_angle):
    """
    Evaluates form quality based on angles.
    Returns: ('PERFECT', 1), ('GOOD', 1), or ('IMPERFECT', -1)
    """
    # Check if both elbows are relatively symmetric (within 15 degrees)
    angle_symmetry = abs(left_angle - right_angle)
    
    # Perfect form: tight elbows, good symmetry
    if elbow_angle < FLARE_THRESHOLD - 20 and angle_symmetry < 15:
        return 'PERFECT', PERFECT_BONUS
    
    # Good form: acceptable elbow position, reasonable symmetry
    elif elbow_angle < FLARE_THRESHOLD and angle_symmetry < 25:
        return 'GOOD', GOOD_BONUS
    
    # Imperfect form: elbows flared or asymmetric
    else:
        return 'IMPERFECT', IMPERFECT_PENALTY

# --- Main OpenCV & MediaPipe Setup ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Increase camera resolution for better detection
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set up MediaPipe Pose with optimized settings
with mp_pose.Pose(
    min_detection_confidence=0.7,  # Increased for better accuracy
    min_tracking_confidence=0.7,   # Increased for stability
    model_complexity=1              # Balance between speed and accuracy
) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Pose detection
        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            
            # Check landmark visibility (confidence)
            left_elbow_vis = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility
            right_elbow_vis = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility
            
            if left_elbow_vis < 0.5 and right_elbow_vis < 0.5:
                raise Exception("Low visibility")

            # Extract landmarks
            left_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
            left_elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
            left_wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)

            right_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
            right_elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
            right_wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)

            # Calculate angles
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Add to history for smoothing
            left_angle_history.append(left_angle)
            right_angle_history.append(right_angle)
            
            # Get smoothed angles
            left_angle_smooth = smooth_angle(left_angle_history)
            right_angle_smooth = smooth_angle(right_angle_history)
            
            # Use minimum angle (most bent elbow)
            elbow_angle = min(left_angle_smooth, right_angle_smooth)

            # --- Enhanced Push-up Counter Logic ---
            # DOWN detection with frame confirmation
            if elbow_angle < DOWN_THRESHOLD:
                consecutive_frames_down += 1
                consecutive_frames_up = 0
                
                if consecutive_frames_down >= FRAME_THRESHOLD and stage == 'UP':
                    stage = "DOWN"
                    # Store both angles for form evaluation
                    form_angle_at_bottom = elbow_angle
                    form_left_angle = left_angle_smooth
                    form_right_angle = right_angle_smooth
                    feedback = "Dropping... keep elbows tucked!"

            # UP detection with frame confirmation
            elif elbow_angle > UP_THRESHOLD:
                consecutive_frames_up += 1
                consecutive_frames_down = 0
                
                if consecutive_frames_up >= FRAME_THRESHOLD and stage == 'DOWN':
                    stage = "UP"
                    
                    # Evaluate form quality
                    form_quality, points = check_form_quality(
                        form_angle_at_bottom, 
                        form_left_angle, 
                        form_right_angle
                    )
                    
                    score += points
                    
                    # Detailed feedback
                    if form_quality == 'PERFECT':
                        feedback = f"✓ PERFECT REP! (+{points}) Great form!"
                    elif form_quality == 'GOOD':
                        feedback = f"✓ GOOD REP! (+{points}) Nice work!"
                    else:
                        feedback = f"✗ IMPERFECT ({points}) Elbows flared - keep them closer!"
                    
                    # Reset
                    form_angle_at_bottom = 0

            # --- Enhanced Visualization ---
            h, w = image.shape[:h]
            
            # Score display with background
            cv2.rectangle(image, (10, 10), (300, 100), (0, 0, 0), -1)
            cv2.rectangle(image, (10, 10), (300, 100), (0, 255, 0), 2)
            cv2.putText(image, f"Score: {score}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

            # Stage indicator
            stage_color = (0, 255, 255) if stage == "DOWN" else (255, 255, 255)
            cv2.putText(image, f"Stage: {stage}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, stage_color, 2, cv2.LINE_AA)

            # Angle display for debugging
            cv2.putText(image, f"Angle: {int(elbow_angle)}", (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2, cv2.LINE_AA)

            # Feedback with background
            feedback_lines = feedback.split('\n')
            y_offset = image.shape[0] - 30
            for line in feedback_lines:
                cv2.rectangle(image, (10, y_offset - 25), (w - 10, y_offset + 5), (0, 0, 0), -1)
                cv2.putText(image, line, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                y_offset -= 35

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        except Exception as e:
            cv2.putText(image, "POSITION YOURSELF IN FRAME", (50, image.shape[0] // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow('Pushup Tracker - Press Q to quit', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()