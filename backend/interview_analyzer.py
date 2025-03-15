import numpy as np
import cv2
import mediapipe as mp
import time
import os
import urllib.request
from collections import deque
import warnings
import librosa
import whisper

# Suppress deprecation warnings from MediaPipe and Protocol Buffers
warnings.filterwarnings('ignore', category=UserWarning, module='mediapipe')
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
warnings.filterwarnings('ignore', message='SymbolDatabase.GetPrototype')
warnings.filterwarnings('ignore', message='PySoundFile failed')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')


class FacialLandmarkDetector:
    def __init__(self, model_path=None):
        """Initialize the facial landmark detector with either a provided model or download one"""
        self.model_path = model_path
        
        # Try to load or download model if needed
        if not self.model_path or not os.path.exists(self.model_path):
            self.model_path = self._download_face_landmarker_model()
            
        # Initialize MediaPipe components
        self._initialize_mediapipe()
    
    def _download_face_landmarker_model(self):
        """Download the face landmarker model if it doesn't exist"""
        model_path = "face_landmarker.task"
        
        print(f"Model path for MediaPipe: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"Downloading face landmarker model (this may take a moment)...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"Successfully downloaded face landmarker model")
                return model_path
            except Exception as e:
                print(f"Error downloading face landmarker model: {e}")
                return None
        else:
            print(f"Face landmarker model already exists")
            return model_path
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe components with multiple fallback approaches"""
        # First, try using the Face Landmarker task approach
        try:
            # Initialize MediaPipe Face Landmarker
            self.mp_face_landmarker = mp.tasks.vision.FaceLandmarker
            self.mp_face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions
            self.mp_base_options = mp.tasks.BaseOptions
            self.mp_running_mode = mp.tasks.vision.RunningMode
            
            # Configure options
            options = self.mp_face_landmarker_options(
                base_options=self.mp_base_options(model_asset_path=self.model_path),
                running_mode=self.mp_running_mode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True
            )
            
            # Create face landmarker
            self.face_landmarker = self.mp_face_landmarker.create_from_options(options)
            print("Successfully initialized Face Landmarker")
            self.using_face_landmarker = True
            
        except Exception as e:
            print(f"Failed to initialize Face Landmarker: {e}")
            raise RuntimeError("Failed to initialize any MediaPipe face detection method")

    
    def detect_landmarks(self, frame):
        """Detect facial landmarks in a frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.face_landmarker.detect(mp_image)
        
        # Check if any faces were detected
        if not detection_result.face_landmarks:
            return None
        
        # Get the first face's landmarks
        face_landmarks = detection_result.face_landmarks[0]
        
        # Convert landmarks to pixel coordinates
        landmarks = []
        for landmark in face_landmarks:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), landmark.z))
        
        # Get blendshapes if available
        blendshapes = None
        if hasattr(detection_result, 'face_blendshapes') and detection_result.face_blendshapes:
            blendshapes = detection_result.face_blendshapes[0]
        
        return {
            'landmarks': landmarks,
            'blendshapes': blendshapes
        }
            
    
    def release(self):
        """Release resources"""
        self.face_landmarker.close()

class GazeDetector:
    def __init__(self, history_size=30):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks indices
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # For tracking gaze movement patterns
        self.gaze_history = deque(maxlen=history_size)
        self.pattern_window = 60
        self.min_pattern_duration = 2.0
        self.natural_movement_threshold = 0.4
        
        # For cheating detection
        self.suspicious_activity_log = []
        self.suspicion_score = 0
        self.last_direction = None
        self.direction_duration = {}
        self.direction_start_time = {}
        self.rapid_shifts = 0
        self.last_shift_time = time.time()
        self.off_screen_threshold = 4.0
        self.video_start_time = time.time()
        
        # Confidence thresholds
        self.confidence_threshold = 0.92
        self.consecutive_frames_threshold = 20
        self.current_behavior_frames = {}
        
        # Pattern tracking
        self.last_significant_direction = None
        self.significant_direction_start = None
        self.pattern_buffer = deque(maxlen=10)
        
        # Reading detection
        self.reading_duration = 0
        self.reading_start_time = None
        self.is_reading = False
        self.reading_pattern_count = 0
        self.min_reading_patterns = 3
        
        # Direction intervals for reading detection
        self.direction_intervals = []
        self.max_reading_interval = 1.5
        
        # Calibration period
        self.calibration_period = 5.0
        self.calibration_gaze_points = []
        
    def _calculate_gaze_direction(self, iris_center, eye_contour):
        # Calculate the eye center by averaging the eye contour points
        eye_center = np.mean(eye_contour, axis=0)
        
        # Calculate the relative position of the iris within the eye
        x_rel = iris_center[0] - eye_center[0]
        y_rel = iris_center[1] - eye_center[1]
        
        # Normalize by the eye width and height
        eye_width = np.max(eye_contour[:, 0]) - np.min(eye_contour[:, 0])
        eye_height = np.max(eye_contour[:, 1]) - np.min(eye_contour[:, 1])
        
        x_norm = x_rel / (eye_width / 2) if eye_width > 0 else 0
        y_norm = y_rel / (eye_height / 2) if eye_height > 0 else 0
        
        # Determine gaze direction
        threshold = 0.2
        
        if x_norm < -threshold:
            h_direction = "LEFT"
        elif x_norm > threshold:
            h_direction = "RIGHT"
        else:
            h_direction = "CENTER"
            
        if y_norm < -threshold:
            v_direction = "UP"
        elif y_norm > threshold:
            v_direction = "DOWN"
        else:
            v_direction = "CENTER"
            
        return h_direction, v_direction, (x_norm, y_norm)
    
    def _is_natural_movement(self, current_norm, prev_norm):
        """Check if movement appears natural (smooth) rather than sudden"""
        if prev_norm is None:
            return True
        
        # Calculate rate of change for both x and y coordinates
        x_movement = abs(current_norm[0] - prev_norm[0])
        y_movement = abs(current_norm[1] - prev_norm[1])
        
        # Use the larger movement rate
        movement_rate = max(x_movement, y_movement)
        return movement_rate <= self.natural_movement_threshold
    
    def _analyze_gaze_pattern(self, h_direction, v_direction, avg_x_norm, avg_y_norm):
        """Analyze gaze patterns for suspicious behavior"""
        current_time = time.time()
        current_direction = f"{h_direction}-{v_direction}"
        elapsed_time = current_time - self.video_start_time
        
        # During calibration period, just collect normal gaze data
        if elapsed_time < self.calibration_period:
            self.calibration_gaze_points.append((avg_x_norm, avg_y_norm))
            return False, 0.0
        
        # Skip if it's just normal center positions
        if (h_direction == "CENTER" and v_direction == "CENTER") or \
           (h_direction == "CENTER" and v_direction == "UP"):
            self.last_significant_direction = None
            self.significant_direction_start = None
            return False, 0.0
        
        # Check if this is a new significant direction
        if current_direction != self.last_significant_direction:
            if self.last_significant_direction is not None:
                # Calculate duration of previous direction
                duration = current_time - self.significant_direction_start
                
                # Track direction changes for reading pattern detection
                if duration < self.max_reading_interval:
                    self.direction_intervals.append((self.last_significant_direction, current_direction, duration))
                    
                    # Check for reading patterns (left-to-right or right-to-left sequences)
                    if len(self.direction_intervals) >= 3:
                        last_three = self.direction_intervals[-3:]
                        left_right_pattern = any(
                            ("LEFT" in i[0] and "RIGHT" in i[1]) or 
                            ("RIGHT" in i[0] and "LEFT" in i[1])
                            for i in last_three
                        )
                        
                        if left_right_pattern:
                            self.reading_pattern_count += 1
                
                if duration >= self.min_pattern_duration:
                    self.pattern_buffer.append((self.last_significant_direction, duration))
            
            self.last_significant_direction = current_direction
            self.significant_direction_start = current_time
        
        # Detect reading behavior based on accumulated patterns
        is_reading = self.reading_pattern_count >= self.min_reading_patterns
        
        # Track reading duration
        if is_reading and not self.is_reading:
            self.reading_start_time = current_time
            self.is_reading = True
        elif not is_reading and self.is_reading:
            if self.reading_start_time:
                self.reading_duration += (current_time - self.reading_start_time)
            self.is_reading = False
            self.reading_start_time = None
        
        # Analyze pattern buffer for suspicious patterns
        if len(self.pattern_buffer) >= 3:
            # Look for alternating patterns (e.g., LEFT-RIGHT-LEFT or UP-DOWN-UP)
            pattern_directions = [p[0] for p in list(self.pattern_buffer)[-3:]]
            pattern_durations = [p[1] for p in list(self.pattern_buffer)[-3:]]
            
            # Check for systematic scanning patterns
            is_systematic = False
            confidence = 0.0
            
            # Only consider suspicious if patterns are clear and sustained
            if all(d >= self.min_pattern_duration for d in pattern_durations):
                # Look for consistent left-right-left scanning (reading)
                if (("LEFT" in pattern_directions[0] and "RIGHT" in pattern_directions[1] and "LEFT" in pattern_directions[2]) or
                    ("RIGHT" in pattern_directions[0] and "LEFT" in pattern_directions[1] and "RIGHT" in pattern_directions[2])):
                    is_systematic = True
                    confidence = 0.93  # High confidence for clear reading patterns
                
                # Less confidence for up-down patterns which may be natural
                elif (("UP" in pattern_directions[0] and "DOWN" in pattern_directions[1] and "UP" in pattern_directions[2]) or
                      ("DOWN" in pattern_directions[0] and "UP" in pattern_directions[1] and "DOWN" in pattern_directions[2])):
                    is_systematic = True
                    confidence = 0.85  # Lower confidence as could be natural behavior
            
            return is_systematic, confidence
        
        return False, 0.0
    
    def _analyze_suspicious_behavior(self, h_direction, v_direction, avg_x_norm, avg_y_norm):
        current_time = time.time()
        elapsed_time = current_time - self.video_start_time
        current_direction = f"{h_direction}-{v_direction}"
        
        # Check for natural vs. sudden movements
        prev_norm = None if len(self.gaze_history) == 0 else self.gaze_history[-1]
        is_natural = self._is_natural_movement((avg_x_norm, avg_y_norm), prev_norm)
        
        # Add to gaze history
        self.gaze_history.append((avg_x_norm, avg_y_norm))
        
        # Analyze gaze patterns
        is_suspicious_pattern, pattern_confidence = self._analyze_gaze_pattern(
            h_direction, v_direction, avg_x_norm, avg_y_norm
        )
        
        # Track direction duration
        if current_direction != self.last_direction:
            if self.last_direction and self.last_direction in self.direction_start_time:
                duration = current_time - self.direction_start_time[self.last_direction]
                self.direction_duration[self.last_direction] = \
                    self.direction_duration.get(self.last_direction, 0) + duration
            
            self.direction_start_time[current_direction] = current_time
            self.last_direction = current_direction
        
        # Only flag as suspicious if there's a persistent, clear pattern
        if is_suspicious_pattern and pattern_confidence >= self.confidence_threshold:
            self.suspicious_activity_log.append({
                'timestamp': elapsed_time,
                'activity': 'Reading pattern detected' if self.reading_pattern_count >= self.min_reading_patterns 
                           else 'Systematic gaze pattern detected',
                'duration': sum(p[1] for p in self.pattern_buffer[-3:]),
                'suspicion_level': 'High',
                'confidence': pattern_confidence,
                'pattern': [p[0] for p in self.pattern_buffer[-3:]]
            })
            self.suspicion_score += 3  # More conservative scoring
        
        # Check for sustained off-screen looking (more conservative)
        extreme_gaze = (not is_natural and 
                      ((h_direction != "CENTER" and abs(avg_x_norm) > 0.8) or
                       (v_direction == "DOWN" and abs(avg_y_norm) > 0.8)))
        
        if extreme_gaze:
            if current_direction in self.current_behavior_frames:
                self.current_behavior_frames[current_direction] += 1
            else:
                self.current_behavior_frames[current_direction] = 1
            
            # Only flag after sustained pattern (more consecutive frames)
            if self.current_behavior_frames[current_direction] >= self.consecutive_frames_threshold:
                duration = current_time - self.direction_start_time[current_direction]
                if duration >= self.min_pattern_duration:
                    confidence = min(0.93, 0.83 + (duration - self.min_pattern_duration) * 0.05)
                    if confidence >= self.confidence_threshold:
                        self.suspicious_activity_log.append({
                            'timestamp': elapsed_time,
                            'activity': f'Sustained {current_direction} gaze',
                            'duration': duration,
                            'suspicion_level': 'High',
                            'confidence': confidence
                        })
                        self.suspicion_score += int(duration * 0.8)  # More conservative scoring
        
        return is_suspicious_pattern, 'High' if is_suspicious_pattern else 'Low'
    
    def process_frame(self, frame):
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        # Process the frame with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return frame, "No face detected", "No face detected", "Low"
        
        # Extract face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to pixel coordinates
        mesh_points = np.array([
            np.multiply([p.x, p.y], [width, height]).astype(int)
            for p in face_landmarks.landmark
        ])
        
        # Extract eye contours
        left_eye_points = mesh_points[self.LEFT_EYE]
        right_eye_points = mesh_points[self.RIGHT_EYE]
        
        # Extract iris points
        left_iris_points = mesh_points[self.LEFT_IRIS]
        right_iris_points = mesh_points[self.RIGHT_IRIS]
        
        # Calculate iris centers
        left_iris_center = np.mean(left_iris_points, axis=0).astype(int)
        right_iris_center = np.mean(right_iris_points, axis=0).astype(int)
        
        # Calculate gaze direction for each eye
        left_h_dir, left_v_dir, left_norm = self._calculate_gaze_direction(left_iris_center, left_eye_points)
        right_h_dir, right_v_dir, right_norm = self._calculate_gaze_direction(right_iris_center, right_eye_points)
        
        # Combine directions (use average of normalized positions)
        avg_x_norm = (left_norm[0] + right_norm[0]) / 2
        avg_y_norm = (left_norm[1] + right_norm[1]) / 2
        
        # Analyze for suspicious behavior
        is_suspicious, suspicion_level = self._analyze_suspicious_behavior(
            left_h_dir, left_v_dir, avg_x_norm, avg_y_norm
        )
        
        # Visualize eye landmarks and iris
        cv2.polylines(frame, [left_eye_points], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye_points], True, (0, 255, 0), 1)
        
        # Draw iris centers
        cv2.circle(frame, tuple(left_iris_center), 2, (0, 0, 255), -1)
        cv2.circle(frame, tuple(right_iris_center), 2, (0, 0, 255), -1)
        
        # Draw gaze direction arrow
        left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
        arrow_length = 60
        arrow_dx = int(arrow_length * avg_x_norm)
        arrow_dy = int(arrow_length * avg_y_norm)
        cv2.arrowedLine(
            frame, 
            tuple(left_eye_center), 
            (left_eye_center[0] + arrow_dx, left_eye_center[1] + arrow_dy),
            (255, 0, 0), 2
        )
        
        # Display information on frame
        cv2.putText(frame, f"Gaze: {left_h_dir}-{left_v_dir}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add suspicion level indicator
        if suspicion_level == "High":
            color = (0, 0, 255)  # Red for high suspicion
        else:
            color = (0, 255, 0)  # Green for low suspicion
            
        cv2.putText(frame, f"Status: {suspicion_level}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame, left_h_dir, left_v_dir, suspicion_level
    
    def generate_report(self):
        """Generate a summary report of suspicious activities with high confidence only"""
        total_duration = time.time() - self.video_start_time
        
        # If reading was active at the end, add final duration
        if self.is_reading and self.reading_start_time:
            self.reading_duration += (time.time() - self.reading_start_time)
        
        # Filter only high-confidence suspicious activities
        high_confidence_activities = [
            activity for activity in self.suspicious_activity_log 
            if activity.get('confidence', 0) >= self.confidence_threshold
        ]
        
        # Calculate percentage of time spent in suspicious directions
        suspicious_time = 0
        for direction, duration in self.direction_duration.items():
            # Consider only clearly suspicious directions
            if ("LEFT" in direction and "LEFT-CENTER" not in direction) or \
               ("RIGHT" in direction and "RIGHT-CENTER" not in direction) or \
               "DOWN" in direction:  # DOWN is highly suspicious (reading notes)
                suspicious_time += duration
        
        suspicious_percentage = (suspicious_time / total_duration) * 100 if total_duration > 0 else 0
        
        # Normalize suspicion score
        base_score = (suspicious_percentage * 0.5) + (len(high_confidence_activities) * 1.5)
        # Apply diminishing returns for longer videos
        if total_duration > 120:  # For videos longer than 2 minutes
            base_score = base_score * (1.0 - (total_duration - 120) / (total_duration * 2))
            
        normalized_score = min(100, max(0, int(base_score)))
        
        # Determine overall suspicion level with higher thresholds
        if normalized_score < 30:
            overall_level = "Low"
        elif normalized_score < 60:
            overall_level = "Medium"
        else:
            overall_level = "High"
            
        # Calculate time spent in each direction
        direction_percentages = {}
        for direction, duration in self.direction_duration.items():
            percentage = (duration / total_duration) * 100
            direction_percentages[direction] = percentage
        
        # Calculate confidence in overall assessment
        direction_confidence = 0.85 if suspicious_percentage > 30 else 0.7
        activity_confidence = min(0.95, len(high_confidence_activities) * 0.04 + 0.7)
        overall_confidence = min(0.95, (direction_confidence + activity_confidence) / 2)
        
        # Include reading-specific metrics
        reading_percentage = (self.reading_duration / total_duration) * 100 if total_duration > 0 else 0
        
        # Categorize behavior for clear reporting
        behavior_assessment = "Normal interview behavior"
        if reading_percentage > 15:
            behavior_assessment = "Significant reading behavior detected"
        elif suspicious_percentage > 30:
            behavior_assessment = "Frequent off-screen glances detected"
        
        report = {
            "total_duration": total_duration,
            "suspicion_score": normalized_score,
            "suspicion_level": overall_level,
            "overall_confidence": overall_confidence,
            "behavior_assessment": behavior_assessment,
            "suspicious_activities": high_confidence_activities,
            "direction_percentages": direction_percentages,
            "reading_duration": self.reading_duration,
            "reading_percentage": reading_percentage
        }
        
        return report
    
    def release(self):
        self.face_mesh.close()

class AudioSyncAnalyzer:
    def __init__(self, history_size=45):
        """Initialize the audio-visual sync analyzer"""
        # Lip landmarks indices
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        
        # For lip sync analysis
        self.lip_movement_history = deque(maxlen=history_size)
        self.audio_activity_history = deque(maxlen=history_size)
        self.sync_threshold = 0.15  # Maximum allowed delay between audio and lip movement
        self.suspicious_segments = []
        
        # For voice analysis
        self.voice_samples = []
        self.transcribed_segments = []  # Store transcribed speech segments
        self.background_voice_segments = []
        
        # Initialize Whisper model
        try:
            self.whisper_model = whisper.load_model("medium")  # Can use "tiny", "base", "small", "medium", "large"
            self.using_whisper = True
            print("Whisper model loaded successfully")
            
            # Buffer for accumulating audio for periodic transcription
            self.audio_buffer = np.array([])
            self.last_transcription_time = time.time()
            self.transcription_interval = 2.0  # Transcribe every 2 seconds
            
            # For detecting reading patterns
            self.speech_fluency_scores = deque(maxlen=10)  # Store recent fluency metrics
            self.speech_naturalness_score = 0.0
            self.speaking_rate_history = deque(maxlen=20)
            self.detected_coaching = False
            
        except Exception as e:
            print(f"Warning: Could not load Whisper model: {e}")
            self.using_whisper = False
            print("Using basic energy-based voice detection")
        
        # Confidence tracking
        self.confidence_threshold = 0.93
        self.min_duration = 1.5
        
        # Lip sync correlation window
        self.correlation_window = 20
        self.lip_audio_delay_range = range(-7, 8)  # Increased range to allow more lag tolerance
        
        # Voice detection thresholds
        self.min_voice_energy = 0.01
        self.min_speech_probability = 0.7
        self.voice_energy_history = deque(maxlen=90)
        
        # For detecting consistent pattern of lip sync issues
        self.sync_issue_history = deque(maxlen=300)
        self.min_sync_issue_percentage = 0.4
        
        # Calibration period
        self.calibration_period = 5.0
        self.calibration_start_time = time.time()
        self.is_calibrating = True
    
    def _extract_lip_movement(self, face_landmarks, frame_shape):
        """Extract lip movement metrics from facial landmarks"""
        height, width = frame_shape[:2]
        
        # Extract lip landmarks
        lip_points = np.array([
            [face_landmarks[idx][0], face_landmarks[idx][1]]
            for idx in self.LIPS if idx < len(face_landmarks)
        ])
        
        if len(lip_points) < 10:  # Need enough points for analysis
            return None
            
        # Calculate lip opening (vertical distance)
        top_lip_y = np.min(lip_points[:, 1])
        bottom_lip_y = np.max(lip_points[:, 1])
        lip_opening = bottom_lip_y - top_lip_y
        
        # Calculate lip width (horizontal distance)
        left_x = np.min(lip_points[:, 0])
        right_x = np.max(lip_points[:, 0])
        lip_width = right_x - left_x
        
        # Calculate mouth area (approximate)
        mouth_area = lip_opening * lip_width
        
        return {
            'opening': lip_opening,
            'width': lip_width,
            'area': mouth_area,
            'points': lip_points
        }
    
    def _analyze_voice_activity(self, audio_segment, sample_rate=16000):
        """Analyze voice activity using Whisper or fallback to energy-based detection"""
        # Ensure audio is in correct format (mono)
        if len(audio_segment.shape) > 1:
            audio_segment = audio_segment.mean(axis=1)
        
        # Calculate audio energy
        energy = np.mean(np.abs(audio_segment)) if len(audio_segment) > 0 else 0
        
        # Store energy for background noise estimation
        self.voice_energy_history.append(energy)
        
        if self.using_whisper:
            # Accumulate audio for periodic transcription
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_segment]) if self.audio_buffer.size else audio_segment
            
            # Periodically transcribe accumulated audio
            current_time = time.time()
            transcription_result = None
            
            if current_time - self.last_transcription_time >= self.transcription_interval and len(self.audio_buffer) > sample_rate:
                # Process with Whisper for transcription
                try:
                    # Normalize audio for Whisper
                    audio_for_whisper = self.audio_buffer / np.max(np.abs(self.audio_buffer)) if np.max(np.abs(self.audio_buffer)) > 0 else self.audio_buffer
                    
                    # Get transcription from Whisper
                    result = self.whisper_model.transcribe(
                        audio_for_whisper, 
                        language="en", 
                        fp16=False,
                        temperature=0.0
                    )
                    
                    transcription_result = result
                    
                    # Reset audio buffer
                    self.audio_buffer = np.array([])
                    self.last_transcription_time = current_time
                    
                    # Extract speech features for analysis
                    text = result["text"].strip()
                    segments = result["segments"]
                    
                    # Store transcription data
                    self.transcribed_segments.append({
                        "timestamp": current_time - self.calibration_start_time,
                        "text": text,
                        "segments": segments
                    })
                    
                    # Analyze speech patterns if there's meaningful text
                    if text and len(text.split()) > 3:
                        # Calculate speech naturalness based on segment durations
                        segment_durations = [s["end"] - s["start"] for s in segments if "end" in s and "start" in s]
                        if segment_durations:
                            # Natural speech has more variation in segment durations
                            naturalness_score = min(1.0, np.std(segment_durations) * 2.5)
                            self.speech_fluency_scores.append(naturalness_score)
                            
                            # Update overall naturalness score
                            self.speech_naturalness_score = np.mean(self.speech_fluency_scores)
                    
                    # Determine speech probability
                    is_speech = 1.0 if text.strip() else 0.0
                    
                except Exception as e:
                    print(f"Warning: Whisper transcription error: {e}")
                    is_speech = float(energy > 0.01)  # Fallback to energy-based detection
            else:
                # Use energy-based detection between transcriptions
                is_speech = float(energy > 0.01)
        else:
            # Fallback to energy-based voice activity detection
            is_speech = float(energy > self.min_voice_energy)
        
        return {
            'is_speech': is_speech,
            'energy': energy,
            'naturalness': getattr(self, 'speech_naturalness_score', 0.5),
            'transcription': transcription_result if 'transcription_result' in locals() else None
        }
    
    def _calculate_cross_correlation(self, signal1, signal2):
        """Calculate normalized cross-correlation between two signals"""
        # Normalize signals
        s1 = (signal1 - np.mean(signal1)) / (np.std(signal1) if np.std(signal1) > 0 else 1)
        s2 = (signal2 - np.mean(signal2)) / (np.std(signal2) if np.std(signal2) > 0 else 1)
        
        # Calculate correlation
        correlation = np.correlate(s1, s2, mode='full')
        
        # Return maximum correlation and its lag
        max_idx = np.argmax(correlation)
        max_corr = correlation[max_idx]
        lag = max_idx - (len(signal1) - 1)
        
        return max_corr, lag
    
    def _detect_lip_sync_issues(self, lip_movement, audio_activity, timestamp):
        """Detect lip sync issues by comparing lip movement with audio over time"""
        current_time = time.time()
        
        # During calibration, just collect data without flagging issues
        if current_time - self.calibration_start_time < self.calibration_period:
            if lip_movement is not None:
                self.lip_movement_history.append(lip_movement['area'])
                self.audio_activity_history.append(audio_activity['energy'])
            return False, 0.0, 0
        
        if lip_movement is None:
            # Track this as a non-issue (face not detected, so no mismatch possible)
            self.sync_issue_history.append(0)
            return False, 0.0, 0
        
        # Store current measurements in history
        self.lip_movement_history.append(lip_movement['area'])
        self.audio_activity_history.append(audio_activity['energy'])
        
        # Need enough history for correlation analysis
        if len(self.lip_movement_history) < self.correlation_window:
            self.sync_issue_history.append(0)
            return False, 0.0, 0
        
        # Get the recent windows for analysis
        lip_window = list(self.lip_movement_history)[-self.correlation_window:]
        audio_window = list(self.audio_activity_history)[-self.correlation_window:]
        
        # Skip analysis if there's not enough audio activity
        if max(audio_window) < 0.02:  # Very quiet audio
            self.sync_issue_history.append(0)
            return False, 0.0, 0
            
        # Skip analysis if there's not enough lip movement
        lip_movement_range = max(lip_window) - min(lip_window)
        if lip_movement_range < 0.1 * max(lip_window):  # Less than 10% variation
            self.sync_issue_history.append(0)
            return False, 0.0, 0
        
        # Calculate first derivative (changes) to better detect synchronization
        lip_changes = np.diff(lip_window)
        audio_changes = np.diff(audio_window)
        
        # Check correlation at different time lags
        best_correlation = -1
        best_lag = 0
        
        for delay in self.lip_audio_delay_range:
            if delay < 0:
                # Audio is ahead of video
                a_window = audio_changes[:delay] if delay != 0 else audio_changes
                l_window = lip_changes[-delay:] if delay != 0 else lip_changes
            else:
                # Video is ahead of audio
                a_window = audio_changes[delay:] if delay != 0 else audio_changes
                l_window = lip_changes[:-delay] if delay != 0 else lip_changes
            
            # Make sure windows are the same length
            min_len = min(len(a_window), len(l_window))
            if min_len < 5:  # Need at least 5 points for meaningful correlation
                continue
                
            a_window = a_window[:min_len]
            l_window = l_window[:min_len]
            
            corr, _ = self._calculate_cross_correlation(l_window, a_window)
            
            if corr > best_correlation:
                best_correlation = corr
                best_lag = delay
        
        # A good lip sync should have high correlation at small lag
        # For natural speech, expect correlation > 0.6
        # Using more generous thresholds for acceptable lag
        is_mismatch = best_correlation < 0.35 or abs(best_lag) > 5  # Increased from 4 to 5
        
        # Calculate confidence based on correlation and lag
        confidence = 1.0 - (best_correlation / 1.0)  # Lower correlation = higher confidence in mismatch
        confidence = confidence * (1.0 + 0.1 * abs(best_lag))  # Adjust for lag magnitude
        
        # Clip to [0, 1] range
        confidence = np.clip(confidence, 0.0, 1.0)
        
        # Track sync issue history
        self.sync_issue_history.append(1 if is_mismatch else 0)
        
        # Only flag as issue if we have a consistent pattern of sync issues
        # and current confidence is high
        recent_issues_percentage = sum(self.sync_issue_history) / len(self.sync_issue_history)
        consistent_pattern = recent_issues_percentage >= self.min_sync_issue_percentage
        
        # Adjust confidence based on consistency of issues
        adjusted_confidence = confidence * min(1.0, recent_issues_percentage * 1.5)
        
        # Only flag high-confidence, consistent issues
        if is_mismatch and adjusted_confidence > self.confidence_threshold and consistent_pattern:
            self.suspicious_segments.append({
                'timestamp': timestamp,
                'type': 'lip_sync_mismatch',
                'confidence': adjusted_confidence,
                'correlation': best_correlation,
                'lag': best_lag,
                'consistency': recent_issues_percentage,
                'duration': 0.03  # 30ms frame
            })
            
        return is_mismatch and consistent_pattern, adjusted_confidence, best_lag
    
    def process_frame(self, frame, audio_segment, timestamp, face_landmarks=None):
        """Process a frame of video and corresponding audio"""
        # Skip if no face landmarks provided
        if face_landmarks is None or len(face_landmarks) < 10:
            return {
                'lip_sync_issue': False,
                'sync_confidence': 0.0,
                'lip_audio_lag': 0,
                'speech_detected': False,
                'naturalness_score': 0.5
            }
        
        # Extract lip movements from landmarks
        lip_data = self._extract_lip_movement(face_landmarks, frame.shape)
        
        # Analyze audio
        voice_activity = self._analyze_voice_activity(audio_segment)
        
        # Check for lip sync issues
        sync_issue, sync_confidence, lag = self._detect_lip_sync_issues(
            lip_data, voice_activity, timestamp
        )
        
        # Add additional information from Whisper if available
        whisper_data = {}
        if self.using_whisper:
            whisper_data = {
                'naturalness_score': getattr(self, 'speech_naturalness_score', 0.5),
                'detected_coaching': getattr(self, 'detected_coaching', False)
            }
        
        return {
            'lip_sync_issue': sync_issue,
            'sync_confidence': sync_confidence,
            'lip_audio_lag': lag,
            'speech_detected': voice_activity['is_speech'] > 0.5,
            'audio_energy': voice_activity['energy'],
            **whisper_data
        }
    
    def generate_report(self):
        """Generate a summary report of detected audio-visual sync issues"""
        # Merge consecutive segments
        merged_segments = []
        current_segment = None
        
        for segment in sorted(self.suspicious_segments, key=lambda x: x['timestamp']):
            if current_segment is None:
                current_segment = segment.copy()
            elif (segment['timestamp'] - (current_segment['timestamp'] + current_segment['duration']) 
                  < self.sync_threshold and segment['type'] == current_segment['type']):
                # Merge segments
                current_segment['duration'] += segment['duration']
                current_segment['confidence'] = max(current_segment['confidence'], 
                                                  segment['confidence'])
            else:
                if current_segment['duration'] >= self.min_duration:
                    merged_segments.append(current_segment)
                current_segment = segment.copy()
        
        if current_segment and current_segment['duration'] >= self.min_duration:
            merged_segments.append(current_segment)
        
        # Calculate overall metrics
        total_suspicious_duration = sum(s['duration'] for s in merged_segments)
        lip_sync_issues = [s for s in merged_segments if s['type'] == 'lip_sync_mismatch']
        
        # Calculate lip sync issue percentage
        total_duration = 0
        lip_sync_percentage = 0
        
        if len(self.sync_issue_history) > 0:
            total_duration = len(self.sync_issue_history) * 0.03  # Assuming 30ms per frame
            lip_sync_percentage = (sum(s['duration'] for s in lip_sync_issues) / total_duration) * 100 if total_duration > 0 else 0
        
        # Speech naturalness analysis
        speech_naturalness = getattr(self, 'speech_naturalness_score', 0.5) * 100
        
        # Transcription samples for the report
        transcription_samples = []
        if hasattr(self, 'transcribed_segments'):
            transcription_samples = [
                {'timestamp': s['timestamp'], 'text': s['text']} 
                for s in self.transcribed_segments[-5:]  # Last 5 segments
            ]
        
        # Overall assessment
        if lip_sync_percentage < 5:
            assessment = "No significant lip sync issues detected"
        elif lip_sync_percentage < 15:
            assessment = "Minor lip sync issues detected"
        else:
            assessment = "Significant lip sync issues detected"
        
        return {
            'total_suspicious_duration': total_suspicious_duration,
            'lip_sync_issues': lip_sync_issues,
            'lip_sync_percentage': lip_sync_percentage,
            'assessment': assessment,
            'speech_naturalness': speech_naturalness,
            'transcription_samples': transcription_samples
        }
    
    def release(self):
        """Release resources"""
        pass

def analyze_interview_video(video_path, output_path=None):
    """
    Analyze an interview video for potential cheating behaviors
    
    Args:
        video_path: Path to the video file
        output_path: Optional path to save the analyzed video
        
    Returns:
        Analysis report
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # Initialize facial landmark detector
    face_detector = FacialLandmarkDetector()
    
    # Initialize gaze detector
    gaze_detector = GazeDetector()
    
    # Initialize audio-visual sync analyzer
    audio_sync_analyzer = AudioSyncAnalyzer()
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} at {fps} FPS, total frames: {total_frames}")
    print("Beginning analysis...")
    
    # Use Whisper to load audio directly from video if possible
    try:
        import whisper
        
        # Load the audio directly using Whisper's internal ffmpeg capability
        audio = whisper.load_audio(video_path)
        # Whisper automatically resamples to 16kHz
        audio_sample_rate = 16000
        
        print(f"Audio extracted: {len(audio)} samples at {audio_sample_rate}Hz")
        has_audio = True
    except Exception as e:
        print(f"Warning: Could not extract audio from video: {e}")
        print("Audio analysis will be skipped")
        has_audio = False
    
    # Create video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Get current timestamp
            timestamp = frame_count / fps
            
            # Process frame for facial landmarks
            face_landmarks_result = face_detector.detect_landmarks(frame)
            
            # Process frame for gaze detection
            processed_frame, h_direction, v_direction, suspicion_level = gaze_detector.process_frame(frame)
            
            # Process audio if available
            audio_analysis_result = {}
            if has_audio:
                # Calculate the corresponding audio segment for this frame
                audio_start_sample = int(timestamp * audio_sample_rate)
                audio_end_sample = int((timestamp + 1/fps) * audio_sample_rate)
                
                # Make sure we don't go past the end of the audio
                if audio_end_sample <= len(audio):
                    audio_segment = audio[audio_start_sample:audio_end_sample]
                    
                    # Process audio and check lip sync
                    face_landmarks = face_landmarks_result['landmarks'] if face_landmarks_result else None
                    audio_analysis_result = audio_sync_analyzer.process_frame(
                        frame, audio_segment, timestamp, face_landmarks
                    )
                    
                    # Add lip sync status to the frame
                    if audio_analysis_result.get('lip_sync_issue', False):
                        cv2.putText(processed_frame, "Lip Sync Issue", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Add speech status to the frame
                    if audio_analysis_result.get('speech_detected', False):
                        cv2.putText(processed_frame, "Speech Detected", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame count
            cv2.putText(processed_frame, f"Frame: {frame_count}/{total_frames}", (10, height - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Write frame to output video if path is provided
            if output_path:
                out.write(processed_frame)
            
            # Display progress periodically
            frame_count += 1
            if frame_count % 100 == 0:
                progress = frame_count/total_frames*100
                print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)...")
        
        # Generate reports
        gaze_report = gaze_detector.generate_report()
        
        # Generate audio report if available
        audio_report = {}
        if has_audio:
            audio_report = audio_sync_analyzer.generate_report()
        
        # Print report summary
        print("\n--- GAZE ANALYSIS REPORT ---")
        print(f"Suspicion Score: {gaze_report['suspicion_score']}/100")
        print(f"Suspicion Level: {gaze_report['suspicion_level']}")
        print(f"Reading Behavior: {gaze_report['reading_percentage']:.1f}% of interview")
        
        if has_audio:
            print("\n--- AUDIO-VISUAL SYNC REPORT ---")
            print(f"Lip Sync Issues: {audio_report['lip_sync_percentage']:.1f}% of interview")
            print(f"Assessment: {audio_report['assessment']}")
            print(f"Speech Naturalness: {audio_report['speech_naturalness']:.1f}%")
            
            # Print sample transcriptions if available
            if audio_report.get('transcription_samples'):
                print("\nSample Transcriptions:")
                for i, trans in enumerate(audio_report['transcription_samples'][:3], 1):
                    timestamp_min = int(trans['timestamp'] // 60)
                    timestamp_sec = int(trans['timestamp'] % 60)
                    print(f"{i}. [{timestamp_min:02d}:{timestamp_sec:02d}] {trans['text']}")
        
        # Combine reports
        combined_report = {
            "gaze_analysis": gaze_report,
            "audio_analysis": audio_report if has_audio else {"available": False}
        }
        
        print("\nAnalysis complete.")
        if output_path:
            print(f"Output saved to {output_path}")
        
        return combined_report
        
    finally:
        # Clean up
        cap.release()
        if output_path:
            out.release()
        gaze_detector.release()
        face_detector.release()
        audio_sync_analyzer.release()

# Main function to demonstrate usage
def main():
    # Path to video file
    video_path = "C:\\Users\\rajpu\\Pictures\\Camera Roll\\WIN_20250314_19_38_03_Pro.mp4"
    
    # Analyze the video
    analyze_interview_video(video_path, "analyzed_interview.mp4")

if __name__ == "__main__":
    main()
