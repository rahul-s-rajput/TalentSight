import warnings
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
from collections import deque
from audio_analysis import AudioVisualAnalyzer
import librosa

# Suppress deprecation warnings from MediaPipe and Protocol Buffers
warnings.filterwarnings('ignore', category=UserWarning, module='mediapipe')
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
warnings.filterwarnings('ignore', message='SymbolDatabase.GetPrototype')
# Suppress librosa warnings
warnings.filterwarnings('ignore', message='PySoundFile failed')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')

# Suppress TensorFlow warnings by redirecting stderr before imports
# This needs to be done before MediaPipe initializes TensorFlow
import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

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
        # Left eye landmarks
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # Right eye landmarks
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Iris landmarks
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # For tracking gaze movement patterns
        self.gaze_history = deque(maxlen=history_size)
        self.pattern_window = 60  # Look at 2-second windows (at 30fps)
        self.min_pattern_duration = 2.0  # More conservative: increased from 1.5 to 2.0 seconds
        self.natural_movement_threshold = 0.4  # More conservative: increased from 0.3 to 0.4
        
        # For cheating detection - more conservative thresholds
        self.suspicious_activity_log = []
        self.suspicion_score = 0
        self.last_direction = None
        self.direction_duration = {}
        self.direction_start_time = {}
        self.rapid_shifts = 0
        self.last_shift_time = time.time()
        self.off_screen_threshold = 4.0  # More conservative: increased from 3.5 to 4.0
        self.video_start_time = time.time()
        
        # Confidence thresholds - more conservative
        self.confidence_threshold = 0.92  # Increased from 0.9 to require higher confidence
        self.consecutive_frames_threshold = 20  # Increased from 15 to require more consecutive frames
        self.current_behavior_frames = {}
        
        # Pattern tracking
        self.last_significant_direction = None
        self.significant_direction_start = None
        self.pattern_buffer = deque(maxlen=10)  # Store recent patterns
        
        # Reading detection - improved
        self.reading_duration = 0  # Initialize reading duration
        self.reading_start_time = None
        self.is_reading = False
        self.reading_pattern_count = 0
        self.min_reading_patterns = 3  # Require at least 3 left-right patterns to confirm reading
        
        # Direction intervals for reading detection
        self.direction_intervals = []
        self.max_reading_interval = 1.5  # Maximum time between direction changes for reading
        
        # Calibration period - assume first 5 seconds are normal behavior
        self.calibration_period = 5.0  # seconds
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
            pattern_directions = [p[0] for p in self.pattern_buffer[-3:]]
            pattern_durations = [p[1] for p in self.pattern_buffer[-3:]]
            
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
                      ((h_direction != "CENTER" and abs(avg_x_norm) > 0.8) or  # Increased threshold
                       (v_direction == "DOWN" and abs(avg_y_norm) > 0.8)))     # Focus more on downward gaze
        
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
            return frame, "No face detected", False, 0
        
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
        
        # Display information on frame
        cv2.putText(frame, f"Gaze: {left_h_dir}-{left_v_dir}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Status: {suspicion_level}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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
            # Consider only clearly suspicious directions (more conservative)
            if ("LEFT" in direction and "LEFT-CENTER" not in direction) or \
               ("RIGHT" in direction and "RIGHT-CENTER" not in direction) or \
               "DOWN" in direction:  # DOWN is highly suspicious (reading notes)
                suspicious_time += duration
        
        suspicious_percentage = (suspicious_time / total_duration) * 100 if total_duration > 0 else 0
        
        # Normalize suspicion score - more conservative scoring system
        base_score = (suspicious_percentage * 0.5) + (len(high_confidence_activities) * 1.5)
        # Apply diminishing returns for longer videos
        if total_duration > 120:  # For videos longer than 2 minutes
            base_score = base_score * (1.0 - (total_duration - 120) / (total_duration * 2))
            
        normalized_score = min(100, max(0, int(base_score)))
        
        # Determine overall suspicion level with higher thresholds
        if normalized_score < 30:  # More conservative: increased from 20 to 30
            overall_level = "Low"
        elif normalized_score < 60:  # More conservative: increased from 50 to 60
            overall_level = "Medium"
        else:
            overall_level = "High"
            
        # Calculate time spent in each direction
        direction_percentages = {}
        for direction, duration in self.direction_duration.items():
            percentage = (duration / total_duration) * 100
            direction_percentages[direction] = percentage
        
        # Calculate confidence in overall assessment
        # More conservative confidence calculation
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


def download_face_landmarker_model():
    """Download the face landmarker model if it doesn't exist"""
    # Use current directory for model to avoid path issues with MediaPipe
    model_path = "face_landmarker.task"
    
    # Print the location for debugging
    print(f"Model path for MediaPipe: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Downloading face landmarker model to {model_path} (this may take a moment)...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"Successfully downloaded face landmarker model to {model_path}")
            return True, model_path
        except Exception as e:
            print(f"Error downloading face landmarker model: {e}")
            return False, None
    else:
        print(f"Face landmarker model already exists at {model_path}")
        return True, model_path

def install_whisper_if_needed():
    """Check if Whisper is installed, install if not"""
    try:
        import whisper
        print("Whisper is already installed")
        return True
    except ImportError:
        print("Whisper not found, installing...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
            print("Successfully installed Whisper")
            return True
        except Exception as e:
            print(f"Error installing Whisper: {e}")
            print("Continuing without Whisper. Audio analysis will be limited.")
            return False

def main():
    # Restore stderr for normal logging
    sys.stderr = stderr
    
    # Download required models
    model_downloaded, model_path = download_face_landmarker_model()
    install_whisper_if_needed()
    
    # Initialize video capture from file
    video_path = "C:\\Users\\rajpu\\Pictures\\Camera Roll\\WIN_20241114_15_29_31_Pro.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Load audio from the video file using librosa
    try:
        print("Loading audio from video (this may take a moment)...")
        audio_data, sample_rate = librosa.load(video_path, sr=16000, mono=True)
        print(f"Successfully loaded audio with sample rate {sample_rate}Hz")
        has_audio = True
    except Exception as e:
        print(f"Warning: Could not load audio from video: {e}")
        print("Running without audio analysis.")
        has_audio = False
        audio_data = None
        sample_rate = 16000  # Default
    
    print("Initializing detectors...")
    # Initialize both detectors
    gaze_detector = GazeDetector()
    audio_visual_analyzer = AudioVisualAnalyzer(model_path if model_downloaded else None)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate audio frame size based on video FPS
    audio_frame_size = int(sample_rate / fps) if has_audio else 0  # Samples per video frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} at {fps} FPS, total frames: {total_frames}")
    print("Beginning analysis...")
    
    frame_count = 0
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Get current timestamp
            timestamp = frame_count / fps
            
            # Process frame for gaze detection
            processed_frame, h_direction, v_direction, suspicion_level = gaze_detector.process_frame(frame)
            
            # Get corresponding audio segment
            if has_audio:
                start_sample = int(timestamp * sample_rate)
                end_sample = min(start_sample + audio_frame_size, len(audio_data))
                
                if start_sample < len(audio_data):
                    audio_segment = audio_data[start_sample:end_sample]
                    
                    # Ensure audio segment is the right size
                    if len(audio_segment) < audio_frame_size:
                        # Pad with zeros if at the end of the file
                        audio_segment = np.pad(audio_segment, 
                                              (0, audio_frame_size - len(audio_segment)),
                                              'constant')
                else:
                    audio_segment = np.zeros(audio_frame_size)
            else:
                audio_segment = np.zeros(audio_frame_size)
            
            # Process frame for audio-visual analysis
            if has_audio:
                av_results = audio_visual_analyzer.process_frame(frame, audio_segment, timestamp)
                lip_sync_status = "Issue" if av_results['lip_sync_issue'] else "OK"
                if av_results['lip_sync_issue']:
                    lip_sync_status += f" (lag: {av_results['lip_audio_lag']} frames)"
                
                # Show additional Whisper data if available
                coaching_status = ""
                if 'detected_coaching' in av_results and av_results['detected_coaching']:
                    coaching_status = " (COACHING DETECTED)"
            else:
                av_results = {'lip_sync_issue': False, 'background_voice': False}
                lip_sync_status = "No Audio"
                coaching_status = ""
            
            # Print progress every 100 frames
            frame_count += 1
            if frame_count % 100 == 0:
                progress = frame_count/total_frames*100
                print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)...")
                if has_audio:
                    print(f"  Gaze: {h_direction}-{v_direction}, Lip Sync: {lip_sync_status}")
                    if 'naturalness_score' in av_results:
                        print(f"  Speech naturalness: {av_results['naturalness_score']:.2f}")
            
            # Display information on frame
            cv2.putText(frame, f"Gaze: {h_direction}-{v_direction}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Lip Sync: {lip_sync_status}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if has_audio:
                cv2.putText(frame, f"Background Voice: {'Detected' if av_results['background_voice'] else 'None'}{coaching_status}", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame (uncomment for visualization)
            # cv2.imshow('Interview Monitoring', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Generate reports from both analyzers
        gaze_report = gaze_detector.generate_report()
        av_report = audio_visual_analyzer.generate_report() if has_audio else None
        
        # Combine reports for overall assessment
        overall_assessment = _generate_combined_assessment(gaze_report, av_report)
        
        # Display comprehensive report
        _display_comprehensive_report(gaze_report, av_report, overall_assessment)
        
    finally:
        # Clean up
        cap.release()
        gaze_detector.release()
        audio_visual_analyzer.release()
        cv2.destroyAllWindows()


def _generate_combined_assessment(gaze_report, av_report):
    """Generate an overall assessment based on both reports"""
    # Default values if audio report is not available
    lip_sync_score = 0
    background_voice_score = 0
    speech_naturalness_score = 0
    reading_from_speech_score = 0
    lip_sync_assessment = "No audio analysis performed"
    background_voice_assessment = "No audio analysis performed"
    speech_naturalness_assessment = "No audio analysis performed"
    reading_from_speech_assessment = "No audio analysis performed"
    
    # If audio report exists, get values
    if av_report:
        lip_sync_score = min(100, av_report['lip_sync_percentage'] * 4)
        background_voice_score = min(100, av_report['background_voice_percentage'] * 8)
        lip_sync_assessment = av_report['lip_sync_assessment']
        background_voice_assessment = av_report['background_voice_assessment']
        
        # Additional Whisper-based metrics if available
        if 'speech_naturalness' in av_report:
            speech_naturalness_score = av_report['speech_naturalness']
            speech_naturalness_assessment = av_report['speech_naturalness_assessment']
        
        if 'reading_likelihood' in av_report:
            reading_from_speech_score = av_report['reading_likelihood']
            reading_from_speech_assessment = av_report['reading_assessment']
    
    # Get gaze scores
    gaze_score = gaze_report['suspicion_score']
    reading_from_gaze_score = min(100, gaze_report['reading_percentage'] * 3)
    
    # Calculate combined reading score (from both gaze and speech analysis)
    combined_reading_score = (reading_from_gaze_score * 0.7 + reading_from_speech_score * 0.3) 
    
    # Overall cheating likelihood assessment - updated weights
    # - Gaze issues: 30%
    # - Reading behavior (combined): 30% 
    # - Lip sync issues: 20%
    # - Background voices: 10%
    # - Speech naturalness: 10%
    
    weighted_score = (
        gaze_score * 0.30 +
        combined_reading_score * 0.30 +
        lip_sync_score * 0.20 +
        background_voice_score * 0.10 +
        (100 - speech_naturalness_score) * 0.10  # Lower naturalness = higher suspicion
    )
    
    # Define thresholds for overall assessment
    if weighted_score < 30:
        integrity_level = "High Integrity"
        integrity_description = "The interview appears to be conducted with high integrity. No significant suspicious behaviors detected."
    elif weighted_score < 60:
        integrity_level = "Possible Integrity Concerns"
        integrity_description = "Some suspicious patterns were detected that may indicate possible integrity issues."
    else:
        integrity_level = "Significant Integrity Concerns"
        integrity_description = "Multiple significant suspicious behaviors were detected that strongly suggest integrity issues."
    
    # Detailed findings
    findings = []
    
    # Add findings about gaze behavior
    if gaze_score < 30:
        findings.append("Normal gaze behavior throughout the interview.")
    elif gaze_score < 60:
        findings.append("Some unusual gaze patterns detected. The applicant occasionally looked away from the camera.")
    else:
        findings.append("Frequent suspicious gaze patterns detected. The applicant often looked away from the camera.")
    
    # Add findings about reading behavior (from gaze)
    if reading_from_gaze_score < 30:
        findings.append("No significant reading behavior detected from eye movements.")
    elif reading_from_gaze_score < 60:
        findings.append("Some evidence of possible reading behavior from eye movements.")
    else:
        findings.append("Strong evidence of systematic reading behavior from eye movements.")
    
    # Add findings about speech patterns if available
    if av_report and 'speech_naturalness' in av_report:
        if speech_naturalness_score > 70:
            findings.append("Natural and spontaneous speech patterns.")
        elif speech_naturalness_score > 40:
            findings.append("Speech patterns show some signs of preparation or scripting.")
        else:
            findings.append("Speech patterns indicate heavily scripted or memorized content.")
    
    # Add findings about lip sync
    if av_report:
        if lip_sync_score < 30:
            findings.append("Audio and video appear well synchronized.")
        elif lip_sync_score < 60:
            findings.append("Some potential issues with audio-video synchronization detected.")
        else:
            findings.append("Significant audio-video synchronization issues detected, suggesting possible synthetic content.")
    
    # Add findings about background voices and coaching
    if av_report:
        coaching_detected = av_report.get('coaching_detected', False)
        if background_voice_score < 30 and not coaching_detected:
            findings.append("No evidence of coaching or background voices.")
        elif background_voice_score < 60 or coaching_detected:
            findings.append("Some potential background voice activity or coaching detected.")
        else:
            findings.append("Significant background voice activity detected, suggesting possible coaching.")
    
    # Add transcription samples if available
    transcription_samples = []
    if av_report and 'transcription_samples' in av_report:
        transcription_samples = av_report['transcription_samples']
    
    return {
        "integrity_level": integrity_level,
        "integrity_description": integrity_description,
        "weighted_score": weighted_score,
        "detailed_findings": findings,
        "transcription_samples": transcription_samples,
        "component_scores": {
            "gaze_score": gaze_score,
            "reading_score_from_gaze": reading_from_gaze_score,
            "reading_score_from_speech": reading_from_speech_score,
            "combined_reading_score": combined_reading_score,
            "lip_sync_score": lip_sync_score,
            "background_voice_score": background_voice_score,
            "speech_naturalness_score": speech_naturalness_score
        },
        "component_assessments": {
            "gaze_assessment": gaze_report['behavior_assessment'],
            "lip_sync_assessment": lip_sync_assessment,
            "background_voice_assessment": background_voice_assessment,
            "speech_naturalness_assessment": speech_naturalness_assessment if 'speech_naturalness' in av_report else "Not analyzed"
        }
    }


def _display_comprehensive_report(gaze_report, av_report, overall_assessment):
    """Display a comprehensive, easy-to-understand report"""
    total_duration = gaze_report['total_duration']
    minutes = int(total_duration // 60)
    seconds = int(total_duration % 60)
    
    print("\n" + "="*80)
    print(f"             INTERVIEW INTEGRITY ANALYSIS REPORT                ")
    print("="*80)
    print(f"Video Duration: {minutes} minutes {seconds} seconds")
    print(f"Overall Integrity Assessment: {overall_assessment['integrity_level']}")
    print(f"Confidence: {gaze_report['overall_confidence']*100:.1f}%")
    print("-"*80)
    print("SUMMARY:")
    print(overall_assessment['integrity_description'])
    print()
    print("KEY FINDINGS:")
    for i, finding in enumerate(overall_assessment['detailed_findings'], 1):
        print(f"{i}. {finding}")
    
    print("\nCOMPONENT SCORES:")
    print(f"  Gaze Behavior Score: {overall_assessment['component_scores']['gaze_score']}/100")
    print(f"  Reading Behavior Score (from eye movements): {overall_assessment['component_scores']['reading_score_from_gaze']}/100")
    if av_report:
        if 'reading_likelihood' in av_report:
            print(f"  Reading Behavior Score (from speech patterns): {overall_assessment['component_scores']['reading_score_from_speech']}/100")
            print(f"  Combined Reading Behavior Score: {overall_assessment['component_scores']['combined_reading_score']:.1f}/100")
        print(f"  Lip Sync Score: {overall_assessment['component_scores']['lip_sync_score']}/100")
        print(f"  Background Voice Score: {overall_assessment['component_scores']['background_voice_score']}/100")
        if 'speech_naturalness' in av_report:
            print(f"  Speech Naturalness Score: {overall_assessment['component_scores']['speech_naturalness_score']}/100")
    print(f"  Overall Weighted Score: {overall_assessment['weighted_score']:.1f}/100")
    
    print("\nDETAILED ANALYSIS:")
    print("\n1. GAZE ANALYSIS")
    print(f"   Assessment: {overall_assessment['component_assessments']['gaze_assessment']}")
    print(f"   Reading behavior detected for {gaze_report['reading_duration']:.1f} seconds " +
          f"({gaze_report['reading_percentage']:.1f}% of interview)")
    
    print("\n   Gaze Direction Breakdown:")
    sorted_directions = sorted(gaze_report['direction_percentages'].items(), 
                              key=lambda x: x[1], reverse=True)
    for direction, percentage in sorted_directions[:5]:  # Show top 5 directions
        if percentage > 1.0:  # Only show significant directions
            print(f"     {direction}: {percentage:.1f}%")
    
    # Include specific suspicious activities if they exist
    if gaze_report['suspicious_activities']:
        print("\n   Suspicious Gaze Activities:")
        for activity in gaze_report['suspicious_activities'][:3]:  # Show top 3
            duration = activity['duration']
            timestamp_min = int(activity['timestamp'] // 60)
            timestamp_sec = int(activity['timestamp'] % 60)
            print(f"     [{timestamp_min:02d}:{timestamp_sec:02d}] {activity['activity']} " +
                  f"(Duration: {duration:.1f}s, Confidence: {activity['confidence']*100:.1f}%)")
    
    if av_report:
        print("\n2. AUDIO-VISUAL ANALYSIS")
        print(f"   Lip Sync Assessment: {overall_assessment['component_assessments']['lip_sync_assessment']}")
        print(f"   Background Voice Assessment: {overall_assessment['component_assessments']['background_voice_assessment']}")
        
        if 'speech_naturalness' in av_report:
            print(f"   Speech Naturalness Assessment: {overall_assessment['component_assessments']['speech_naturalness_assessment']}")
        
        if 'transcription_samples' in overall_assessment and overall_assessment['transcription_samples']:
            print("\n   Speech Transcription Samples:")
            for sample in overall_assessment['transcription_samples']:
                timestamp_min = int(sample['timestamp'] // 60)
                timestamp_sec = int(sample['timestamp'] % 60)
                print(f"     [{timestamp_min:02d}:{timestamp_sec:02d}] \"{sample['text']}\"")
        
        if av_report['lip_sync_issues']:
            print("\n   Lip Sync Issues:")
            for issue in av_report['lip_sync_issues'][:3]:  # Show top 3
                timestamp_min = int(issue['timestamp'] // 60)
                timestamp_sec = int(issue['timestamp'] % 60)
                print(f"     [{timestamp_min:02d}:{timestamp_sec:02d}] Duration: {issue['duration']:.2f}s, " +
                      f"Confidence: {issue['confidence']*100:.1f}%" +
                      f"{', Correlation: ' + str(issue.get('correlation', 'N/A'))[:5] if 'correlation' in issue else ''}" +
                      f"{', Lag: ' + str(issue.get('lag', 'N/A')) + ' frames' if 'lag' in issue else ''}")
        
        if av_report['background_voices']:
            print("\n   Background Voice Detections:")
            for detection in av_report['background_voices'][:3]:  # Show top 3
                timestamp_min = int(detection['timestamp'] // 60)
                timestamp_sec = int(detection['timestamp'] % 60)
                detection_type = detection.get('detected_method', 'audio_analysis')
                print(f"     [{timestamp_min:02d}:{timestamp_sec:02d}] Duration: {detection['duration']:.2f}s, " +
                      f"Confidence: {detection['confidence']*100:.1f}%" +
                      (f" (Detected through: {detection_type})" if 'detected_method' in detection else ""))
    else:
        print("\n=== AUDIO ANALYSIS ===")
        print("No audio analysis performed - could not extract audio from video.")
    
    print("\nRECOMMENDATION:")
    if overall_assessment['weighted_score'] < 30:
        print("This interview appears to be conducted with integrity. No further review necessary.")
    elif overall_assessment['weighted_score'] < 60:
        print("Some suspicious patterns were detected. Consider reviewing the flagged sections.")
    else:
        print("Significant suspicious behaviors detected. Recommend thorough manual review.")
    
    print("="*80)


if __name__ == "__main__":
    main()
