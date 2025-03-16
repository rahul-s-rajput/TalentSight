import numpy as np
import librosa
import sounddevice as sd
import cv2
import warnings
import time
import torch
import os
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture
import mediapipe as mp
from collections import deque

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='mediapipe')
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
warnings.filterwarnings('ignore', message='SymbolDatabase.GetPrototype')
warnings.filterwarnings('ignore', message='PySoundFile failed')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')

class AudioVisualAnalyzer:
    def __init__(self, model_path=None):
        # Initialize MediaPipe Face Landmarker
        self.mp_face_landmarker = mp.tasks.vision.FaceLandmarker
        self.mp_face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions
        self.mp_base_options = mp.tasks.BaseOptions
        self.mp_running_mode = mp.tasks.vision.RunningMode
        
        # Store provided model path
        self.model_path = model_path
        
        # Try to load face landmarker model with multiple fallback approaches
        self._load_face_landmarker_model()
        
        # Lip landmarks indices (for Face Mesh fallback)
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        
        # For lip sync analysis - more conservative
        self.lip_movement_history = deque(maxlen=45)  # Increased to 1.5 seconds at 30fps
        self.audio_activity_history = deque(maxlen=45)
        self.sync_threshold = 0.15  # Maximum allowed delay between audio and lip movement
        self.suspicious_segments = []
        
        # For voice analysis
        self.voice_samples = []
        self.transcribed_segments = []  # Store transcribed speech segments
        self.background_voice_segments = []
        
        # Initialize Whisper model
        try:
            import whisper
            self.whisper_model = whisper.load_model("base")  # Can use "tiny", "base", "small", "medium", "large"
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
            print("Falling back to basic audio analysis")
            self.using_whisper = False
            # Initialize WebRTC VAD for fallback voice activity detection
            import webrtcvad
            self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (highest)
        
        # Confidence tracking - more conservative thresholds
        self.confidence_threshold = 0.93
        self.min_duration = 1.5
        
        # Lip sync correlation window
        self.correlation_window = 20
        self.lip_audio_delay_range = range(-7, 8)
        
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
        
    def _load_face_landmarker_model(self):
        """Try multiple approaches to load the face landmarker model"""
        # First approach: Try with provided model path
        if self.model_path and os.path.exists(self.model_path):
            try:
                print(f"Approach 1 - Loading face landmarker model from provided path: {self.model_path}")
                options = self.mp_face_landmarker_options(
                    base_options=self.mp_base_options(model_asset_path='face_landmarker.task'),
                    running_mode=self.mp_running_mode.VIDEO,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_face_blendshapes=True
                )
                self.face_landmarker = self.mp_face_landmarker.create_from_options(options)
                print("Face landmarker model loaded successfully (Approach 1)")
                return
            except Exception as e:
                print(f"Approach 1 failed: {e}")
        else:
            print(f"Approach 1 skipped: No valid model path provided")
        
        # Second approach: Try using model in the current directory
        try:
            model_path = "face_landmarker.task"
            
            if os.path.exists(model_path):
                print(f"Approach 2 - Loading face landmarker model from current directory: {model_path}")
                options = self.mp_face_landmarker_options(
                    base_options=self.mp_base_options(model_asset_path=model_path),
                    running_mode=self.mp_running_mode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_face_blendshapes=True
                )
                self.face_landmarker = self.mp_face_landmarker.create_from_options(options)
                print("Face landmarker model loaded successfully (Approach 2)")
                return
            else:
                print(f"Approach 2 skipped: Model not found in current directory")
        except Exception as e:
            print(f"Approach 2 failed: {e}")
        
        # Third approach: Try with direct URL
        try:
            model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            print(f"Approach 3 - Loading face landmarker model from URL directly: {model_url}")
            options = self.mp_face_landmarker_options(
                base_options=self.mp_base_options(model_asset_path=model_url),
                running_mode=self.mp_running_mode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True
            )
            self.face_landmarker = self.mp_face_landmarker.create_from_options(options)
            print("Face landmarker model loaded successfully (Approach 3)")
            return
        except Exception as e:
            print(f"Approach 3 failed: {e}")
            
        # Fallback to traditional Face Mesh if all approaches fail
        print("Falling back to traditional Face Mesh")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def _extract_lip_movement(self, frame):
        """Extract lip movement metrics from a video frame using Face Landmarker or Face Mesh"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        # Use Face Landmarker if available, otherwise fall back to Face Mesh
        if hasattr(self, 'face_landmarker'):
            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.face_landmarker.detect(mp_image)
            
            # Check if any faces were detected
            if not detection_result.face_landmarks:
                return None
            
            # Get the first face's landmarks
            face_landmarks = detection_result.face_landmarks[0]
            
            # Extract lip landmarks (specific indices for lips in landmarker)
            # These indices are different from Face Mesh - using a subset focused on mouth
            lip_indices = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405]
            
            lip_points = np.array([
                [face_landmarks[idx].x * width, face_landmarks[idx].y * height]
                for idx in lip_indices if idx < len(face_landmarks)
            ])
            
            # If we have blendshapes, use them for more accurate lip movement detection
            if hasattr(detection_result, 'face_blendshapes') and detection_result.face_blendshapes:
                blendshapes = detection_result.face_blendshapes[0]
                
                # Extract relevant mouth movement blendshapes
                mouth_open_score = 0
                jawOpen_idx = next((i for i, b in enumerate(blendshapes) if b.category_name == "jawOpen"), None)
                mouthClose_idx = next((i for i, b in enumerate(blendshapes) if b.category_name == "mouthClose"), None)
                mouthFunnel_idx = next((i for i, b in enumerate(blendshapes) if b.category_name == "mouthFunnel"), None)
                mouthPucker_idx = next((i for i, b in enumerate(blendshapes) if b.category_name == "mouthPucker"), None)
                
                if jawOpen_idx is not None:
                    mouth_open_score += blendshapes[jawOpen_idx].score
                if mouthClose_idx is not None:
                    mouth_open_score -= blendshapes[mouthClose_idx].score
                if mouthFunnel_idx is not None:
                    mouth_open_score += blendshapes[mouthFunnel_idx].score * 0.5
                if mouthPucker_idx is not None:
                    mouth_open_score += blendshapes[mouthPucker_idx].score * 0.3
                
                # Get lip opening and width from blendshapes
                lip_opening = max(0, mouth_open_score) * 30  # Scale for compatibility
                
                # Check for lip points if available
                if len(lip_points) > 4:
                    # Calculate lip width from points
                    leftmost = min(lip_points[:, 0])
                    rightmost = max(lip_points[:, 0])
                    lip_width = rightmost - leftmost
                else:
                    # Estimate from blendshapes
                    lip_width = 30  # Default value
                    
                    if mouthPucker_idx is not None:
                        lip_width -= blendshapes[mouthPucker_idx].score * 10
                    if mouthFunnel_idx is not None:
                        lip_width -= blendshapes[mouthFunnel_idx].score * 5
            else:
                # If no blendshapes, calculate from points
                if len(lip_points) < 8:  # Need enough points
                    return None
                    
                # Find top and bottom lip points
                top_y = min(lip_points[:, 1])
                bottom_y = max(lip_points[:, 1])
                lip_opening = bottom_y - top_y
                
                # Find left and right mouth corners
                left_x = min(lip_points[:, 0])
                right_x = max(lip_points[:, 0])
                lip_width = right_x - left_x
        else:
            # Fall back to Face Mesh
            results = self.face_mesh.process(rgb_frame)
            if not results.multi_face_landmarks:
                return None
            
            # Extract face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert landmarks to pixel coordinates
            lip_points = np.array([
                [face_landmarks.landmark[idx].x * width,
                 face_landmarks.landmark[idx].y * height]
                for idx in self.LIPS
            ])
            
            # Calculate lip opening (vertical distance)
            top_lip = np.mean(lip_points[2:4], axis=0)
            bottom_lip = np.mean(lip_points[8:10], axis=0)
            lip_opening = np.linalg.norm(top_lip - bottom_lip)
            
            # Calculate lip width (horizontal distance)
            lip_width = np.linalg.norm(lip_points[0] - lip_points[6])
            
        # Calculate mouth area (approximate)
        mouth_area = lip_opening * lip_width
        
        return {
            'opening': lip_opening,
            'width': lip_width,
            'area': mouth_area,
            'points': lip_points
        }
    
    def _analyze_voice_activity(self, audio_segment, sample_rate=16000):
        """Analyze voice activity using Whisper or fallback to WebRTC VAD"""
        # Ensure audio is in correct format (mono)
        if len(audio_segment.shape) > 1:
            audio_segment = audio_segment.mean(axis=1)
        
        # Calculate audio energy for any approach
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
                        temperature=0.0,
                        word_timestamps=True,
                        no_speech_threshold=0.6
                    )
                    
                    transcription_result = result
                    
                    # Reset audio buffer
                    self.audio_buffer = np.array([])
                    self.last_transcription_time = current_time
                    
                    # Extract speech features for analysis
                    text = result["text"].strip()
                    segments = result["segments"]
                    words = []
                    
                    # Extract words with timestamps if available
                    for segment in segments:
                        if "words" in segment:
                            words.extend(segment["words"])
                    
                    # Store transcription data
                    self.transcribed_segments.append({
                        "timestamp": current_time - self.calibration_start_time,
                        "text": text,
                        "segments": segments,
                        "words": words
                    })
                    
                    # Analyze speech patterns
                    if words and len(words) > 1:
                        # Calculate speaking rate (words per second)
                        duration = segments[-1]["end"] - segments[0]["start"] if segments else 1.0
                        words_per_sec = len(words) / max(duration, 0.1)
                        self.speaking_rate_history.append(words_per_sec)
                        
                        # Calculate word gap consistency (natural speech has variable gaps)
                        word_gaps = []
                        for i in range(1, len(words)):
                            if "start" in words[i] and "end" in words[i-1]:
                                gap = words[i]["start"] - words[i-1]["end"]
                                word_gaps.append(gap)
                        
                        gap_consistency = np.std(word_gaps) if word_gaps else 0
                        
                        # More natural speech has higher variability in gaps
                        naturalness_score = min(1.0, gap_consistency * 5)
                        self.speech_fluency_scores.append(naturalness_score)
                        
                        # Update overall naturalness score
                        self.speech_naturalness_score = np.mean(self.speech_fluency_scores) if self.speech_fluency_scores else 0.5
                        
                        # Detect potential coaching (keywords, prompts, etc.)
                        lower_text = text.lower()
                        coaching_keywords = ["say", "tell them", "mention", "answer", "talk about", "respond"]
                        if any(keyword in lower_text for keyword in coaching_keywords):
                            self.detected_coaching = True
                    
                    # Determine speech probability
                    is_speech = 1.0 if text.strip() else 0.0
                    
                except Exception as e:
                    print(f"Warning: Whisper transcription error: {e}")
                    is_speech = float(energy > 0.01)  # Fallback to energy-based detection
            else:
                # Use energy-based detection between transcriptions
                is_speech = float(energy > 0.01)
        else:
            # Fallback to WebRTC VAD
            # Normalize audio for VAD
            audio_segment_int16 = (audio_segment * 32768).astype(np.int16)
            
            # Split into 30ms frames (required by WebRTC VAD)
            frame_length = int(0.03 * sample_rate)
            frames = [audio_segment_int16[i:i+frame_length] 
                     for i in range(0, len(audio_segment_int16), frame_length)]
            
            # Detect voice activity for each frame
            voice_activity = [
                self.vad.is_speech(frame.tobytes(), sample_rate)
                for frame in frames if len(frame) == frame_length
            ]
            
            is_speech = np.mean(voice_activity) if voice_activity else 0
        
        return {
            'is_speech': is_speech,
            'energy': energy,
            'naturalness': getattr(self, 'speech_naturalness_score', 0.5),
            'transcription': transcription_result
        }
    
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
            if min_len < 5:  # Need at least 5 points for meaningful correlation (was 3)
                continue
                
            a_window = a_window[:min_len]
            l_window = l_window[:min_len]
            
            corr, _ = self._calculate_cross_correlation(l_window, a_window)
            
            if corr > best_correlation:
                best_correlation = corr
                best_lag = delay
        
        # A good lip sync should have high correlation at small lag
        # For natural speech, expect correlation > 0.6
        # More conservative thresholds - 0.4 -> 0.35, lag 3 -> 4
        is_mismatch = best_correlation < 0.35 or abs(best_lag) > 4
        
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
    
    def _identify_background_voices(self, audio_segment, timestamp):
        """Identify potential background voices"""
        # Using Whisper-based approach if available
        if self.using_whisper and hasattr(self, 'detected_coaching') and self.detected_coaching:
            confidence = 0.95  # High confidence if we detected coaching keywords
            
            self.background_voice_segments.append({
                'timestamp': timestamp,
                'type': 'background_voice',
                'confidence': confidence,
                'duration': 0.5,  # Assuming 0.5 second duration
                'detected_method': 'whisper_keywords'
            })
            
            return True, confidence
            
        # Get voice activity information
        voice_activity = self._analyze_voice_activity(audio_segment)
        
        # Calculate noise floor (dynamic threshold based on recent audio)
        if len(self.voice_energy_history) > 30:
            # Use the 10th percentile as an estimate of the noise floor
            noise_floor = np.percentile(self.voice_energy_history, 10)
            # Set minimum energy threshold as 3x the noise floor
            min_energy_threshold = max(self.min_voice_energy, noise_floor * 3)
        else:
            min_energy_threshold = self.min_voice_energy
        
        # Skip if not enough energy (likely silence or noise)
        if voice_activity['energy'] < min_energy_threshold:
            return False, 0.0
        
        # Skip if speech probability is too low
        if voice_activity['is_speech'] < self.min_speech_probability:
            return False, 0.0
        
        # Collect voice samples for GMM model
        if len(self.voice_samples) < 100:  # Need enough samples to train GMM
            # Only add clear speech samples to training data
            if voice_activity['is_speech'] > 0.8 and voice_activity['energy'] > min_energy_threshold * 2:
                self.voice_samples.append(audio_segment)
            return False, 0.0
        
        if not hasattr(self, 'gmm') or self.gmm is None:
            # Train GMM on collected voice samples
            try:
                X = np.vstack(self.voice_samples)
                self.gmm = GaussianMixture(n_components=2, random_state=0)
                self.gmm.fit(X.reshape(-1, 1))
            except Exception as e:
                print(f"Warning: Failed to train voice model: {e}")
                return False, 0.0
        
        # Extract audio features (simple energy in different frequency bands)
        try:
            # Calculate spectrogram and extract frequency bands
            D = np.abs(librosa.stft(audio_segment, n_fft=512, hop_length=256))
            
            # Use mean energy in different bands as features
            bands = np.mean(D, axis=1)
            bands = bands / np.max(bands) if np.max(bands) > 0 else bands
            
            # Predict if current segment matches primary speaker
            prediction = self.gmm.predict(bands.reshape(-1, 1))
            prob = self.gmm.predict_proba(bands.reshape(-1, 1))
            
            # Check if there's significant presence of secondary voice
            # Only consider as background voice if it's substantially different
            has_background_voice = np.any(prediction != prediction[0]) and \
                                np.std(prediction) > 0.3
                                
            confidence = np.max(prob[prediction != prediction[0]]) if has_background_voice else 0.0
            
            if has_background_voice and confidence > self.confidence_threshold:
                self.background_voice_segments.append({
                    'timestamp': timestamp,
                    'type': 'background_voice',
                    'confidence': confidence,
                    'energy': voice_activity['energy'],
                    'speech_prob': voice_activity['is_speech'],
                    'duration': len(audio_segment) / 16000  # assuming 16kHz sample rate
                })
            
            return has_background_voice, confidence
        except Exception as e:
            print(f"Warning: Error in background voice detection: {e}")
            return False, 0.0
    
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
    
    def process_frame(self, frame, audio_segment, timestamp):
        """Process a frame of video and corresponding audio"""
        # Extract lip movements
        lip_data = self._extract_lip_movement(frame)
        
        # Analyze audio
        voice_activity = self._analyze_voice_activity(audio_segment)
        
        # Check for lip sync issues
        sync_issue, sync_confidence, lag = self._detect_lip_sync_issues(
            lip_data, voice_activity, timestamp
        )
        
        # Check for background voices
        background_voice, voice_confidence = self._identify_background_voices(
            audio_segment, timestamp
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
            'background_voice': background_voice,
            'voice_confidence': voice_confidence,
            **whisper_data
        }
    
    def generate_report(self):
        """Generate a summary report of detected issues"""
        # Combine and analyze all suspicious segments
        all_segments = self.suspicious_segments + self.background_voice_segments
        all_segments.sort(key=lambda x: x['timestamp'])
        
        # Merge consecutive segments
        merged_segments = []
        current_segment = None
        
        for segment in all_segments:
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
        background_voices = [s for s in merged_segments if s['type'] == 'background_voice']
        
        # Calculate lip sync issue percentage
        total_duration = 0
        lip_sync_percentage = 0
        background_voice_percentage = 0
        
        if len(self.sync_issue_history) > 0:
            total_duration = len(self.sync_issue_history) * 0.03  # Assuming 30ms per frame
            lip_sync_percentage = (sum(s['duration'] for s in lip_sync_issues) / total_duration) * 100
            background_voice_percentage = (sum(s['duration'] for s in background_voices) / total_duration) * 100
        
        # Additional speech analysis if using Whisper
        speech_naturalness = 0.5
        reading_likelihood = 0.0
        coaching_detected = False
        transcription_samples = []
        
        if self.using_whisper:
            speech_naturalness = getattr(self, 'speech_naturalness_score', 0.5)
            coaching_detected = getattr(self, 'detected_coaching', False)
            
            # Calculate reading likelihood based on speech patterns
            speaking_rates = list(self.speaking_rate_history) if hasattr(self, 'speaking_rate_history') else []
            if speaking_rates:
                rate_consistency = 1.0 - min(0.5, np.std(speaking_rates) / max(np.mean(speaking_rates), 0.1))
                fluency_scores = list(self.speech_fluency_scores) if hasattr(self, 'speech_fluency_scores') else []
                avg_fluency = np.mean(fluency_scores) if fluency_scores else 0.5
                
                # Reading is characterized by consistent speaking rate and lower naturalness
                reading_likelihood = (rate_consistency * 0.7 + (1.0 - avg_fluency) * 0.3) * 100
            
            # Get sample transcriptions for the report
            if hasattr(self, 'transcribed_segments'):
                transcription_samples = [
                    {'timestamp': s['timestamp'], 'text': s['text']} 
                    for s in self.transcribed_segments[-5:]  # Last 5 segments
                ]
        
        # Overall assessment - more conservative
        lip_sync_assessment = "No significant lip sync issues detected"
        if lip_sync_percentage > 5:
            lip_sync_assessment = "Possible lip sync issues detected"
        if lip_sync_percentage > 15:
            lip_sync_assessment = "Significant lip sync issues detected"
            
        background_voice_assessment = "No background voices detected"
        if background_voice_percentage > 3 or coaching_detected:
            background_voice_assessment = "Possible background voices or coaching detected"
        if background_voice_percentage > 10:
            background_voice_assessment = "Significant background voice activity detected"
        
        speech_naturalness_assessment = "Natural speech patterns"
        if speech_naturalness < 0.4:
            speech_naturalness_assessment = "Somewhat mechanical speech patterns"
        if speech_naturalness < 0.25:
            speech_naturalness_assessment = "Very mechanical or scripted speech patterns"
        
        reading_assessment = "No evidence of reading from notes"
        if reading_likelihood > 30:
            reading_assessment = "Possible reading from prepared notes"
        if reading_likelihood > 60:
            reading_assessment = "Strong evidence of reading from prepared notes"
        
        return {
            'total_suspicious_duration': total_suspicious_duration,
            'lip_sync_issues': lip_sync_issues,
            'background_voices': background_voices,
            'total_lip_sync_duration': sum(s['duration'] for s in lip_sync_issues),
            'total_background_voice_duration': sum(s['duration'] for s in background_voices),
            'lip_sync_percentage': lip_sync_percentage,
            'background_voice_percentage': background_voice_percentage,
            'lip_sync_assessment': lip_sync_assessment,
            'background_voice_assessment': background_voice_assessment,
            'speech_naturalness': speech_naturalness * 100,
            'speech_naturalness_assessment': speech_naturalness_assessment,
            'reading_likelihood': reading_likelihood,
            'reading_assessment': reading_assessment,
            'coaching_detected': coaching_detected,
            'transcription_samples': transcription_samples
        }
    
    def release(self):
        """Release resources"""
        if hasattr(self, 'face_landmarker'):
            self.face_landmarker.close()
        elif hasattr(self, 'face_mesh'):
            self.face_mesh.close() 