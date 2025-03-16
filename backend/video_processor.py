import os
import cv2
import numpy as np
import torch
import soundfile as sf
import librosa
import tempfile
import moviepy.editor as mp
from qai_hub_models.models.whisper_base_en import Model as WhisperBaseEn
from pyannote.audio import Pipeline
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from speechbrain.pretrained import EncoderClassifier
import warnings
import time
from collections import deque
warnings.filterwarnings("ignore")

class VideoProcessor:
    def __init__(self):
        print("Initializing VideoProcessor...")
        # Initialize gaze detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Load the Whisper model through QAI Hub Models
        print("Loading Whisper model...")
        self.whisper_model = WhisperBaseEn.from_pretrained()
        
        # Load speaker embedding model for diarization
        print("Loading speaker embedding model...")
        self.speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        print("Models loaded successfully!")
    
    def extract_audio(self, video_path):
        """Extract audio from video file"""
        print(f"Extracting audio from video: {video_path}")
        
        # Create a temporary directory for audio extraction
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "extracted_audio.wav")
        
        # Extract audio using moviepy
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=16000)
        
        # Load audio for processing
        audio, sr = sf.read(audio_path)
        
        # Ensure audio is mono and in the correct sample rate
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        return audio, sr, audio_path
    
    def detect_gaze(self, video_path, output_path=None):
        """Detect gaze in video and optionally save processed video"""
        print(f"Detecting gaze in video: {video_path}")
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Initialize variables for gaze tracking
        gaze_results = []
        frame_count = 0
        video_start_time = time.time()
        
        # For tracking gaze patterns
        gaze_history = deque(maxlen=30)
        last_direction = None
        direction_duration = {}
        direction_start_time = {}
        suspicious_activity_log = []
        suspicion_score = 0
        
        # For reading detection
        reading_duration = 0
        reading_start_time = None
        is_reading = False
        reading_pattern_count = 0
        
        # Direction intervals for reading detection
        direction_intervals = []
        max_reading_interval = 1.5
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the frame
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            current_time = time.time()
            elapsed_time = current_time - video_start_time
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Region of interest for eyes
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Detect eyes in the face
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                for (ex, ey, ew, eh) in eyes:
                    # Draw rectangle around eyes
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    
                    # Calculate center of eye
                    eye_center_x = x + ex + ew // 2
                    eye_center_y = y + ey + eh // 2
                    
                    # Calculate relative position within the frame
                    rel_x = eye_center_x / frame_width
                    rel_y = eye_center_y / frame_height
                    
                    # Estimate gaze direction with more precision
                    if rel_x < 0.3:
                        h_direction = "LEFT"
                    elif rel_x > 0.7:
                        h_direction = "RIGHT"
                    else:
                        h_direction = "CENTER"
                    
                    if rel_y < 0.3:
                        v_direction = "UP"
                    elif rel_y > 0.7:
                        v_direction = "DOWN"
                    else:
                        v_direction = "CENTER"
                    
                    # Combine directions
                    current_direction = f"{h_direction}-{v_direction}"
                    
                    # Track direction duration
                    if current_direction != last_direction:
                        if last_direction and last_direction in direction_start_time:
                            duration = current_time - direction_start_time[last_direction]
                            direction_duration[last_direction] = direction_duration.get(last_direction, 0) + duration
                        
                        direction_start_time[current_direction] = current_time
                        last_direction = current_direction
                        
                        # Track direction changes for reading pattern detection
                        if last_direction is not None:
                            duration = current_time - direction_start_time.get(last_direction, current_time)
                            if duration < max_reading_interval:
                                direction_intervals.append((last_direction, current_direction, duration))
                                
                                # Check for reading patterns (left-to-right or right-to-left sequences)
                                if len(direction_intervals) >= 3:
                                    last_three = direction_intervals[-3:]
                                    left_right_pattern = any(
                                        ("LEFT" in i[0] and "RIGHT" in i[1]) or 
                                        ("RIGHT" in i[0] and "LEFT" in i[1])
                                        for i in last_three
                                    )
                                    
                                    if left_right_pattern:
                                        reading_pattern_count += 1
                    
                    # Detect reading behavior
                    is_current_reading = reading_pattern_count >= 3
                    
                    # Track reading duration
                    if is_current_reading and not is_reading:
                        reading_start_time = current_time
                        is_reading = True
                    elif not is_current_reading and is_reading:
                        if reading_start_time:
                            reading_duration += (current_time - reading_start_time)
                        is_reading = False
                        reading_start_time = None
                    
                    # Check for suspicious patterns
                    if (h_direction != "CENTER" and v_direction == "DOWN") or \
                       (h_direction == "LEFT" and v_direction != "CENTER") or \
                       (h_direction == "RIGHT" and v_direction != "CENTER"):
                        # Potential cheating behavior - looking down or to sides
                        duration = current_time - direction_start_time.get(current_direction, current_time)
                        if duration > 1.5:  # Sustained for more than 1.5 seconds
                            suspicious_activity_log.append({
                                'timestamp': elapsed_time,
                                'activity': f'Sustained {current_direction} gaze',
                                'duration': duration,
                                'suspicion_level': 'High'
                            })
                            suspicion_score += 1
                    
                    # Add text with gaze direction
                    cv2.putText(frame, current_direction, (eye_center_x, eye_center_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Store result
                    gaze_results.append({
                        "frame": frame_count,
                        "timestamp": frame_count / fps,
                        "direction": current_direction,
                        "x_rel": rel_x,
                        "y_rel": rel_y
                    })
                    
                    # Add to gaze history
                    gaze_history.append((rel_x, rel_y))
            
            # Write the processed frame
            if output_path:
                out.write(frame)
            
            frame_count += 1
        
        # Release resources
        cap.release()
        if output_path:
            out.release()
        
        # Calculate total duration
        total_duration = frame_count / fps
        
        # If reading was active at the end, add final duration
        if is_reading and reading_start_time:
            reading_duration += (time.time() - reading_start_time)
        
        # Calculate percentage of time spent in suspicious directions
        suspicious_time = 0
        for direction, duration in direction_duration.items():
            # Consider only clearly suspicious directions
            if ("LEFT" in direction and "LEFT-CENTER" not in direction) or \
               ("RIGHT" in direction and "RIGHT-CENTER" not in direction) or \
               "DOWN" in direction:  # DOWN is highly suspicious (reading notes)
                suspicious_time += duration
        
        suspicious_percentage = (suspicious_time / total_duration) * 100 if total_duration > 0 else 0
        
        # Calculate time spent in each direction
        direction_percentages = {}
        for direction, duration in direction_duration.items():
            percentage = (duration / total_duration) * 100
            direction_percentages[direction] = percentage
        
        # Calculate reading percentage
        reading_percentage = (reading_duration / total_duration) * 100 if total_duration > 0 else 0
        
        # Normalize suspicion score
        base_score = (suspicious_percentage * 0.5) + (len(suspicious_activity_log) * 1.5)
        # Apply diminishing returns for longer videos
        if total_duration > 120:  # For videos longer than 2 minutes
            base_score = base_score * (1.0 - (total_duration - 120) / (total_duration * 2))
        
        normalized_score = min(100, max(0, int(base_score)))
        
        # Determine overall suspicion level
        if normalized_score < 30:
            overall_level = "Low"
        elif normalized_score < 60:
            overall_level = "Medium"
        else:
            overall_level = "High"
        
        # Categorize behavior for clear reporting
        behavior_assessment = "Normal interview behavior"
        if reading_percentage > 15:
            behavior_assessment = "Significant reading behavior detected"
        elif suspicious_percentage > 30:
            behavior_assessment = "Frequent off-screen glances detected"
        
        # Create gaze report
        gaze_report = {
            "total_duration": total_duration,
            "suspicion_score": normalized_score,
            "suspicion_level": overall_level,
            "behavior_assessment": behavior_assessment,
            "suspicious_activities": suspicious_activity_log,
            "direction_percentages": direction_percentages,
            "reading_duration": reading_duration,
            "reading_percentage": reading_percentage,
            "gaze_results": gaze_results
        }
        
        print(f"Processed {frame_count} frames for gaze detection")
        return gaze_results, gaze_report
    
    def transcribe_audio(self, audio, sample_rate, num_speakers=2):
        """Transcribe audio with speaker diarization using demo module"""
        print(f"Starting transcription with {num_speakers} speakers...")
        
        # Save audio to a temporary file
        import tempfile
        import soundfile as sf
        import io
        from contextlib import redirect_stdout
        
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, audio, sample_rate)
        
        # Use the demo module directly
        from qai_hub_models.models.whisper_base_en import demo
        import sys
        
        # Store original args and replace with our args
        original_argv = sys.argv
        sys.argv = ['demo.py', '--audio_file', temp_file.name]
        
        # Capture the output from demo.main()
        f = io.StringIO()
        with redirect_stdout(f):
            demo.main()
        
        # Restore original args
        sys.argv = original_argv
        
        # Get the captured output and extract the transcription
        output = f.getvalue()
        transcription_line = None
        for line in output.split('\n'):
            if line.startswith("Transcription:"):
                transcription_line = line
                break
        
        if transcription_line:
            # Extract the transcription text
            transcription = transcription_line.replace("Transcription:", "").strip()
            print(f"Transcription: {transcription}")
            
            # Split the transcription into sentences
            import re
            sentences = re.split(r'(?<=[.!?])\s+', transcription)
            
            # Create segments based on sentences
            segments = []
            current_time = 0
            avg_speaking_rate = 3  # words per second - adjust as needed
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                # Estimate duration based on word count
                word_count = len(sentence.split())
                duration = max(1, word_count / avg_speaking_rate)  # Ensure minimum duration
                
                segments.append({
                    "start": current_time,
                    "end": current_time + duration,
                    "text": sentence
                })
                
                current_time += duration
        else:
            # Fallback to a single segment if transcription not found
            print("Warning: Could not extract transcription from demo output")
            segments = [{
                "start": 0,
                "end": len(audio) / sample_rate,
                "text": "Transcription not available"
            }]
        
        # Step 2: Generate speaker embeddings for each segment
        print("Generating speaker embeddings...")
        embeddings = []
        
        for segment in segments:
            start_sample = int(segment["start"] * sample_rate)
            end_sample = int(segment["end"] * sample_rate)
            
            # Ensure we don't go out of bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            # Extract audio segment
            segment_audio = audio[start_sample:end_sample]
            
            # Skip if segment is too short
            if len(segment_audio) < 1000:  # Arbitrary minimum length
                segment_audio = np.pad(segment_audio, (0, 1000 - len(segment_audio)))
            
            # Get embedding using SpeechBrain
            with torch.no_grad():
                embedding = self.speaker_model.encode_batch(torch.tensor(segment_audio).unsqueeze(0))
                embeddings.append(embedding.squeeze().cpu().numpy())
        
        # Step 3: Cluster the embeddings to identify speakers
        print(f"Clustering embeddings to identify {num_speakers} speakers...")
    
        # Special case for only one speaker or one embedding
        if num_speakers == 1 or len(embeddings) <= 1:
            # Assign all segments to the same speaker
            labels = [0] * len(embeddings)
        else:
            # Use clustering for multiple speakers
            clustering = AgglomerativeClustering(n_clusters=num_speakers)
            labels = clustering.fit_predict(embeddings)
        
        # Step 4: Assign speakers to segments
        for i, segment in enumerate(segments):
            if i < len(labels):  # Safety check
                segment["speaker"] = f"Speaker {labels[i] + 1}"
            else:
                segment["speaker"] = "Speaker 1"  # Default
        
        return segments
    
    def process_video(self, video_path, output_video_path=None, output_transcript_path=None, output_gaze_report_path=None, num_speakers=2):
        """Process video: detect gaze, transcribe audio, and perform speaker diarization"""
        # Step 1: Detect gaze
        print("\n=== STEP 1: GAZE DETECTION ===")
        gaze_results, gaze_report = self.detect_gaze(video_path, output_video_path)
        
        # Step 2: Extract audio from video
        print("\n=== STEP 2: AUDIO EXTRACTION ===")
        audio, sample_rate, audio_path = self.extract_audio(video_path)
        
        # Step 3: Transcribe audio with speaker diarization
        print("\n=== STEP 3: TRANSCRIPTION WITH SPEAKER DIARIZATION ===")
        segments = self.transcribe_audio(audio, sample_rate, num_speakers)
        
        # Step 4: Print and save results
        print("\n=== RESULTS ===")
        self.print_transcript(segments)
        
        # Print gaze report summary
        print("\n=== GAZE ANALYSIS REPORT ===")
        print(f"Suspicion Score: {gaze_report['suspicion_score']}/100")
        print(f"Suspicion Level: {gaze_report['suspicion_level']}")
        print(f"Behavior Assessment: {gaze_report['behavior_assessment']}")
        print(f"Reading Behavior: {gaze_report['reading_percentage']:.1f}% of interview")
        
        if output_transcript_path:
            self.save_transcript(segments, gaze_results, output_transcript_path)
        
        if output_gaze_report_path:
            self.save_gaze_report(gaze_report, output_gaze_report_path)
        
        print(f"\nProcessing complete!")
        return segments, gaze_report
    
    def print_transcript(self, segments):
        """Print transcript with speaker labels"""
        print("\n=== TRANSCRIPTION WITH SPEAKER DIARIZATION ===\n")
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            speaker = segment["speaker"]
            
            print(f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}")
    
    def save_transcript(self, segments, gaze_results, output_path):
        """Save transcript with speaker labels and gaze information to a file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== TRANSCRIPTION WITH SPEAKER DIARIZATION AND GAZE INFORMATION ===\n\n")
            
            # Create a lookup for gaze information
            gaze_lookup = {}
            for gaze in gaze_results:
                timestamp = gaze["timestamp"]
                gaze_lookup[timestamp] = gaze["direction"]
            
            for segment in segments:
                start = segment["start"]
                end = segment["end"]
                text = segment["text"]
                speaker = segment["speaker"]
                
                # Find closest gaze information
                closest_gaze = "unknown"
                for timestamp in gaze_lookup:
                    if start <= timestamp <= end:
                        closest_gaze = gaze_lookup[timestamp]
                        break
                
                f.write(f"[{start:.2f}s - {end:.2f}s] {speaker} ({closest_gaze}): {text}\n")
        
        print(f"Transcript saved to {output_path}")

    def save_gaze_report(self, gaze_report, output_path):
        """Save gaze analysis report to a file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== GAZE ANALYSIS REPORT ===\n\n")
            f.write(f"Total Duration: {gaze_report['total_duration']:.2f} seconds\n")
            f.write(f"Suspicion Score: {gaze_report['suspicion_score']}/100\n")
            f.write(f"Suspicion Level: {gaze_report['suspicion_level']}\n")
            f.write(f"Behavior Assessment: {gaze_report['behavior_assessment']}\n")
            f.write(f"Reading Duration: {gaze_report['reading_duration']:.2f} seconds ({gaze_report['reading_percentage']:.1f}%)\n\n")
            
            f.write("Direction Percentages:\n")
            for direction, percentage in gaze_report['direction_percentages'].items():
                f.write(f"  {direction}: {percentage:.1f}%\n")
            
            f.write("\nSuspicious Activities:\n")
            if gaze_report['suspicious_activities']:
                for i, activity in enumerate(gaze_report['suspicious_activities'], 1):
                    f.write(f"{i}. [{activity['timestamp']:.2f}s] {activity['activity']} ")
                    f.write(f"(Duration: {activity['duration']:.2f}s, Level: {activity['suspicion_level']})\n")
            else:
                f.write("  No suspicious activities detected\n")
        
        print(f"Gaze analysis report saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process video with gaze detection and audio transcription using Qualcomm Whisper-Base-En")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--speakers", type=int, default=2, help="Number of speakers in the audio (default: 2)")
    parser.add_argument("--output_video", help="Path to save processed video (optional)")
    parser.add_argument("--output_transcript", help="Path to save transcript (optional)")
    parser.add_argument("--output_gaze_report", help="Path to save gaze analysis report (optional)")
    
    args = parser.parse_args()
    
    processor = VideoProcessor()
    processor.process_video(
        args.video, 
        args.output_video, 
        args.output_transcript,
        args.output_gaze_report,
        args.speakers
    )
