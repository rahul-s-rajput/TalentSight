import os
import cv2
import numpy as np
import torch
import soundfile as sf
import librosa
import tempfile
import moviepy.editor as mp
from qai_hub_models.models.whisper_base_en import Model as WhisperBaseEn
from scipy.spatial.distance import cdist
import warnings
import time
from collections import deque
import requests
import json
import yaml
import subprocess
import sys
from pathlib import Path
warnings.filterwarnings("ignore")

def load_config():
    """Load configuration from config.yaml file"""
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {
            "api_key": os.environ.get("ANYTHING_LLM_API_KEY", ""),
            "model_server_base_url": os.environ.get("ANYTHING_LLM_BASE_URL", "http://localhost:3001/api"),
            "workspace_slug": os.environ.get("ANYTHING_LLM_WORKSPACE", "default")
        }

def test_auth():
    """Test authentication with AnythingLLM API"""
    config = load_config()
    
    # Print config for debugging
    print(f"Using API URL: {config['model_server_base_url']}/auth")
    print(f"API Key (first 5 chars): {config['api_key'][:5] if config['api_key'] else 'None'}...")
    
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {config['api_key']}"
    }
    try:
        response = requests.get(
            f"{config['model_server_base_url']}/auth", 
            headers=headers
        )
        print(f"Auth response status: {response.status_code}")
        print(f"Auth response body: {response.text[:100]}...")  # Print first 100 chars of response
        return response.status_code == 200
    except Exception as e:
        print(f"Authentication test failed: {e}")
        return False

class VideoProcessor:
    def __init__(self):
        print("Initializing VideoProcessor...")
        # Initialize gaze detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Load the Whisper model through QAI Hub Models
        print("Loading Whisper model...")
        self.whisper_model = WhisperBaseEn.from_pretrained()
        
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
    
    def process_interview_transcript(self, raw_transcript):
        """
        Takes raw transcript from Whisper model and uses AnythingLLM API to identify speakers 
        and format the conversation properly.
        
        Args:
            raw_transcript (str): The raw transcript text from Whisper model
            
        Returns:
            str: Formatted transcript with "Interviewer:" and "Applicant:" labels
        """
        # First test authentication
        if not test_auth():
            return "Error: Authentication failed. Check your API key and server URL."
        
        # Load configuration
        config = load_config()
        
        # Use the chat endpoint instead of the workspace endpoint
        chat_url = f"{config['model_server_base_url']}/workspace/{config['workspace_slug']}/chat"
        
        # Set up the headers with API key
        headers = {
            "accept":"application/json",
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        # Prepare the prompt for the LLM
        prompt = f"""
        Below is a raw interview transcript. Identify which lines are spoken by the interviewer 
        and which are spoken by the applicant/candidate. Format the transcript with "Interviewer:" 
        and "Applicant:" labels at the beginning of each speaking turn.
        
        Raw transcript:
        {raw_transcript}
        
        Return ONLY the formatted transcript with proper speaker labels.
        """
        
        # Prepare data payload
        data = {
            "message": prompt,
            "mode": "chat",
            "sessionId": "interview-transcription-session",
            "attachments": [],
            "history": []  # No history needed for single prompt
        }
        
        try:
            response = requests.post(chat_url, headers=headers, json=data, timeout=300)
            
            if response.status_code == 200:
                response_json = response.json()
                formatted_transcript = response_json.get('textResponse', '')
                return formatted_transcript
            else:
                return f"Error: Received status code {response.status_code} from AnythingLLM API"
        except Exception as e:
            return f"Error: Failed to process transcript - {str(e)}"
    
    def detect_gaze(self, video_path, output_video_path, frame_skip=3):
        """Detect gaze in video and optionally save processed video
        
        Args:
            video_path: Path to the video file
            output_video_path: Path to save the processed video (optional)
            frame_skip: Process only every Nth frame to speed up analysis (default: 3)
        """
        print(f"Detecting gaze in video: {video_path} (processing every {frame_skip} frames)")
        
        # Validate video path
        if not video_path:
            raise ValueError("No video path provided")
        
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found at path: {video_path}")
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Check FPS before division
        if fps <= 0:
            self.logger.warning(f"Invalid FPS value ({fps}). Setting to default value of 30.")
            fps = 30.0  # Set a reasonable default
        
        # Initialize video writer if output path is provided
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps/frame_skip, (frame_width, frame_height))
        
        # Initialize variables for gaze tracking
        gaze_results = []
        frame_count = 0
        processed_count = 0
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
            
            # Skip frames to speed up processing
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            processed_count += 1
            
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
            if output_video_path:
                out.write(frame)
            
            frame_count += 1
        
        # Release resources
        cap.release()
        if output_video_path:
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
        
        print(f"Processed {processed_count} frames for gaze detection")
        return gaze_results, gaze_report
    
    def transcribe_audio(self, audio, sample_rate, num_speakers=2):
        """Transcribe audio and identify speakers using AnythingLLM"""
        print(f"Starting transcription...")
        
        # Step 1: Transcribe audio using Whisper model (keep this part)
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
            raw_transcription = transcription_line.replace("Transcription:", "").strip()
            print(f"Raw Transcription: {raw_transcription}")
            
            # Step 2: Use AnythingLLM to format transcript with speaker labels
            formatted_transcript = self.process_interview_transcript(raw_transcription)
            print("Formatted transcript with speaker labels:")
            print(formatted_transcript)
            
            # Step 3: Parse the formatted transcript into the expected segment format
            segments = []
            current_time = 0
            avg_speaking_rate = 3  # words per second - adjust as needed
            
            # Process each line of the formatted transcript
            for line in formatted_transcript.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Extract speaker and text
                if line.startswith("Interviewer:"):
                    speaker = "Interviewer"
                    text = line[len("Interviewer:"):].strip()
                elif line.startswith("Applicant:"):
                    speaker = "Applicant"
                    text = line[len("Applicant:"):].strip()
                else:
                    # Fallback if line doesn't start with expected speaker label
                    speaker = f"Speaker {1 if 'Interviewer' in segments else 2}"
                    text = line
                
                # Estimate duration based on word count
                word_count = len(text.split())
                duration = max(1, word_count / avg_speaking_rate)  # Ensure minimum duration
                
                segments.append({
                    "start": current_time,
                    "end": current_time + duration,
                    "text": text,
                    "speaker": speaker
                })
                
                current_time += duration
        else:
            # Fallback to a single segment if transcription not found
            print("Warning: Could not extract transcription from demo output")
            segments = [{
                "start": 0,
                "end": len(audio) / sample_rate,
                "text": "Transcription not available",
                "speaker": "Speaker 1"
            }]
        
        return segments
    
    def process_video(self, video_path, output_video_path=None, output_transcript_path=None, 
                     output_gaze_report_path=None, output_evaluation_path=None,
                     num_speakers=2, frame_skip=3, job_description=None):
        """Process video: detect gaze, transcribe audio, perform speaker diarization, and evaluate interview
        
        Args:
            video_path: Path to the video file
            output_video_path: Path to save processed video (optional)
            output_transcript_path: Path to save transcript (optional)
            output_gaze_report_path: Path to save gaze analysis report (optional)
            output_evaluation_path: Path to save interview evaluation (optional)
            num_speakers: Number of speakers in the audio (default: 2)
            frame_skip: Process only every Nth frame to speed up analysis (default: 3)
            job_description: Custom job description for evaluation (optional)
        
        Returns:
            tuple: (segments, gaze_report, evaluation_results)
        """
        # Step 1: Detect gaze
        print("\n=== STEP 1: GAZE DETECTION ===")
        gaze_results, gaze_report = self.detect_gaze(video_path, output_video_path, frame_skip)
        
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
        
        # Save transcript to file if path provided
        if output_transcript_path:
            self.save_transcript(segments, gaze_results, output_transcript_path)
        
        # Save gaze report if path provided
        if output_gaze_report_path:
            self.save_gaze_report(gaze_report, output_gaze_report_path)
        
        # Step 5: Always evaluate interview using prompt.py
        print("\n=== STEP 5: INTERVIEW EVALUATION ===")
        # Convert segments to formatted transcript text
        transcript_text = ""
        for segment in segments:
            speaker = segment["speaker"]
            text = segment["text"]
            transcript_text += f"{speaker}: {text}\n\n"
        
        # Send transcript to prompt.py for evaluation
        evaluation_results = self.evaluate_transcript_with_prompt(transcript_text, job_description)
        
        # Print evaluation summary
        print("\n=== EVALUATION SUMMARY ===")
        if "error" in evaluation_results:
            print(f"Evaluation error: {evaluation_results['error']}")
        else:
            try:
                overall_score = evaluation_results.get("Overall_Score", "N/A")
                print(f"Overall Score: {overall_score}")
                
                # Print STAR Method scores if available
                star_scores = evaluation_results.get("STAR_Method_Scores", {})
                if star_scores:
                    print("\nSTAR Method Scores:")
                    for criterion, score in star_scores.items():
                        print(f"  {criterion}: {score}")
                
                # Print Three Cs scores if available
                three_cs_scores = evaluation_results.get("Three_Cs_Scores", {})
                if three_cs_scores:
                    print("\nThree Cs Scores:")
                    for criterion, score in three_cs_scores.items():
                        print(f"  {criterion}: {score}")
                
                # Print summary points if available
                summary = evaluation_results.get("Feedback", {}).get("Summary", [])
                if summary:
                    print("\nSummary:")
                    for point in summary:
                        print(f"  • {point}")
            except Exception as e:
                print(f"Error displaying evaluation summary: {e}")
        
        # Save evaluation results if path provided
        if output_evaluation_path:
            with open(output_evaluation_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2)
            print(f"Evaluation results saved to {output_evaluation_path}")
        
        print(f"\nProcessing complete!")
        return segments, gaze_report, evaluation_results
    
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

    def evaluate_transcript_with_prompt(self, transcript_text, job_description=None):
        """
        Send the transcript to prompt.py for redaction and evaluation
        
        Args:
            transcript_text (str): The transcript text with speaker labels
            job_description (str, optional): Custom job description for evaluation
        
        Returns:
            dict: The evaluation results from prompt.py
        """
        print("\n=== SENDING TRANSCRIPT TO PROMPT.PY FOR EVALUATION ===")
        
        # Save transcript to a temporary file
        temp_dir = tempfile.mkdtemp()
        transcript_path = os.path.join(temp_dir, "transcript.txt")
        
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript_text)
        
        # Prepare command to run prompt.py
        # Use absolute path to prompt.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_py_path = os.path.join(script_dir, "prompt.py")
        
        # Set PYTHONIOENCODING environment variable to handle Unicode in console output
        my_env = os.environ.copy()
        my_env["PYTHONIOENCODING"] = "utf-8"
        
        cmd = [sys.executable, prompt_py_path, "--transcript", transcript_path]
        
        # Add job description if provided
        if job_description:
            job_desc_path = os.path.join(temp_dir, "job_description.txt")
            with open(job_desc_path, 'w', encoding='utf-8') as f:
                f.write(job_description)
            cmd.extend(["--job_description", job_desc_path])
        
        # Add output path for evaluation results
        eval_output_path = os.path.join(temp_dir, "evaluation_results.json")
        cmd.extend(["--output", eval_output_path])
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run prompt.py as a subprocess with the modified environment
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=my_env)
            print("Prompt.py execution completed successfully")
            
            # Load evaluation results
            if os.path.exists(eval_output_path):
                with open(eval_output_path, 'r', encoding='utf-8') as f:
                    evaluation_results = json.load(f)
                return evaluation_results
            else:
                print(f"Warning: Evaluation output file not found at {eval_output_path}")
                return {"error": "Evaluation output file not found"}
        
        except subprocess.CalledProcessError as e:
            print(f"Error running prompt.py: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return {"error": f"Failed to run prompt.py: {str(e)}"}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {"error": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process video with gaze detection and audio transcription using Qualcomm Whisper-Base-En")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--speakers", type=int, default=2, help="Number of speakers in the audio (default: 2)")
    parser.add_argument("--output_video", help="Path to save processed video (optional)")
    parser.add_argument("--output_transcript", help="Path to save transcript (optional)")
    parser.add_argument("--output_gaze_report", help="Path to save gaze analysis report (optional)")
    parser.add_argument("--output_evaluation", help="Path to save interview evaluation (optional)")
    parser.add_argument("--frame_skip", type=int, default=3, help="Process only every Nth frame (default: 3)")
    parser.add_argument("--job_description", help="Path to custom job description file (optional)")
    
    args = parser.parse_args()
    
    # Load job description if provided
    job_description = None
    if args.job_description and os.path.exists(args.job_description):
        with open(args.job_description, 'r', encoding='utf-8') as f:
            job_description = f.read()
    
    processor = VideoProcessor()
    segments, gaze_report, evaluation_results = processor.process_video(
        args.video, 
        args.output_video, 
        args.output_transcript,
        args.output_gaze_report,
        args.output_evaluation,
        args.speakers,
        args.frame_skip,
        job_description
    )
    
    # If no output_evaluation was specified but we want to save the results anyway
    if not args.output_evaluation:
        default_output = f"{os.path.splitext(args.video)[0]}_evaluation.json"
        with open(default_output, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"Evaluation results also saved to {default_output}")

