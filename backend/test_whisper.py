import whisper
from pyannote.audio import Pipeline
import torch
import pandas as pd
import os
import gc
from pathlib import Path
import numpy as np

# Clean up memory
torch.cuda.empty_cache()
gc.collect()

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
audio_file = "audio.mp3"
video_filename = "THE Millennial Job Interview.mp4"

# Optional: Extract audio from video
# os.system(f"ffmpeg -i \"{video_filename}\" -vn -acodec mp3 {audio_file}")
# print(f"Audio saved to {audio_file}")

# 1. Transcribe with standard Whisper
print("Loading Whisper model...")
model = whisper.load_model("medium")  # Standard Whisper model

# Load audio
print("Loading audio...")
audio = whisper.load_audio(audio_file)

# Transcribe
print("Transcribing audio...")
result = model.transcribe(audio, verbose=False)

# Free up memory
del model
gc.collect()
torch.cuda.empty_cache()

# Format the result to match what we need
segments = result["segments"]

# 2. Perform speaker diarization with Pyannote
def load_pipeline_from_pretrained(path_to_config: str | Path) -> Pipeline:
    path_to_config = Path(path_to_config)

    print(f"Loading pyannote pipeline from {path_to_config}...")
    cwd = Path.cwd().resolve()  # store current working directory
    cd_to = path_to_config.parent.parent.resolve()

    print(f"Changing working directory to {cd_to}")
    os.chdir(cd_to)

    pipeline = Pipeline.from_pretrained(path_to_config)

    print(f"Changing working directory back to {cwd}")
    os.chdir(cwd)

    return pipeline

# Load diarization model
PATH_TO_CONFIG = "models/pyannote_diarization_config.yaml"
diarize_model = load_pipeline_from_pretrained(PATH_TO_CONFIG)

# Perform diarization
print("Performing speaker diarization...")
diarize_segments = diarize_model(audio_file)

# Convert to dataframe for easier processing
diarize_segments_list = []
for turn, _, speaker in diarize_segments.itertracks(yield_label=True):
    diarize_segments_list.append({
        'segment': turn,
        'speaker': speaker,
        'start': turn.start,
        'end': turn.end
    })
diarize_df = pd.DataFrame(diarize_segments_list)

# 3. Custom function to assign speakers to transcript segments
def assign_speakers_to_segments(transcript_segments, diarize_df):
    """More accurate speaker assignment based on WhisperX approach"""
    result_with_speakers = []
    
    # Sort diarization segments by time
    diarize_df = diarize_df.sort_values(by=['start', 'end'])
    
    for seg in transcript_segments:
        # Get all words with timestamps if available
        if 'words' in seg:
            words = seg['words']
            word_speakers = []
            
            for word in words:
                word_start = word['start']
                word_end = word['end']
                
                # Find overlapping diarization segments
                overlap_scores = []
                for _, row in diarize_df.iterrows():
                    # Calculate intersection
                    intersection = max(0, min(word_end, row['end']) - max(word_start, row['start']))
                    if intersection > 0:
                        # Calculate IoU (Intersection over Union)
                        word_duration = word_end - word_start
                        overlap_scores.append((row['speaker'], intersection / word_duration))
                
                # Assign speaker with highest overlap score
                if overlap_scores:
                    overlap_scores.sort(key=lambda x: x[1], reverse=True)
                    word_speakers.append(overlap_scores[0][0])
                else:
                    word_speakers.append(None)
            
            # Determine most common speaker for this segment
            from collections import Counter
            if word_speakers:
                speaker_counts = Counter(filter(None, word_speakers))
                if speaker_counts:
                    most_common_speaker = speaker_counts.most_common(1)[0][0]
                    
                    # Create new segment with speaker info
                    new_segment = seg.copy()
                    new_segment['speaker'] = most_common_speaker
                    result_with_speakers.append(new_segment)
                else:
                    result_with_speakers.append(seg)
            else:
                result_with_speakers.append(seg)
        else:
            # For segments without word-level timestamps, use original algorithm
            start_time = seg['start']
            end_time = seg['end']
            
            # Calculate total overlap durations by speaker
            speaker_overlaps = {}
            for _, row in diarize_df.iterrows():
                overlap_start = max(start_time, row['start'])
                overlap_end = min(end_time, row['end'])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > 0:
                    speaker = row['speaker']
                    if speaker in speaker_overlaps:
                        speaker_overlaps[speaker] += overlap_duration
                    else:
                        speaker_overlaps[speaker] = overlap_duration
            
            # Find speaker with maximum overlap
            if speaker_overlaps:
                max_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
                segment_duration = end_time - start_time
                
                # Only assign if overlap is significant (at least 30% of segment)
                if speaker_overlaps[max_speaker] >= 0.3 * segment_duration:
                    new_segment = seg.copy()
                    new_segment['speaker'] = max_speaker
                    result_with_speakers.append(new_segment)
                else:
                    # Mark as uncertain if no speaker has significant overlap
                    result_with_speakers.append(seg)
            else:
                result_with_speakers.append(seg)
    
    return result_with_speakers


# Assign speakers to segments
print("Assigning speakers to transcript...")
segments_with_speakers = assign_speakers_to_segments(segments, diarize_df)

# Create final result structure
final_result = {
    "text": result["text"],
    "segments": segments_with_speakers,
    "language": result["language"]
}

# Function to save transcript in speaker format
def save_transcript(result, output_file="transcript_2.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        current_speaker = None
        current_text = ""
        
        for segment in result["segments"]:
            if "speaker" in segment:
                speaker = segment["speaker"]
                text = segment["text"].strip()
                
                # If this is a new speaker, write the previous speaker's text
                if current_speaker is not None and speaker != current_speaker:
                    f.write(f"Speaker {current_speaker}: {current_text}\n\n")
                    current_text = text
                # If this is the same speaker, concatenate the text
                elif current_speaker is not None:
                    current_text += " " + text
                # If this is the first speaker
                else:
                    current_text = text
                    
                current_speaker = speaker
            else:
                # Handle segments without speaker labels
                if current_speaker is not None:
                    # Write the previous speaker's text before switching to unknown
                    f.write(f"Speaker {current_speaker}: {current_text}\n\n")
                    current_speaker = None
                    current_text = ""
                
                text = segment["text"].strip()
                f.write(f"Unknown Speaker: {text}\n\n")
        
        # Write the last speaker's text if there is any
        if current_speaker is not None and current_text:
            f.write(f"Speaker {current_speaker}: {current_text}\n\n")
            
    print(f"Transcript saved to {output_file}")

# Save the transcript
save_transcript(final_result)
print("Process completed!")
