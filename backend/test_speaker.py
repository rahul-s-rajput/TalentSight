import whisperx
from pyannote.audio import Pipeline
import gc 
import os
import torch
import pandas as pd
from pathlib import Path
torch.cuda.empty_cache()
gc.collect()
device = "cuda" 
audio_file = "audio.mp3"
video_filename = "THE Millennial Job Interview.mp4"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
# os.system(f"ffmpeg -i \"{video_filename}\" -vn -acodec mp3 {audio_file}")
# print(f"Audio saved to {audio_file}")
# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
model_dir = "C:\\Users\\rajpu\\Downloads\\Talent Sight\\backend"
model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"]) # before alignment

# # delete model if low on GPU resources

gc.collect()
torch.cuda.empty_cache() 
del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"]) # after alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a
def load_pipeline_from_pretrained(path_to_config: str | Path) -> Pipeline:
    path_to_config = Path(path_to_config)

    print(f"Loading pyannote pipeline from {path_to_config}...")
    # the paths in the config are relative to the current working directory
    # so we need to change the working directory to the model path
    # and then change it back

    cwd = Path.cwd().resolve()  # store current working directory

    # first .parent is the folder of the config, second .parent is the folder containing the 'models' folder
    cd_to = path_to_config.parent.parent.resolve()

    print(f"Changing working directory to {cd_to}")
    os.chdir(cd_to)

    pipeline = Pipeline.from_pretrained(path_to_config)

    print(f"Changing working directory back to {cwd}")
    os.chdir(cwd)

    return pipeline
# 3. Assign speaker labels
PATH_TO_CONFIG = "models/pyannote_diarization_config.yaml"
diarize_model = load_pipeline_from_pretrained(PATH_TO_CONFIG)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio_file)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
# Convert the Pyannote diarization output to WhisperX expected format
diarize_segments_list = []
for turn, _, speaker in diarize_segments.itertracks(yield_label=True):
    diarize_segments_list.append({
        'segment': turn,
        'speaker': speaker,
        'start': turn.start,
        'end': turn.end
    })
diarize_df = pd.DataFrame(diarize_segments_list)
result = whisperx.assign_word_speakers(diarize_df, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs

# Function to save transcript in speaker format
def save_transcript(result, output_file="transcript.txt"):
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
save_transcript(result)