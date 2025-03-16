import re
import json
import yaml
import spacy
import requests
import argparse
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
import os
import logging

# Add near the top of the file
logging.basicConfig(filename='debug_responses.log', level=logging.DEBUG, 
                   format='%(asctime)s - %(message)s')

# Load the English NLP model for named entity recognition
# At the beginning of your prompt.py file:
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        print("Installing required spaCy model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"], check=True)
        nlp = spacy.load("en_core_web_md")
except ImportError:
    print("spaCy not found. Installing spaCy and required model...")
    import subprocess
    subprocess.run(["pip", "install", "spacy"], check=True)
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"], check=True)
    import spacy
    nlp = spacy.load("en_core_web_md")


# Default job description
DEFAULT_JOB_DESCRIPTION = """
Data Analyst Position
Requirements:
- 3+ years of experience in data analysis
- Proficiency in SQL, Python, and data visualization tools
- Experience with statistical analysis and predictive modeling
- Strong problem-solving and critical thinking skills
- Excellent communication skills, with ability to present complex findings to non-technical audiences
- Bachelor's degree in Statistics, Mathematics, Computer Science, or related field

Responsibilities:
- Analyze large datasets to identify trends and insights
- Create and maintain dashboards and reports
- Collaborate with cross-functional teams to support data-driven decisions
- Develop and implement data quality protocols
- Present findings and recommendations to stakeholders
"""

# Auth functions
def load_config():
    """Load configuration from config.yaml file"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def test_auth():
    """Test authentication with AnythingLLM server"""
    config = load_config()
    
    # Construct the auth URL
    auth_url = f"{config['model_server_base_url']}/auth"
    
    # Set up the headers with API key
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json"
    }
    
    # Make the request
    response = requests.get(auth_url, headers=headers)
    
    # Check if authentication was successful
    if response.status_code == 200:
        # Use ASCII instead of Unicode emoji to avoid encoding issues
        print("✓ Authentication successful")
        return True
    else:
        # Use ASCII instead of Unicode emoji
        print(f"✗ Authentication failed: {response.status_code}")
        print(response.text)
        return False

# PII redaction functions
def identify_pii(text):
    """Identify PII using regex patterns and NER."""
    # Process text with spaCy for named entity recognition
    doc = nlp(text)

    # Create dictionary to store PII with their positions
    pii_dict = {}

    # Find speaker labels (Interviewer:, Candidate:, etc.) to exclude them
    speaker_pattern = r'^\s*(Interviewer|Candidate|Applicant|Interviewee|Speaker)\s*:'
    speaker_positions = []
    for match in re.finditer(speaker_pattern, text, re.MULTILINE):
        speaker_positions.append((match.start(), match.end()))

    # Extract named entities from spaCy
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "FAC"]:
            # Check if this entity overlaps with any speaker label
            entity_positions = [(m.start(), m.end()) for m in re.finditer(re.escape(ent.text), text)]
            valid_positions = []

            for start, end in entity_positions:
                is_speaker = False
                for sp_start, sp_end in speaker_positions:
                    # If there's overlap with a speaker position, skip this position
                    if not (end <= sp_start or start >= sp_end):
                        is_speaker = True
                        break
                if not is_speaker:
                    valid_positions.append((start, end))

            if valid_positions:  # Only add if we have valid positions
                pii_dict[ent.text] = {
                    "type": ent.label_,
                    "positions": valid_positions
                }

    # Regex patterns for common PII
    patterns = {
        # Company names (look for common suffixes)
        "COMPANY": r'\b[A-Z][a-zA-Z0-9\s]*\s(?:Inc|LLC|Ltd|Corp|Corporation|Company)\b',
        # Company names without suffixes but with distinctive patterns
        "POTENTIAL_COMPANY": r'\b(?:[A-Z][a-zA-Z0-9]+){2,}(?:\s[A-Z][a-zA-Z0-9]+)*\b',
        # Money amounts
        "MONEY": r'\$[\d,.]+\s?(?:million|thousand|billion)?|\b\d+(?:\.\d+)?\s?(?:million|thousand|billion)?\s?dollars\b',
        # Percentages
        "PERCENTAGE": r'\b\d+(?:\.\d+)?%\b',
        # Specific dates
        "DATE": r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
        # Years of experience
        "YEARS": r'\b\d+\+?\s+years?\b',
        # Email addresses
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        # Phone numbers
        "PHONE": r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
    }

    # Find matches for each pattern
    for pii_type, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            matched_text = match.group()
            start, end = match.start(), match.end()

            # Skip if this matches a speaker label
            is_speaker = False
            for sp_start, sp_end in speaker_positions:
                if not (end <= sp_start or start >= sp_end):
                    is_speaker = True
                    break
            if is_speaker:
                continue

            # Check if this isn't already captured by spaCy
            if not any(matched_text in key for key in pii_dict.keys()):
                if matched_text not in pii_dict:
                    pii_dict[matched_text] = {
                        "type": pii_type,
                        "positions": []
                    }
                pii_dict[matched_text]["positions"].append((start, end))

    return pii_dict

def redact_transcript(text, pii_dict):
    """Create a redacted version of the transcript."""
    # Create a list of characters to build the redacted text
    chars = list(text)

    # Sort PII positions from end to beginning to avoid index shifting
    all_positions = []
    for pii, info in pii_dict.items():
        for pos in info["positions"]:
            all_positions.append((pos[0], pos[1], info["type"]))

    all_positions.sort(reverse=True)

    # Replace PII with redacted markers
    for start, end, pii_type in all_positions:
        redacted_text = f"[{pii_type}]"
        chars[start:end] = list(redacted_text)

    return ''.join(chars)

def process_transcript(text):
    """Process transcript text and return redacted version."""
    # Identify PII
    pii_dict = identify_pii(text)

    # Generate redacted transcript
    redacted_transcript = redact_transcript(text, pii_dict)

    # Return the redacted text
    return redacted_transcript

def load_transcript(file_path):
    """Load transcript from file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Interview evaluation functions
def extract_responses(transcript):
    """Extract candidate responses from the transcript with more lenient parsing"""
    responses = []
    lines = transcript.split('\n')
    current_response = ""
    in_candidate_response = False  # Flag to track if we're in a candidate response
    
    logging.debug("Starting to extract responses from transcript of length: %d", len(transcript))
    logging.debug("First 300 chars of transcript: %s", transcript[:300] if len(transcript) > 300 else transcript)
    
    # Define more patterns for identifying speakers
    interviewer_patterns = [
        r'^(?i)\s*interviewer\s*:', 
        r'^(?i)\s*question\s*:',
        r'^(?i)\s*interviewer\s*\d*\s*:'
    ]
    
    candidate_patterns = [
        r'^(?i)\s*candidate\s*:', 
        r'^(?i)\s*applicant\s*:', 
        r'^(?i)\s*interviewee\s*:', 
        r'^(?i)\s*speaker\s*:',
        r'^(?i)\s*answer\s*:',
        r'^(?i)\s*response\s*:',
        r'^(?i)\s*candidate\s*\d*\s*:'
    ]
    
    # Function to check if a line matches any pattern in a list
    def matches_pattern(line, patterns):
        return any(re.match(pattern, line) for pattern in patterns)
    
    # Try to determine the speaker format from the transcript
    sample_lines = [line for line in lines[:20] if line.strip()]  # First 20 non-empty lines
    logging.debug("Sample lines for pattern detection: %s", sample_lines[:5])
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Log every line being processed for debugging
        logging.debug("Processing line: %s", line[:50] if len(line) > 50 else line)
        
        # Check if this is an interviewer line
        if matches_pattern(line, interviewer_patterns):
            in_candidate_response = False
            logging.debug("Found interviewer line")
            continue
        
        # Check if this is a candidate line
        if matches_pattern(line, candidate_patterns):
            # If we were already in a candidate response, save the previous one
            if current_response:
                responses.append(current_response.strip())
                logging.debug("Added response: %s", current_response[:50] + "..." if len(current_response) > 50 else current_response)
            
            # Start a new response with the text after the colon
            colon_pos = line.find(':')
            if colon_pos != -1:
                current_response = line[colon_pos + 1:].strip()
            else:
                current_response = line.strip()  # Fallback if no colon found
            
            in_candidate_response = True
            logging.debug("Started new response: %s", current_response[:50] + "..." if len(current_response) > 50 else current_response)
            continue
        
        # If we're in the middle of a candidate response and this isn't a speaker line, add it
        if in_candidate_response:
            current_response += " " + line
            continue
        
        # If we reach this point, we have a line without a clear speaker prefix
        # As a fallback, treat alternating speakers, where interviewer starts
        if i > 0 and len(responses) == 0 and not current_response:
            # If we haven't found any responses yet using the prefixes, try a different approach
            if i % 2 == 1:  # Odd lines could be candidate responses (assuming interviewer starts)
                if current_response:
                    current_response += " " + line
                else:
                    current_response = line
                    in_candidate_response = True
            elif current_response:  # End of a candidate response
                responses.append(current_response.strip())
                logging.debug("Added response from alternating pattern: %s", 
                             current_response[:50] + "..." if len(current_response) > 50 else current_response)
                current_response = ""
                in_candidate_response = False
    
    # Add the last response if there is one
    if current_response:
        responses.append(current_response.strip())
        logging.debug("Added final response: %s", current_response[:50] + "..." if len(current_response) > 50 else current_response)

    # Super fallback: if still no responses found, just take everything as one response
    if len(responses) == 0 and transcript.strip():
        logging.debug("No structured responses found, using entire transcript as one response")
        responses.append(transcript.strip())
    
    logging.debug("Extracted %d responses total", len(responses))
    
    # Debug: Write the responses to a file
    with open('extracted_responses.txt', 'w', encoding='utf-8') as f:
        f.write(f"Found {len(responses)} responses:\n\n")
        for i, response in enumerate(responses):
            f.write(f"Response {i+1}:\n{response}\n\n")
    
    return responses

def create_evaluation_prompt(transcript, job_description):
    """Create a prompt for Llama to evaluate the candidate"""
    # Extract candidate responses
    responses = extract_responses(transcript)
    
    # Debug print to show extracted responses
    print("\n=== EXTRACTED RESPONSES ===")
    for i, response in enumerate(responses):
        print(f"Response {i+1}: {response[:100]}..." if len(response) > 100 else f"Response {i+1}: {response}")
    
    # Join responses for prompt
    responses_text = "\n\n".join(responses)
    
    prompt = f"""
You are an interview evaluation expert. You need to evaluate a candidate's interview responses for a job.

Job Description:
{job_description}

Candidate Responses:
{responses_text}

Evaluate the candidate using the STAR method (Situation, Task, Action, Result) and the Three Cs (Credibility, Competence, Confidence).

For each of these criteria, provide a score out of 10 and specific feedback.

Format your response as a JSON object with the following structure:
{{
  "STAR_Method_Scores": {{
    "Situation": "score/10",
    "Task": "score/10",
    "Action": "score/10",
    "Result": "score/10",
    "Average": "average/10"
  }},
  "Three_Cs_Scores": {{
    "Credibility": "score/10",
    "Competence": "score/10",
    "Confidence": "score/10",
    "Average": "average/10"
  }},
  "Overall_Score": "overall/10",
  "Feedback": {{
    "Overall_Evaluation_Score": "overall/10",
    "STAR_Method_Analysis": {{
      "Strengths": [
        "List of strengths in STAR method"
      ],
      "Areas_for_Improvement": [
        "List of areas for improvement in STAR method"
      ]
    }},
    "Three_Cs_Analysis": {{
      "Strengths": [
        "List of strengths in Three Cs"
      ],
      "Areas_for_Improvement": [
        "List of areas for improvement in Three Cs"
      ]
    }},
    "Summary": [
      "Overall summary points about the candidate"
    ]
  }}
}}

Your response must be a valid JSON object exactly matching the structure above. Do not include any text outside the JSON object.
"""
    return prompt

def send_prompt_to_llama(prompt_text):
    """Send a prompt to the model using direct chat mode"""
    # First test authentication
    if not test_auth():
        return {"error": "Authentication failed. Check your API key and server URL.", "raw_response": None}
    
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
    
    # Prepare the data with mode set to "chat"
    data = {
        "message": prompt_text,
        "mode": "chat",
        "sessionId": "interview-evaluation-session",
        "attachments": [],
        "history": []  # No history needed for single prompt
    }
    
    # Function to repair common JSON formatting issues
    def repair_json(json_str):
        """Attempt to repair common JSON formatting issues"""
        # Replace score formats like 8.5/10 with "8.5/10"
        json_str = re.sub(r'(\d+(?:\.\d+)?)/10', r'"\1/10"', json_str)
        
        # Fix pluralization issue with STAR_Method_Score -> STAR_Method_Scores
        json_str = json_str.replace('"STAR_Method_Score"', '"STAR_Method_Scores"')
        
        # Fix missing commas between properties
        json_str = re.sub(r'"\s*}\s*"', '", "', json_str)
        json_str = re.sub(r'"\s*{\s*"', '", "', json_str)
        json_str = re.sub(r'(\d+)\s*"', r'\1, "', json_str)
        
        # Fix property names without quotes
        json_str = re.sub(r'{\s*(\w+)\s*:', r'{ "\1":', json_str)
        json_str = re.sub(r',\s*(\w+)\s*:', r', "\1":', json_str)
        
        # Fix missing commas after values
        json_str = re.sub(r'(true|false|null|\d+|\"\w+\")\s*"', r'\1, "', json_str)
        
        # Fix trailing commas in arrays and objects
        json_str = re.sub(r',\s*]', r']', json_str)
        json_str = re.sub(r',\s*}', r'}', json_str)
        
        # Fix missing quotes around property values
        json_str = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,}])', r': "\1"\2', json_str)
        
        return json_str

    
    # Make the request with increased timeout
    try:
        print("Sending request to LLM server...")
        response = requests.post(chat_url, headers=headers, json=data, timeout=300)
        
        if response.status_code == 200:
            print(f"Received response with status code {response.status_code}")
            response_json = response.json()
            
            # Extract text response based on direct chat endpoint format
            response_text = response_json.get('textResponse', '')
            
            # Debug information
            print(f"Response length: {len(response_text)}")
            if not response_text:
                print("Warning: Received empty response from LLM server")
                return {"error": "Empty response from LLM server", "raw_response": response_json}
            
            # Print first 100 characters of the response for debugging
            print(f"Response preview: {response_text[:100]}...")
            
            # Save raw response for debugging
            print(f"Full raw response: {response_text}")
            
            # Try multiple approaches to extract and fix JSON
            
            # 1. First try to find JSON with regex
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                try:
                    # Try parsing as is
                    evaluation = json.loads(json_str)
                    # Add raw response to the result dictionary
                    evaluation['raw_response'] = response_text
                    return evaluation
                except json.JSONDecodeError as e:
                    # Try repairing common issues
                    try:
                        repaired_json = repair_json(json_str)
                        evaluation = json.loads(repaired_json)
                        # Add raw response to the result dictionary
                        evaluation['raw_response'] = response_text
                        return evaluation
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON after repair: {repaired_json}")
                        # Continue to next approaches
            
            # 2. Try using a more lenient JSON parser
            try:
                import demjson3
                evaluation = demjson3.decode(json_str)
                # Add raw response to the result dictionary
                evaluation['raw_response'] = response_text
                return evaluation
            except (ImportError, NameError, Exception):
                # demjson3 not available or other error
                pass
                
            # 3. Last resort - extract and repair the JSON structure manually
            # Extract all key-value pairs and reconstruct
            try:
                # This is a simplified approach - you may need more sophisticated parsing
                key_value_pattern = r'"([^"]+)"\s*:\s*("([^"]+)"|(\d+(?:\.\d+)?)|(\{.*?\})|(\[.*?\])|true|false|null)'
                matches = re.findall(key_value_pattern, response_text)
                if matches:
                    reconstructed_json = "{"
                    for i, match in enumerate(matches):
                        key = match[0]
                        value = match[1]
                        reconstructed_json += f'"{key}": {value}'
                        if i < len(matches) - 1:
                            reconstructed_json += ", "
                    reconstructed_json += "}"
                    
                    try:
                        evaluation = json.loads(reconstructed_json)
                        # Add raw response to the result dictionary
                        evaluation['raw_response'] = response_text
                        return evaluation
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                print(f"Error during manual JSON reconstruction: {str(e)}")
            
            # If all approaches fail, return error
            return {"error": "Failed to parse JSON response", "raw_response": response_text}

        else:
            print(f"Request failed with status {response.status_code}")
            return {"error": f"Request failed with status {response.status_code}", "raw_response": response.text}
    except Exception as e:
        print(f"Exception during request: {str(e)}")
        return {"error": f"Request failed: {str(e)}", "raw_response": "Exception occurred"}

def evaluate_interview(transcript, job_description=DEFAULT_JOB_DESCRIPTION):
    """Evaluate an interview transcript using Llama 1 8B"""
    # Redact PII from transcript
    redacted_transcript = process_transcript(transcript)
    
    # Create prompt for evaluation
    evaluation_prompt = create_evaluation_prompt(redacted_transcript, job_description)
    
    # Send to Llama for evaluation
    llama_response = send_prompt_to_llama(evaluation_prompt)
    
    # The llama_response is already a dictionary, so we can return it directly
    # No need to parse JSON from a string
    return llama_response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TalentSight Interview Evaluation System")
    parser.add_argument("--transcript", help="Path to interview transcript file")
    parser.add_argument("--job_description", help="Path to job description file")
    parser.add_argument("--output", help="Path to save evaluation results")
    parser.add_argument("--redact_only", action="store_true", help="Only redact PII without evaluation")
    
    args = parser.parse_args()
    
    # Load transcript
    if args.transcript:
        transcript = load_transcript(args.transcript)
    else:
        # Example transcript for testing
        transcript = """
Interviewer: Can you tell me about a time when you had to analyze a complex dataset?

Candidate: At DataTech Solutions, I was faced with a challenging project analyzing customer churn data spanning 3 years with over 50 variables and 200,000 records. The dataset had numerous missing values and inconsistencies, making it particularly difficult to work with. My responsibility was to identify the key factors driving customer attrition and develop actionable recommendations to reduce it. I approached this methodically by first cleaning the data using Python and pandas, implementing imputation techniques for missing values. I then conducted exploratory data analysis using visualization tools like Matplotlib and Seaborn to identify patterns. I applied statistical methods including correlation analysis and chi-square tests, followed by machine learning algorithms like logistic regression and random forests to build predictive models. This analysis revealed three primary factors accounting for 70% of our churn: service outages, price increases, and lack of customer engagement after the first 90 days. Based on these findings, I developed a targeted retention program that reduced churn by 23% in the following quarter, which translated to approximately $1.2 million in saved annual recurring revenue.
"""
    
    # Load job description if provided
    if args.job_description:
        with open(args.job_description, 'r', encoding='utf-8') as f:
            job_description = f.read()
    else:
        job_description = DEFAULT_JOB_DESCRIPTION
    
    # Process transcript (redact PII)
    redacted_transcript = process_transcript(transcript)
    print("\n=== REDACTED TRANSCRIPT ===")
    print(redacted_transcript)
    
    # Skip evaluation if redact_only flag is set
    if not args.redact_only:
        print("\n=== EVALUATING CANDIDATE ===")
        
        # Debug print for extracted responses
        print("\n=== EXTRACTED RESPONSES ===")
        responses = extract_responses(transcript)
        for i, response in enumerate(responses):
            print(f"Response {i+1}: {response[:100]}..." if len(response) > 100 else f"Response {i+1}: {response}")
        
        evaluation = evaluate_interview(transcript, job_description)
        
        # Print evaluation
        print("\n=== EVALUATION RESULTS ===")
        
        # Print the raw response in all cases
        print("\n=== RAW RESPONSE ===")
        print(evaluation.get('raw_response', 'No raw response available'))
        
        # Remove raw_response from the printed JSON for cleaner output
        eval_for_display = evaluation.copy()
        if 'raw_response' in eval_for_display:
            del eval_for_display['raw_response']
        
        # Print formatted JSON result
        print("\n=== PARSED JSON RESULT ===")
        print(json.dumps(eval_for_display, indent=2))
        
        # Save results if output path provided
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(evaluation, f, indent=2)
            print(f"\nResults saved to {args.output}")
