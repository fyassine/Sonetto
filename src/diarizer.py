import os
import io
import json
import time
import azure.cognitiveservices.speech as speechsdk
from flask import jsonify
import requests  # Add this import for REST API calls
import groq

# Import will happen at runtime to avoid circular imports
# from relay import sessions, AZURE_SPEECH_KEY, AZURE_SPEECH_REGION

def process_audio_file(audio_file_path):
    """Process audio file for transcription.
    
    Args:
        audio_file_path: Path to the audio file or raw audio data
        
    Returns:
        Processed audio file object ready for transcription
    """
    
    try:
        audio_file = None
        # Check if input is a file path or raw audio data
        if isinstance(audio_file_path, str) and os.path.exists(audio_file_path):
            # Input is a file path
            with open(audio_file_path, 'rb') as f:
                audio_file = io.BytesIO(f.read())
                audio_file.name = os.path.basename(audio_file_path)
        else:
            # Input is raw audio data
            audio_file = io.BytesIO(audio_file_path)
            audio_file.name = 'audio.wav'

        return audio_file

    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        return None

def transcribe_all_speakers(audio_file_path):
    """Transcribe audio and identify different speakers using REST API.
    
    Args:
        audio_file_path: Path to the audio file to transcribe
        
    Returns:
        Dictionary mapping speaker IDs to lists of transcribed text segments
    """
    # Import the Azure Speech credentials from relay.py at runtime
    # to avoid circular imports
    from relay import AZURE_SPEECH_KEY, AZURE_SPEECH_REGION
    
    try:
        # Dictionary to store speaker-specific transcriptions
        speaker_transcriptions = {}
        
        print("Using REST API for speech recognition...")
        with open(audio_file_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        
        # REST API endpoint and headers
        endpoint = f"https://{AZURE_SPEECH_REGION}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
            "Content-Type": "audio/wav",
            "Accept": "application/json"
        }
        params = {
            "language": "en-US"
        }
        
        # Send POST request to REST API
        response = requests.post(endpoint, headers=headers, params=params, data=audio_data)
        response.raise_for_status()
        result = response.json()
        
        # Extract recognized text
        if "DisplayText" in result:
            recognized_text = result["DisplayText"]
            print(f"REST API recognized: {recognized_text}")
            speaker_transcriptions["speaker_1"] = [recognized_text]
        else:
            print("REST API did not return recognized text.")
            speaker_transcriptions["speaker_1"] = [""]
        
        return speaker_transcriptions

    except Exception as e:
        print(f"Error during speech recognition: {str(e)}")
        # Return an empty dictionary instead of a Flask response
        return {"speaker_1": [f"Error: {str(e)}"]}

def pick_relevant_speaker(audio_file_path, session_id=None):
    """Diarize audio and identify the most relevant speaker.
    
    Args:
        audio_file_path: Path to the audio file to process
        session_id: Optional session ID to retrieve session information
        
    Returns:
        The relevant speaker information or None if no relevant speaker was found
    """
    # Import the sessions dictionary from relay.py at runtime
    # to avoid circular imports
    from relay import sessions
    
    # Get speaker transcriptions using Azure Speech SDK
    speaker_transcriptions = transcribe_all_speakers(audio_file_path)
        
    # Format results and identify the relevant speaker
    diarized_text = []
    relevant_speaker = None

    # Analyze each speaker's text to find the most relevant one
    for speaker_id, texts in speaker_transcriptions.items():
        combined_text = " ".join(texts)
        
        # Skip empty text
        if not combined_text or combined_text.strip() == "":
            print(f"Speaker {speaker_id} has empty text, skipping analysis")
            speaker_entry = {
                "speaker_id": speaker_id,
                "text": combined_text,
                "is_ordering_food": False,
                "order_details": ""
            }
            diarized_text.append(speaker_entry)
            continue
        
        # Use Groq to analyze if this speaker is ordering food
        prompt = f"""Analyze this text and determine if the speaker is ordering food. 
        Return a JSON with two fields:
        - is_ordering_food (boolean): true if the speaker is clearly ordering food
        - order_details (string): if is_ordering_food is true, extract the order details
        
        Text to analyze: {combined_text}"""
        
        try:
            client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
            )
            
            # Add robust JSON parsing with fallback
            try:
                # First try to find JSON in the response if it's not a clean JSON
                import re
                content = completion.choices[0].message.content
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    analysis = json.loads(json_str)
                else:
                    analysis = json.loads(content)
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Invalid JSON from Groq API for speaker {speaker_id}: {e}. Using default values.")
                analysis = {"is_ordering_food": False, "order_details": ""}
            
            speaker_entry = {
                "speaker_id": speaker_id,
                "text": combined_text,
                "is_ordering_food": analysis.get("is_ordering_food", False),
                "order_details": analysis.get("order_details", "")
            }
            
            diarized_text.append(speaker_entry)
            print(f"Speaker {speaker_id}: {speaker_entry}")
            
            if analysis.get("is_ordering_food", False):
                relevant_speaker = speaker_entry
                print(f"Found food ordering speaker: {speaker_id}")
                break
            
        except Exception as e:
            print(f"Error analyzing speaker {speaker_id}: {str(e)}")
            diarized_text.append({
                "speaker_id": speaker_id,
                "text": combined_text,
                "is_ordering_food": False,
                "order_details": ""
            })
    
    # If no relevant speaker was found but we have speakers with non-empty text, use the first one with non-empty text
    if not relevant_speaker and diarized_text:
        for speaker in diarized_text:
            if speaker["text"] and speaker["text"].strip() != "":
                relevant_speaker = speaker
                print(f"No ordering speaker found, using speaker with non-empty text: {speaker['speaker_id']}")
                break
    
    # If still no relevant speaker, use the first one regardless of text
    if not relevant_speaker and diarized_text:
        relevant_speaker = diarized_text[0]
        print(f"No speaker with non-empty text found, using first speaker: {relevant_speaker['speaker_id']}")
        
    # If we still don't have a relevant speaker, create a default one
    if not relevant_speaker:
        relevant_speaker = {
            "speaker_id": "default_speaker",
            "text": "No speech detected"
        }
        diarized_text.append(relevant_speaker)
        print("No speakers detected, using default speaker")
    
    # If session_id is provided and valid, send the results via websocket
    if session_id and session_id in sessions:
        ws = sessions[session_id].get("websocket")
        if ws:
            message = {
                "event": "recognized",
                "diarized_text": diarized_text,
                "relevant_speaker": relevant_speaker,
                "language": sessions[session_id]["language"]
            }
            ws.send(json.dumps(message))
            print(f"Sent diarized text with {len(diarized_text)} speakers via websocket")
    
    return relevant_speaker