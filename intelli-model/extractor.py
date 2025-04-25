import os
import textract
import speech_recognition as sr
from bs4 import BeautifulSoup
import requests
from typing import Optional

def extract_text(path):
    """
    Extracts text from various file formats (PDF, DOCX, TXT, SRT, MP4, MKV).
    """
    ext = os.path.splitext(path)[1].lower()
    
    if ext in ['.mp4', '.mkv']:
        audio_path = path.replace(ext, ".wav")
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)

    # If the file is a document (PDF, DOCX, TXT), extract text
    elif ext in ['.pdf', '.docx', '.txt']:
        return textract.process(path).decode("utf-8")
    
    # If the file is a subtitle file (SRT), return its contents
    elif ext == '.srt':
        with open(path, 'r') as f:
            return f.read()

    return ""

def extract_text_from_website(url: str) -> Optional[str]:
    """
    Extracts plain text from a given website URL.
    Returns None if extraction fails for any reason.
    """
    if not url or not isinstance(url, str):
        print("Invalid URL provided")
        return None
        
    try:
        response = requests.get(url)
        response.raise_for_status()  
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from common content elements
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article', 'section', 'div.content'])
        
        # If no content found with those tags, try getting all visible text
        if not paragraphs:
            # Alternative approach to get visible text
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator=' ')
            return ' '.join(text.split())
        
        text = ' '.join([p.get_text(separator=' ') for p in paragraphs])
        cleaned_text = ' '.join(text.split())  # Remove extra whitespace
        
        return cleaned_text if cleaned_text else None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error extracting text: {e}")
        return None