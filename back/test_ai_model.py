import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from load_model import generate_phi3_response
from transcribe_audio import transcribe_audio

mock_response = "This is a generated response. It should be concise and limited to a few sentences."

def test_generate_phi3_response():
    prompt = "Hello, how are you?"
    with patch('load_model.generate_ollama_response', return_value=mock_response) as mock_generate:
        response = generate_phi3_response(prompt)
        assert response == mock_response

def test_generate_phi3_response_none():
    prompt = "Hello, how are you?"
    with patch('load_model.generate_ollama_response', return_value=None) as mock_generate:
        response = generate_phi3_response(prompt)
        assert response is None
        mock_generate.assert_called_once_with('phi3', prompt)

@pytest.mark.asyncio
async def test_transcribe_audio():
    file_path = "test_audio.wav"
    
    mock_segments = [MagicMock(text="Hello world")]
    mock_info = MagicMock()
    
    with patch('faster_whisper.WhisperModel.transcribe', return_value=(mock_segments, mock_info)) as mock_transcribe, \
         patch('metrics.log_metrics_transcription_time', return_value=0.1) as mock_log:
        
        start_time = time.time()
        response, transcription_time = await transcribe_audio(file_path)
        end_time = time.time()

        assert response == "Hello world"
        assert end_time - start_time >= transcription_time


if __name__ == '__main__':
    pytest.main()