import unittest
from unittest.mock import MagicMock, patch, mock_open
from unittest import mock
import numpy as np
from datetime import datetime, timedelta
import pytest
import os
import base64
from io import BytesIO
from fastapi import UploadFile
from back.audio_utils import (save_audio, save_user_audio, file_to_base64, 
                              delete_audio_file, process_audio_file, is_valid_audio_file, 
                              butter_lowpass, lowpass_filter)

from back.metrics import (write_to_csv, log_metrics_transcription_time,
                                 log_response_time_phi3, log_total_time, log_custom_metric)

class TestAudioFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists('./audio'):
            os.makedirs('./audio')
        if not os.path.exists('./audio/user'):
            os.makedirs('./audio/user')

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('./audio'):
            for root, dirs, files in os.walk('./audio', topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir('./audio')

    def test_save_audio(self):
        audio = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        file_path = './audio/test.wav'
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        save_audio(audio, file_path)
        self.assertTrue(os.path.exists(file_path))
        
        os.remove(file_path)

    @pytest.mark.asyncio
    @patch("aiofiles.open", new_callable=mock_open)
    async def test_save_user_audio(mock_file):
        mock_upload = MagicMock(spec=UploadFile)
        mock_upload.read.return_value = b"Fake audio data"
        mock_upload.filename = "test.wav"

        file_path = await save_user_audio(mock_upload)
        
        mock_file.assert_called_once_with(file_path, "wb")
        mock_file().write.assert_called_once_with(b"Fake audio data")
        
        os.remove(file_path)

    def test_file_to_base64(self):
        audio_data = b"Fake audio data"
        with patch("builtins.open", mock_open(read_data=audio_data)) as mock_file:
            base64_str = file_to_base64("fake_path.wav")
            mock_file.assert_called_once_with("fake_path.wav", "rb")
            self.assertEqual(base64.b64encode(audio_data).decode("utf-8"), base64_str)

    @patch("os.remove")
    @patch("os.path.exists", return_value=True)
    def test_delete_audio_file(self, mock_exists, mock_remove):
        file_path = "test.wav"
        delete_audio_file(file_path)
        mock_remove.assert_called_once_with(file_path)
        
    @patch("os.path.exists", return_value=False)
    def test_delete_audio_file_not_exists(self, mock_exists):
        file_path = "test.wav"
        with patch("builtins.print") as mock_print:
            delete_audio_file(file_path)
            mock_print.assert_called_once_with(f"File {file_path} does not exist.")

    @patch("subprocess.run")
    @patch("soundfile.read")
    @patch("soundfile.write")
    def test_process_audio_file(self, mock_write, mock_read, mock_run):
        mock_read.return_value = (np.zeros(44100), 44100)
        audio_path = "./audio/test.wav"
        filename = "test.wav"
        output_path = process_audio_file(audio_path, filename)
        mock_run.assert_called_once_with(['ffmpeg', '-i', audio_path, '-acodec', 'pcm_s16le', '-ar', '44100', './audio/user/converted_test.wav'])
        mock_write.assert_called_once()
        self.assertEqual(output_path, "./audio/user/denoised_converted_test.wav")

    def test_is_valid_audio_file(self):
        mock_upload = MagicMock(spec=UploadFile)
        mock_upload.filename = "test.wav"
        self.assertTrue(is_valid_audio_file(mock_upload))

        mock_upload.filename = "test.txt"
        self.assertFalse(is_valid_audio_file(mock_upload))

    def test_butter_lowpass(self):
        cutoff = 1000
        fs = 44100
        order = 6
        b, a = butter_lowpass(cutoff, fs, order)
        self.assertEqual(len(b), order + 1)
        self.assertEqual(len(a), order + 1)

    def test_lowpass_filter(self):
        data = np.random.randn(1000)
        cutoff = 1000
        fs = 44100
        order = 6
        filtered_data = lowpass_filter(data, cutoff, fs, order)
        self.assertEqual(len(filtered_data), len(data))

class TestMetricsLogging(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open)
    @patch("csv.writer")
    def test_write_to_csv(self, mock_csv_writer, mock_file):
        row = ["2023-01-01T12:00:00", "test_metric", 0.123, 50, 10]
        mock_file.side_effect = [mock_open(read_data="").return_value, mock_open().return_value]
        write_to_csv(row)
        mock_file.assert_any_call('metrics.csv', mode='a', newline='')
        mock_csv_writer().writerow.assert_called_with(row)
        
    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("builtins.open", new_callable=mock_open)
    @patch("csv.writer")
    def test_log_metrics_transcription_time(self, mock_csv_writer, mock_file, mock_cpu, mock_memory):
        mock_cpu.return_value = 10
        mock_memory.return_value.percent = 50
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=1)
        transcription_time = log_metrics_transcription_time(start_time, end_time)
        self.assertEqual(transcription_time, timedelta(seconds=1))
        mock_csv_writer().writerow.assert_called_with([mock.ANY, "def Transcription faster whisper time", transcription_time, 50, 10])

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("builtins.open", new_callable=mock_open)
    @patch("csv.writer")
    def test_log_response_time_phi3(self, mock_csv_writer, mock_file, mock_cpu, mock_memory):
        mock_cpu.return_value = 10
        mock_memory.return_value.percent = 50
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=2)
        response_time = log_response_time_phi3(start_time, end_time)
        self.assertEqual(response_time, timedelta(seconds=2))
        mock_csv_writer().writerow.assert_called_with([mock.ANY, "def Phi 3 response generation time", response_time, 50, 10])

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("builtins.open", new_callable=mock_open)
    @patch("csv.writer")
    def test_log_total_time(self, mock_csv_writer, mock_file, mock_cpu, mock_memory):
        mock_cpu.return_value = 10
        mock_memory.return_value.percent = 50
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=3)
        total_time = log_total_time(start_time, end_time)
        self.assertEqual(total_time, timedelta(seconds=3))
        mock_csv_writer().writerow.assert_any_call([mock.ANY, "def Total processing time", total_time, 50, 10])

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("back.metrics_logger.write_to_csv")
    @patch("asyncio.to_thread")
    @patch("datetime.datetime")
    @pytest.mark.asyncio
    async def test_log_custom_metric(self, mock_datetime, mock_to_thread, mock_write_to_csv, mock_cpu, mock_memory):
        mock_cpu.return_value = 10
        mock_memory.return_value.percent = 50
        mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)
        metric_name = "Custom metric"
        metric_value = 123
        await log_custom_metric(metric_name, metric_value)
        mock_to_thread.assert_awaited_once_with(write_to_csv, ["2023-01-01T12:00:00", metric_name, metric_value, 50, 10])


if __name__ == '__main__':
    unittest.main()
