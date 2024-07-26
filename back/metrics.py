import psutil
import csv
from datetime import datetime
import asyncio

def write_to_csv(row, filename='metrics.csv'):
    file_exists = False
    try:
        with open(filename, 'r') as csvfile:
            file_exists = True
    except FileNotFoundError:
        file_exists = False
    
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "metric", "value", "gpu_usage", "cpu_usage"])
        writer.writerow(row)

def log_metrics_transcription_time(start_time, end_time):
    transcription_time = end_time - start_time
    gpu_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    timestamp = datetime.now().isoformat()
    write_to_csv([timestamp, "def Transcription faster whisper time", transcription_time, gpu_usage, cpu_usage])
    return transcription_time

def log_response_time_phi3(start_time, end_time):
    response_time = end_time - start_time
    gpu_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    timestamp = datetime.now().isoformat()
    write_to_csv([timestamp, "def Phi 3 response generation time", response_time, gpu_usage, cpu_usage])
    return response_time

def log_total_time(start_time, end_time):
    total_time = end_time - start_time
    gpu_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    timestamp = datetime.now().isoformat()
    write_to_csv([timestamp, "def Total processing time", total_time, gpu_usage, cpu_usage])
    write_to_csv("\n")
    return total_time

async def log_custom_metric(metric_name, metric_value):
    gpu_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    timestamp = datetime.now().isoformat()
    await asyncio.to_thread(write_to_csv, [timestamp, metric_name, metric_value, gpu_usage, cpu_usage])
