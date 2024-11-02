import psutil
import csv
from datetime import datetime
import asyncio
import traceback

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
            writer.writerow(["timestamp", "metric", "value", "gpu_usage", "cpu_usage", "memory_usage", "disk_usage"])
        writer.writerow(row)

def log_metrics_transcription_time(start_time, end_time):
    transcription_time = end_time - start_time
    gpu_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().used / (1024 ** 3)
    disk_usage = psutil.disk_usage('/').percent
    timestamp = datetime.now().isoformat()
    write_to_csv([timestamp, "Transcription Time", transcription_time, gpu_usage, cpu_usage, memory_usage, disk_usage])
    return transcription_time

def log_response_time_phi3(start_time, end_time):
    response_time = end_time - start_time
    gpu_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().used / (1024 ** 3)
    disk_usage = psutil.disk_usage('/').percent
    timestamp = datetime.now().isoformat()
    write_to_csv([timestamp, "Response Generation Time", response_time, gpu_usage, cpu_usage, memory_usage, disk_usage])
    return response_time

def log_total_time(start_time, end_time):
    total_time = end_time - start_time
    gpu_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().used / (1024 ** 3)
    disk_usage = psutil.disk_usage('/').percent
    timestamp = datetime.now().isoformat()
    write_to_csv([timestamp, "Total Processing Time", total_time, gpu_usage, cpu_usage, memory_usage, disk_usage])
    return total_time

async def log_custom_metric(metric_name, metric_value):
    gpu_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().used / (1024 ** 3)
    disk_usage = psutil.disk_usage('/').percent
    timestamp = datetime.now().isoformat()
    await asyncio.to_thread(write_to_csv, [timestamp, metric_name, metric_value, gpu_usage, cpu_usage, memory_usage, disk_usage])

def log_error(exception):
    timestamp = datetime.now().isoformat()
    error_message = f"Error: {str(exception)}"
    traceback_info = traceback.format_exc()
    write_to_csv([timestamp, "Error", error_message, "N/A", "N/A", "N/A", "N/A"])
    write_to_csv([timestamp, "Traceback", traceback_info, "N/A", "N/A", "N/A", "N/A"])
