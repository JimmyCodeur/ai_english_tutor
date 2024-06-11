from datetime import datetime

def log_conversation(prompt, response):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("./log/conversation_logs.txt", "a") as log_file:
        log_file.write(f"[{current_time}] Prompt: {prompt}\n")
        log_file.write(f"[{current_time}] Response: {response}\n\n")