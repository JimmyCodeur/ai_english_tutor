import subprocess

command = "tts --list_models"
result = subprocess.run(command, shell=True, capture_output=True, text=True)
print(result.stdout)
