import subprocess

# Commande à exécuter
command = "tts --list_models"

# Exécution de la commande et capture de la sortie
result = subprocess.run(command, shell=True, capture_output=True, text=True)

# Affichage des résultats
print(result.stdout)
